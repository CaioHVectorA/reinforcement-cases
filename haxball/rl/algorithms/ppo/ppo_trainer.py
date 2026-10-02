"""
Proximal Policy Optimization (PPO) algorithm for HaxBall RL.
Supports decoupled observation space, continuous/discrete actions,
Generalized Advantage Estimation (GAE), and checkpoint management.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.rl.models.mlp_policy import ActorCriticMLP
from haxball.rl.models.entity_attention import EntityAttentionPolicy

class PPOTrainer:
    def __init__(
        self,
        env: HaxBallEnv,
        policy_type: str = "mlp",  # "mlp" or "attention"
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_steps: int = 2048,
        batch_size: int = 64,
        update_epochs: int = 10,
        device: str = "cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.device = torch.device(device)

        is_discrete = hasattr(env.action_space, "n")
        act_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

        if policy_type == "attention":
            self.agent = EntityAttentionPolicy(
                embed_dim=64, num_heads=4, act_dim=act_dim, is_discrete=is_discrete
            ).to(self.device)
        else:
            self.agent = ActorCriticMLP(
                obs_dim=env.observation_space.shape[0],
                act_dim=act_dim,
                is_discrete=is_discrete,
                hidden_dim=128
            ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

    def train_iteration(self) -> Dict[str, float]:
        """Performs one rollout and PPO update."""
        obs_dim = self.env.observation_space.shape[0]
        is_discrete = hasattr(self.env.action_space, "n")
        act_shape = () if is_discrete else (self.env.action_space.shape[0],)

        obs_buffer = torch.zeros((self.num_steps, obs_dim), device=self.device)
        actions_buffer = torch.zeros((self.num_steps, *act_shape), device=self.device)
        logprobs_buffer = torch.zeros(self.num_steps, device=self.device)
        rewards_buffer = torch.zeros(self.num_steps, device=self.device)
        dones_buffer = torch.zeros(self.num_steps, device=self.device)
        values_buffer = torch.zeros(self.num_steps, device=self.device)

        obs, _ = self.env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        episode_rewards = []
        episode_win_count = 0
        episode_count = 0
        cur_reward = 0.0

        for step in range(self.num_steps):
            obs_buffer[step] = obs_t

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs_t.unsqueeze(0))
                values_buffer[step] = value.flatten()

            actions_buffer[step] = action.squeeze(0)
            logprobs_buffer[step] = logprob.squeeze(0)

            act_input = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = self.env.step(act_input)
            done = terminated or truncated

            rewards_buffer[step] = reward
            dones_buffer[step] = float(done)
            cur_reward += reward

            if done:
                episode_rewards.append(cur_reward)
                episode_count += 1
                if info.get("winner") == 1:
                    episode_win_count += 1
                cur_reward = 0.0
                next_obs, _ = self.env.reset()

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        # GAE calculation
        with torch.no_grad():
            next_value = self.agent.get_value(obs_t.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer, device=self.device)
            lastgaelam = 0.0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + self.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        # PPO optimization epochs
        b_inds = np.arange(self.num_steps)
        total_loss_accum = 0.0
        num_batches = 0

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.num_steps, self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    obs_buffer[mb_inds], actions_buffer[mb_inds]
                )
                logratio = newlogprob - logprobs_buffer[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += loss.item()
                num_batches += 1

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        win_rate = (episode_win_count / episode_count * 100) if episode_count else 0.0

        return {
            "mean_reward": mean_reward,
            "win_rate": win_rate,
            "loss": total_loss_accum / max(1, num_batches),
            "episodes": episode_count
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.agent.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
