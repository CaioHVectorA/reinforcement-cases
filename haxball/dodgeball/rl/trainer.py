"""
Multi-Agent PPO Self-Play Trainer for HaxBall Dodgeball.
"""

from __future__ import annotations
import os
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from core.vector import Vec2
    from core.constants import Team, GameState
    from core.dodgeball_game import DodgeballGame
    from rl.observations import DodgeballObservationBuilder
    from rl.rewards import DodgeballRewardEngine
    from rl.mlp_policy import ActorCriticMLP
except (ImportError, ValueError):
    from ..core.vector import Vec2
    from ..core.constants import Team, GameState
    from ..core.dodgeball_game import DodgeballGame
    from .observations import DodgeballObservationBuilder
    from .rewards import DodgeballRewardEngine
    from .mlp_policy import ActorCriticMLP

class DodgeballSelfPlayTrainer:
    def __init__(
        self,
        game: DodgeballGame,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 512,
        batch_size: int = 64,
        update_epochs: int = 4,
        device: str = "cpu"
    ):
        self.game = game
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.device = torch.device(device)
        self.lr = lr

        self.obs_builder = DodgeballObservationBuilder(
            max_teammates=1,
            max_opponents=2
        )
        self.reward_engine = DodgeballRewardEngine()

        self.obs_dim = self.obs_builder.observation_dimension
        self.act_dim = 3
        self.policy = ActorCriticMLP(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=128
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        self.num_agents = max(1, len(self.game.players))
        self.reset_storage()

        self.total_env_steps = 0
        self.total_iterations = 0
        self.wall_elims = 0
        self.recent_rewards: List[float] = []
        self.last_stats: Dict[str, Any] = {
            "iteration": 0,
            "total_steps": 0,
            "mean_reward": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "wall_elims": 0,
            "red_score": 0,
            "blue_score": 0,
        }

    def reset_storage(self):
        self.num_agents = max(1, len(self.game.players))
        self.buf_obs = torch.zeros((self.rollout_steps, self.num_agents, self.obs_dim), device=self.device)
        self.buf_actions = torch.zeros((self.rollout_steps, self.num_agents, self.act_dim), device=self.device)
        self.buf_logprobs = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.buf_rewards = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.buf_dones = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.buf_values = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.rollout_ptr = 0

    def step_simulation(self) -> Dict[str, Any]:
        players = self.game.players
        num_p = len(players)
        if num_p != self.num_agents:
            self.reset_storage()

        obs_list = [self.obs_builder.build_observation(self.game, p) for p in players]
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions_t, logprobs_t, _, values_t = self.policy.get_action_and_value(obs_tensor)

        actions_np = actions_t.cpu().numpy()

        inputs: Dict[int, Tuple[float, float, bool]] = {}
        for i, p in enumerate(players):
            if not self.game.is_alive(p.player_id):
                inputs[p.player_id] = (0.0, 0.0, False)
                continue
            ori = 1.0 if p.team == Team.RED else -1.0
            mx = float(np.clip(actions_np[i, 0], -1.0, 1.0)) * ori
            my = float(np.clip(actions_np[i, 1], -1.0, 1.0))
            kick = bool(actions_np[i, 2] > 0.0)
            inputs[p.player_id] = (mx, my, kick)

        step_info = self.game.step(inputs)
        self.total_env_steps += 1

        rewards_dict = self.reward_engine.compute_rewards(self.game, step_info)
        rewards_list = [rewards_dict.get(p.player_id, 0.0) for p in players]

        self.wall_elims += len(step_info.get("eliminations", []))
        done_flag = 1.0 if (step_info.get("round_over", False) or step_info.get("game_over", False)) else 0.0

        idx = self.rollout_ptr
        self.buf_obs[idx] = obs_tensor
        self.buf_actions[idx] = actions_t
        self.buf_logprobs[idx] = logprobs_t
        self.buf_rewards[idx] = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        self.buf_dones[idx] = done_flag
        self.buf_values[idx] = values_t.flatten()

        self.rollout_ptr += 1

        if self.rollout_ptr >= self.rollout_steps:
            train_stats = self._update_ppo()
            self.rollout_ptr = 0
            self.total_iterations += 1
            self.last_stats.update(train_stats)

        self.last_stats["total_steps"] = self.total_env_steps
        self.last_stats["red_score"] = self.game.red_score
        self.last_stats["blue_score"] = self.game.blue_score
        self.last_stats["wall_elims"] = self.wall_elims

        return self.last_stats

    def _update_ppo(self) -> Dict[str, float]:
        with torch.no_grad():
            next_obs_list = [self.obs_builder.build_observation(self.game, p) for p in self.game.players]
            next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=self.device)
            next_values = self.policy.get_value(next_obs_tensor).flatten()

            advantages = torch.zeros_like(self.buf_rewards)
            lastgaelam = torch.zeros(self.num_agents, device=self.device)

            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    nextvalues = next_values
                else:
                    nextvalues = self.buf_values[t + 1]
                nextnonterminal = 1.0 - self.buf_dones[t]

                delta = self.buf_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buf_values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.buf_values

        b_obs = self.buf_obs.reshape(-1, self.obs_dim)
        b_actions = self.buf_actions.reshape(-1, self.act_dim)
        b_logprobs = self.buf_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        total_samples = self.rollout_steps * self.num_agents
        indices = np.arange(total_samples)

        pg_losses, v_losses, ent_losses = [], [], []

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                logratio = newlogprob - b_logprobs[mb_idx]
                ratio = torch.exp(logratio)

                mb_advantages = b_advantages[mb_idx]
                surr1 = -mb_advantages * ratio
                surr2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(surr1, surr2).mean()

                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        mean_reward = float(self.buf_rewards.mean().item())
        self.recent_rewards.append(mean_reward)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)

        return {
            "iteration": self.total_iterations + 1,
            "mean_reward": float(np.mean(self.recent_rewards)),
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "total_steps": self.total_env_steps,
            "total_iterations": self.total_iterations,
            "wall_elims": self.wall_elims,
        }, path)

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.total_env_steps = ckpt.get("total_steps", 0)
        self.total_iterations = ckpt.get("total_iterations", 0)
        return True
