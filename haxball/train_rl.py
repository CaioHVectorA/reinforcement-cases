"""
PPO (Proximal Policy Optimization) Training Pipeline for HaxBall.
Supports training against baseline bots (Heuristic, WallRebound) and self-play.
Includes Actor-Critic network, GAE-lambda, mini-batch updates, model checkpoints,
and evaluation routines.
"""

from __future__ import annotations
import os
import time
import argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.bots import HeuristicBot, WallReboundBot, GoalieBot

# ---------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticContinuous(nn.Module):
    def __init__(self, obs_dim: int = 20, act_dim: int = 3):
        super().__init__()
        # Shared or dual MLP trunk
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)

        return action, log_prob, entropy, value

# ---------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------

class PPOTrainer:
    def __init__(
        self,
        env: HaxBallEnv,
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

        self.agent = ActorCriticContinuous(
            obs_dim=env.obs_dim,
            act_dim=env.action_space.shape[0]
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

    def train(self, total_timesteps: int = 50000, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)

        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        num_iterations = total_timesteps // self.num_steps
        global_step = 0
        episode_rewards = []
        episode_win_count = 0
        episode_count = 0

        print("\n" + "=" * 60)
        print(f"  STARTING PPO TRAINING: {total_timesteps:,} TIMESTEPS")
        print(f"  Map: {self.env.stadium.name} | Device: {self.device}")
        print("=" * 60)

        for iteration in range(1, num_iterations + 1):
            obs_buffer = torch.zeros((self.num_steps, self.env.obs_dim), device=self.device)
            actions_buffer = torch.zeros((self.num_steps, 3), device=self.device)
            logprobs_buffer = torch.zeros(self.num_steps, device=self.device)
            rewards_buffer = torch.zeros(self.num_steps, device=self.device)
            dones_buffer = torch.zeros(self.num_steps, device=self.device)
            values_buffer = torch.zeros(self.num_steps, device=self.device)

            cur_ep_reward = 0.0

            # 1. Rollout Collection
            for step in range(self.num_steps):
                global_step += 1
                obs_buffer[step] = obs

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(obs.unsqueeze(0))
                    values_buffer[step] = value.flatten()

                actions_buffer[step] = action
                logprobs_buffer[step] = logprob

                # Environment step
                act_np = action.cpu().numpy()[0]
                next_obs_np, reward, terminated, truncated, info = self.env.step(act_np)
                done = terminated or truncated

                rewards_buffer[step] = reward
                dones_buffer[step] = float(done)
                cur_ep_reward += reward

                if done:
                    episode_rewards.append(cur_ep_reward)
                    episode_count += 1
                    if info.get("winner") == 1:  # Red won
                        episode_win_count += 1
                    cur_ep_reward = 0.0
                    next_obs_np, _ = self.env.reset()

                obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)

            # 2. GAE Advantage Calculation
            with torch.no_grad():
                next_value = self.agent.get_value(obs.unsqueeze(0)).reshape(1, -1)
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

            # Flatten batch for optimization
            b_obs = obs_buffer
            b_actions = actions_buffer
            b_logprobs = logprobs_buffer
            b_advantages = advantages
            b_returns = returns
            b_values = values_buffer

            # 3. PPO Update Epochs
            b_inds = np.arange(self.num_steps)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.num_steps, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Policy loss
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                    # Total loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # Logging
            recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0.0
            win_rate = (episode_win_count / episode_count * 100) if episode_count > 0 else 0.0
            print(
                f"[Iter {iteration:03d}/{num_iterations}] Step: {global_step:06d} | "
                f"Mean Reward: {recent_reward:+.2f} | Win Rate: {win_rate:4.1f}% | "
                f"Episodes: {episode_count}"
            )

            # Save checkpoint every 10 iterations or at the end
            if iteration % 10 == 0 or iteration == num_iterations:
                save_path = os.path.join(save_dir, f"haxball_ppo_iter_{iteration}.pt")
                torch.save(self.agent.state_dict(), save_path)
                print(f" -> Checkpoint saved to: {save_path}")

        print("\nTraining completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train HaxBall RL Agent with PPO")
    parser.add_argument("--map", choices=["futsal", "classic", "dodgeball"], default="futsal")
    parser.add_argument("--bot", choices=["heuristic", "wall", "goalie"], default="heuristic")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()

    map_path = os.path.join(os.path.dirname(__file__), "maps", f"{args.map}.hbs")

    opp_bot = WallReboundBot() if args.bot == "wall" else HeuristicBot()
    env = HaxBallEnv(stadium_file=map_path, opponent_bot=opp_bot, max_steps=1500)

    trainer = PPOTrainer(env=env, num_steps=2048, batch_size=64, update_epochs=8)
    trainer.train(total_timesteps=args.timesteps, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
