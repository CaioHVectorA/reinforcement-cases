"""
Standard Value-Based Reinforcement Learning: Deep Q-Network (DQN).
Implements experience replay, target networks, and epsilon-greedy exploration
as a classical RL benchmark compared against PPO.
"""

from __future__ import annotations
import os
import random
from collections import deque
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.rl.models.mlp_policy import layer_init

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, num_actions))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNTrainer:
    def __init__(
        self,
        env: HaxBallEnv,
        lr: float = 5e-4,
        gamma: float = 0.99,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        device: str = "cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device(device)

        self.obs_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.n if hasattr(env.action_space, "n") else 18

        self.q_net = QNetwork(self.obs_dim, self.num_actions).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.total_steps = 0

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def train_step(self, num_env_steps: int = 1000) -> Dict[str, float]:
        """Runs environment steps, stores experience, and updates Q-network."""
        obs, _ = self.env.reset()
        episode_rewards = []
        cur_reward = 0.0
        losses = []

        for _ in range(num_env_steps):
            self.total_steps += 1
            action = self.select_action(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.append((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            cur_reward += reward

            # Optimize Q-Network if enough samples
            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

                s_t = torch.tensor(np.array(s_batch), dtype=torch.float32, device=self.device)
                a_t = torch.tensor(a_batch, dtype=torch.long, device=self.device).unsqueeze(1)
                r_t = torch.tensor(r_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
                ns_t = torch.tensor(np.array(ns_batch), dtype=torch.float32, device=self.device)
                d_t = torch.tensor(d_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

                # Current Q(s, a)
                curr_q = self.q_net(s_t).gather(1, a_t)

                # Target Q(s', a')
                with torch.no_grad():
                    max_next_q = self.target_net(ns_t).max(dim=1, keepdim=True)[0]
                    target_q = r_t + (1.0 - d_t) * self.gamma * max_next_q

                loss = nn.functional.mse_loss(curr_q, target_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            # Update target network
            if self.total_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if done:
                episode_rewards.append(cur_reward)
                cur_reward = 0.0
                obs, _ = self.env.reset()

        # Decay epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        mean_loss = float(np.mean(losses)) if losses else 0.0

        return {
            "mean_reward": mean_reward,
            "loss": mean_loss,
            "epsilon": self.eps,
            "episodes": len(episode_rewards)
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
