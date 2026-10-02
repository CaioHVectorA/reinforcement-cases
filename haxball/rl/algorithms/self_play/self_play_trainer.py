"""
Recursive 2v2 Multi-Agent Self-Play Trainer for HaxBall RL.
Supports:
1. Decentralized execution with Parameter Sharing (all agents share the policy from their ego perspective)
2. Team-Play Reward Engine integration (assists, passes, anti-clustering spacing, defensive anchor)
3. Online On-Policy PPO with GAE-lambda
4. High-speed multi-step execution (up to 100x acceleration per frame)
5. Instant policy re-initialization ("Ficar Burro") to observe learning from scratch
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.rewards.reward_engine import TeamPlayRewardEngine
from haxball.rl.models.mlp_policy import ActorCriticMLP

class SelfPlay2v2Trainer:
    def __init__(
        self,
        game: HaxBallGame,
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

        # Observation builder & Reward engine
        self.obs_builder = DecoupledObservationBuilder()
        self.reward_engine = TeamPlayRewardEngine()
        self.reward_engine.reset(self.game)

        # Policy Network
        self.obs_dim = self.obs_builder.obs_dim  # 61 dimensions
        self.act_dim = 3  # (move_x, move_y, kick)
        self.policy = ActorCriticMLP(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            is_discrete=False,
            hidden_dim=128
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

        # Rollout Storage
        self.num_agents = len(self.game.players)
        self.reset_storage()

        # Telemetry & Metrics
        self.total_env_steps = 0
        self.total_iterations = 0
        self.goals_red = 0
        self.goals_blue = 0
        self.total_passes = 0
        self.total_assists = 0
        self.recent_rewards: List[float] = []
        self.recent_losses: List[float] = []
        self.last_stats: Dict[str, Any] = {
            "iteration": 0,
            "total_steps": 0,
            "mean_reward": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "red_score": 0,
            "blue_score": 0,
            "passes": 0,
            "assists": 0
        }

    def reset_storage(self):
        """Initializes empty rollout tensors for all agents."""
        self.num_agents = max(1, len(self.game.players))
        self.buf_obs = torch.zeros((self.rollout_steps, self.num_agents, self.obs_dim), device=self.device)
        self.buf_actions = torch.zeros((self.rollout_steps, self.num_agents, self.act_dim), device=self.device)
        self.buf_logprobs = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.buf_rewards = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.buf_dones = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.buf_values = torch.zeros((self.rollout_steps, self.num_agents), device=self.device)
        self.rollout_ptr = 0

    def reset_policy_to_random(self):
        """Re-initializes all policy weights to completely random, making models 'burros' again."""
        self.policy = ActorCriticMLP(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            is_discrete=False,
            hidden_dim=128
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        self.reset_storage()
        self.total_env_steps = 0
        self.total_iterations = 0
        self.goals_red = 0
        self.goals_blue = 0
        self.total_passes = 0
        self.total_assists = 0
        self.recent_rewards.clear()
        self.recent_losses.clear()
        self.last_stats = {
            "iteration": 0,
            "total_steps": 0,
            "mean_reward": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "red_score": 0,
            "blue_score": 0,
            "passes": 0,
            "assists": 0
        }

    def step_simulation(self) -> Dict[str, Any]:
        """
        Executes one environment simulation tick:
        1. Queries policy for all agents
        2. Steps game physics
        3. Computes team rewards
        4. Accumulates rollout buffer
        5. Performs PPO update when buffer is full
        """
        players = self.game.players
        num_p = len(players)
        if num_p != self.num_agents:
            self.reset_storage()
            num_p = len(players)

        # 1. Gather observation for each player from its ego perspective
        obs_list = []
        for p in players:
            obs = self.obs_builder.build_observation(self.game, p)
            obs_list.append(obs)
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)

        # 2. Policy forward pass (Batch across all players)
        with torch.no_grad():
            actions_t, logprobs_t, _, values_t = self.policy.get_action_and_value(obs_tensor)

        actions_np = actions_t.cpu().numpy()

        # 3. Decode inputs for physics engine
        inputs: Dict[int, Tuple[float, float, bool]] = {}
        for i, p in enumerate(players):
            mx = float(np.clip(actions_np[i, 0], -1.0, 1.0))
            my = float(np.clip(actions_np[i, 1], -1.0, 1.0))
            kick = bool(actions_np[i, 2] > 0.0)
            inputs[p.player_id] = (mx, my, kick)

        # 4. Step game physics
        step_info = self.game.step(inputs)
        self.total_env_steps += 1

        # 5. Compute Team-Play Rewards
        team_rewards = self.reward_engine.compute_team_rewards(self.game, step_info)
        rewards_list = [team_rewards.get(p.player_id, 0.0) for p in players]

        # Goal tracking
        if step_info.get("goal_scored", False):
            if step_info.get("scoring_team") == Team.RED:
                self.goals_red += 1
            elif step_info.get("scoring_team") == Team.BLUE:
                self.goals_blue += 1

        # Check for passes/assists recorded by reward engine
        if self.reward_engine.passer_candidate_id is not None:
            self.total_passes += 1

        done_flag = 1.0 if (step_info.get("goal_scored", False) or step_info.get("game_over", False)) else 0.0

        # 6. Store in rollout buffer
        idx = self.rollout_ptr
        self.buf_obs[idx] = obs_tensor
        self.buf_actions[idx] = actions_t
        self.buf_logprobs[idx] = logprobs_t
        self.buf_rewards[idx] = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        self.buf_dones[idx] = done_flag
        self.buf_values[idx] = values_t.flatten()

        self.rollout_ptr += 1

        # If goal was scored, restart round smoothly
        if step_info.get("goal_scored", False):
            self.game.reset_round()
            self.reward_engine.reset(self.game)

        # 7. Rollout buffer full -> Run PPO Update
        if self.rollout_ptr >= self.rollout_steps:
            train_stats = self._update_ppo()
            self.rollout_ptr = 0
            self.total_iterations += 1
            self.last_stats.update(train_stats)

        self.last_stats["total_steps"] = self.total_env_steps
        self.last_stats["red_score"] = self.game.red_score
        self.last_stats["blue_score"] = self.game.blue_score
        self.last_stats["passes"] = self.total_passes

        return self.last_stats

    def _update_ppo(self) -> Dict[str, float]:
        """Calculates GAE-lambda advantages and optimizes policy parameters."""
        with torch.no_grad():
            # Estimate next value
            next_obs_list = [self.obs_builder.build_observation(self.game, p) for p in self.game.players]
            next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=self.device)
            next_values = self.policy.get_value(next_obs_tensor).flatten()

            advantages = torch.zeros_like(self.buf_rewards)
            lastgaelam = torch.zeros(self.num_agents, device=self.device)

            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    nextnonterminal = 1.0 - self.buf_dones[t]
                    nextvalues = next_values
                else:
                    nextnonterminal = 1.0 - self.buf_dones[t + 1]
                    nextvalues = self.buf_values[t + 1]

                delta = self.buf_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buf_values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.buf_values

        # Flatten rollout across time and agents: (rollout_steps * num_agents, ...)
        b_obs = self.buf_obs.reshape(-1, self.obs_dim)
        b_actions = self.buf_actions.reshape(-1, self.act_dim)
        b_logprobs = self.buf_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.buf_values.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = b_obs.shape[0]
        batch_size = min(self.batch_size, total_samples)

        pg_losses = []
        v_losses = []
        entropies = []

        for _ in range(self.update_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                mb_inds = indices[start:end]

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Policy loss with PPO clipping
                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

        mean_reward = self.buf_rewards.mean().item()
        self.recent_rewards.append(mean_reward)
        mean_loss = float(np.mean(pg_losses))

        return {
            "iteration": self.total_iterations + 1,
            "mean_reward": mean_reward,
            "policy_loss": mean_loss,
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies))
        }

    def step_multistep(self, count: int) -> Dict[str, Any]:
        """
        Executes N simulation ticks consecutively.
        Enables 1x to 100x acceleration per frame!
        """
        stats = self.last_stats
        for _ in range(max(1, count)):
            stats = self.step_simulation()
        return stats
