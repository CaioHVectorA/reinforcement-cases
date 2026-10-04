"""
Neural network policy architectures for HaxBall RL.
Supports Continuous (Gaussian) and Discrete (Categorical) Actor-Critic models.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, is_discrete: bool = False, hidden_dim: int = 128):
        super().__init__()
        self.is_discrete = is_discrete
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Actor head
        if is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        if self.is_discrete:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        else:
            action_mean = self.actor(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)

            if action is None:
                action = probs.sample()

            log_prob = probs.log_prob(action).sum(-1)
            entropy = probs.entropy().sum(-1)
            value = self.critic(x)
            return action, log_prob, entropy, value
