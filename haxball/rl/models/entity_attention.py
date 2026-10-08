"""
Entity-Attention Policy for HaxBall RL.
Uses Multi-Head Self-Attention over players and the ball to achieve true
permutation-invariance and seamless scaling from 1v1 up to 5v5.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from haxball.rl.models.mlp_policy import layer_init

class EntityAttentionPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        act_dim: int = 3,
        is_discrete: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.is_discrete = is_discrete
        self.act_dim = act_dim

        # Entity feature encoders:
        # Global/Ball: 8 features -> embed_dim
        self.ball_encoder = nn.Sequential(
            layer_init(nn.Linear(8, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, embed_dim))
        )
        # Ego: 8 features -> embed_dim
        self.ego_encoder = nn.Sequential(
            layer_init(nn.Linear(8, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, embed_dim))
        )
        # Player (Teammate / Opponent): 5 features -> embed_dim
        self.player_encoder = nn.Sequential(
            layer_init(nn.Linear(5, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, embed_dim))
        )

        # Multi-Head Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # Policy & Value heads
        if is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(embed_dim * 2, embed_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(embed_dim, act_dim), std=0.01)
            )
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(embed_dim * 2, embed_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(embed_dim, act_dim), std=0.01)
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(embed_dim * 2, embed_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(embed_dim, 1), std=1.0)
        )

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parses flat observation vector from DecoupledObservationBuilder
        into entity tokens: [B, num_entities, embed_dim]
        """
        # x is [B, obs_dim]
        ball_feat = x[:, 0:8]
        ego_feat = x[:, 8:16]

        e_ball = self.ball_encoder(ball_feat).unsqueeze(1)
        e_ego = self.ego_encoder(ego_feat).unsqueeze(1)

        tokens = [e_ball, e_ego]

        # Remaining features are chunks of 5
        curr = 16
        while curr + 5 <= x.shape[1]:
            p_feat = x[:, curr:curr+5]
            e_p = self.player_encoder(p_feat).unsqueeze(1)
            tokens.append(e_p)
            curr += 5

        # Concatenate along entity dimension: [B, num_entities, embed_dim]
        return torch.cat(tokens, dim=1)

    def forward_repr(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._extract_tokens(x)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)

        # Context is ego token concatenated with mean pooled field representation
        ego_token = tokens[:, 1, :]
        field_pooled = tokens.mean(dim=1)
        return torch.cat([ego_token, field_pooled], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.forward_repr(x)
        return self.actor(context)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        context = self.forward_repr(x)
        return self.critic(context)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        context = self.forward_repr(x)

        if self.is_discrete:
            logits = self.actor(context)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(context)
        else:
            action_mean = self.actor(context)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)

            if action is None:
                action = probs.sample()

            log_prob = probs.log_prob(action).sum(-1)
            entropy = probs.entropy().sum(-1)
            value = self.critic(context)
            return action, log_prob, entropy, value
