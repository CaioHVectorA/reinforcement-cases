"""
Models package for HaxBall RL.
"""

from haxball.rl.models.mlp_policy import ActorCriticMLP
from haxball.rl.models.entity_attention import EntityAttentionPolicy

__all__ = ["ActorCriticMLP", "EntityAttentionPolicy"]
