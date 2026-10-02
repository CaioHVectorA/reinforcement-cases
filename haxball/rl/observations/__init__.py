"""
Observation builders package for HaxBall RL.
"""

from haxball.rl.observations.base import BaseObservationBuilder
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder

__all__ = ["BaseObservationBuilder", "DecoupledObservationBuilder"]
