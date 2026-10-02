"""
Abstract base class for HaxBall RL observation builders.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from gymnasium import spaces
from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame

class BaseObservationBuilder(ABC):
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Returns gymnasium space for the observation."""
        pass

    @abstractmethod
    def build_observation(self, game: HaxBallGame, player: Disc) -> np.ndarray:
        """Constructs state observation vector for a given player."""
        pass
