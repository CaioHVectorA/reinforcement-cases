"""
Base bot class for HaxBall agents and baselines.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame

class BaseBot(ABC):
    def __init__(self, name: str = "BaseBot"):
        self.name = name

    @abstractmethod
    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        """
        Calculates action for the bot.
        Returns: (move_x: float, move_y: float, kick: bool)
        where move_x and move_y are in [-1.0, 1.0].
        """
        pass

    def reset(self):
        """Clears any per-match memory (called when a new match/opponent starts)."""
        return None
