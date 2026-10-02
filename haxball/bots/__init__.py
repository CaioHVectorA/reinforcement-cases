"""
Bots package for HaxBall agents and baselines.
"""

from haxball.bots.base_bot import BaseBot
from haxball.bots.heuristic_bot import HeuristicBot
from haxball.bots.wall_rebound_bot import WallReboundBot
from haxball.bots.goalie_bot import GoalieBot

__all__ = ["BaseBot", "HeuristicBot", "WallReboundBot", "GoalieBot"]
