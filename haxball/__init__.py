"""
HaxBall: 2D Physics Simulator, Stadium Engine & Reinforcement Learning Environment.
"""

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, CollisionMask
from haxball.core.disc import Disc
from haxball.core.segment import Vertex, Segment
from haxball.core.stadium import Stadium
from haxball.core.physics_engine import PhysicsEngine
from haxball.core.game import HaxBallGame

__version__ = "1.0.0"

__all__ = [
    "Vec2",
    "Team",
    "GameState",
    "CollisionMask",
    "Disc",
    "Vertex",
    "Segment",
    "Stadium",
    "PhysicsEngine",
    "HaxBallGame",
]
