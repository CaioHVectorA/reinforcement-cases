"""
Constants and definitions for the HaxBall simulation engine.
Faithful to the original HaxBall (.hbs) specifications and physics parameters.
"""

from enum import IntEnum, auto

# Collision Group Bitmasks
class CollisionMask:
    NONE = 0
    BALL = 1 << 0
    RED = 1 << 1
    BLUE = 1 << 2
    WALL = 1 << 3
    RED_KO = 1 << 4
    BLUE_KO = 1 << 5
    SCORE = 1 << 6
    C0 = 1 << 7
    C1 = 1 << 8
    C2 = 1 << 9
    C3 = 1 << 10
    ALL = (1 << 11) - 1

COLLISION_FLAG_MAP = {
    "none": CollisionMask.NONE,
    "ball": CollisionMask.BALL,
    "red": CollisionMask.RED,
    "blue": CollisionMask.BLUE,
    "wall": CollisionMask.WALL,
    "redKO": CollisionMask.RED_KO,
    "blueKO": CollisionMask.BLUE_KO,
    "score": CollisionMask.SCORE,
    "c0": CollisionMask.C0,
    "c1": CollisionMask.C1,
    "c2": CollisionMask.C2,
    "c3": CollisionMask.C3,
    "all": CollisionMask.ALL,
}

def parse_collision_flags(flags) -> int:
    """Parses a list of flag strings or an integer into a bitmask."""
    if isinstance(flags, int):
        return flags
    if isinstance(flags, str):
        flags = [flags]
    if not isinstance(flags, (list, tuple)):
        return CollisionMask.ALL

    mask = 0
    for flag in flags:
        f_lower = flag.strip()
        if f_lower in COLLISION_FLAG_MAP:
            mask |= COLLISION_FLAG_MAP[f_lower]
        elif flag == "redKO":
            mask |= CollisionMask.RED_KO
        elif flag == "blueKO":
            mask |= CollisionMask.BLUE_KO
    return mask

class Team(IntEnum):
    NONE = 0
    RED = 1
    BLUE = 2

class GameState(IntEnum):
    KICKOFF_RED = auto()
    KICKOFF_BLUE = auto()
    PLAYING = auto()
    GOAL_CELEBRATION = auto()
    GAME_OVER = auto()

# Default Classic HaxBall Physics Parameters
DEFAULT_PLAYER_PHYSICS = {
    "radius": 15.0,
    "bCoef": 0.5,
    "invMass": 0.5,
    "damping": 0.96,
    "acceleration": 0.1,
    "kickingAcceleration": 0.07,
    "kickingDamping": 0.96,
    "kickStrength": 5.0,
    "kickMargin": 4.0,       # Distance threshold buffer for kicking ball
    "kickBack": 0.0,
}

DEFAULT_BALL_PHYSICS = {
    "radius": 10.0,
    "bCoef": 0.5,
    "invMass": 1.0,
    "damping": 0.99,
    "color": "FFFFFF",
    "cMask": ["all"],
    "cGroup": ["ball"],
}

# Futsal Specialized Physics (Faster, smaller, lighter ball, high responsiveness)
FUTSAL_PLAYER_PHYSICS = {
    "radius": 15.0,
    "bCoef": 0.5,
    "invMass": 0.5,
    "damping": 0.96,
    "acceleration": 0.11,
    "kickingAcceleration": 0.08,
    "kickingDamping": 0.96,
    "kickStrength": 5.4,
    "kickMargin": 4.0,
    "kickBack": 0.0,
}

FUTSAL_BALL_PHYSICS = {
    "radius": 7.0,          # Smaller ball
    "bCoef": 0.4,           # Controlled futsal bounce
    "invMass": 1.35,        # Lighter ball for fast speed & tight control
    "damping": 0.988,       # High speed persistence
    "color": "FFD700",      # Yellow futsal ball
    "cMask": ["all"],
    "cGroup": ["ball"],
}

FPS = 60
TIMESTEP = 1.0 / FPS
