"""
Constants and definitions for HaxBall Dodgeball simulation engine.
"""

from enum import IntEnum, auto

class CollisionMask:
    NONE = 0
    BALL = 1 << 0
    RED = 1 << 1
    BLUE = 1 << 2
    WALL = 1 << 3
    SCORE = 1 << 4
    ALL = (1 << 5) - 1

COLLISION_FLAG_MAP = {
    "none": CollisionMask.NONE,
    "ball": CollisionMask.BALL,
    "red": CollisionMask.RED,
    "blue": CollisionMask.BLUE,
    "wall": CollisionMask.WALL,
    "all": CollisionMask.ALL,
}

def parse_collision_flags(flags) -> int:
    if isinstance(flags, int):
        return flags
    if isinstance(flags, str):
        flags = [flags]
    if not isinstance(flags, (list, tuple)):
        return CollisionMask.ALL

    mask = 0
    for flag in flags:
        f_lower = str(flag).strip().lower()
        if f_lower in COLLISION_FLAG_MAP:
            mask |= COLLISION_FLAG_MAP[f_lower]
        elif f_lower == "all":
            mask |= CollisionMask.ALL
    return mask

class Team(IntEnum):
    NONE = 0
    RED = 1
    BLUE = 2

class GameState(IntEnum):
    PLAYING = 1
    ROUND_CELEBRATION = 2
    GAME_OVER = 3

FPS = 60.0

DEFAULT_PLAYER_PHYSICS = {
    "radius": 15.0,
    "bCoef": 0.5,
    "invMass": 0.5,
    "damping": 0.96,
    "acceleration": 0.12,
    "kickingAcceleration": 0.09,
    "kickingDamping": 0.96,
    "kickStrength": 6.0,
    "kickMargin": 4.0,
    "kickBack": 0.0,
}

# Dodgeball ball physics: bCoef 0.70 to lose speed when bouncing on walls!
DEFAULT_BALL_PHYSICS = {
    "radius": 9.0,
    "bCoef": 0.72,
    "invMass": 1.1,
    "damping": 0.988,
    "color": "FF4466",
    "cMask": ["all"],
    "cGroup": ["ball"],
}
