"""
Disc representation for players and balls in HaxBall Dodgeball.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from .vector import Vec2
from .constants import Team, CollisionMask

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    clean = hex_str.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        return (255, 255, 255)
    try:
        return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))
    except ValueError:
        return (255, 255, 255)

class Disc:
    __slots__ = (
        "pos", "speed", "radius", "bCoef", "invMass", "damping",
        "color", "color_rgb", "cMask", "cGroup", "trait", "name",
        "is_player", "player_id", "team", "player_number",
        "acceleration", "kicking_acceleration", "kicking_damping",
        "kick_strength", "kick_margin", "kick_back",
        "is_kicking", "kick_flash"
    )

    def __init__(
        self,
        pos: Vec2,
        speed: Optional[Vec2] = None,
        radius: float = 10.0,
        bCoef: float = 0.5,
        invMass: float = 1.0,
        damping: float = 0.99,
        color: str = "FFFFFF",
        cMask: int = CollisionMask.ALL,
        cGroup: int = CollisionMask.WALL,
        trait: str = "",
        name: str = "Disc"
    ):
        self.pos = pos.copy()
        self.speed = speed.copy() if speed is not None else Vec2(0.0, 0.0)
        self.radius = float(radius)
        self.bCoef = float(bCoef)
        self.invMass = float(invMass)
        self.damping = float(damping)
        self.color = color
        self.color_rgb = hex_to_rgb(color)
        self.cMask = cMask
        self.cGroup = cGroup
        self.trait = trait
        self.name = name

        # Player properties
        self.is_player = False
        self.player_id = 0
        self.team = Team.NONE
        self.player_number = 0
        self.acceleration = 0.12
        self.kicking_acceleration = 0.09
        self.kicking_damping = 0.96
        self.kick_strength = 6.0
        self.kick_margin = 4.0
        self.kick_back = 0.0
        self.is_kicking = False
        self.kick_flash = 0

    @property
    def is_static(self) -> bool:
        return self.invMass <= 1e-9

    def can_collide_with(self, other: Disc) -> bool:
        return bool((self.cGroup & other.cMask) and (other.cGroup & self.cMask))
