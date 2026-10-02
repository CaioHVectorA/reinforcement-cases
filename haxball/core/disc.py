"""
Disc rigid body representation for HaxBall.
All dynamic entities (ball, players) and static obstacles (goal posts) are Discs.
"""

from __future__ import annotations
from typing import Optional, Tuple
from haxball.core.vector import Vec2
from haxball.core.constants import CollisionMask, Team, parse_collision_flags

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex string (e.g. 'FFFFFF', 'E56E56', '#5689E5') to RGB tuple."""
    clean = hex_str.lstrip("#")
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
        "color_hex", "color_rgb", "cMask", "cGroup",
        "is_player", "team", "player_id", "player_number",
        "is_kicking", "kick_flash", "acceleration",
        "kicking_acceleration", "kicking_damping",
        "kick_strength", "kick_margin", "kick_back",
        "trait", "name"
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
        cGroup: int = CollisionMask.BALL,
        is_player: bool = False,
        team: Team = Team.NONE,
        player_id: int = 0,
        player_number: int = 0,
        acceleration: float = 0.1,
        kicking_acceleration: float = 0.07,
        kicking_damping: float = 0.96,
        kick_strength: float = 5.0,
        kick_margin: float = 4.0,
        kick_back: float = 0.0,
        trait: str = "",
        name: str = ""
    ):
        self.pos = pos.copy()
        self.speed = speed.copy() if speed is not None else Vec2(0.0, 0.0)
        self.radius = float(radius)
        self.bCoef = float(bCoef)
        self.invMass = float(invMass)
        self.damping = float(damping)

        self.color_hex = color
        self.color_rgb = hex_to_rgb(color)
        self.cMask = cMask
        self.cGroup = cGroup

        self.is_player = is_player
        self.team = team
        self.player_id = player_id
        self.player_number = player_number
        self.is_kicking = False
        self.kick_flash = 0

        self.acceleration = float(acceleration)
        self.kicking_acceleration = float(kicking_acceleration)
        self.kicking_damping = float(kicking_damping)
        self.kick_strength = float(kick_strength)
        self.kick_margin = float(kick_margin)
        self.kick_back = float(kick_back)

        self.trait = trait
        self.name = name

    @property
    def is_static(self) -> bool:
        return self.invMass <= 1e-9

    def can_collide_with(self, other: Disc) -> bool:
        """Standard HaxBall collision group/mask matching."""
        return bool((self.cGroup & other.cMask) and (other.cGroup & self.cMask))

    def copy(self) -> Disc:
        d = Disc(
            pos=self.pos.copy(),
            speed=self.speed.copy(),
            radius=self.radius,
            bCoef=self.bCoef,
            invMass=self.invMass,
            damping=self.damping,
            color=self.color_hex,
            cMask=self.cMask,
            cGroup=self.cGroup,
            is_player=self.is_player,
            team=self.team,
            player_id=self.player_id,
            player_number=self.player_number,
            acceleration=self.acceleration,
            kicking_acceleration=self.kicking_acceleration,
            kicking_damping=self.kicking_damping,
            kick_strength=self.kick_strength,
            kick_margin=self.kick_margin,
            kick_back=self.kick_back,
            trait=self.trait,
            name=self.name
        )
        d.is_kicking = self.is_kicking
        d.kick_flash = self.kick_flash
        return d
