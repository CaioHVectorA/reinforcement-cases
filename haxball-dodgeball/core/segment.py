"""
Segment geometry definitions for HaxBall Dodgeball walls.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
from .vector import Vec2
from .constants import CollisionMask
from .disc import hex_to_rgb

class Vertex:
    __slots__ = ("pos", "bCoef", "cMask", "cGroup", "trait")

    def __init__(self, pos: Vec2, bCoef: float = 0.75, cMask: int = CollisionMask.ALL, cGroup: int = CollisionMask.WALL, trait: str = ""):
        self.pos = pos.copy()
        self.bCoef = float(bCoef)
        self.cMask = cMask
        self.cGroup = cGroup
        self.trait = trait

class Segment:
    __slots__ = ("p0", "p1", "bCoef", "cMask", "cGroup", "vis", "color_hex", "color_rgb", "trait", "name")

    def __init__(
        self,
        p0: Vec2,
        p1: Vec2,
        bCoef: float = 0.75,
        cMask: int = CollisionMask.ALL,
        cGroup: int = CollisionMask.WALL,
        vis: bool = True,
        color: str = "FFFFFF",
        trait: str = "",
        name: str = ""
    ):
        self.p0 = p0.copy()
        self.p1 = p1.copy()
        self.bCoef = float(bCoef)
        self.cMask = cMask
        self.cGroup = cGroup
        self.vis = vis
        self.color_hex = color
        self.color_rgb = hex_to_rgb(color)
        self.trait = trait
        self.name = name

    def closest_point(self, pt: Vec2) -> Tuple[Vec2, Vec2]:
        """Returns (closest_point, normal_pointing_to_pt)."""
        v = self.p1 - self.p0
        v_len_sq = v.length_sq()
        if v_len_sq < 1e-9:
            q = self.p0.copy()
        else:
            t = max(0.0, min(1.0, (pt - self.p0).dot(v) / v_len_sq))
            q = self.p0 + v * t

        diff = pt - q
        dist = diff.length()
        if dist > 1e-7:
            normal = diff / dist
        else:
            normal = Vec2(-v.y, v.x).normalized()
        return q, normal
