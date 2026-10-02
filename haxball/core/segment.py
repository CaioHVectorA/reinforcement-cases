"""
Vertex and Segment geometry definitions for HaxBall.
Supports straight walls and curved arc walls faithful to HaxBall stadium physics.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
from haxball.core.vector import Vec2
from haxball.core.constants import CollisionMask
from haxball.core.disc import hex_to_rgb

class Vertex:
    __slots__ = ("pos", "bCoef", "cMask", "cGroup", "trait", "name")

    def __init__(
        self,
        pos: Vec2,
        bCoef: float = 1.0,
        cMask: int = CollisionMask.ALL,
        cGroup: int = CollisionMask.WALL,
        trait: str = "",
        name: str = ""
    ):
        self.pos = pos.copy()
        self.bCoef = float(bCoef)
        self.cMask = cMask
        self.cGroup = cGroup
        self.trait = trait
        self.name = name

    def copy(self) -> Vertex:
        return Vertex(
            pos=self.pos.copy(),
            bCoef=self.bCoef,
            cMask=self.cMask,
            cGroup=self.cGroup,
            trait=self.trait,
            name=self.name
        )

class Segment:
    __slots__ = (
        "p0", "p1", "bCoef", "cMask", "cGroup", "curve", "bias",
        "vis", "color_hex", "color_rgb", "trait", "name",
        # Precomputed arc cache
        "is_curved", "arc_center", "arc_radius", "arc_start_angle", "arc_span_angle"
    )

    def __init__(
        self,
        p0: Vec2,
        p1: Vec2,
        bCoef: float = 1.0,
        cMask: int = CollisionMask.ALL,
        cGroup: int = CollisionMask.WALL,
        curve: float = 0.0,
        bias: float = 0.0,
        vis: bool = True,
        color: str = "C7E59B",
        trait: str = "",
        name: str = ""
    ):
        self.p0 = p0.copy()
        self.p1 = p1.copy()
        self.bCoef = float(bCoef)
        self.cMask = cMask
        self.cGroup = cGroup
        self.curve = float(curve)
        self.bias = float(bias)
        self.vis = vis
        self.color_hex = color
        self.color_rgb = hex_to_rgb(color)
        self.trait = trait
        self.name = name

        self.is_curved = abs(self.curve) > 1e-4
        self.arc_center = Vec2(0, 0)
        self.arc_radius = 0.0
        self.arc_start_angle = 0.0
        self.arc_span_angle = 0.0

        if self.is_curved:
            self._precompute_arc()

    def _precompute_arc(self):
        """Precomputes arc circle center, radius, and angular span for fast collision checks."""
        v = self.p1 - self.p0
        chord_len = v.length()
        if chord_len < 1e-7:
            self.is_curved = False
            return

        curve_rad = math.radians(self.curve)
        half_angle = curve_rad / 2.0
        sin_half = math.sin(half_angle)
        tan_half = math.tan(half_angle)

        if abs(sin_half) < 1e-7 or abs(tan_half) < 1e-7:
            self.is_curved = False
            return

        radius = abs((chord_len / 2.0) / sin_half)
        dist_to_center = (chord_len / 2.0) / tan_half

        # Midpoint of chord
        mid = (self.p0 + self.p1) * 0.5
        u_chord = v.normalized()
        # Perpendicular vector pointing to the left of chord
        perp = Vec2(-u_chord.y, u_chord.x)

        # Center is offset perpendicular to the chord
        center = mid - perp * dist_to_center

        self.arc_center = center
        self.arc_radius = radius

        # Angles from center to endpoints
        a0 = math.atan2(self.p0.y - center.y, self.p0.x - center.x)
        a1 = math.atan2(self.p1.y - center.y, self.p1.x - center.x)

        span = a1 - a0
        # Normalize span to match curve sign and direction
        if self.curve > 0:
            while span < 0:
                span += 2 * math.pi
            while span > 2 * math.pi:
                span -= 2 * math.pi
        else:
            while span > 0:
                span -= 2 * math.pi
            while span < -2 * math.pi:
                span += 2 * math.pi

        self.arc_start_angle = a0
        self.arc_span_angle = span

    def closest_point(self, pt: Vec2) -> Tuple[Vec2, Vec2]:
        """
        Finds closest point on segment to `pt`.
        Returns (closest_point, normal_pointing_to_pt).
        """
        if not self.is_curved:
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
                # Pt is exactly on segment, use segment perpendicular
                normal = Vec2(-v.y, v.x).normalized()
            return q, normal

        # Curved arc case
        center = self.arc_center
        diff_from_center = pt - center
        dist_from_center = diff_from_center.length()

        if dist_from_center < 1e-7:
            q = self.p0.copy()
            diff = pt - q
            normal = Vec2(0, 1) if diff.length() < 1e-7 else diff.normalized()
            return q, normal

        pt_angle = math.atan2(diff_from_center.y, diff_from_center.x)
        rel_angle = pt_angle - self.arc_start_angle

        # Normalize relative angle to [0, 2pi) or (-2pi, 0] depending on span sign
        if self.arc_span_angle > 0:
            while rel_angle < 0:
                rel_angle += 2 * math.pi
            while rel_angle >= 2 * math.pi:
                rel_angle -= 2 * math.pi
            in_span = 0 <= rel_angle <= self.arc_span_angle
        else:
            while rel_angle > 0:
                rel_angle -= 2 * math.pi
            while rel_angle <= -2 * math.pi:
                rel_angle += 2 * math.pi
            in_span = self.arc_span_angle <= rel_angle <= 0

        if in_span:
            # Closest point is directly on the circle arc
            u_rad = diff_from_center / dist_from_center
            q = center + u_rad * self.arc_radius
        else:
            # Check closest endpoint
            d0 = pt.distance_to_sq(self.p0)
            d1 = pt.distance_to_sq(self.p1)
            q = self.p0 if d0 < d1 else self.p1

        diff = pt - q
        dist = diff.length()
        if dist > 1e-7:
            normal = diff / dist
        else:
            normal = diff_from_center.normalized()
        return q, normal
