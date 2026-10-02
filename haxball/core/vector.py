"""
2D Vector mathematics utility for HaxBall physics.
Designed for performance, clarity, and safety.
"""

from __future__ import annotations
import math
from typing import Tuple, Union

class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    @classmethod
    def from_iterable(cls, coords) -> Vec2:
        return cls(coords[0], coords[1])

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def copy(self) -> Vec2:
        return Vec2(self.x, self.y)

    def __repr__(self) -> str:
        return f"Vec2({self.x:.3f}, {self.y:.3f})"

    def __add__(self, other: Union[Vec2, Tuple[float, float]]) -> Vec2:
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        return Vec2(self.x + other[0], self.y + other[1])

    def __radd__(self, other: Union[Vec2, Tuple[float, float]]) -> Vec2:
        return self.__add__(other)

    def __sub__(self, other: Union[Vec2, Tuple[float, float]]) -> Vec2:
        if isinstance(other, Vec2):
            return Vec2(self.x - other.x, self.y - other.y)
        return Vec2(self.x - other[0], self.y - other[1])

    def __rsub__(self, other: Union[Vec2, Tuple[float, float]]) -> Vec2:
        if isinstance(other, Vec2):
            return Vec2(other.x - self.x, other.y - self.y)
        return Vec2(other[0] - self.x, other[1] - self.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        inv = 1.0 / scalar
        return Vec2(self.x * inv, self.y * inv)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return False
        return math.isclose(self.x, other.x, abs_tol=1e-7) and math.isclose(self.y, other.y, abs_tol=1e-7)

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def distance_to_sq(self, other: Vec2) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def distance_to(self, other: Vec2) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vec2) -> float:
        """2D cross product: returns z-component scalar (self.x * other.y - self.y * other.x)."""
        return self.x * other.y - self.y * other.x

    def normalized(self) -> Vec2:
        l = self.length()
        if l < 1e-9:
            return Vec2(0.0, 0.0)
        inv = 1.0 / l
        return Vec2(self.x * inv, self.y * inv)

    def perp(self) -> Vec2:
        """Returns perpendicular vector rotated 90 degrees CCW (-y, x)."""
        return Vec2(-self.y, self.x)

    def rotate(self, angle_rad: float) -> Vec2:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def clamp_length(self, max_len: float) -> Vec2:
        l_sq = self.length_sq()
        if l_sq > max_len * max_len:
            return self.normalized() * max_len
        return self.copy()
