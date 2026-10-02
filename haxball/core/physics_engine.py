"""
Accurate 2D physics engine for HaxBall.
Simulates player motion with kick damping, kick rate limiting, elastic/inelastic collisions,
curved and straight segment reflections, infinite planes, and kickoff barriers.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from haxball.core.vector import Vec2
from haxball.core.constants import CollisionMask, Team
from haxball.core.disc import Disc
from haxball.core.segment import Segment
from haxball.core.stadium import Stadium, Plane

class PhysicsEngine:
    def __init__(self, stadium: Stadium):
        self.stadium = stadium
        self.discs: List[Disc] = []
        self.players: List[Disc] = []
        self.ball: Optional[Disc] = None
        self.segments: List[Segment] = list(stadium.segments)
        self.planes: List[Plane] = list(stadium.planes)
        self.solver_iterations: int = 3

        # Kick cooldown tracker per player (in ticks)
        self.kick_cooldowns: Dict[int, int] = {}

    def reset_with_entities(self, ball: Disc, players: List[Disc], static_discs: Optional[List[Disc]] = None):
        """Sets active entities in physics world."""
        self.ball = ball
        self.players = players
        self.discs = [ball] + players
        if static_discs:
            self.discs.extend(static_discs)
        else:
            self.discs.extend(self.stadium.discs)

        for p in self.players:
            self.kick_cooldowns[p.player_id] = 0

    def apply_player_inputs(self, inputs: Dict[int, Tuple[float, float, bool]]):
        """
        Applies input commands (move_x, move_y, kick) for each player by player_id.
        In HaxBall:
        When holding kick: acceleration = kickingAcceleration, damping = kickingDamping.
        Otherwise: acceleration = acceleration, damping = damping.
        """
        for p in self.players:
            cmd = inputs.get(p.player_id, (0.0, 0.0, False))
            move_x, move_y, kick = cmd

            move_vec = Vec2(move_x, move_y)
            if move_vec.length_sq() > 1.0:
                move_vec = move_vec.normalized()

            # Decrement cooldown
            cd = self.kick_cooldowns.get(p.player_id, 0)
            if cd > 0:
                self.kick_cooldowns[p.player_id] = cd - 1

            if kick:
                acc = p.kicking_acceleration
                damp = p.kicking_damping
                p.is_kicking = True
            else:
                acc = p.acceleration
                damp = p.damping
                p.is_kicking = False

            if p.kick_flash > 0:
                p.kick_flash -= 1

            # HaxBall velocity update equation
            p.speed = (p.speed + move_vec * acc) * damp

    def step(self, inputs: Dict[int, Tuple[float, float, bool]]) -> Dict[str, Any]:
        """
        Executes a single physics tick (1/60s).
        Returns a dict of events.
        """
        events = {"kicks": []}

        # 1. Update player velocities from inputs
        self.apply_player_inputs(inputs)

        # 2. Apply damping to ball and other dynamic discs
        if self.ball:
            self.ball.speed = self.ball.speed * self.ball.damping

        for d in self.discs:
            if not d.is_player and d != self.ball and not d.is_static:
                d.speed = d.speed * d.damping

        # 3. Integrate positions
        for d in self.discs:
            if not d.is_static:
                d.pos = d.pos + d.speed

        # 4. Kicking mechanic (with rate limit to avoid continuous multi-frame spam)
        if self.ball:
            for p in self.players:
                if p.is_kicking:
                    dist = p.pos.distance_to(self.ball.pos)
                    reach = p.radius + self.ball.radius + p.kick_margin
                    cd = self.kick_cooldowns.get(p.player_id, 0)

                    if dist <= reach and cd == 0:
                        diff = self.ball.pos - p.pos
                        norm = diff.normalized() if diff.length_sq() > 1e-9 else Vec2(1.0, 0.0)

                        # Kick impulse
                        self.ball.speed = self.ball.speed + norm * p.kick_strength
                        if p.kick_back > 0:
                            p.speed = p.speed - norm * p.kick_back

                        # Trigger kick flash & cooldown (approx 200ms / 12 ticks)
                        p.kick_flash = 8
                        self.kick_cooldowns[p.player_id] = 12

                        events["kicks"].append({
                            "player_id": p.player_id,
                            "team": p.team,
                            "pos": p.pos.to_tuple()
                        })

        # 5. Collision resolution solver
        for _ in range(self.solver_iterations):
            self._resolve_disc_collisions()
            self._resolve_segment_collisions()
            self._resolve_plane_collisions()

        return events

    def _resolve_disc_collisions(self):
        n = len(self.discs)
        for i in range(n):
            d_a = self.discs[i]
            for j in range(i + 1, n):
                d_b = self.discs[j]

                if not d_a.can_collide_with(d_b):
                    continue

                diff = d_a.pos - d_b.pos
                dist_sq = diff.length_sq()
                r_sum = d_a.radius + d_b.radius
                if dist_sq < r_sum * r_sum:
                    dist = diff.length()
                    if dist > 1e-9:
                        normal = diff / dist
                    else:
                        normal = Vec2(1.0, 0.0)

                    penetration = r_sum - dist
                    w_a = d_a.invMass
                    w_b = d_b.invMass
                    w_tot = w_a + w_b

                    if w_tot > 1e-9:
                        # Position correction
                        d_a.pos = d_a.pos + normal * (penetration * (w_a / w_tot))
                        d_b.pos = d_b.pos - normal * (penetration * (w_b / w_tot))

                        # Velocity impulse
                        v_rel = d_a.speed - d_b.speed
                        v_n = v_rel.dot(normal)
                        if v_n < 0:
                            e = d_a.bCoef * d_b.bCoef
                            j_mag = -(1.0 + e) * v_n / w_tot
                            d_a.speed = d_a.speed + normal * (j_mag * w_a)
                            d_b.speed = d_b.speed - normal * (j_mag * w_b)

    def _resolve_segment_collisions(self):
        for d in self.discs:
            if d.is_static:
                continue

            for seg in self.segments:
                if not ((d.cGroup & seg.cMask) and (seg.cGroup & d.cMask)):
                    continue

                q, normal = seg.closest_point(d.pos)
                diff = d.pos - q
                dist = diff.length()

                if dist < d.radius:
                    penetration = d.radius - dist
                    d.pos = d.pos + normal * penetration

                    v_n = d.speed.dot(normal)
                    if v_n < 0:
                        e = d.bCoef * seg.bCoef
                        j_mag = -(1.0 + e) * v_n
                        d.speed = d.speed + normal * j_mag

    def _resolve_plane_collisions(self):
        for d in self.discs:
            if d.is_static:
                continue

            for plane in self.planes:
                if not ((d.cGroup & plane.cMask) and (plane.cGroup & d.cMask)):
                    continue

                dist = d.pos.dot(plane.normal) - plane.dist
                if dist < d.radius:
                    penetration = d.radius - dist
                    d.pos = d.pos + plane.normal * penetration

                    v_n = d.speed.dot(plane.normal)
                    if v_n < 0:
                        e = d.bCoef * plane.bCoef
                        d.speed = d.speed - plane.normal * ((1.0 + e) * v_n)
