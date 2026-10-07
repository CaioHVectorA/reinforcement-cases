"""
Specialized 2D Physics Engine for HaxBall Dodgeball.
Physics behaviors requested:
1. Ball loses velocity when bouncing on walls (energy dissipation / bounce restitution).
2. Players DO NOT die upon touching the ball; instead, the ball applies a kinetic knockback force.
3. Light, responsive, authentic player movement with full momentum retention.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from .vector import Vec2
from .constants import CollisionMask, Team
from .disc import Disc
from .segment import Segment
from .stadium import Stadium

class PhysicsEngine:
    def __init__(self, stadium: Stadium):
        self.stadium = stadium
        self.discs: List[Disc] = []
        self.players: List[Disc] = []
        self.ball: Optional[Disc] = None
        self.segments: List[Segment] = list(stadium.segments)
        self.solver_iterations: int = 2

        self.kick_cooldowns: Dict[int, int] = {}

    def reset_with_entities(self, ball: Disc, players: List[Disc]):
        self.ball = ball
        self.players = players
        self.discs = [ball] + players
        for p in self.players:
            self.kick_cooldowns[p.player_id] = 0

    def apply_player_inputs(self, inputs: Dict[int, Tuple[float, float, bool]]):
        for p in self.players:
            cmd = inputs.get(p.player_id, (0.0, 0.0, False))
            mx, my, kick = cmd

            move_vec = Vec2(mx, my)
            if move_vec.length_sq() > 1.0:
                move_vec = move_vec.normalized()

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

            # Authentic HaxBall velocity integration
            p.speed = (p.speed + move_vec * acc) * damp

    def step(self, inputs: Dict[int, Tuple[float, float, bool]]) -> Dict[str, Any]:
        events = {"kicks": [], "ball_player_impacts": []}

        # 1. Apply inputs
        self.apply_player_inputs(inputs)

        # 2. Air resistance damping on dynamic ball
        if self.ball:
            self.ball.speed = self.ball.speed * self.ball.damping

        # 3. Kicking mechanics (Crisp, authentic kick impulse)
        if self.ball:
            for p in self.players:
                if p.is_kicking:
                    dist = p.pos.distance_to(self.ball.pos)
                    reach = p.radius + self.ball.radius + p.kick_margin
                    cd = self.kick_cooldowns.get(p.player_id, 0)

                    if dist <= reach and cd == 0:
                        diff = self.ball.pos - p.pos
                        norm = diff.normalized() if diff.length_sq() > 1e-9 else Vec2(1.0, 0.0)

                        # Timed Rebote / Deflection Mechanic:
                        # If the ball was traveling towards the player, a timed kick catches it and
                        # fires it back with explosive momentum!
                        incoming_speed_towards_player = -self.ball.speed.dot(norm)
                        rebote_bonus = max(0.0, incoming_speed_towards_player * 0.4)
                        total_kick_speed = p.kick_strength + rebote_bonus

                        self.ball.speed = norm * total_kick_speed

                        p.kick_flash = 6
                        self.kick_cooldowns[p.player_id] = 4

                        events["kicks"].append({
                            "player_id": p.player_id,
                            "team": p.team,
                            "pos": p.pos.to_tuple(),
                            "ball_speed": self.ball.speed.length(),
                            "is_rebote": incoming_speed_towards_player > 2.0
                        })

        # 4. Integrate positions and resolve collisions
        for d in self.discs:
            if not d.is_static:
                d.pos = d.pos + d.speed

        for _ in range(self.solver_iterations):
            self._resolve_disc_collisions(events)
            self._resolve_segment_collisions()

        return events

    def _resolve_disc_collisions(self, events: Dict[str, Any]):
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
                    normal = diff / dist if dist > 1e-9 else Vec2(1.0, 0.0)
                    penetration = r_sum - dist

                    is_player_ball = (d_a.is_player and d_b == self.ball) or (d_b.is_player and d_a == self.ball)

                    if is_player_ball:
                        player_disc = d_a if d_a.is_player else d_b
                        ball_disc = d_b if d_a.is_player else d_a
                        p_norm = normal if d_a.is_player else -normal  # Points from ball to player

                        # 1. POSITION SEPARATION:
                        # THE PLAYER CANNOT MOVE THE BALL!
                        # The ball is immovable to walking bodies: 100% of overlap separation pushes the PLAYER, NOT the ball!
                        player_disc.pos = player_disc.pos + p_norm * penetration

                        # 2. VELOCITY RESOLUTION:
                        # If player walks into the ball without kicking, the player CANNOT push the ball forward!
                        # The player is stopped/repelled:
                        p_vel_into_ball = player_disc.speed.dot(-p_norm)
                        if p_vel_into_ball > 0:
                            player_disc.speed = player_disc.speed - (-p_norm) * (p_vel_into_ball * 1.3)

                        # If the ball is moving with lethal/fast velocity into the player (PLAYER HIT BY BALL):
                        ball_vel_into_player = ball_disc.speed.dot(p_norm)
                        incoming_ball_speed = ball_disc.speed.length()

                        if incoming_ball_speed > 2.0 and ball_vel_into_player > 0.3:
                            # MASSIVE KNOCKBACK on the player:
                            knockback_force = max(7.0, incoming_ball_speed * 1.7)
                            player_disc.speed = player_disc.speed + p_norm * knockback_force

                            # Ball ricochets slightly / loses momentum on body impact
                            ball_disc.speed = ball_disc.speed * 0.35 - p_norm * 2.0

                            events["ball_player_impacts"].append({
                                "player_id": player_disc.player_id,
                                "team": player_disc.team,
                                "impact_speed": incoming_ball_speed
                            })
                    else:
                        w_a = d_a.invMass
                        w_b = d_b.invMass
                        w_tot = w_a + w_b

                        if w_tot > 1e-9:
                            # Position correction to separate overlapping discs
                            d_a.pos = d_a.pos + normal * (penetration * (w_a / w_tot))
                            d_b.pos = d_b.pos - normal * (penetration * (w_b / w_tot))

                            # Elastic collision impulse
                            v_rel = d_a.speed - d_b.speed
                            v_n = v_rel.dot(normal)

                            if v_n < 0:
                                e = d_a.bCoef * d_b.bCoef
                                j_mag = -(1.0 + e) * v_n / w_tot
                                d_a.speed = d_a.speed + normal * (j_mag * w_a)
                                d_b.speed = d_b.speed - normal * (j_mag * w_b)

    def _resolve_segment_collisions(self):
        """
        Segment wall collisions with speed loss on bounce!
        """
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
                        # Restitution: bCoef_disc * bCoef_wall (loses speed on bounce!)
                        e = d.bCoef * seg.bCoef
                        j_mag = -(1.0 + e) * v_n
                        d.speed = d.speed + normal * j_mag

                        # Extra energy dissipation on walls for the ball:
                        # Ball loses tangential speed when bouncing against walls
                        if d == self.ball:
                            d.speed = d.speed * 0.94
