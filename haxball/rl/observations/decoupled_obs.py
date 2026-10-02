"""
Universal Decoupled Observation Builder for HaxBall RL.
Decoupled from stadium map size (via continuous boundary normalization)
and decoupled from player count (supporting 1v1, 2v2, 3v3, 5v5 via masked entity slots).
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from gymnasium import spaces

from haxball.core.vector import Vec2
from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.rl.observations.base import BaseObservationBuilder

class DecoupledObservationBuilder(BaseObservationBuilder):
    def __init__(
        self,
        max_teammates: int = 4,
        max_opponents: int = 5,
        max_vel: float = 15.0
    ):
        self.max_teammates = max_teammates
        self.max_opponents = max_opponents
        self.max_vel = max_vel

        # 8 ball/global + 8 ego + (max_teammates * 5) + (max_opponents * 5)
        self.obs_dim = 16 + (self.max_teammates * 5) + (self.max_opponents * 5)

    def get_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def build_observation(self, game: HaxBallGame, player: Disc) -> np.ndarray:
        stad = game.stadium
        ball = game.ball

        # Map half-dimensions for normalization
        w = max(10.0, stad.bg_width)
        h = max(10.0, stad.bg_height)
        diag = math.hypot(w, h)

        # Team perspective: Red attacks towards +X, Blue attacks towards -X
        is_red = (player.team == Team.RED)
        attack_sign = 1.0 if is_red else -1.0

        # Opponent and Own goal positions
        opp_goal_x = attack_sign * w
        own_goal_x = -attack_sign * w

        # 1. Global & Ball Features (8)
        if ball:
            b_x = (ball.pos.x * attack_sign) / w
            b_y = ball.pos.y / h
            b_vx = (ball.speed.x * attack_sign) / self.max_vel
            b_vy = ball.speed.y / self.max_vel
            dist_ball_to_opp_goal = math.hypot(w - ball.pos.x * attack_sign, ball.pos.y) / diag
            dist_ball_to_own_goal = math.hypot(-w - ball.pos.x * attack_sign, ball.pos.y) / diag
        else:
            b_x = b_y = b_vx = b_vy = 0.0
            dist_ball_to_opp_goal = dist_ball_to_own_goal = 0.0

        score_diff = float(game.red_score - game.blue_score) * attack_sign
        time_ratio = float(game.time_seconds) / max(60.0, float(game.time_limit_secs)) if game.time_limit_secs > 0 else 0.0

        global_features = [
            b_x, b_y, b_vx, b_vy,
            dist_ball_to_opp_goal, dist_ball_to_own_goal,
            score_diff, time_ratio
        ]

        # 2. Ego Player Features (8)
        p_x = (player.pos.x * attack_sign) / w
        p_y = player.pos.y / h
        p_vx = (player.speed.x * attack_sign) / self.max_vel
        p_vy = player.speed.y / self.max_vel

        if ball:
            rel_b_x = ((ball.pos.x - player.pos.x) * attack_sign) / w
            rel_b_y = (ball.pos.y - player.pos.y) / h
            dist_to_ball = player.pos.distance_to(ball.pos) / diag
            reach = player.radius + ball.radius + player.kick_margin
            can_kick = 1.0 if player.pos.distance_to(ball.pos) <= reach else 0.0
        else:
            rel_b_x = rel_b_y = dist_to_ball = can_kick = 0.0

        ego_features = [
            p_x, p_y, p_vx, p_vy,
            rel_b_x, rel_b_y, dist_to_ball, can_kick
        ]

        # 3. Teammates Features (max_teammates * 5)
        teammates = [p for p in game.players if p.team == player.team and p.player_id != player.player_id]
        # Sort by distance to self
        teammates.sort(key=lambda t: player.pos.distance_to_sq(t.pos))

        tm_features: List[float] = []
        for i in range(self.max_teammates):
            if i < len(teammates):
                tm = teammates[i]
                rel_x = ((tm.pos.x - player.pos.x) * attack_sign) / w
                rel_y = (tm.pos.y - player.pos.y) / h
                tm_vx = (tm.speed.x * attack_sign) / self.max_vel
                tm_vy = tm.speed.y / self.max_vel
                active = 1.0
                tm_features.extend([rel_x, rel_y, tm_vx, tm_vy, active])
            else:
                # Padded slot
                tm_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 4. Opponents Features (max_opponents * 5)
        opponents = [p for p in game.players if p.team != player.team]
        opponents.sort(key=lambda o: player.pos.distance_to_sq(o.pos))

        opp_features: List[float] = []
        for i in range(self.max_opponents):
            if i < len(opponents):
                opp = opponents[i]
                rel_x = ((opp.pos.x - player.pos.x) * attack_sign) / w
                rel_y = (opp.pos.y - player.pos.y) / h
                opp_vx = (opp.speed.x * attack_sign) / self.max_vel
                opp_vy = opp.speed.y / self.max_vel
                active = 1.0
                opp_features.extend([rel_x, rel_y, opp_vx, opp_vy, active])
            else:
                # Padded slot
                opp_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.array(global_features + ego_features + tm_features + opp_features, dtype=np.float32)
        return obs
