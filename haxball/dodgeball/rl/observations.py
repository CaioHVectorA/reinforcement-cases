"""
Observation Builder for HaxBall Dodgeball RL.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional
from gymnasium import spaces
try:
    from core.vector import Vec2
    from core.constants import Team
    from core.disc import Disc
    from core.dodgeball_game import DodgeballGame
except (ImportError, ValueError):
    from ..core.vector import Vec2
    from ..core.constants import Team
    from ..core.disc import Disc
    from ..core.dodgeball_game import DodgeballGame

class DodgeballObservationBuilder:
    def __init__(self, max_teammates: int = 1, max_opponents: int = 2):
        self.max_teammates = max_teammates
        self.max_opponents = max_opponents
        self._dim = 10 + 8 + (max_teammates * 6) + (max_opponents * 6)

    @property
    def observation_dimension(self) -> int:
        return self._dim

    def get_observation_space(self) -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self._dim,), dtype=np.float32)

    def build_observation(self, game: DodgeballGame, ego_player: Disc) -> np.ndarray:
        bg_w = max(50.0, game.stadium.bg_width)
        bg_h = max(50.0, game.stadium.bg_height)
        v_max = 15.0

        is_red = (ego_player.team == Team.RED)
        ori = 1.0 if is_red else -1.0

        obs = []

        # 1. Ego State & Deadly Wall Proximity
        ego_x = ego_player.pos.x * ori
        ego_y = ego_player.pos.y
        ego_vx = ego_player.speed.x * ori
        ego_vy = ego_player.speed.y

        dist_top_wall = max(0.0, (bg_h - ego_player.radius - ego_y) / bg_h)
        dist_bottom_wall = max(0.0, (bg_h - ego_player.radius + ego_y) / bg_h)
        dist_back_wall = max(0.0, (bg_w - ego_player.radius + ego_x) / bg_w)
        dist_center = max(0.0, -ego_x / bg_w)

        can_kick = 1.0 if ego_player.pos.distance_to(game.ball.pos) <= (ego_player.radius + game.ball.radius + ego_player.kick_margin) else 0.0
        is_alive = 1.0 if game.is_alive(ego_player.player_id) else 0.0

        obs.extend([
            ego_x / bg_w,
            ego_y / bg_h,
            ego_vx / v_max,
            ego_vy / v_max,
            dist_top_wall,
            dist_bottom_wall,
            dist_back_wall,
            dist_center,
            can_kick,
            is_alive
        ])

        # 2. Ball Kinematics & Ballistic Threat
        b_x = game.ball.pos.x * ori
        b_y = game.ball.pos.y
        b_vx = game.ball.speed.x * ori
        b_vy = game.ball.speed.y
        b_speed = game.ball.speed.length()

        rel_bx = (b_x - ego_x) / bg_w
        rel_by = (b_y - ego_y) / bg_h
        b_in_own_half = 1.0 if b_x < 0.0 else 0.0

        diff_ball_ego = ego_player.pos - game.ball.pos
        dist_ball_ego = diff_ball_ego.length()

        time_to_impact = 1.0
        is_threat = 0.0

        if b_speed > 1.0 and dist_ball_ego > 1e-4:
            approach_speed = game.ball.speed.dot(diff_ball_ego / dist_ball_ego)
            if approach_speed > 1.0:
                is_threat = 1.0
                time_to_impact = min(1.0, (dist_ball_ego / max(1.0, approach_speed)) / 60.0)

        obs.extend([
            rel_bx,
            rel_by,
            b_vx / v_max,
            b_vy / v_max,
            min(1.0, b_speed / v_max),
            b_in_own_half,
            is_threat,
            time_to_impact
        ])

        # 3. Teammates
        teammates = [p for p in game.players if p.team == ego_player.team and p.player_id != ego_player.player_id]
        teammates.sort(key=lambda p: p.pos.distance_to(ego_player.pos))

        for i in range(self.max_teammates):
            if i < len(teammates):
                tm = teammates[i]
                tm_x = tm.pos.x * ori
                tm_y = tm.pos.y
                tm_alive = 1.0 if game.is_alive(tm.player_id) else 0.0
                rel_tm_x = (tm_x - ego_x) / bg_w
                rel_tm_y = (tm_y - ego_y) / bg_h

                in_line_of_fire = 0.0
                if tm_alive > 0.5 and (tm_x > ego_x):
                    if abs(tm_y - ego_y) < (tm.radius + ego_player.radius + 15.0):
                        in_line_of_fire = 1.0

                obs.extend([
                    rel_tm_x,
                    rel_tm_y,
                    tm.speed.x * ori / v_max,
                    tm.speed.y / v_max,
                    tm_alive,
                    in_line_of_fire
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 4. Opponents
        opponents = [p for p in game.players if p.team != ego_player.team]
        opponents.sort(key=lambda p: p.pos.distance_to(ego_player.pos))

        for i in range(self.max_opponents):
            if i < len(opponents):
                opp = opponents[i]
                opp_x = opp.pos.x * ori
                opp_y = opp.pos.y
                opp_alive = 1.0 if game.is_alive(opp.player_id) else 0.0
                rel_opp_x = (opp_x - ego_x) / bg_w
                rel_opp_y = (opp_y - ego_y) / bg_h

                opp_top = max(0.0, (bg_h - opp.radius - opp.pos.y) / bg_h)
                opp_bot = max(0.0, (bg_h - opp.radius + opp.pos.y) / bg_h)
                opp_dist_deadly = min(opp_top, opp_bot)

                obs.extend([
                    rel_opp_x,
                    rel_opp_y,
                    opp.speed.x * ori / v_max,
                    opp.speed.y / v_max,
                    opp_alive,
                    opp_dist_deadly
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)
