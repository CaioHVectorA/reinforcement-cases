"""
DodgeBot Baseline for HaxBall Dodgeball.
Tactical bot specializing in:
1. Staying safe from deadly walls (wall repulsion).
2. Evading fast incoming balls to avoid suffering knockback into the wall.
3. Sniping opponents with lead aim to knock them into their deadly walls.
4. Anti-friendly fire & corridor clearance for teammates.
"""

from __future__ import annotations
import math
from typing import Tuple, Any
try:
    from core.vector import Vec2
    from core.constants import Team
    from core.disc import Disc
except (ImportError, ValueError):
    from ..core.vector import Vec2
    from ..core.constants import Team
    from ..core.disc import Disc

class DodgeBot:
    def __init__(self, name: str = "DodgeBot_Pro"):
        self.name = name
        self.wall_safety_margin = 42.0

    def act(self, game: Any, player: Disc) -> Tuple[float, float, bool]:
        if hasattr(game, "is_alive") and not game.is_alive(player.player_id):
            return 0.0, 0.0, False

        bg_w = game.stadium.bg_width
        bg_h = game.stadium.bg_height
        ball = game.ball
        is_red = (player.team == Team.RED)

        b_speed = ball.speed.length()
        diff_to_ball = ball.pos - player.pos
        dist_to_ball = diff_to_ball.length()

        # 1. Evasion: Ball approaching fast?
        is_incoming = False
        if b_speed > 2.0 and dist_to_ball > 1e-4:
            approach_speed = ball.speed.dot(-diff_to_ball.normalized())
            if approach_speed > 1.2:
                is_incoming = True

        reach = player.radius + ball.radius + player.kick_margin
        if is_incoming:
            if dist_to_ball <= reach:
                # REBOTE TIMING: Chuta a bola de volta na direção do adversário!
                opponents = [p for p in game.players if p.team != player.team]
                if hasattr(game, "alive_players"):
                    opponents = [p for p in opponents if p.player_id in game.alive_players]
                target_opp = opponents[0] if opponents else None
                aim_pos = target_opp.pos if target_opp else Vec2(-200 if is_red else 200, 0)
                shot_vec = (aim_pos - ball.pos).normalized() if (aim_pos - ball.pos).length_sq() > 1e-4 else Vec2(-1.0 if is_red else 1.0, 0)
                return shot_vec.x, shot_vec.y, True

            ball_dir = ball.speed.normalized()
            perp_1 = Vec2(-ball_dir.y, ball_dir.x)
            perp_2 = Vec2(ball_dir.y, -ball_dir.x)

            pos_1 = player.pos + perp_1 * 50.0
            pos_2 = player.pos + perp_2 * 50.0

            margin_1 = min(bg_h - abs(pos_1.y), bg_w - abs(pos_1.x))
            margin_2 = min(bg_h - abs(pos_2.y), bg_w - abs(pos_2.x))

            best_perp = perp_1 if margin_1 >= margin_2 else perp_2
            move = self._apply_wall_avoidance(player, best_perp, bg_w, bg_h)
            return move.x, move.y, False

        # 2. Check if a teammate is closer and about to shoot
        teammates = [p for p in game.players if p.team == player.team and p.player_id != player.player_id]
        if hasattr(game, "alive_players"):
            teammates = [p for p in teammates if p.player_id in game.alive_players]

        teammate_on_ball = None
        for tm in teammates:
            if tm.pos.distance_to(ball.pos) < dist_to_ball:
                teammate_on_ball = tm
                break

        if teammate_on_ball is not None and teammate_on_ball.pos.distance_to(ball.pos) < 35.0:
            flank_y = 65.0 if player.pos.y > 0 else -65.0
            target_pos = Vec2(player.pos.x, flank_y)
            move_vec = (target_pos - player.pos).normalized() if (target_pos - player.pos).length() > 5.0 else Vec2(0, 0)
            move = self._apply_wall_avoidance(player, move_vec, bg_w, bg_h)
            return move.x, move.y, False

        # 3. Offense: Shoot ball to knock opponent into deadly wall!
        ball_on_own_half = (ball.pos.x < 0) if is_red else (ball.pos.x > 0)
        reach = player.radius + ball.radius + player.kick_margin

        if dist_to_ball <= reach:
            opponents = [p for p in game.players if p.team != player.team]
            if hasattr(game, "alive_players"):
                opponents = [p for p in opponents if p.player_id in game.alive_players]

            target_opp = opponents[0] if opponents else None
            if target_opp:
                target_pos = target_opp.pos + target_opp.speed * 12.0
            else:
                target_pos = Vec2(200 if is_red else -200, 0)

            shot_dir = (target_pos - ball.pos).normalized()

            # Anti-friendly fire
            has_friendly_block = False
            for tm in teammates:
                rel = tm.pos - ball.pos
                if rel.length() > 5.0 and shot_dir.dot(rel.normalized()) > 0.88:
                    has_friendly_block = True
                    break

            if has_friendly_block:
                wall_target_y = (bg_h - 10.0) if player.pos.y < 0 else (-bg_h + 10.0)
                bank_target = Vec2(0.0, wall_target_y)
                shot_dir = (bank_target - ball.pos).normalized()

            to_ball = (ball.pos - player.pos).normalized()
            return to_ball.x, to_ball.y, True

        if ball_on_own_half or abs(ball.pos.x) < 50.0:
            move_vec = (ball.pos - player.pos).normalized()
        else:
            home_x = -150.0 if is_red else 150.0
            target_pos = Vec2(home_x, ball.pos.y * 0.5)
            move_vec = (target_pos - player.pos).normalized() if (target_pos - player.pos).length() > 10.0 else Vec2(0, 0)

        move = self._apply_wall_avoidance(player, move_vec, bg_w, bg_h)
        return move.x, move.y, False

    def _apply_wall_avoidance(self, player: Disc, move_vec: Vec2, bg_w: float, bg_h: float) -> Vec2:
        safe_m = self.wall_safety_margin
        is_red = (player.team == Team.RED)

        res = move_vec.copy()

        dist_top = bg_h - player.radius - player.pos.y
        if dist_top < safe_m:
            res.y += -(safe_m - dist_top) / safe_m * 1.6

        dist_bot = bg_h - player.radius + player.pos.y
        if dist_bot < safe_m:
            res.y += (safe_m - dist_bot) / safe_m * 1.6

        if is_red:
            dist_back = bg_w - player.radius + player.pos.x
            if dist_back < safe_m:
                res.x += (safe_m - dist_back) / safe_m * 1.6
        else:
            dist_back = bg_w - player.radius - player.pos.x
            if dist_back < safe_m:
                res.x -= (safe_m - dist_back) / safe_m * 1.6

        dist_div = abs(player.pos.x)
        if dist_div < 20.0:
            res.x += (-0.5 if is_red else 0.5)

        if res.length_sq() > 1.0:
            res = res.normalized()
        return res
