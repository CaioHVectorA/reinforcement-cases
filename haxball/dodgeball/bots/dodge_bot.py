"""
DodgeBot Baseline for HaxBall Dodgeball.
Tactical bot specializing in:
1. Staying safe from deadly walls (wall repulsion).
2. Evading fast incoming balls to avoid suffering knockback into the wall.
3. Sniping opponents with lead aim to knock them into their deadly walls.
4. Timed explosive rebotes on incoming projectiles.
5. Dynamic active stance (never stationary or frozen).
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
        self.wall_safety_margin = 18.0

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
        reach = player.radius + ball.radius + player.kick_margin

        # 1. Evasion & Timed Rebote on incoming projectiles
        is_incoming = False
        if b_speed > 2.5 and dist_to_ball > 1e-4:
            approach_speed = ball.speed.dot(-diff_to_ball.normalized())
            if approach_speed > 1.2:
                is_incoming = True

        if is_incoming:
            # Se a bola entrou no alcance de chute: REBOTE EXPLOSIVO!
            if dist_to_ball <= reach:
                opponents = [p for p in game.players if p.team != player.team]
                if hasattr(game, "alive_players"):
                    opponents = [p for p in opponents if p.player_id in game.alive_players]
                target_opp = opponents[0] if opponents else None
                aim_pos = target_opp.pos if target_opp else Vec2(-200 if not is_red else 200, 0)
                shot_vec = (aim_pos - ball.pos).normalized() if (aim_pos - ball.pos).length_sq() > 1e-4 else Vec2(-1.0 if not is_red else 1.0, 0)
                return shot_vec.x, shot_vec.y, True

            # Se ainda não está no alcance de chute: esquiva lateral perpendicular
            ball_dir = ball.speed.normalized()
            perp_1 = Vec2(-ball_dir.y, ball_dir.x)
            perp_2 = Vec2(ball_dir.y, -ball_dir.x)

            pos_1 = player.pos + perp_1 * 40.0
            pos_2 = player.pos + perp_2 * 40.0

            margin_1 = min(bg_h - abs(pos_1.y), bg_w - abs(pos_1.x))
            margin_2 = min(bg_h - abs(pos_2.y), bg_w - abs(pos_2.x))

            best_perp = perp_1 if margin_1 >= margin_2 else perp_2
            move = self._apply_wall_avoidance(player, best_perp, bg_w, bg_h)
            return move.x, move.y, False

        # 2. Check teammates (Anti-crowding & clearance)
        teammates = [p for p in game.players if p.team == player.team and p.player_id != player.player_id]
        if hasattr(game, "alive_players"):
            teammates = [p for p in teammates if p.player_id in game.alive_players]

        teammate_on_ball = None
        for tm in teammates:
            if tm.pos.distance_to(ball.pos) < dist_to_ball:
                teammate_on_ball = tm
                break

        if teammate_on_ball is not None and teammate_on_ball.pos.distance_to(ball.pos) < 30.0:
            flank_y = 60.0 if player.pos.y > 0 else -60.0
            target_pos = Vec2(player.pos.x, flank_y)
            move_vec = (target_pos - player.pos).normalized() if (target_pos - player.pos).length() > 5.0 else Vec2(0, 0)
            move = self._apply_wall_avoidance(player, move_vec, bg_w, bg_h)
            return move.x, move.y, False

        # 3. Offense vs Stance
        ball_on_own_half = (ball.pos.x < 0) if is_red else (ball.pos.x > 0)
        ball_near_divider = abs(ball.pos.x) < 40.0

        if dist_to_ball <= reach:
            # CHUTE OFENSIVO: Disparar contra o oponente!
            opponents = [p for p in game.players if p.team != player.team]
            if hasattr(game, "alive_players"):
                opponents = [p for p in opponents if p.player_id in game.alive_players]

            target_opp = opponents[0] if opponents else None
            if target_opp:
                # Mirar com antecipação (lead aim)
                target_pos = target_opp.pos + target_opp.speed * 8.0
            else:
                target_pos = Vec2(-200 if not is_red else 200, 0)

            shot_dir = (target_pos - ball.pos).normalized() if (target_pos - ball.pos).length_sq() > 1e-4 else Vec2(-1.0 if not is_red else 1.0, 0)

            # Evitar fogo amigo em 2v2
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

            return shot_dir.x, shot_dir.y, True

        if ball_on_own_half or ball_near_divider:
            # Bola no campo próprio ou disputável na rede: CORRE DIRETO PRA BOLA
            move_vec = (ball.pos - player.pos).normalized()
        else:
            # Bola no campo adversário: POSTURA ATIVA (Stance / Shadowing)
            # O bot se move continuamente, espelhando o Y da bola e jukando pra não ficar parado
            base_x = -90.0 if is_red else 90.0
            ticks = getattr(game, "time_ticks", 0)
            juke = math.sin(ticks * 0.08) * 25.0
            target_y = max(-bg_h * 0.75, min(bg_h * 0.75, ball.pos.y * 0.80 + juke))
            target_pos = Vec2(base_x, target_y)
            diff = target_pos - player.pos
            move_vec = diff.normalized() if diff.length() > 6.0 else Vec2(0, 0)

        move = self._apply_wall_avoidance(player, move_vec, bg_w, bg_h)
        return move.x, move.y, False

    def _apply_wall_avoidance(self, player: Disc, move_vec: Vec2, bg_w: float, bg_h: float) -> Vec2:
        safe_m = self.wall_safety_margin
        is_red = (player.team == Team.RED)

        res = move_vec.copy()

        # Paredes externas mortais (topo e base)
        dist_top = bg_h - player.radius - player.pos.y
        if dist_top < safe_m:
            res.y = min(res.y, 0.0) - (safe_m - dist_top) / safe_m

        dist_bot = bg_h - player.radius + player.pos.y
        if dist_bot < safe_m:
            res.y = max(res.y, 0.0) + (safe_m - dist_bot) / safe_m

        # Parede de fundo mortal
        if is_red:
            dist_back = bg_w - player.radius + player.pos.x
            if dist_back < safe_m:
                res.x = max(res.x, 0.0) + (safe_m - dist_back) / safe_m
        else:
            dist_back = bg_w - player.radius - player.pos.x
            if dist_back < safe_m:
                res.x = min(res.x, 0.0) - (safe_m - dist_back) / safe_m

        # Divisória central (não é mortal, só impede cruzamento)
        dist_div = abs(player.pos.x) - player.radius
        if dist_div < 6.0:
            if is_red and res.x > 0:
                res.x = 0.0
            elif not is_red and res.x < 0:
                res.x = 0.0

        if res.length_sq() > 1.0:
            res = res.normalized()
        return res
