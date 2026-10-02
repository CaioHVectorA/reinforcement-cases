"""
Wall Rebound Bot ("Tabela Master") for HaxBall.
Demonstrates competitive 'tryhard' mechanics by calculating wall bank shots
and ricochets to bypass defenders and score.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
from haxball.core.vector import Vec2
from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot

class WallReboundBot(BaseBot):
    def __init__(self, name: str = "WallReboundBot"):
        super().__init__(name=name)

    def _is_path_blocked(self, start: Vec2, end: Vec2, obstacles: list[Disc], radius_margin: float = 25.0) -> bool:
        """Checks if any opponent obstacle intersects line segment between start and end."""
        path = end - start
        path_len_sq = path.length_sq()
        if path_len_sq < 1e-9:
            return False

        for obs in obstacles:
            to_obs = obs.pos - start
            t = max(0.0, min(1.0, to_obs.dot(path) / path_len_sq))
            closest_pt = start + path * t
            if closest_pt.distance_to(obs.pos) < (obs.radius + radius_margin):
                return True
        return False

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        ball = game.ball
        if not ball:
            return (0.0, 0.0, False)

        stad = game.stadium
        is_red = (player.team == Team.RED)

        opp_goal_x = stad.bg_width if is_red else -stad.bg_width
        target_goal = Vec2(opp_goal_x, 0.0)

        # Get list of opponents
        opponents = [p for p in game.players if p.team != player.team]

        # Check if direct line to goal is blocked by an opponent
        direct_blocked = self._is_path_blocked(ball.pos, target_goal, opponents)

        aim_target = target_goal

        if direct_blocked:
            # Calculate Wall Bounce (Tabela / Bank Shot)
            # Wall Y boundaries (top and bottom)
            wall_y_top = stad.bg_height
            wall_y_bottom = -stad.bg_height

            # Choose the wall closer to the ball
            wall_y = wall_y_top if ball.pos.y >= 0 else wall_y_bottom

            # Optical reflection method:
            # Virtual mirrored goal has y = 2 * wall_y - goal_y
            mirrored_goal_y = 2.0 * wall_y - target_goal.y
            mirrored_target = Vec2(target_goal.x, mirrored_goal_y)

            # Intersection of line (ball -> mirrored_target) with wall_y
            t = (wall_y - ball.pos.y) / (mirrored_target.y - ball.pos.y + 1e-7)
            bounce_x = ball.pos.x + (mirrored_target.x - ball.pos.x) * t

            # Ensure bounce point is within court length
            if abs(bounce_x) < stad.bg_width * 0.95:
                bounce_point = Vec2(bounce_x, wall_y)
                # Verify that path to bounce point is clear
                if not self._is_path_blocked(ball.pos, bounce_point, opponents):
                    aim_target = bounce_point

        # Positioning behind ball along the aim target
        pred_ball_pos = ball.pos + ball.speed * 4.0
        aim_dir = (aim_target - pred_ball_pos).normalized()
        behind_distance = player.radius + ball.radius + 10.0
        desired_pos = pred_ball_pos - aim_dir * behind_distance

        # Wrong side check
        is_wrong_side = (player.pos.x > ball.pos.x) if is_red else (player.pos.x < ball.pos.x)

        if is_wrong_side:
            # Evade ball and reposition
            bypass_y = 65.0 if player.pos.y > ball.pos.y else -65.0
            recovery_target = Vec2(ball.pos.x - (45.0 if is_red else -45.0), ball.pos.y + bypass_y)
            move_dir = (recovery_target - player.pos).normalized()
            kick = False
        else:
            move_dir = (desired_pos - player.pos)
            if move_dir.length() < 16.0:
                move_dir = (ball.pos - player.pos).normalized()
            else:
                move_dir = move_dir.normalized()

            # Kicking logic
            dist_to_ball = player.pos.distance_to(ball.pos)
            reach = player.radius + ball.radius + player.kick_margin + 2.0
            kick = False

            if dist_to_ball <= reach:
                shot_vec = (ball.pos - player.pos).normalized()
                alignment = shot_vec.dot(aim_dir)
                if alignment > 0.3:
                    kick = True

        return (move_dir.x, move_dir.y, kick)
