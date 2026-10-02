"""
Heuristic Bot for HaxBall.
Plays with positioning, ball trajectory anticipation, defensive recoveries,
and goal targeting.
"""

from __future__ import annotations
import math
from typing import Tuple
from haxball.core.vector import Vec2
from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot

class HeuristicBot(BaseBot):
    def __init__(self, name: str = "HeuristicBot"):
        super().__init__(name=name)

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        ball = game.ball
        if not ball:
            return (0.0, 0.0, False)

        stad = game.stadium
        is_red = (player.team == Team.RED)

        # Target opponent goal and own goal coordinates
        # Default goal is at stadium border X
        opp_goal_x = stad.bg_width if is_red else -stad.bg_width
        own_goal_x = -stad.bg_width if is_red else stad.bg_width
        target_goal = Vec2(opp_goal_x, 0.0)
        own_goal = Vec2(own_goal_x, 0.0)

        # Anticipate ball position based on current velocity (6 frames ahead)
        pred_ball_pos = ball.pos + ball.speed * 6.0

        # Vector from predicted ball to opponent goal
        to_goal = (target_goal - pred_ball_pos).normalized()

        # Desired position behind ball
        behind_distance = player.radius + ball.radius + 12.0
        desired_pos = pred_ball_pos - to_goal * behind_distance

        # Check if bot is on the wrong side (between ball and target goal)
        is_wrong_side = (player.pos.x > ball.pos.x) if is_red else (player.pos.x < ball.pos.x)

        if is_wrong_side:
            # Circle around ball to avoid kicking into own net
            # Choose bypass route (above or below ball based on current y)
            bypass_y = 60.0 if player.pos.y > ball.pos.y else -60.0
            recovery_target = Vec2(ball.pos.x - (40.0 if is_red else -40.0), ball.pos.y + bypass_y)
            move_dir = (recovery_target - player.pos).normalized()
            kick = False
        else:
            # Drive straight into position behind ball, pushing it towards goal
            move_dir = (desired_pos - player.pos)
            dist_to_desired = move_dir.length()
            if dist_to_desired < 15.0:
                # Attack the ball directly towards goal!
                move_dir = (ball.pos - player.pos).normalized()
            else:
                move_dir = move_dir.normalized()

            # Kicking decision:
            # Only kick if within reach AND ball is between player and target goal
            dist_to_ball = player.pos.distance_to(ball.pos)
            reach = player.radius + ball.radius + player.kick_margin + 2.0
            kick = False

            if dist_to_ball <= reach:
                angle_to_ball = (ball.pos - player.pos).normalized()
                shot_alignment = angle_to_ball.dot(to_goal)
                if shot_alignment > 0.35:  # Reasonably aligned with goal
                    kick = True

        return (move_dir.x, move_dir.y, kick)
