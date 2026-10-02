"""
Goalkeeper Bot for HaxBall.
Maintains defensive line, guards the goal mouth, and clears dangerous balls.
"""

from __future__ import annotations
from typing import Tuple
from haxball.core.vector import Vec2
from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot

class GoalieBot(BaseBot):
    def __init__(self, name: str = "GoalieBot"):
        super().__init__(name=name)

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        ball = game.ball
        if not ball:
            return (0.0, 0.0, False)

        stad = game.stadium
        is_red = (player.team == Team.RED)

        # Own goal position
        goal_x = -stad.bg_width if is_red else stad.bg_width
        goal_pos = Vec2(goal_x, 0.0)

        # Goalie patrol X position (slightly in front of goal line)
        patrol_x = goal_x + (40.0 if is_red else -40.0)

        # Predict ball trajectory
        pred_ball = ball.pos + ball.speed * 8.0

        # Bisector / projection: clamp goalie Y between goal post bounds (-60 to 60)
        target_y = max(-65.0, min(65.0, pred_ball.y * 0.6))
        desired_pos = Vec2(patrol_x, target_y)

        # Distance to ball
        dist_to_ball = player.pos.distance_to(ball.pos)

        # If ball is close to own goal area, attack and clear it!
        in_danger_zone = (ball.pos.x < -stad.bg_width * 0.65) if is_red else (ball.pos.x > stad.bg_width * 0.65)

        if in_danger_zone and dist_to_ball < 80.0:
            move_dir = (ball.pos - player.pos).normalized()
            kick = dist_to_ball <= (player.radius + ball.radius + player.kick_margin + 2.0)
        else:
            to_desired = desired_pos - player.pos
            move_dir = to_desired.normalized() if to_desired.length() > 5.0 else Vec2(0, 0)
            kick = False

        return (move_dir.x, move_dir.y, kick)
