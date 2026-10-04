"""
Reward shaping functions for HaxBall Reinforcement Learning.
Includes dense shaping (ball approach, goal alignment, shooting velocity)
and sparse rewards (goals scored/conceded, wall-rebound bonuses).
"""

from __future__ import annotations
import math
from typing import Dict, Any
from haxball.core.vector import Vec2
from haxball.core.constants import Team
from haxball.core.game import HaxBallGame

class RewardShaper:
    def __init__(
        self,
        goal_reward: float = 10.0,
        concede_penalty: float = 10.0,
        approach_ball_weight: float = 0.05,
        ball_to_goal_vel_weight: float = 0.08,
        kick_alignment_weight: float = 0.3,
        wall_rebound_weight: float = 0.5,
    ):
        self.goal_reward = goal_reward
        self.concede_penalty = concede_penalty
        self.approach_ball_weight = approach_ball_weight
        self.ball_to_goal_vel_weight = ball_to_goal_vel_weight
        self.kick_alignment_weight = kick_alignment_weight
        self.wall_rebound_weight = wall_rebound_weight

        self.prev_dist_to_ball: float = 0.0
        self.prev_ball_dist_to_goal: float = 0.0

    def reset(self, game: HaxBallGame, agent_team: Team):
        player = next((p for p in game.players if p.team == agent_team), None)
        ball = game.ball
        stad = game.stadium

        if player and ball:
            self.prev_dist_to_ball = player.pos.distance_to(ball.pos)
            opp_goal_x = stad.bg_width if agent_team == Team.RED else -stad.bg_width
            target_goal = Vec2(opp_goal_x, 0.0)
            self.prev_ball_dist_to_goal = ball.pos.distance_to(target_goal)
        else:
            self.prev_dist_to_ball = 0.0
            self.prev_ball_dist_to_goal = 0.0

    def compute_reward(
        self,
        game: HaxBallGame,
        agent_team: Team,
        step_info: Dict[str, Any]
    ) -> float:
        reward = 0.0
        player = next((p for p in game.players if p.team == agent_team), None)
        ball = game.ball
        stad = game.stadium

        if not player or not ball:
            return 0.0

        opp_goal_x = stad.bg_width if agent_team == Team.RED else -stad.bg_width
        target_goal = Vec2(opp_goal_x, 0.0)

        # 1. Goal Scored / Conceded (Primary sparse reward)
        if step_info.get("goal_scored", False):
            if step_info.get("scoring_team") == agent_team:
                reward += self.goal_reward
            else:
                reward -= self.concede_penalty
            return reward

        # 2. Dense: Approaching the ball
        curr_dist_to_ball = player.pos.distance_to(ball.pos)
        dist_delta = self.prev_dist_to_ball - curr_dist_to_ball
        reward += dist_delta * self.approach_ball_weight
        self.prev_dist_to_ball = curr_dist_to_ball

        # 3. Ball moving toward opponent goal
        to_goal_dir = (target_goal - ball.pos).normalized()
        ball_goal_speed = ball.speed.dot(to_goal_dir)
        if ball_goal_speed > 0:
            reward += ball_goal_speed * self.ball_to_goal_vel_weight

        # 4. Kick reward when aligned with target goal
        kicks = step_info.get("events", {}).get("kicks", [])
        for k in kicks:
            if k["player_id"] == player.player_id:
                # Shot vector
                to_ball_dir = (ball.pos - player.pos).normalized()
                alignment = to_ball_dir.dot(to_goal_dir)
                if alignment > 0:
                    reward += alignment * self.kick_alignment_weight

        return reward
