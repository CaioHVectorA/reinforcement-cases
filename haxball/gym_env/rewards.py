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
        touch_ball_weight: float = 0.10,
        defensive_position_weight: float = 0.03,
        whiff_kick_penalty: float = 0.01,
    ):
        self.goal_reward = goal_reward
        self.concede_penalty = concede_penalty
        self.approach_ball_weight = approach_ball_weight
        self.ball_to_goal_vel_weight = ball_to_goal_vel_weight
        self.kick_alignment_weight = kick_alignment_weight
        self.touch_ball_weight = touch_ball_weight
        self.defensive_position_weight = defensive_position_weight
        self.whiff_kick_penalty = whiff_kick_penalty

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

        is_red = (agent_team == Team.RED)
        opp_goal_x = stad.bg_width if is_red else -stad.bg_width
        own_goal_x = -stad.bg_width if is_red else stad.bg_width
        target_goal = Vec2(opp_goal_x, 0.0)
        own_goal = Vec2(own_goal_x, 0.0)

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

        # 4. Ball Touch / Contact Bonus
        reach = player.radius + ball.radius + player.kick_margin
        collisions = step_info.get("events", {}).get("disc_ball_collisions", [])
        for c in collisions:
            if c.get("player_id") == player.player_id:
                reward += self.touch_ball_weight

        # 5. Kick reward when aligned vs Whiff Penalty for air kicks
        kicks = step_info.get("events", {}).get("kicks", [])
        made_kick = False
        for k in kicks:
            if k["player_id"] == player.player_id:
                made_kick = True
                to_ball_dir = (ball.pos - player.pos).normalized()
                alignment = to_ball_dir.dot(to_goal_dir)
                if alignment > 0:
                    reward += alignment * self.kick_alignment_weight

        # Penalize pressing kick when nowhere near the ball (air kick / whiff spam)
        if player.is_kicking and not made_kick and curr_dist_to_ball > (reach + 10.0):
            reward -= self.whiff_kick_penalty

        # 6. Penalizar severamente camperar dentro do próprio gol (Anti-Goal-Camping)
        p_depth = player.pos.x if is_red else -player.pos.x
        if p_depth < -stad.bg_width + 15.0:
            reward -= 0.05  # Punição por entrar na própria rede e ficar parado lá

        # 7. Posicionamento Defensivo Ativo (apenas quando a bola estiver no campo de defesa)
        ball_in_defense = (ball.pos.x < 0.0) if is_red else (ball.pos.x > 0.0)
        if ball_in_defense and p_depth > -stad.bg_width + 30.0:
            dist_player_to_own_goal = player.pos.distance_to(own_goal)
            dist_ball_to_own_goal = ball.pos.distance_to(own_goal)
            if dist_player_to_own_goal < dist_ball_to_own_goal:
                reward += self.defensive_position_weight * 0.5

        return reward
