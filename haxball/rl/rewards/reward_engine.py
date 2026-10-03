"""
Comprehensive Multi-Agent Team-Play Reward Engine for HaxBall RL.
Supports:
1. Shared Team Rewards (Global Utility for 1v1, 2v2, 3v3, 5v5)
2. Pass Completion & Assist Tracking
3. Anti-Clustering Spacing Penalty (breaks symmetric kindergarten soccer)
4. Defensive Covering & Anchor Role Incentives
5. Ball Progression & Shot Alignment toward Opponent Goal
6. Wall Rebound ("Tabela") Incentives
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
from haxball.core.vector import Vec2
from haxball.core.constants import Team
from haxball.core.game import HaxBallGame
from haxball.core.disc import Disc

class TeamPlayRewardEngine:
    def __init__(
        self,
        goal_reward: float = 15.0,
        concede_penalty: float = 15.0,
        assist_reward: float = 5.0,
        pass_completed_reward: float = 3.0,
        reception_reward: float = 2.0,
        interception_reward: float = 1.5,
        approach_ball_weight: float = 0.06,
        ball_to_goal_vel_weight: float = 0.12,
        kick_alignment_weight: float = 0.45,
        wall_rebound_weight: float = 0.5,
        touch_ball_bonus: float = 0.04,
        spacing_penalty_weight: float = 0.03,
        defensive_cover_reward: float = 0.02,
        cluster_threshold_dist: float = 75.0,
        pass_max_ticks: int = 50
    ):
        self.goal_reward = goal_reward
        self.concede_penalty = concede_penalty
        self.assist_reward = assist_reward
        self.pass_completed_reward = pass_completed_reward
        self.reception_reward = reception_reward
        self.interception_reward = interception_reward
        self.approach_ball_weight = approach_ball_weight
        self.ball_to_goal_vel_weight = ball_to_goal_vel_weight
        self.kick_alignment_weight = kick_alignment_weight
        self.wall_rebound_weight = wall_rebound_weight
        self.touch_ball_bonus = touch_ball_bonus
        self.spacing_penalty_weight = spacing_penalty_weight
        self.defensive_cover_reward = defensive_cover_reward
        self.cluster_threshold_dist = cluster_threshold_dist
        self.pass_max_ticks = pass_max_ticks

        # State tracking
        self.prev_dist_to_ball: Dict[int, float] = {}
        self.last_kicking_player_id: Optional[int] = None
        self.last_kicking_team: Optional[Team] = None
        self.last_kick_tick: int = 0
        self.passer_candidate_id: Optional[int] = None
        self.pass_candidate_tick: int = 0
        self.assist_candidates: Dict[Team, Optional[int]] = {Team.RED: None, Team.BLUE: None}

    def reset(self, game: HaxBallGame):
        self.prev_dist_to_ball.clear()
        for p in game.players:
            if game.ball:
                self.prev_dist_to_ball[p.player_id] = p.pos.distance_to(game.ball.pos)
        self.last_kicking_player_id = None
        self.last_kicking_team = None
        self.last_kick_tick = 0
        self.passer_candidate_id = None
        self.pass_candidate_tick = 0
        self.assist_candidates = {Team.RED: None, Team.BLUE: None}

    def compute_team_rewards(
        self,
        game: HaxBallGame,
        step_info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Computes per-player rewards for all players in the match.
        Returns a dictionary mapping player_id -> scalar reward.
        """
        rewards: Dict[int, float] = {p.player_id: 0.0 for p in game.players}
        ball = game.ball
        stad = game.stadium
        current_tick = game.time_ticks

        if not ball:
            return rewards

        # 1. Sparse Goal / Concede & Assists
        if step_info.get("goal_scored", False):
            scoring_team = step_info.get("scoring_team", Team.NONE)
            conceding_team = Team.BLUE if scoring_team == Team.RED else Team.RED

            for p in game.players:
                if p.team == scoring_team:
                    rewards[p.player_id] += self.goal_reward
                    # Assist reward
                    if self.assist_candidates[scoring_team] == p.player_id:
                        rewards[p.player_id] += self.assist_reward
                elif p.team == conceding_team:
                    rewards[p.player_id] -= self.concede_penalty

            # Reset touch history after goal
            self.passer_candidate_id = None
            self.assist_candidates = {Team.RED: None, Team.BLUE: None}
            return rewards

        # 2. Touch / Kick Events: Pass completion, assists & interceptions
        kicks = step_info.get("events", {}).get("kicks", [])
        for k in kicks:
            pid = k["player_id"]
            player = next((p for p in game.players if p.player_id == pid), None)
            if not player:
                continue

            # Check if this touch is a completed pass from a teammate
            if self.last_kicking_player_id is not None and self.last_kicking_player_id != pid:
                if self.last_kicking_team == player.team:
                    ticks_since_kick = current_tick - self.last_kick_tick
                    if ticks_since_kick <= self.pass_max_ticks:
                        # Team pass completed!
                        passer_id = self.last_kicking_player_id
                        rewards[passer_id] += self.pass_completed_reward
                        rewards[pid] += self.reception_reward
                        self.assist_candidates[player.team] = passer_id
                else:
                    # Interception! Stole ball from opponent
                    rewards[pid] += self.interception_reward
                    self.assist_candidates[player.team] = None

            self.last_kicking_player_id = pid
            self.last_kicking_team = player.team
            self.last_kick_tick = current_tick

            # Kick direction toward opponent goal
            attack_sign = 1.0 if player.team == Team.RED else -1.0
            opp_goal = Vec2(attack_sign * stad.bg_width, 0.0)
            to_goal_dir = (opp_goal - ball.pos).normalized()
            to_ball_dir = (ball.pos - player.pos).normalized()
            shot_alignment = to_ball_dir.dot(to_goal_dir)

            if shot_alignment > 0:
                rewards[pid] += shot_alignment * self.kick_alignment_weight

            # Wall rebound incentive: kicking obliquely at side walls
            if abs(to_ball_dir.y) > 0.4 and ball.speed.length() > 3.0:
                rewards[pid] += self.wall_rebound_weight

        # 3. Dense Positional Rewards per Player
        for player in game.players:
            pid = player.player_id
            attack_sign = 1.0 if player.team == Team.RED else -1.0
            opp_goal = Vec2(attack_sign * stad.bg_width, 0.0)
            own_goal = Vec2(-attack_sign * stad.bg_width, 0.0)

            # A. Dense ball approach (only for closest teammate to ball to prevent clustering)
            teammates = [p for p in game.players if p.team == player.team]
            closest_teammate = min(teammates, key=lambda t: t.pos.distance_to_sq(ball.pos))
            curr_dist = player.pos.distance_to(ball.pos)
            prev_dist = self.prev_dist_to_ball.get(pid, curr_dist)

            if player.player_id == closest_teammate.player_id:
                # Primary presser: gets reward for closing down ball
                dist_delta = prev_dist - curr_dist
                rewards[pid] += dist_delta * self.approach_ball_weight

                # Ball contact/possession bonus
                if curr_dist <= (player.radius + ball.radius + 6.0):
                    rewards[pid] += self.touch_ball_bonus

            self.prev_dist_to_ball[pid] = curr_dist

            # B. Anti-clustering spacing penalty with teammates
            for other_tm in teammates:
                if other_tm.player_id != pid:
                    tm_dist = player.pos.distance_to(other_tm.pos)
                    if tm_dist < self.cluster_threshold_dist:
                        # Crowding penalty
                        crowd_factor = (self.cluster_threshold_dist - tm_dist) / self.cluster_threshold_dist
                        rewards[pid] -= crowd_factor * self.spacing_penalty_weight

            # C. Defensive covering bonus
            # If ball is in our defensive half, reward the backline player for staying between ball and goal
            is_defensive_situation = (ball.pos.x * attack_sign) < 0
            if is_defensive_situation and len(teammates) > 1:
                # Find the deepest player on our team
                deepest_teammate = min(teammates, key=lambda t: t.pos.x * attack_sign)
                if player.player_id == deepest_teammate.player_id:
                    # Check if this player is between ball and our goal line
                    if (player.pos.x * attack_sign) < (ball.pos.x * attack_sign):
                        rewards[pid] += self.defensive_cover_reward

            # D. Ball moving toward opponent goal (shared team benefit)
            to_goal_dir = (opp_goal - ball.pos).normalized()
            ball_goal_speed = ball.speed.dot(to_goal_dir)
            if ball_goal_speed > 0:
                rewards[pid] += ball_goal_speed * self.ball_to_goal_vel_weight

        return rewards
