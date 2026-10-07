"""
Reward Engine for HaxBall Dodgeball RL.
Designed for the knockback-into-wall elimination mechanic.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
try:
    from core.vector import Vec2
    from core.constants import Team
    from core.dodgeball_game import DodgeballGame
except (ImportError, ValueError):
    from ..core.vector import Vec2
    from ..core.constants import Team
    from ..core.dodgeball_game import DodgeballGame

class DodgeballRewardEngine:
    def __init__(
        self,
        kill_reward: float = 10.0,
        death_penalty: float = 10.0,
        wall_suicide_penalty: float = 12.0,
        evasion_bonus: float = 1.5,
        knockback_hit_bonus: float = 2.0,
    ):
        self.kill_reward = kill_reward
        self.death_penalty = death_penalty
        self.wall_suicide_penalty = wall_suicide_penalty
        self.evasion_bonus = evasion_bonus
        self.knockback_hit_bonus = knockback_hit_bonus

    def compute_rewards(self, game: DodgeballGame, step_info: Dict[str, Any]) -> Dict[int, float]:
        rewards: Dict[int, float] = {p.player_id: 0.0 for p in game.players}
        bg_w = game.stadium.bg_width
        bg_h = game.stadium.bg_height

        # 1. Elimination events (occurring ONLY on wall impact!)
        for elim in step_info.get("eliminations", []):
            p_id = elim["player_id"]
            reason = elim["reason"]
            credited_team = elim["credited_team"]

            if reason == "wall_suicide":
                rewards[p_id] -= self.wall_suicide_penalty
            else:
                rewards[p_id] -= self.death_penalty

            for opp in game.players:
                if opp.team == credited_team and game.is_alive(opp.player_id):
                    rewards[opp.player_id] += self.kill_reward

        # 2. Continuous hazard avoidance shaping
        for p in game.players:
            if not game.is_alive(p.player_id):
                continue

            r = p.radius
            if p.team == Team.RED:
                dist_top = bg_h - r - p.pos.y
                dist_bot = bg_h - r + p.pos.y
                dist_back = bg_w - r + p.pos.x
            else:
                dist_top = bg_h - r - p.pos.y
                dist_bot = bg_h - r + p.pos.y
                dist_back = bg_w - r - p.pos.x

            min_dist_wall = min(dist_top, dist_bot, dist_back)
            if min_dist_wall < 35.0:
                overlap = (35.0 - max(0.0, min_dist_wall)) / 35.0
                rewards[p.player_id] -= 0.05 * overlap

        # 3. Successful knockback impact on opponent (pushing them with the ball!)
        for impact in step_info.get("events", {}).get("ball_player_impacts", []):
            p_id = impact["player_id"]
            team = impact["team"]
            speed = impact["impact_speed"]

            if speed > 2.5:
                # Credit the opponent who kicked this forceful ball!
                for opp in game.players:
                    if opp.team != team and opp.player_id == game.last_kicker_id:
                        rewards[opp.player_id] += self.knockback_hit_bonus

        return rewards
