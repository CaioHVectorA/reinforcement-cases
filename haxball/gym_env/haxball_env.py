"""
Gymnasium environment for HaxBall Reinforcement Learning.
Compatible with Stable-Baselines3, CleanRL, and custom PyTorch agents.
"""

from __future__ import annotations
import math
from typing import Optional, Dict, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot
from haxball.bots.heuristic_bot import HeuristicBot
from haxball.gym_env.rewards import RewardShaper

class HaxBallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        stadium_file: Optional[str] = None,
        stadium: Optional[Stadium] = None,
        opponent_bot: Optional[BaseBot] = None,
        discrete_actions: bool = False,
        render_mode: Optional[str] = None,
        max_steps: int = 2000,
        score_limit: int = 3,
        reward_shaper: Optional[RewardShaper] = None
    ):
        super().__init__()

        # Load stadium
        if stadium is not None:
            self.stadium = stadium
        elif stadium_file is not None:
            self.stadium = Stadium.load_from_file(stadium_file)
        else:
            # Default to Classic
            self.stadium = Stadium()

        self.opponent_bot = opponent_bot
        self.discrete_actions = discrete_actions
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.score_limit = score_limit
        self.reward_shaper = reward_shaper or RewardShaper()

        self.game = HaxBallGame(
            stadium=self.stadium,
            score_limit=self.score_limit,
            time_limit_secs=180,
            red_players_count=1,
            blue_players_count=1
        )

        self.renderer = None
        self.current_step = 0

        # Action space
        if self.discrete_actions:
            # 8 movement directions + stop = 9, combined with kick (True/False) = 18 actions
            self.action_space = spaces.Discrete(18)
        else:
            # [move_x, move_y, kick] in [-1.0, 1.0]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

        # Observation space dimension: 20 features
        # (normalized positions, velocities, relative distances, alignments)
        self.obs_dim = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def _decode_action(self, action) -> Tuple[float, float, bool]:
        """Converts gym action to (move_x, move_y, kick)."""
        if self.discrete_actions:
            act_idx = int(action)
            kick = (act_idx >= 9)
            dir_idx = act_idx % 9

            # 9 directions: 0: center, 1: up, 2: up-right, 3: right, 4: down-right,
            # 5: down, 6: down-left, 7: left, 8: up-left
            dirs = [
                (0.0, 0.0),
                (0.0, 1.0),
                (0.707, 0.707),
                (1.0, 0.0),
                (0.707, -0.707),
                (0.0, -1.0),
                (-0.707, -0.707),
                (-1.0, 0.0),
                (-0.707, 0.707)
            ]
            mx, my = dirs[dir_idx]
            return (mx, my, kick)
        else:
            mx = float(np.clip(action[0], -1.0, 1.0))
            my = float(np.clip(action[1], -1.0, 1.0))
            kick = bool(action[2] > 0.0)
            return (mx, my, kick)

    def _get_obs(self) -> np.ndarray:
        ball = self.game.ball
        red_player = next((p for p in self.game.players if p.team == Team.RED), None)
        blue_player = next((p for p in self.game.players if p.team == Team.BLUE), None)

        w = self.stadium.width
        h = self.stadium.height
        diag = math.hypot(w, h)
        max_vel = 15.0

        target_goal_x = self.stadium.bg_width
        own_goal_x = -self.stadium.bg_width

        # Safe defaults
        b_x, b_y = (ball.pos.x / w, ball.pos.y / h) if ball else (0.0, 0.0)
        b_vx, b_vy = (ball.speed.x / max_vel, ball.speed.y / max_vel) if ball else (0.0, 0.0)

        p1_x, p1_y = (red_player.pos.x / w, red_player.pos.y / h) if red_player else (0.0, 0.0)
        p1_vx, p1_vy = (red_player.speed.x / max_vel, red_player.speed.y / max_vel) if red_player else (0.0, 0.0)

        p2_x, p2_y = (blue_player.pos.x / w, blue_player.pos.y / h) if blue_player else (0.0, 0.0)
        p2_vx, p2_vy = (blue_player.speed.x / max_vel, blue_player.speed.y / max_vel) if blue_player else (0.0, 0.0)

        # Relatives for agent (Red player)
        rel_ball_x = (ball.pos.x - red_player.pos.x) / w if (ball and red_player) else 0.0
        rel_ball_y = (ball.pos.y - red_player.pos.y) / h if (ball and red_player) else 0.0
        dist_to_ball = red_player.pos.distance_to(ball.pos) / diag if (ball and red_player) else 0.0

        can_kick = 0.0
        if red_player and ball:
            reach = red_player.radius + ball.radius + red_player.kick_margin
            can_kick = 1.0 if red_player.pos.distance_to(ball.pos) <= reach else 0.0

        rel_opp_x = (blue_player.pos.x - red_player.pos.x) / w if (red_player and blue_player) else 0.0
        rel_opp_y = (blue_player.pos.y - red_player.pos.y) / h if (red_player and blue_player) else 0.0

        # Ball to target goal alignment
        if ball:
            to_goal = Vec2(target_goal_x - ball.pos.x, 0.0 - ball.pos.y).normalized()
            to_ball = (ball.pos - red_player.pos).normalized() if red_player else Vec2(0, 0)
            shot_alignment = to_ball.dot(to_goal)
        else:
            shot_alignment = 0.0

        obs = np.array([
            b_x, b_y, b_vx, b_vy,
            p1_x, p1_y, p1_vx, p1_vy,
            p2_x, p2_y, p2_vx, p2_vy,
            rel_ball_x, rel_ball_y, dist_to_ball,
            can_kick,
            rel_opp_x, rel_opp_y,
            shot_alignment,
            float(self.game.red_score - self.game.blue_score)
        ], dtype=np.float32)

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.game.reset_match()
        self.reward_shaper.reset(self.game, Team.RED)
        self.current_step = 0

        obs = self._get_obs()
        info = {
            "red_score": self.game.red_score,
            "blue_score": self.game.blue_score
        }
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Red agent action
        red_cmd = self._decode_action(action)
        red_player = next((p for p in self.game.players if p.team == Team.RED), None)

        inputs = {}
        if red_player:
            inputs[red_player.player_id] = red_cmd

        # Blue opponent action
        blue_player = next((p for p in self.game.players if p.team == Team.BLUE), None)
        if blue_player:
            if self.opponent_bot:
                blue_cmd = self.opponent_bot.act(self.game, blue_player)
            else:
                blue_cmd = (0.0, 0.0, False)
            inputs[blue_player.player_id] = blue_cmd

        # Step game
        step_info = self.game.step(inputs)

        # Compute reward for Red agent
        reward = self.reward_shaper.compute_reward(self.game, Team.RED, step_info)

        # Check termination & truncation
        terminated = step_info["game_over"]
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = {
            "red_score": self.game.red_score,
            "blue_score": self.game.blue_score,
            "goal_scored": step_info["goal_scored"],
            "scoring_team": step_info["scoring_team"],
            "winner": step_info["winner"]
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.renderer is None:
            from haxball.renderer.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer(self.game)

        self.renderer.render()

    def close(self):
        if self.renderer is not None:
            import pygame
            pygame.quit()
            self.renderer = None
