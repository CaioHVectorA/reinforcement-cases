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
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler

class HaxBallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        stadium_file: Optional[str] = None,
        stadium: Optional[Stadium] = None,
        opponent_bot: Optional[BaseBot] = None,
        discrete_actions: bool = True,
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

        self.action_handler = ActionHandler()
        self.obs_builder = DecoupledObservationBuilder()

        # Action space
        if self.discrete_actions:
            self.action_space = spaces.Discrete(18)
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

        # 61-dimension universal observation space
        self.obs_dim = self.obs_builder.obs_dim
        self.observation_space = self.obs_builder.get_observation_space()

    def _decode_action(self, action) -> Tuple[float, float, bool]:
        """Converts gym action to (move_x, move_y, kick)."""
        if self.discrete_actions or isinstance(action, (int, np.integer)):
            return self.action_handler.decode_discrete(int(action))
        else:
            return self.action_handler.decode_continuous(np.array(action, dtype=np.float32))

    def _get_obs(self) -> np.ndarray:
        red_player = next((p for p in self.game.players if p.team == Team.RED), None)
        if red_player is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        return self.obs_builder.build_observation(self.game, red_player)

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
