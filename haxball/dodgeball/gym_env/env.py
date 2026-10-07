"""
Gymnasium environment for HaxBall Dodgeball.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from core.vector import Vec2
    from core.constants import Team, GameState
    from core.stadium import Stadium
    from core.dodgeball_game import DodgeballGame
    from rl.observations import DodgeballObservationBuilder
    from rl.rewards import DodgeballRewardEngine
except (ImportError, ValueError):
    from ..core.vector import Vec2
    from ..core.constants import Team, GameState
    from ..core.stadium import Stadium
    from ..core.dodgeball_game import DodgeballGame
    from ..rl.observations import DodgeballObservationBuilder
    from ..rl.rewards import DodgeballRewardEngine

DEFAULT_MAP = os.path.join(os.path.dirname(__file__), "..", "maps", "dodgeball.hbs")

class DodgeballEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        stadium_file: Optional[str] = None,
        red_players_count: int = 1,
        blue_players_count: int = 1,
        opponent_bot: Optional[Any] = None,
        max_steps: int = 3000,
        score_limit: int = 5,
    ):
        super().__init__()
        map_path = stadium_file or DEFAULT_MAP
        self.stadium = Stadium.load_from_file(map_path)

        self.red_players_count = red_players_count
        self.blue_players_count = blue_players_count
        self.opponent_bot = opponent_bot
        self.max_steps = max_steps
        self.score_limit = score_limit

        self.game = DodgeballGame(
            stadium=self.stadium,
            score_limit=score_limit,
            time_limit_secs=180,
            red_players_count=red_players_count,
            blue_players_count=blue_players_count,
        )

        self.obs_builder = DodgeballObservationBuilder(
            max_teammates=max(1, red_players_count - 1),
            max_opponents=max(1, blue_players_count)
        )
        self.reward_engine = DodgeballRewardEngine()

        self.observation_space = self.obs_builder.get_observation_space()
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.game.reset_match()

        ego = self.game.players[0]
        obs = self.obs_builder.build_observation(self.game, ego)
        info = {
            "red_score": self.game.red_score,
            "blue_score": self.game.blue_score,
            "alive_count": len(self.game.alive_players)
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        ego = self.game.players[0]

        mx = float(np.clip(action[0], -1.0, 1.0))
        my = float(np.clip(action[1], -1.0, 1.0))
        kick = bool(action[2] > 0.0)
        inputs = {ego.player_id: (mx, my, kick)}

        for p in self.game.players:
            if p.player_id == ego.player_id:
                continue
            if self.opponent_bot is not None and p.team != ego.team:
                inputs[p.player_id] = self.opponent_bot.act(self.game, p)
            else:
                inputs[p.player_id] = (0.0, 0.0, False)

        step_info = self.game.step(inputs)
        rewards_dict = self.reward_engine.compute_rewards(self.game, step_info)
        reward = rewards_dict.get(ego.player_id, 0.0)

        terminated = (self.game.state == GameState.GAME_OVER)
        truncated = (self.current_step >= self.max_steps)

        obs = self.obs_builder.build_observation(self.game, ego)
        info = {
            "red_score": self.game.red_score,
            "blue_score": self.game.blue_score,
            "eliminations": step_info["eliminations"]
        }
        return obs, reward, terminated, truncated, info
