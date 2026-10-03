"""
Action spaces and decoders for HaxBall RL.
Supports Continuous, Discrete, MultiDiscrete, and direct HaxBall Online Headless API format.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Union
import numpy as np
from gymnasium import spaces

class ActionHandler:
    @staticmethod
    def get_continuous_space() -> spaces.Box:
        return spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    @staticmethod
    def get_discrete_space() -> spaces.Discrete:
        # 9 movement directions * 2 kick states = 18
        return spaces.Discrete(18)

    @staticmethod
    def get_multidiscrete_space() -> spaces.MultiDiscrete:
        # [x (-1, 0, 1), y (-1, 0, 1), kick (0, 1)]
        return spaces.MultiDiscrete([3, 3, 2])

    @staticmethod
    def decode_continuous(action: np.ndarray) -> Tuple[float, float, bool]:
        mx = float(np.clip(action[0], -1.0, 1.0))
        my = float(np.clip(action[1], -1.0, 1.0))
        kick = bool(action[2] > 0.0)
        return (mx, my, kick)

    @staticmethod
    def encode_discrete(mx: float, my: float, kick: bool) -> int:
        x_idx = 1 if abs(mx) < 0.3 else (2 if mx > 0 else 0)
        y_idx = 1 if abs(my) < 0.3 else (2 if my > 0 else 0)
        move_idx = y_idx * 3 + x_idx  # 0 to 8
        return move_idx + (9 if kick else 0)

    @staticmethod
    def decode_discrete(action: Union[int, np.integer]) -> Tuple[float, float, bool]:
        act_idx = int(action)
        kick = (act_idx >= 9)
        dir_idx = act_idx % 9

        x_idx = dir_idx % 3
        y_idx = dir_idx // 3
        mx = float(x_idx - 1)
        my = float(y_idx - 1)

        # Normalize diagonal length
        if mx != 0.0 and my != 0.0:
            mx *= 0.7071
            my *= 0.7071

        return (mx, my, kick)

    @staticmethod
    def decode_multidiscrete(action: np.ndarray) -> Tuple[float, float, bool]:
        # action[0] in {0, 1, 2} -> {-1.0, 0.0, 1.0}
        # action[1] in {0, 1, 2} -> {-1.0, 0.0, 1.0}
        # action[2] in {0, 1} -> {False, True}
        mx = float(action[0] - 1)
        my = float(action[1] - 1)
        kick = bool(action[2] == 1)
        return (mx, my, kick)

    @staticmethod
    def to_haxball_headless_format(mx: float, my: float, kick: bool) -> Dict[str, Any]:
        """
        Translates (mx, my, kick) to the official HaxBall Headless Host API format
        e.g. for room.setPlayerInputs({ xdir: -1 | 0 | 1, ydir: -1 | 0 | 1, kick: boolean })
        """
        xdir = 1 if mx > 0.3 else (-1 if mx < -0.3 else 0)
        # Note: in Haxball web canvas, ydir = 1 is down, -1 is up.
        # In our physics, positive Y is up, so ydir = -1 is up.
        ydir = -1 if my > 0.3 else (1 if my < -0.3 else 0)
        return {
            "xdir": xdir,
            "ydir": ydir,
            "kick": kick
        }
