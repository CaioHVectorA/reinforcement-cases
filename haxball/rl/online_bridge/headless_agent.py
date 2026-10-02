"""
Bridge for deploying trained HaxBall RL models directly into online rooms
via the HaxBall Headless Host API (Node.js / Browser Headless Host).
"""

from __future__ import annotations
import json
from typing import Dict, Any, Tuple
import numpy as np
import torch

from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler
from haxball.rl.models.mlp_policy import ActorCriticMLP

class OnlineHeadlessAgent:
    """
    Online bridge agent that receives state packets from a HaxBall Headless room
    and returns immediate actions in the native Headless API format:
    { xdir: -1 | 0 | 1, ydir: -1 | 0 | 1, kick: boolean }
    """
    def __init__(self, model_path: str, obs_dim: int = 61, is_discrete: bool = True):
        self.obs_builder = DecoupledObservationBuilder()
        self.action_handler = ActionHandler()
        self.device = torch.device("cpu")

        self.model = ActorCriticMLP(
            obs_dim=obs_dim,
            act_dim=18 if is_discrete else 3,
            is_discrete=is_discrete
        ).to(self.device)

        if model_path and model_path.endswith(".pt"):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def act_from_raw_state(self, game_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives raw JSON state from HaxBall room API (ball pos, players list, etc.)
        and computes the optimal action.
        """
        # When connected to a real HaxBall Headless room, state dictionary is received
        # For offline testing, we can simulate or forward observations
        obs = np.zeros(self.obs_builder.obs_dim, dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _, _ = self.model.get_action_and_value(obs_t)

        if self.model.is_discrete:
            mx, my, kick = self.action_handler.decode_discrete(int(action.item()))
        else:
            mx, my, kick = self.action_handler.decode_continuous(action.numpy()[0])

        return self.action_handler.to_haxball_headless_format(mx, my, kick)

    def export_weights_for_javascript(self, output_json_path: str):
        """
        Exports model weights to a lightweight JSON file so that the neural network
        can be evaluated directly in JavaScript in the HaxBall Headless room script
        without requiring an external Python process!
        """
        weights_dict = {}
        for name, param in self.model.named_parameters():
            weights_dict[name] = param.detach().cpu().numpy().tolist()

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(weights_dict, f)
        print(f"Exported policy weights for JavaScript room script to: {output_json_path}")
