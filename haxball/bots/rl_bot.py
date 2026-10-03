"""
RL Bot that executes actions using a trained PyTorch policy checkpoint (.pt).
Supports decoupled observation space (61 dimensions) and standard observation space.
"""

from __future__ import annotations
import os
from typing import Tuple, Optional
import numpy as np
import torch

from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.models.mlp_policy import ActorCriticMLP

class RLBot(BaseBot):
    """
    Bot driven by a trained Reinforcement Learning neural network policy (.pt checkpoint).
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        policy: Optional[torch.nn.Module] = None,
        name: str = "RLBot",
        obs_dim: int = 61,
        act_dim: int = 3,
        device: str = "cpu"
    ):
        super().__init__(name=name)
        self.device = torch.device(device)
        self.obs_builder = DecoupledObservationBuilder()

        if policy is not None:
            self.policy = policy.to(self.device)
        else:
            self.policy = ActorCriticMLP(
                obs_dim=obs_dim,
                act_dim=act_dim,
                is_discrete=False,
                hidden_dim=128
            ).to(self.device)

            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.policy.load_state_dict(state_dict)
                print(f"[{self.name}] Carregou com sucesso os pesos de: {model_path}")
            elif model_path:
                print(f"[{self.name}] AVISO: Arquivo de modelo não encontrado: {model_path}")

        self.policy.eval()

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        """Calculates action for the bot using neural network inference."""
        obs = self.obs_builder.build_observation(game, player)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_mean = self.policy.actor(obs_tensor).cpu().numpy()[0]

        attack_sign = 1.0 if player.team == Team.RED else -1.0

        # Action components: [move_x, move_y, kick_logit] in ego-centric frame
        move_x = float(np.clip(action_mean[0], -1.0, 1.0)) * attack_sign
        move_y = float(np.clip(action_mean[1], -1.0, 1.0))
        kick = bool(action_mean[2] > 0.0)

        return (move_x, move_y, kick)
