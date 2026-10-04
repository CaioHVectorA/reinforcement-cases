"""
RL Bot that executes actions using any trained PyTorch policy checkpoint (.pt).
Supports:
- Behavioral Cloning models (e.g. HaxBallExpertPolicy with 18 discrete actions)
- Decoupled Entity-Attention Transformer models
- Standard ActorCriticMLP models (continuous or discrete)
- Custom state_dict or serialized torch.nn.Module
"""

from __future__ import annotations
import os
import math
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

from haxball.core.constants import Team
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.core.vector import Vec2
from haxball.bots.base_bot import BaseBot
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.models.mlp_policy import ActorCriticMLP
from haxball.rl.models.entity_attention import EntityAttentionPolicy
from haxball.rl.actions.action_space import ActionHandler

class HaxBallExpertPolicy(nn.Module):
    """Behavioral Cloning / Imitation Learning Network with Full Multi-Agent Field Vision."""
    def __init__(self, obs_dim: int = 61, num_actions: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RLBot(BaseBot):
    """
    Bot driven by a trained Reinforcement Learning / Imitation Learning neural network.
    Automatically adapts to checkpoint architecture, observation dimensions, and action formats.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        policy: Optional[torch.nn.Module] = None,
        name: str = "RLBot",
        device: str = "cpu"
    ):
        super().__init__(name=name)
        self.device = torch.device(device)
        self.obs_builder = DecoupledObservationBuilder()
        self.model_type = "mlp"
        self.expected_obs_dim = 61
        self.is_discrete_18 = False
        self.policy: Optional[torch.nn.Module] = None

        if policy is not None:
            self.policy = policy.to(self.device)
            self.policy.eval()
        elif model_path:
            self.load_model(model_path)
        else:
            # Default policy
            self.policy = ActorCriticMLP(
                obs_dim=61,
                act_dim=3,
                is_discrete=False,
                hidden_dim=128
            ).to(self.device)
            self.policy.eval()

    def load_model(self, model_path: str) -> bool:
        """
        Dynamically loads and reconstructs model from any .pt checkpoint file.
        """
        if not os.path.exists(model_path):
            print(f"[{self.name}] AVISO: Arquivo não encontrado: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state_dict if wrapped
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            elif isinstance(checkpoint, nn.Module):
                self.policy = checkpoint.to(self.device)
                self.policy.eval()
                self._inspect_model()
                print(f"[{self.name}] Modelo nn.Module carregado com sucesso de: {model_path}")
                return True
            else:
                print(f"[{self.name}] Formato de checkpoint desconhecido.")
                return False

            keys = list(state_dict.keys())
            
            # 1. Check if it's HaxBallExpertPolicy (net.0.weight, net.8.weight)
            if any(k.startswith("net.") for k in keys):
                in_dim = state_dict["net.0.weight"].shape[1]
                out_dim = state_dict[keys[-1]].shape[0] if "weight" in keys[-1] else state_dict[keys[-2]].shape[0]
                self.expected_obs_dim = in_dim
                self.is_discrete_18 = (out_dim == 18)
                self.model_type = "expert_bc"
                
                # Check if layer norm is present
                if any("1.weight" in k for k in keys):
                    self.policy = HaxBallExpertPolicy(obs_dim=in_dim, num_actions=out_dim).to(self.device)
                else:
                    # Dynamic Linear Net
                    layers = []
                    weight_keys = [k for k in keys if k.endswith(".weight")]
                    for idx, wk in enumerate(weight_keys):
                        w = state_dict[wk]
                        layers.append(nn.Linear(w.shape[1], w.shape[0]))
                        if idx < len(weight_keys) - 1:
                            layers.append(nn.ReLU())
                    self.policy = nn.Sequential(*layers).to(self.device)

                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                print(f"[{self.name}] Behavioral Cloning Net carregada! In: {in_dim}, Out: {out_dim} (18={self.is_discrete_18})")
                return True

            # 2. Check if it's EntityAttentionPolicy
            elif any("ball_encoder" in k for k in keys) or any("attention" in k for k in keys):
                self.model_type = "entity_attention"
                out_key = [k for k in keys if "actor" in k and "weight" in k][-1]
                out_dim = state_dict[out_key].shape[0]
                self.is_discrete_18 = (out_dim == 18)
                self.expected_obs_dim = 61
                self.policy = EntityAttentionPolicy(
                    act_dim=out_dim,
                    is_discrete=self.is_discrete_18
                ).to(self.device)
                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                print(f"[{self.name}] Entity-Attention Policy carregada com sucesso!")
                return True

            # 3. Check if it's ActorCriticMLP
            elif any("actor.0.weight" in k for k in keys) or any("actor" in k for k in keys):
                self.model_type = "actor_critic"
                in_dim = state_dict["actor.0.weight"].shape[1]
                out_keys = [k for k in keys if k.startswith("actor.") and k.endswith(".weight")]
                last_key = out_keys[-1]
                out_dim = state_dict[last_key].shape[0]
                self.expected_obs_dim = in_dim
                self.is_discrete_18 = (out_dim == 18)
                
                hidden_dim = state_dict["actor.0.weight"].shape[0]
                self.policy = ActorCriticMLP(
                    obs_dim=in_dim,
                    act_dim=out_dim,
                    is_discrete=self.is_discrete_18,
                    hidden_dim=hidden_dim
                ).to(self.device)
                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                print(f"[{self.name}] ActorCriticMLP carregado! In: {in_dim}, Out: {out_dim}")
                return True

            # 4. Fallback generic linear loading
            else:
                # Try loading into current policy
                if self.policy is not None:
                    self.policy.load_state_dict(state_dict, strict=False)
                    self.policy.eval()
                    print(f"[{self.name}] Pesos carregados parcialmente no modelo atual.")
                    return True

        except Exception as e:
            print(f"[{self.name}] Erro ao carregar modelo {model_path}: {e}")
            return False

        return False

    def _inspect_model(self):
        """Inspects custom loaded nn.Module to infer expected obs dim and action space."""
        try:
            # Test with 61 and 44
            for test_dim in [61, 44]:
                dummy = torch.zeros((1, test_dim), dtype=torch.float32, device=self.device)
                try:
                    out = self.policy(dummy)
                    if isinstance(out, tuple):
                        out = out[0]
                    self.expected_obs_dim = test_dim
                    self.is_discrete_18 = (out.shape[-1] == 18)
                    return
                except Exception:
                    pass
        except Exception:
            pass

    def _build_obs_vector(self, game: HaxBallGame, player: Disc) -> np.ndarray:
        """Constructs observation vector tailored to the model's expected dimension."""
        if self.expected_obs_dim == 61:
            return self.obs_builder.build_observation(game, player)

        # Build 44-dim egocentric vector
        ball = game.ball
        stad = game.stadium
        w = max(10.0, stad.bg_width)
        h = max(10.0, stad.bg_height)
        is_red = (player.team == Team.RED)
        attack_sign = 1.0 if is_red else -1.0

        obs_vec = np.zeros(self.expected_obs_dim, dtype=np.float32)
        if ball:
            ball_dx = (ball.pos.x - player.pos.x) * attack_sign
            ball_dy = ball.pos.y - player.pos.y
            ball_vx = ball.speed.x * attack_sign
            ball_vy = ball.speed.y
            obs_vec[0] = ball_dx / 400.0
            obs_vec[1] = ball_dy / 200.0
            obs_vec[2] = ball_vx / 10.0
            obs_vec[3] = ball_vy / 10.0

        # Opponent info
        opp = next((p for p in game.players if p.team != player.team), None)
        if opp:
            obs_vec[4] = ((opp.pos.x - player.pos.x) * attack_sign) / 400.0
            obs_vec[5] = (opp.pos.y - player.pos.y) / 200.0

        # Reach / Kick
        if ball and player.pos.distance_to(ball.pos) < (player.radius + ball.radius + 10.0):
            obs_vec[6] = 1.0

        return obs_vec

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        """Calculates action for the bot using neural network inference."""
        if self.policy is None:
            return (0.0, 0.0, False)

        obs = self._build_obs_vector(game, player)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if hasattr(self.policy, "actor"):
                out = self.policy.actor(obs_tensor)
            elif hasattr(self.policy, "get_action_and_value"):
                out, _, _, _ = self.policy.get_action_and_value(obs_tensor)
            else:
                out = self.policy(obs_tensor)

        if isinstance(out, tuple):
            out = out[0]

        is_red = (player.team == Team.RED)
        attack_sign = 1.0 if is_red else -1.0

        # 1. Discrete 18 actions
        if self.is_discrete_18 or out.shape[-1] == 18:
            act_idx = int(torch.argmax(out, dim=-1).cpu().item())
            mx, my, kick = ActionHandler.decode_discrete(act_idx)
            # Ego-centric attack sign inversion
            mx *= attack_sign
            return (mx, my, kick)

        # 2. Continuous 3 actions [mx, my, kick_logit]
        action_arr = out.cpu().numpy()[0]
        mx = float(np.clip(action_arr[0], -1.0, 1.0)) * attack_sign
        my = float(np.clip(action_arr[1], -1.0, 1.0))
        kick = bool(action_arr[2] > 0.0)

        return (mx, my, kick)
