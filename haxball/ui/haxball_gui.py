"""
HaxBall Futsal Studio - Arena Oficial com Painel de Checkboxes & Visual Autêntico.

Características Principais:
1. Visual 100% Idêntico à Referência Oficial de Futsal:
   - Quadra cinza fosca uniforme (#424D55 / rgb(66, 77, 85)) em toda a extensão.
   - Linhas brancas sólidas, marcações de out-of-bounds (ticks) fora das linhas.
   - 4 pontos amarelos (#FFCC00) nos cantos, 2 pontos de pênalti brancos em cada metade.
   - Traves autênticas: Rosa/Vermelha (#FF7B7B) na esquerda, Azul Celeste (#4DA3FF) na direita com trilhos guias.
   - Bola de futsal pequena laranja (#FFA000) com aro escuro.
   - Avatares com números e nicks renderizados abaixo (Umbabaraum, özil, lucasfera15, alex atacante, etc.).
2. Painel Interativo com Checkboxes:
   - Configuração de cada slot por CHECKBOXES intuitivos [✔] / [ ].
   - Seleção de Formato (3v3, 2v2, 1v1) e Velocidade por checkboxes.
   - Alternância rápida com tecla 'C', 'Tab' ou botão no topo.
3. Bots Humanizados & Clusterização de Funções:
   - Bot Fixo: Âncora defensiva, protege o corredor central e faz saída pelas alas.
   - Bot Ala: Abertura de espaço nas laterais, triangulação e tabelas nas paredes.
   - Bot Atacante / Pivô: Pressão alta agressiva e finalização rápida.
   - Inércia e suavização temporal para eliminar tremores robóticos ("reativos demais").
4. Áudio 100% Desativado (Silêncio Absoluto).
"""

from __future__ import annotations
import os
import sys
import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import pygame
import torch

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, FPS
from haxball.core.disc import Disc, hex_to_rgb
from haxball.core.segment import Segment
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot
from haxball.bots.heuristic_bot import HeuristicBot
from haxball.bots.wall_rebound_bot import WallReboundBot
from haxball.bots.futsal_3v3_team import Futsal3v3Bot, Futsal3v3Coordinator
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler
from haxball.rl.models.entity_attention import EntityAttentionPolicy
from haxball.rl.models.mlp_policy import ActorCriticMLP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = PROJECT_ROOT / "haxball" / "maps"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

STADIUM_CATALOG = {
    "futsal_3v3": {
        "title": "Futsal 3v3 GLH",
        "file": str(MAP_DIR / "futsal_3v3.hbs"),
        "desc": "Quadra Cinza Oficial Bazinga (550x240). Tabelas perfeitas e alta velocidade.",
        "default_format": 3
    },
    "futsal_2v2": {
        "title": "Futsal 2v2 Arena",
        "file": str(MAP_DIR / "futsal_2v2.hbs"),
        "desc": "Quadra Cinza Equilibrada (450x200). Espaçamento tático e transições.",
        "default_format": 2
    },
    "futsal_1v1": {
        "title": "Futsal 1v1 Rápido",
        "file": str(MAP_DIR / "futsal.hbs"),
        "desc": "Duelo individual (380x180). Ataque e defesa imediatos.",
        "default_format": 1
    }
}

CONTROLLER_OPTIONS = [
    ("human", "Humano", (255, 235, 50)),
    ("bc_ai", "IA BC", (255, 120, 120)),
    ("ppo_rl", "IA RL", (120, 225, 150)),
    ("bot_fixo", "Bot Fixo", (100, 185, 255)),
    ("bot_ala", "Bot Ala", (160, 215, 255)),
    ("bot_press", "Bot Atacante", (255, 145, 80)),
    ("bot_wall", "Bot Tabela", (195, 145, 255)),
    ("idle", "Inativo", (130, 135, 145)),
]

DEFAULT_NICKNAMES = {
    Team.RED: ["Umbabaraum", "özil", "lucasfera15"],
    Team.BLUE: ["alex atacante", "gimendez", "falcao12"]
}

SPEED_OPTIONS = [0.5, 1.0, 2.0, 5.0]


class HaxBallStudioApp:
    def __init__(self, width: int = 1240, height: int = 680):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("HaxBall Futsal - Arena Oficial com Painel de Checkboxes")

        self.clock = pygame.time.Clock()
        self.running = True
        self.is_paused = False
        self.speed_idx = 1  # 1.0x

        # Painel de Checkboxes (overlay modal)
        self.show_checkbox_panel = False

        # Fontes do Estilo HaxBall Oficial
        self.font_nick = pygame.font.SysFont("Verdana", 11, bold=False)
        self.font_num = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_score = pygame.font.SysFont("Trebuchet MS", 22, bold=True)
        self.font_timer = pygame.font.SysFont("Lucida Console", 18, bold=True)
        self.font_banner = pygame.font.SysFont("Trebuchet MS", 28, bold=True)
        self.font_ui_title = pygame.font.SysFont("Trebuchet MS", 15, bold=True)
        self.font_ui_text = pygame.font.SysFont("Arial", 12, bold=False)
        self.font_ui_bold = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_check = pygame.font.SysFont("Arial", 11, bold=True)

        self.kick_ripples: List[Dict[str, Any]] = []
        self.last_state: Optional[GameState] = None

        # Formato e Estádio
        self.current_stadium_key = "futsal_3v3"
        self.players_per_team = 3  # 3v3 default

        # Controladores por Slot (Default: Red 1 Humano, Red 2 e 3 IA BC; Blue 1 Fixo, Blue 2 Ala, Blue 3 Press)
        self.slot_controllers = {
            Team.RED: ["human", "bc_ai", "bc_ai"],
            Team.BLUE: ["bot_fixo", "bot_ala", "bot_press"]
        }

        # Nomes dos jogadores
        self.nicknames = dict(DEFAULT_NICKNAMES)

        # Observações e Ações
        self.obs_builder = DecoupledObservationBuilder()
        self.action_handler = ActionHandler()

        # Modelos de IA e Bots
        self._init_models_and_bots()

        # Estatísticas de Telemetria
        self.red_possession_ticks = 0
        self.blue_possession_ticks = 0
        self.red_shots = 0
        self.blue_shots = 0

        # Suavização de Ação do BC e RL (Momentum humano)
        self._bc_prev_action: Dict[int, Tuple[float, float]] = {}
        self._rl_prev_action: Dict[int, Tuple[float, float]] = {}

        # Botões e Checkboxes clicáveis da UI
        self.ui_buttons: List[Dict[str, Any]] = []

        # Inicializa partida
        self._init_game()

    def _init_models_and_bots(self):
        # 1. Behavioral Cloning (Entity Attention)
        self.bc_model = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True)
        bc_path = CHECKPOINT_DIR / "bc_futsal_3v3.pt"
        if bc_path.exists():
            try:
                self.bc_model.load_state_dict(torch.load(str(bc_path), map_location="cpu"))
            except Exception as e:
                print(f"[Studio] Aviso ao carregar BC: {e}")
        self.bc_model.eval()

        # 2. PPO Reinforcement Learning (Entity Attention)
        self.rl_model = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True)
        rl_path = CHECKPOINT_DIR / "haxball_rl_best.pt"
        if not rl_path.exists():
            rl_path = CHECKPOINT_DIR / "haxball_rl_potente.pt"
        if rl_path.exists():
            try:
                self.rl_model.load_state_dict(torch.load(str(rl_path), map_location="cpu"))
                print(f"[Studio] Modelo RL carregado com sucesso: {rl_path.name}")
            except Exception as e:
                print(f"[Studio] Aviso ao carregar RL: {e}")
        self.rl_model.eval()

        # 3. Coordenadores e Bots Especialistas com Humanização
        self.red_coord = Futsal3v3Coordinator(Team.RED)
        self.blue_coord = Futsal3v3Coordinator(Team.BLUE)

        # 4. Bots com papéis clusterizados
        self.heuristic_bot = HeuristicBot(name="Heuristic")
        self.wall_rebound_bot = WallReboundBot(name="WallRebound")

    def _init_game(self):
        stadium_info = STADIUM_CATALOG[self.current_stadium_key]
        stadium = Stadium.load_from_file(stadium_info["file"])
        self.game = HaxBallGame(
            stadium=stadium,
            score_limit=5,
            time_limit_secs=300,
            red_players_count=self.players_per_team,
            blue_players_count=self.players_per_team
        )
        self.red_possession_ticks = 0
        self.blue_possession_ticks = 0
        self.red_shots = 0
        self.blue_shots = 0
        self._calc_camera()

    def _calc_camera(self):
        # Enquadramento maximizado: a quadra preenche quase toda a janela
        margin_x = 35.0
        margin_y = 22.0
        header_h = 28.0

        avail_w = self.width - margin_x * 2.0
        avail_h = self.height - header_h - margin_y * 2.0

        stad = self.game.stadium
        scale_x = avail_w / (stad.width * 2.0)
        scale_y = avail_h / (stad.height * 2.0)
        self.scale = min(scale_x, scale_y)

        self.center_x = self.width / 2.0
        self.center_y = header_h + (avail_h / 2.0) + margin_y

    def world_to_screen(self, vec: Vec2) -> Tuple[int, int]:
        sx = int(self.center_x + vec.x * self.scale)
        sy = int(self.center_y - vec.y * self.scale)
        return (sx, sy)

    def world_len_to_screen(self, length: float) -> int:
        return max(1, int(round(length * self.scale)))

    def set_slot_controller(self, team: Team, slot_idx: int, controller_key: str):
        if slot_idx < len(self.slot_controllers[team]):
            self.slot_controllers[team][slot_idx] = controller_key

    def set_format(self, n_players: int):
        if n_players in (1, 2, 3) and n_players != self.players_per_team:
            self.players_per_team = n_players
            self._init_game()

    def set_stadium(self, stad_key: str):
        if stad_key in STADIUM_CATALOG and stad_key != self.current_stadium_key:
            self.current_stadium_key = stad_key
            self._init_game()

    def step_simulation(self):
        keys = pygame.key.get_pressed()

        # Inputs humanos (WASD / Setas)
        mx = 0.0
        my = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            mx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            mx += 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            my += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            my -= 1.0

        kick = bool(
            keys[pygame.K_SPACE] or
            keys[pygame.K_x] or
            keys[pygame.K_c] or
            keys[pygame.K_LSHIFT] or
            keys[pygame.K_RSHIFT]
        )

        red_players = [p for p in self.game.players if p.team == Team.RED]
        blue_players = [p for p in self.game.players if p.team == Team.BLUE]

        inputs_dict: Dict[int, Tuple[float, float, bool]] = {}

        for p_idx, p in enumerate(red_players):
            ctrl = self.slot_controllers[Team.RED][p_idx] if p_idx < len(self.slot_controllers[Team.RED]) else "idle"
            inputs_dict[p.player_id] = self._get_action_for_player(p, ctrl, mx, my, kick, Team.RED, p_idx)

        for p_idx, p in enumerate(blue_players):
            ctrl = self.slot_controllers[Team.BLUE][p_idx] if p_idx < len(self.slot_controllers[Team.BLUE]) else "idle"
            inputs_dict[p.player_id] = self._get_action_for_player(p, ctrl, mx, my, kick, Team.BLUE, p_idx)

        # Passo da física
        step_info = self.game.step(inputs_dict)

        # Rastreamento de finalizações
        events = step_info.get("events", {})
        kicks = events.get("kicks", [])
        if kicks:
            for k in kicks:
                kx, ky = self.world_to_screen(Vec2.from_iterable(k["pos"]))
                self.kick_ripples.append({"x": kx, "y": ky, "radius": 14, "alpha": 255})
                if k["team"] == Team.RED:
                    self.red_shots += 1
                else:
                    self.blue_shots += 1

        self.last_state = step_info.get("state")

        # Posse de bola
        ball = self.game.ball
        if ball:
            min_r = min((p.pos.distance_to(ball.pos) for p in red_players), default=999)
            min_b = min((p.pos.distance_to(ball.pos) for p in blue_players), default=999)
            if min_r < min_b and min_r < 90:
                self.red_possession_ticks += 1
            elif min_b < min_r and min_b < 90:
                self.blue_possession_ticks += 1

    def _get_action_for_player(
        self,
        player: Disc,
        ctrl: str,
        human_mx: float,
        human_my: float,
        human_kick: bool,
        team: Team,
        slot_idx: int
    ) -> Tuple[float, float, bool]:
        if ctrl == "human":
            return (human_mx, human_my, human_kick)

        elif ctrl == "bc_ai":
            # Inferência BC Calibrada com Suavização de Inércia
            with torch.no_grad():
                obs = self.obs_builder.build_observation(self.game, player)
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                logits = self.bc_model.actor(self.bc_model.forward_repr(obs_t))[0]
                probs = torch.softmax(logits, dim=-1)

                kick_prob = probs[9:].sum().item()
                # Limiar marginal calibrado: dispara chute se intenção for > 22%
                if kick_prob > 0.22:
                    act_idx = 9 + torch.argmax(probs[9:]).item()
                else:
                    act_idx = torch.argmax(probs[:9]).item()

                raw_act = self.action_handler.decode_discrete(act_idx)
                rx, ry, rkick = raw_act

                # Inversão necessária para o time Azul (Ego-perspective para coordenadas globais)
                attack_sign = 1.0 if player.team == Team.RED else -1.0
                world_rx = rx * attack_sign

                # Suavização de movimento para evitar tremores robóticos
                prev_x, prev_y = self._bc_prev_action.get(player.player_id, (0.0, 0.0))
                smooth_x = prev_x * 0.65 + world_rx * 0.35
                smooth_y = prev_y * 0.65 + ry * 0.35
                self._bc_prev_action[player.player_id] = (smooth_x, smooth_y)

                return (smooth_x, smooth_y, rkick)

        elif ctrl == "ppo_rl":
            with torch.no_grad():
                obs = self.obs_builder.build_observation(self.game, player)
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                logits = self.rl_model.actor(self.rl_model.forward_repr(obs_t))[0]
                probs = torch.softmax(logits, dim=-1)

                kick_prob = probs[9:].sum().item()
                if kick_prob > 0.22:
                    act_idx = 9 + torch.argmax(probs[9:]).item()
                else:
                    act_idx = torch.argmax(probs[:9]).item()

                raw_act = self.action_handler.decode_discrete(act_idx)
                rx, ry, rkick = raw_act

                # Inversão necessária para o time Azul (Ego-perspective para coordenadas globais)
                attack_sign = 1.0 if player.team == Team.RED else -1.0
                world_rx = rx * attack_sign

                prev_x, prev_y = self._rl_prev_action.get(player.player_id, (0.0, 0.0))
                smooth_x = prev_x * 0.65 + world_rx * 0.35
                smooth_y = prev_y * 0.65 + ry * 0.35
                self._rl_prev_action[player.player_id] = (smooth_x, smooth_y)

                return (smooth_x, smooth_y, rkick)

        elif ctrl == "bot_fixo":
            coord = self.red_coord if team == Team.RED else self.blue_coord
            return coord.get_action(self.game, player, role="fixo")

        elif ctrl == "bot_ala":
            coord = self.red_coord if team == Team.RED else self.blue_coord
            return coord.get_action(self.game, player, role="ala")

        elif ctrl == "bot_press":
            coord = self.red_coord if team == Team.RED else self.blue_coord
            return coord.get_action(self.game, player, role="press")

        elif ctrl == "bot_wall":
            return self.wall_rebound_bot.act(self.game, player)

        # "idle"
        return (0.0, 0.0, False)

    def render(self):
        self.ui_buttons.clear()

        # 1. Canvas com a Cor Oficial de Futsal (#424D55 / rgb(66, 77, 85))
        futsal_gray = (66, 77, 85)
        self.screen.fill(futsal_gray)

        stad = self.game.stadium

        # 2. Retângulo Principal da Quadra
        pw = self.world_len_to_screen(stad.bg_width * 2.0)
        ph = self.world_len_to_screen(stad.bg_height * 2.0)
        pitch_rect = pygame.Rect(
            int(self.center_x - pw / 2.0),
            int(self.center_y - ph / 2.0),
            pw,
            ph
        )

        # Preenchimento uniforme (sem corte de tons externos)
        pygame.draw.rect(self.screen, futsal_gray, pitch_rect)

        # Linha Externa Branca Sólida (#FFFFFF, width 2)
        line_color = (255, 255, 255)
        pygame.draw.rect(self.screen, line_color, pitch_rect, width=2)

        # 3. Marcações de Out-of-bounds (Ticks sutis fora das linhas)
        self._draw_boundary_ticks(pitch_rect)

        # 4. Linha Central e Círculo Central
        c_top = (int(self.center_x), pitch_rect.top)
        c_bottom = (int(self.center_x), pitch_rect.bottom)
        pygame.draw.line(self.screen, line_color, c_top, c_bottom, width=2)

        ko_rad = self.world_len_to_screen(stad.bg_kickoff_radius)
        center_pt = (int(self.center_x), int(self.center_y))
        pygame.draw.circle(self.screen, line_color, center_pt, ko_rad, width=2)
        pygame.draw.circle(self.screen, line_color, center_pt, 3)

        # Áreas de Futsal (D-Arcs)
        self._draw_futsal_penalty_arcs(pitch_rect)

        # Pontos de Pênalti Duplos (2 em cada metade ao longo do eixo central)
        self._draw_penalty_spots()

        # Trilhos Guias Azuis na Trave Direita
        self._draw_goal_rails(pitch_rect)

        # 4 Cantos com Pontos Amarelos (#FFCC00)
        corner_rad = 3
        c_color = (255, 204, 0)
        corners = [
            (pitch_rect.left, pitch_rect.top),
            (pitch_rect.right, pitch_rect.top),
            (pitch_rect.left, pitch_rect.bottom),
            (pitch_rect.right, pitch_rect.bottom)
        ]
        for c in corners:
            pygame.draw.circle(self.screen, c_color, c, corner_rad)
            pygame.draw.circle(self.screen, (20, 20, 20), c, corner_rad, width=1)

        # 5. Segmentos e Paredes
        for seg in self.game.physics.segments:
            if not seg.vis or seg.trait == "goalNet":
                continue
            color = seg.color_rgb
            if seg.is_curved:
                self._draw_curved_segment(seg, color)
            else:
                p0_s = self.world_to_screen(seg.p0)
                p1_s = self.world_to_screen(seg.p1)
                pygame.draw.line(self.screen, color, p0_s, p1_s, width=2)

        # 6. Traves Oficiais (Rosa/Vermelho na esquerda, Azul na direita)
        self._draw_goal_posts()

        # 7. Ondas de Impacto de Chute
        new_ripples = []
        for rip in self.kick_ripples:
            surf = pygame.Surface((rip["radius"] * 2 + 4, rip["radius"] * 2 + 4), pygame.SRCALPHA)
            alpha = max(0, int(rip["alpha"]))
            pygame.draw.circle(surf, (255, 255, 255, alpha), (rip["radius"] + 2, rip["radius"] + 2), rip["radius"], width=2)
            self.screen.blit(surf, (rip["x"] - rip["radius"] - 2, rip["y"] - rip["radius"] - 2))
            rip["radius"] += 2
            rip["alpha"] -= 28
            if rip["alpha"] > 0:
                new_ripples.append(rip)
        self.kick_ripples = new_ripples

        # 8. Bola e Jogadores com Estilo Autêntico
        if self.game.ball:
            self._draw_futsal_ball(self.game.ball)

        for p in self.game.players:
            self._draw_futsal_player(p)

        # 9. Placar Minimalista no Topo
        self._draw_minimal_scoreboard()

        # 10. Botão Flutuante do Painel de Checkboxes
        self._draw_toggle_button()

        # 11. Modal de Checkboxes (se aberto)
        if self.show_checkbox_panel:
            self._draw_checkbox_modal()

        pygame.display.flip()

    def _draw_boundary_ticks(self, rect: pygame.Rect):
        tick_len = 10
        pygame.draw.line(self.screen, (255, 255, 255), (int(self.center_x), rect.top - tick_len), (int(self.center_x), rect.top), width=2)
        pygame.draw.line(self.screen, (255, 255, 255), (int(self.center_x), rect.bottom), (int(self.center_x), rect.bottom + tick_len), width=2)

        for offset_x in [-self.world_len_to_screen(180), self.world_len_to_screen(180)]:
            x = int(self.center_x + offset_x)
            pygame.draw.line(self.screen, (255, 255, 255), (x, rect.top - tick_len), (x, rect.top), width=2)
            pygame.draw.line(self.screen, (255, 255, 255), (x, rect.bottom), (x, rect.bottom + tick_len), width=2)

        y_mid = int(self.center_y)
        pygame.draw.line(self.screen, (255, 255, 255), (rect.left - tick_len, y_mid), (rect.left, y_mid), width=2)
        pygame.draw.line(self.screen, (255, 255, 255), (rect.right, y_mid), (rect.right + tick_len, y_mid), width=2)

    def _draw_futsal_penalty_arcs(self, rect: pygame.Rect):
        line_color = (255, 255, 255)
        arc_w = self.world_len_to_screen(125.0)
        arc_h = self.world_len_to_screen(160.0)

        l_box = pygame.Rect(rect.left - arc_w, int(self.center_y - arc_h), arc_w * 2, arc_h * 2)
        pygame.draw.arc(self.screen, line_color, l_box, -math.pi / 2, math.pi / 2, width=2)

        r_box = pygame.Rect(rect.right - arc_w, int(self.center_y - arc_h), arc_w * 2, arc_h * 2)
        pygame.draw.arc(self.screen, line_color, r_box, math.pi / 2, 3 * math.pi / 2, width=2)

    def _draw_penalty_spots(self):
        p_color = (255, 255, 255)
        spot_rad = 3

        s1 = self.world_to_screen(Vec2(-380.0, 0.0))
        s2 = self.world_to_screen(Vec2(-250.0, 0.0))
        pygame.draw.circle(self.screen, p_color, s1, spot_rad)
        pygame.draw.circle(self.screen, p_color, s2, spot_rad)

        s3 = self.world_to_screen(Vec2(250.0, 0.0))
        s4 = self.world_to_screen(Vec2(380.0, 0.0))
        pygame.draw.circle(self.screen, p_color, s3, spot_rad)
        pygame.draw.circle(self.screen, p_color, s4, spot_rad)

    def _draw_goal_rails(self, rect: pygame.Rect):
        blue_rail_col = (74, 163, 255)
        rail_len = self.world_len_to_screen(45.0)
        y_top = self.world_to_screen(Vec2(550.0, 80.0))[1]
        y_bot = self.world_to_screen(Vec2(550.0, -80.0))[1]

        pygame.draw.line(self.screen, blue_rail_col, (rect.right, y_top), (rect.right + rail_len, y_top), width=2)
        pygame.draw.line(self.screen, blue_rail_col, (rect.right, y_bot), (rect.right + rail_len, y_bot), width=2)

    def _draw_goal_posts(self):
        post_rad = 5

        # Trave Esquerda (Pink/Red #FF7B7B)
        red_post_col = (255, 123, 123)
        p_l_top = self.world_to_screen(Vec2(-550.0, 80.0))
        p_l_bot = self.world_to_screen(Vec2(-550.0, -80.0))
        pygame.draw.circle(self.screen, red_post_col, p_l_top, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_l_top, post_rad, width=1)
        pygame.draw.circle(self.screen, red_post_col, p_l_bot, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_l_bot, post_rad, width=1)

        # Trave Direita (Sky Blue #4DA3FF)
        blue_post_col = (77, 163, 255)
        p_r_top = self.world_to_screen(Vec2(550.0, 80.0))
        p_r_bot = self.world_to_screen(Vec2(550.0, -80.0))
        pygame.draw.circle(self.screen, blue_post_col, p_r_top, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_r_top, post_rad, width=1)
        pygame.draw.circle(self.screen, blue_post_col, p_r_bot, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_r_bot, post_rad, width=1)

    def _draw_futsal_ball(self, ball: Disc):
        center_s = self.world_to_screen(ball.pos)
        rad_s = self.world_len_to_screen(ball.radius)

        # Bola Laranja Oficial (#FFA000)
        ball_col = (255, 160, 0)
        pygame.draw.circle(self.screen, ball_col, center_s, rad_s)
        # Borda escura fina
        pygame.draw.circle(self.screen, (20, 22, 25), center_s, rad_s, width=2)
        # Núcleo sutil
        pygame.draw.circle(self.screen, (230, 130, 0), center_s, max(1, rad_s // 3))

    def _draw_futsal_player(self, p: Disc):
        cx, cy = self.world_to_screen(p.pos)
        rad_s = self.world_len_to_screen(p.radius)

        # Anel de chute branco
        if p.is_kicking or p.kick_flash > 0:
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), rad_s + 4, width=3)

        # Corpo do Avatar
        if p.team == Team.RED:
            # Vermelho com anel branco interno
            pygame.draw.circle(self.screen, (211, 47, 47), (cx, cy), rad_s)
            pygame.draw.circle(self.screen, (20, 20, 20), (cx, cy), rad_s, width=2)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, rad_s - 4), width=1)
        else:
            # Azul com detalhe amarelo superior (como na foto de referência)
            pygame.draw.circle(self.screen, (21, 101, 192), (cx, cy), rad_s)
            y_box = pygame.Rect(cx - rad_s, cy - rad_s, rad_s * 2, rad_s)
            pygame.draw.arc(self.screen, (255, 214, 0), y_box, 0, math.pi, width=3)
            pygame.draw.circle(self.screen, (20, 20, 20), (cx, cy), rad_s, width=2)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, rad_s - 4), width=1)

        # Número da Camisa
        num_str = str(p.player_number)
        num_surf = self.font_num.render(num_str, True, (255, 255, 255))
        self.screen.blit(num_surf, num_surf.get_rect(center=(cx, cy)))

        # Nickname renderizado abaixo do avatar
        team_nicks = self.nicknames.get(p.team, [])
        idx = p.player_number - 1
        nick = team_nicks[idx] if idx < len(team_nicks) else f"Player_{p.player_number}"

        ctrl = self.slot_controllers[p.team][idx] if idx < len(self.slot_controllers[p.team]) else "idle"
        text_col = (255, 235, 60) if ctrl == "human" else (240, 240, 240)

        nick_surf = self.font_nick.render(nick, True, text_col)
        self.screen.blit(nick_surf, (cx - nick_surf.get_width() // 2, cy + rad_s + 4))

    def _draw_curved_segment(self, seg: Segment, color: Tuple[int, int, int]):
        steps = 18
        pts = []
        center = seg.arc_center
        start_a = seg.arc_start_angle
        span_a = seg.arc_span_angle
        radius = seg.arc_radius
        for i in range(steps + 1):
            t = i / steps
            ang = start_a + span_a * t
            pt = center + Vec2(math.cos(ang), math.sin(ang)) * radius
            pts.append(self.world_to_screen(pt))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, width=2)

    def _draw_minimal_scoreboard(self):
        r_score = str(self.game.red_score)
        b_score = str(self.game.blue_score)
        timer_str = self.game.time_string

        pill_w = 210
        pill_h = 28
        pill_rect = pygame.Rect(int(self.center_x - pill_w / 2.0), 4, pill_w, pill_h)
        pygame.draw.rect(self.screen, (32, 38, 44), pill_rect, border_radius=5)
        pygame.draw.rect(self.screen, (60, 72, 84), pill_rect, width=1, border_radius=5)

        r_txt = self.font_score.render(r_score, True, (255, 115, 115))
        self.screen.blit(r_txt, (pill_rect.left + 22, pill_rect.centery - r_txt.get_height() // 2))

        t_txt = self.font_timer.render(timer_str, True, (240, 240, 240))
        self.screen.blit(t_txt, t_txt.get_rect(center=pill_rect.center))

        b_txt = self.font_score.render(b_score, True, (110, 180, 255))
        self.screen.blit(b_txt, (pill_rect.right - 22 - b_txt.get_width(), pill_rect.centery - b_txt.get_height() // 2))

        if self.game.state == GameState.GOAL_CELEBRATION:
            t_str = "RED" if self.game.last_goal_team == Team.RED else "BLUE"
            col = (255, 115, 115) if self.game.last_goal_team == Team.RED else (110, 180, 255)
            banner = self.font_banner.render(f"GOL! {t_str} MARCOU!", True, col)
            b_r = banner.get_rect(center=(int(self.center_x), int(self.center_y - 65)))
            pygame.draw.rect(self.screen, (24, 28, 34), b_r.inflate(36, 16), border_radius=6)
            pygame.draw.rect(self.screen, col, b_r.inflate(36, 16), width=2, border_radius=6)
            self.screen.blit(banner, b_r)

    def _draw_toggle_button(self):
        btn_w = 260
        btn_h = 28
        btn_rect = pygame.Rect(20, 6, btn_w, btn_h)

        bg_col = (45, 95, 175) if self.show_checkbox_panel else (32, 40, 52)
        pygame.draw.rect(self.screen, bg_col, btn_rect, border_radius=5)
        pygame.draw.rect(self.screen, (70, 95, 130), btn_rect, width=1, border_radius=5)

        lbl = "⚙ OCULTAR CHECKBOXES [C]" if self.show_checkbox_panel else "⚙ CONFIGURAR JOGADORES [C]"
        txt = self.font_check.render(lbl, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=btn_rect.center))

        self.ui_buttons.append({"rect": btn_rect, "action": "toggle_panel"})

    def _draw_checkbox_modal(self):
        """Desenha o painel moderno e intuitivo de checkboxes para configurar slots e modos."""
        modal_w = 1140
        modal_h = 580
        modal_x = (self.width - modal_w) // 2
        modal_y = 42

        # Fundo do Modal
        modal_surf = pygame.Surface((modal_w, modal_h), pygame.SRCALPHA)
        modal_surf.fill((22, 27, 36, 245))  # Vidro escuro semi-transparente
        self.screen.blit(modal_surf, (modal_x, modal_y))

        modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)
        pygame.draw.rect(self.screen, (65, 85, 115), modal_rect, width=2, border_radius=8)

        # Cabeçalho do Modal
        header_txt = self.font_ui_title.render("PAINEL DE CONFIGURAÇÃO DE JOGADORES & IA (SELEÇÃO POR CHECKBOXES)", True, (255, 255, 255))
        self.screen.blit(header_txt, (modal_x + 20, modal_y + 16))

        # Botão Fechar [X]
        close_rect = pygame.Rect(modal_x + modal_w - 90, modal_y + 12, 75, 26)
        pygame.draw.rect(self.screen, (180, 55, 55), close_rect, border_radius=4)
        close_txt = self.font_check.render("FECHAR [C]", True, (255, 255, 255))
        self.screen.blit(close_txt, close_txt.get_rect(center=close_rect.center))
        self.ui_buttons.append({"rect": close_rect, "action": "toggle_panel"})

        # Linha Divisória
        pygame.draw.line(self.screen, (50, 65, 85), (modal_x + 20, modal_y + 48), (modal_x + modal_w - 20, modal_y + 48), width=1)

        cur_y = modal_y + 58

        # --- SEÇÃO 1: FORMATO DA PARTIDA & VELOCIDADE (CHECKBOXES) ---
        fmt_lbl = self.font_ui_bold.render("Formato da Partida:", True, (200, 215, 235))
        self.screen.blit(fmt_lbl, (modal_x + 24, cur_y + 2))

        chk_x = modal_x + 165
        for fmt, label in [(3, "3v3 Futsal"), (2, "2v2 Futsal"), (1, "1v1 Futsal")]:
            is_checked = (self.players_per_team == fmt)
            box_rect = pygame.Rect(chk_x, cur_y, 16, 16)
            self._draw_checkbox_widget(box_rect, is_checked)
            txt = self.font_ui_text.render(label, True, (255, 255, 255) if is_checked else (170, 180, 195))
            self.screen.blit(txt, (chk_x + 22, cur_y))

            hit_rect = pygame.Rect(chk_x, cur_y - 2, 110, 20)
            self.ui_buttons.append({"rect": hit_rect, "action": "set_format", "val": fmt})
            chk_x += 125

        # Velocidade
        spd_lbl = self.font_ui_bold.render("Velocidade:", True, (200, 215, 235))
        self.screen.blit(spd_lbl, (modal_x + 600, cur_y + 2))

        s_chk_x = modal_x + 695
        for s_idx, spd_val in enumerate(SPEED_OPTIONS):
            is_checked = (self.speed_idx == s_idx)
            box_rect = pygame.Rect(s_chk_x, cur_y, 16, 16)
            self._draw_checkbox_widget(box_rect, is_checked)
            txt = self.font_ui_text.render(f"{spd_val}x", True, (255, 215, 60) if is_checked else (170, 180, 195))
            self.screen.blit(txt, (s_chk_x + 22, cur_y))

            hit_rect = pygame.Rect(s_chk_x, cur_y - 2, 65, 20)
            self.ui_buttons.append({"rect": hit_rect, "action": "set_speed", "val": s_idx})
            s_chk_x += 75

        cur_y += 36
        pygame.draw.line(self.screen, (50, 65, 85), (modal_x + 20, cur_y), (modal_x + modal_w - 20, cur_y), width=1)
        cur_y += 14

        # --- SEÇÃO 2: SLOTS TIME VERMELHO (RED) ---
        red_title = self.font_ui_bold.render("🔴 TIME VERMELHO (RED)", True, (255, 115, 115))
        self.screen.blit(red_title, (modal_x + 24, cur_y))
        cur_y += 24

        for s_idx in range(self.players_per_team):
            cur_ctrl = self.slot_controllers[Team.RED][s_idx]
            nick = self.nicknames[Team.RED][s_idx] if s_idx < len(self.nicknames[Team.RED]) else f"Red_{s_idx+1}"

            row_lbl = self.font_ui_bold.render(f"Slot {s_idx+1} ({nick}):", True, (230, 230, 230))
            self.screen.blit(row_lbl, (modal_x + 36, cur_y + 2))

            opt_x = modal_x + 215
            for opt_key, opt_name, opt_color in CONTROLLER_OPTIONS:
                is_checked = (cur_ctrl == opt_key)
                b_rect = pygame.Rect(opt_x, cur_y, 15, 15)
                self._draw_checkbox_widget(b_rect, is_checked)

                txt_col = opt_color if is_checked else (160, 170, 185)
                t_surf = self.font_check.render(opt_name, True, txt_col)
                self.screen.blit(t_surf, (opt_x + 20, cur_y))

                hit_w = t_surf.get_width() + 26
                hit_rect = pygame.Rect(opt_x, cur_y - 2, hit_w, 20)
                self.ui_buttons.append({"rect": hit_rect, "action": "set_slot", "team": Team.RED, "slot": s_idx, "ctrl": opt_key})

                opt_x += hit_w + 12

            cur_y += 32

        cur_y += 8
        pygame.draw.line(self.screen, (50, 65, 85), (modal_x + 20, cur_y), (modal_x + modal_w - 20, cur_y), width=1)
        cur_y += 14

        # --- SEÇÃO 3: SLOTS TIME AZUL (BLUE) ---
        blue_title = self.font_ui_bold.render("🔵 TIME AZUL (BLUE)", True, (110, 180, 255))
        self.screen.blit(blue_title, (modal_x + 24, cur_y))
        cur_y += 24

        for s_idx in range(self.players_per_team):
            cur_ctrl = self.slot_controllers[Team.BLUE][s_idx]
            nick = self.nicknames[Team.BLUE][s_idx] if s_idx < len(self.nicknames[Team.BLUE]) else f"Blue_{s_idx+1}"

            row_lbl = self.font_ui_bold.render(f"Slot {s_idx+1} ({nick}):", True, (230, 230, 230))
            self.screen.blit(row_lbl, (modal_x + 36, cur_y + 2))

            opt_x = modal_x + 215
            for opt_key, opt_name, opt_color in CONTROLLER_OPTIONS:
                is_checked = (cur_ctrl == opt_key)
                b_rect = pygame.Rect(opt_x, cur_y, 15, 15)
                self._draw_checkbox_widget(b_rect, is_checked)

                txt_col = opt_color if is_checked else (160, 170, 185)
                t_surf = self.font_check.render(opt_name, True, txt_col)
                self.screen.blit(t_surf, (opt_x + 20, cur_y))

                hit_w = t_surf.get_width() + 26
                hit_rect = pygame.Rect(opt_x, cur_y - 2, hit_w, 20)
                self.ui_buttons.append({"rect": hit_rect, "action": "set_slot", "team": Team.BLUE, "slot": s_idx, "ctrl": opt_key})

                opt_x += hit_w + 12

            cur_y += 32

        # Dica no rodapé do modal
        cur_y = modal_y + modal_h - 45
        pygame.draw.line(self.screen, (50, 65, 85), (modal_x + 20, cur_y), (modal_x + modal_w - 20, cur_y), width=1)
        tip_text = "Dica: Você pode misturar como quiser (ex: Você + 2 IAs vs 3 Bots, ou 6 IAs em auto-jogo). Pressione 'C' para fechar e ver o jogo."
        tip_surf = self.font_ui_text.render(tip_text, True, (160, 175, 195))
        self.screen.blit(tip_surf, (modal_x + 24, cur_y + 12))

    def _draw_checkbox_widget(self, rect: pygame.Rect, checked: bool):
        """Desenha uma caixinha de checkbox nítida [✔] ou [ ]."""
        border_col = (75, 150, 240) if checked else (85, 100, 120)
        bg_col = (30, 80, 160) if checked else (35, 42, 54)

        pygame.draw.rect(self.screen, bg_col, rect, border_radius=3)
        pygame.draw.rect(self.screen, border_col, rect, width=1, border_radius=3)

        if checked:
            # Símbolo de check
            p1 = (rect.left + 3, rect.top + 7)
            p2 = (rect.left + 6, rect.top + 11)
            p3 = (rect.left + 12, rect.top + 3)
            pygame.draw.line(self.screen, (255, 255, 255), p1, p2, width=2)
            pygame.draw.line(self.screen, (255, 255, 255), p2, p3, width=2)

    def handle_click(self, pos: Tuple[int, int]):
        for btn in self.ui_buttons:
            if btn["rect"].collidepoint(pos):
                act = btn["action"]
                if act == "toggle_panel":
                    self.show_checkbox_panel = not self.show_checkbox_panel
                elif act == "set_format":
                    self.set_format(btn["val"])
                elif act == "set_speed":
                    self.speed_idx = btn["val"]
                elif act == "set_slot":
                    self.set_slot_controller(btn["team"], btn["slot"], btn["ctrl"])
                break

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.show_checkbox_panel:
                            self.show_checkbox_panel = False
                        else:
                            self.running = False
                    elif event.key in (pygame.K_c, pygame.K_TAB):
                        self.show_checkbox_panel = not self.show_checkbox_panel
                    elif event.key == pygame.K_r:
                        self.game.reset_round()
                        self._init_models_and_bots()
                    elif event.key in (pygame.K_p, pygame.K_PAUSE):
                        self.is_paused = not self.is_paused
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        self.set_format(int(event.unicode))
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.speed_idx = min(len(SPEED_OPTIONS) - 1, self.speed_idx + 1)
                    elif event.key == pygame.K_MINUS:
                        self.speed_idx = max(0, self.speed_idx - 1)

            if not self.is_paused:
                spd = SPEED_OPTIONS[self.speed_idx]
                if spd <= 1.0:
                    self.step_simulation()
                else:
                    for _ in range(int(spd)):
                        self.step_simulation()

            self.render()

            target_fps = 60 if SPEED_OPTIONS[self.speed_idx] >= 1.0 else 30
            self.clock.tick(target_fps)

        pygame.quit()


HaxBallApp = HaxBallStudioApp
HaxBallGUI = HaxBallStudioApp

def main():
    app = HaxBallStudioApp()
    app.run()


if __name__ == "__main__":
    main()
