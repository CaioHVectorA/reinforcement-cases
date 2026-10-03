"""
Authentic HaxBall GUI Suite & Recursive Self-Play Training Center.
Faithful to original HaxBall game look-and-feel:
- Real-time 2v2 (and NvN) live recursive self-play training directly on pitch.
- Speed acceleration up to 100x (run 100 physics & policy steps per frame!).
- Cognitive progression phases: from "burro" (random actions) to ball seeking,
  spacing, pass completion, and defensive covering.
- "Ficar Burro (Reset)" button to reset policy weights back to zero on the fly.
- Pre-trained checkpoints loader dashboard (.pt model selection & management).
- Expanded official map catalog: 2v2 Futsal, 3v3 Futsal (7899), 5v5 Futsal (9362),
  Micro 1v1, Big Stadium, Small Classic, Classic, and Dodgeball.
- Full hybrid controls: WASD or Arrow Keys + Space/X/C/Shift.
"""

from __future__ import annotations
import os
import sys
import time
import math
import threading
import pygame
from typing import Dict, Tuple, Optional, Any, List
import torch

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, FPS
from haxball.core.disc import hex_to_rgb
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots import NPC_BOTS, GAUNTLET_ORDER, BaseBot, RLBot
from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.rl.algorithms.ppo.ppo_trainer import PPOTrainer
from haxball.rl.algorithms.standard_rl.dqn_trainer import DQNTrainer
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer
from haxball.ui.widgets import (
    ICON_DISPATCH, draw_icon_stadium, draw_icon_lightning,
    draw_icon_robot, draw_icon_user, draw_icon_reset,
    draw_icon_play, draw_icon_pause, draw_icon_brain, draw_icon_help
)

MAP_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")

STADIUM_CATALOG = {
    "futsal_2v2": {
        "title": "Futsal 2v2 Arena (Ideal p/ Treino)",
        "file": os.path.join(MAP_DIR, "futsal_2v2.hbs"),
        "desc": "Quadra balanceada (450x200) para 4 jogadores. Paredes elásticas (bCoef 1.25) e condução de bola colada.",
        "format": 2
    },
    "futsal_3v3": {
        "title": "Futsal 3x3 GLH (Oficial 7899)",
        "file": os.path.join(MAP_DIR, "futsal_3v3.hbs"),
        "desc": "O mapa oficial mais jogado do mundo por Bazinga!. Bola ultraleve (6.3), alta velocidade e tabelas perfeitas.",
        "format": 3
    },
    "futsal_5v5": {
        "title": "Futsal 5x5 GLH (Oficial 9362)",
        "file": os.path.join(MAP_DIR, "futsal_5v5.hbs"),
        "desc": "Mapa oficial expandido (1080x532) para 10 jogadores, com áreas e traves de futsal regulamentares.",
        "format": 5
    },
    "micro_1v1": {
        "title": "Micro 1v1 Arena (Duelo Rápido)",
        "file": os.path.join(MAP_DIR, "micro_1v1.hbs"),
        "desc": "Arena ultracompacta (340x160) com transições imediatas, ricochetes brutais e ritmo frenético de 1 contra 1.",
        "format": 1
    },
    "big_stadium": {
        "title": "Big Stadium Oficial (Campo Aberto)",
        "file": os.path.join(MAP_DIR, "big_stadium.hbs"),
        "desc": "Campo amplo de grama (840x400) para futebol tático, passes longos em profundidade e cruzamentos.",
        "format": 3
    },
    "small_classic": {
        "title": "Small Classic (Futebol Rápido)",
        "file": os.path.join(MAP_DIR, "small_classic.hbs"),
        "desc": "Campo clássico reduzido (360x180) para partidas rápidas com física tradicional de grama.",
        "format": 2
    },
    "classic": {
        "title": "Classic Stadium Oficial",
        "file": os.path.join(MAP_DIR, "classic.hbs"),
        "desc": "Campo oficial padrão de HaxBall com traves arredondadas e física de futebol tradicional.",
        "format": 1
    },
    "dodgeball": {
        "title": "Dodgeball Arena (Queimada)",
        "file": os.path.join(MAP_DIR, "dodgeball.hbs"),
        "desc": "Quadra de queimada com barreira central que bloqueia jogadores e permite a passagem da bola.",
        "format": 1
    }
}

SPEED_LEVELS = [1, 2, 5, 10, 25, 50, 100]

class HaxBallApp:
    def __init__(
        self,
        width: int = 1280,
        height: int = 768,
        mode: str = "self_play",
        model_path: Optional[str] = None,
        bot_key: str = "rl"
    ):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("HaxBall RL Studio - End-to-End Control Center & Battle Arena")

        self.clock = pygame.time.Clock()
        self.running = True
        self.is_paused = False

        # Modes: "self_play" or "human"
        self.play_mode = mode
        self.speed_multiplier = 1

        # Fonts
        self.font_score = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_time = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font_hud = pygame.font.SysFont("Verdana", 13, bold=True)
        self.font_regular = pygame.font.SysFont("Arial", 13)
        self.font_bold = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_title = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font_player = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_telemetry = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 11)

        # Modals
        self.show_stadium_modal = False
        self.show_rl_modal = False
        self.show_help_modal = False
        self.rl_modal_tab = "models"  # "models", "training", "bots"

        # Checkpoints & Model State
        self.active_checkpoint_name = "Nenhum"
        self.checkpoint_feedback = ""
        self.feedback_time = 0.0

        # Current Stadium & Match
        self.current_stadium_key = "futsal_2v2"
        self.team_format = 2
        self.self_play_trainer: Optional[SelfPlay2v2Trainer] = None
        self._init_game(self.current_stadium_key, self.team_format)

        # Auto-detect initial model path
        default_model = model_path
        if default_model is None:
            for candidate in ["checkpoints/fase5_pro_master_1M.pt", "checkpoints/meu_haxball_bot.pt"]:
                if os.path.exists(candidate):
                    default_model = candidate
                    break

        # Bots Catalog
        self.model_path = default_model
        self.bot_catalog: Dict[str, BaseBot] = {
            "rl": RLBot(model_path=default_model, name="RL Bot (Trained)")
        }
        for k, (cls, label, title, desc, col) in NPC_BOTS.items():
            self.bot_catalog[k] = cls(name=title.split(" ")[0])

        self.gauntlet_mode = (bot_key == "gauntlet")
        self.gauntlet_index = 0
        if self.gauntlet_mode:
            self.active_bot_key = GAUNTLET_ORDER[0]
            self.team_format = 1
        else:
            self.active_bot_key = bot_key if bot_key in self.bot_catalog else "press"

        if default_model and os.path.exists(default_model):
            self.load_checkpoint(default_model)

        # Background RL Trainer
        self.training_thread: Optional[threading.Thread] = None
        self.training_active = False
        self.training_status = "Inativo"
        self.training_algo = "PPO"
        self.training_policy = "mlp"
        self.training_metrics = {"step": 0, "reward": 0.0, "win_rate": 0.0, "loss": 0.0}

    def _init_game(self, stadium_key: str, players_count: int = 2):
        self.current_stadium_key = stadium_key
        stadium_info = STADIUM_CATALOG[stadium_key]
        stadium = Stadium.load_from_file(stadium_info["file"])

        self.team_format = players_count
        self.game = HaxBallGame(
            stadium=stadium,
            score_limit=3,
            time_limit_secs=180,
            red_players_count=players_count,
            blue_players_count=players_count
        )

        # Initialize or reconnect self-play trainer
        old_policy = self.self_play_trainer.policy if self.self_play_trainer else None
        self.self_play_trainer = SelfPlay2v2Trainer(game=self.game, rollout_steps=256)
        if old_policy is not None:
            try:
                self.self_play_trainer.policy.load_state_dict(old_policy.state_dict())
            except Exception:
                pass

        self._calc_camera()

    def _calc_camera(self):
        top_offset = 64
        bottom_bar_h = 60
        avail_w = self.width - 60
        avail_h = self.height - top_offset - bottom_bar_h - 30

        stad = self.game.stadium
        scale_x = avail_w / (stad.width * 2.0)
        scale_y = avail_h / (stad.height * 2.0)
        self.scale = min(scale_x, scale_y)

        self.center_x = self.width / 2.0
        self.center_y = top_offset + avail_h / 2.0 + 8

    def world_to_screen(self, vec: Vec2) -> Tuple[int, int]:
        sx = int(self.center_x + vec.x * self.scale)
        sy = int(self.center_y - vec.y * self.scale)
        return (sx, sy)

    def world_len_to_screen(self, length: float) -> int:
        return max(1, int(round(length * self.scale)))

    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        cp_dir = "checkpoints"
        if not os.path.exists(cp_dir):
            return []
        files = [f for f in os.listdir(cp_dir) if f.endswith(".pt")]

        tier_info = {
            "fase1_iniciante_25k.pt": {"title": "Fase 1: Iniciante (25k)", "desc": "Reflexos básicos de perseguição da bola", "badge": "25k", "color": (235, 150, 40)},
            "fase2_amador_70k.pt": {"title": "Fase 2: Amador (70k)", "desc": "Alinhamento ao gol e chutes direcionados", "badge": "70k", "color": (220, 200, 50)},
            "fase3_intermediario_200k.pt": {"title": "Fase 3: Intermediário (200k)", "desc": "Cobertura de trave e finalizações perigosas", "badge": "200k", "color": (58, 142, 230)},
            "fase4_veterano_500k.pt": {"title": "Fase 4: Veterano (500k)", "desc": "Passes intencionais, desarmes e tabelas", "badge": "500k", "color": (155, 100, 230)},
            "fase5_pro_master_1M.pt": {"title": "Fase 5: Pro Master (1M)", "desc": "Inteligência máxima com antecipação", "badge": "1M ★", "color": (60, 210, 120)},
            "meu_haxball_bot.pt": {"title": "Modelo Atual Recente", "desc": "Último checkpoint salvo do auto-confronto", "badge": "Recente", "color": (80, 190, 160)},
        }

        results = []
        ordered_keys = ["fase1_iniciante_25k.pt", "fase2_amador_70k.pt", "fase3_intermediario_200k.pt", "fase4_veterano_500k.pt", "fase5_pro_master_1M.pt", "meu_haxball_bot.pt"]

        for k in ordered_keys:
            if k in files:
                info = tier_info[k]
                full_p = os.path.join(cp_dir, k)
                results.append({
                    "filename": k,
                    "title": info["title"],
                    "desc": info["desc"],
                    "badge": info["badge"],
                    "color": info["color"],
                    "size_kb": int(os.path.getsize(full_p) / 1024)
                })

        for f in sorted(files):
            if f not in ordered_keys:
                full_p = os.path.join(cp_dir, f)
                results.append({
                    "filename": f,
                    "title": f"Custom: {f}",
                    "desc": "Modelo customizado salvo",
                    "badge": "Custom",
                    "color": (160, 170, 185),
                    "size_kb": int(os.path.getsize(full_p) / 1024)
                })

        return results

    def load_checkpoint(self, filename: str):
        full_path = os.path.join("checkpoints", filename) if not os.path.isabs(filename) else filename
        if os.path.exists(full_path):
            state_dict = torch.load(full_path, map_location="cpu")
            if "rl" in self.bot_catalog:
                self.bot_catalog["rl"].policy.load_state_dict(state_dict)
                self.bot_catalog["rl"].policy.eval()
            if self.self_play_trainer:
                self.self_play_trainer.policy.load_state_dict(state_dict)
            self.active_checkpoint_name = os.path.basename(full_path)
            self.active_bot_key = "rl"
            self.checkpoint_feedback = f"Carregado: {self.active_checkpoint_name}"
            self.feedback_time = time.time()
            print(f"[GUI] Checkpoint carregado com sucesso: {self.active_checkpoint_name}")

    def save_current_checkpoint(self, custom_name: Optional[str] = None):
        os.makedirs("checkpoints", exist_ok=True)
        if custom_name is None:
            custom_name = f"treino_gui_{int(time.time())}.pt"
        full_path = os.path.join("checkpoints", custom_name)
        if self.self_play_trainer:
            torch.save(self.self_play_trainer.policy.state_dict(), full_path)
            self.active_checkpoint_name = custom_name
            self.checkpoint_feedback = f"Salvo: {custom_name}"
            self.feedback_time = time.time()
            print(f"[GUI] Checkpoint salvo em: {full_path}")

    def cycle_bot(self):
        keys = list(self.bot_catalog.keys())
        if "gauntlet" not in keys:
            keys.append("gauntlet")
        idx = keys.index(self.active_bot_key) if self.active_bot_key in keys else (keys.index("gauntlet") if getattr(self, "gauntlet_mode", False) else 0)
        nxt_key = keys[(idx + 1) % len(keys)]
        if nxt_key == "gauntlet":
            self.gauntlet_mode = True
            self.gauntlet_index = 0
            self.active_bot_key = GAUNTLET_ORDER[0]
            self.checkpoint_feedback = f"Gauntlet: {self.active_bot_key.upper()} (1/{len(GAUNTLET_ORDER)})"
        else:
            self.gauntlet_mode = False
            self.active_bot_key = nxt_key
            self.checkpoint_feedback = f"Oponente: {nxt_key.upper()}"
        self.feedback_time = time.time()

    def get_player_inputs(self) -> Dict[int, Tuple[float, float, bool]]:
        keys = pygame.key.get_pressed()
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

        kick = (
            keys[pygame.K_SPACE] or
            keys[pygame.K_x] or
            keys[pygame.K_c] or
            keys[pygame.K_LSHIFT] or
            keys[pygame.K_RSHIFT]
        )

        inputs: Dict[int, Tuple[float, float, bool]] = {}

        # Red Team (Player 0 is Human)
        red_players = [p for p in self.game.players if p.team == Team.RED]
        for i, p in enumerate(red_players):
            if i == 0:
                inputs[p.player_id] = (mx, my, kick)
            else:
                bot = self.bot_catalog["heuristic"]
                inputs[p.player_id] = bot.act(self.game, p)

        # Blue Team (Opponent Bots)
        blue_players = [p for p in self.game.players if p.team == Team.BLUE]
        for i, p in enumerate(blue_players):
            bot = self.bot_catalog.get(self.active_bot_key, self.bot_catalog["rl"])
            inputs[p.player_id] = bot.act(self.game, p)

        return inputs

    def update(self):
        if self.is_paused or self.show_stadium_modal or self.show_rl_modal or self.show_help_modal:
            return

        if self.play_mode == "self_play":
            if self.self_play_trainer:
                self.self_play_trainer.step_multistep(self.speed_multiplier)
        else:
            for _ in range(self.speed_multiplier):
                inputs = self.get_player_inputs()
                step_info = self.game.step(inputs)

            if getattr(self, "gauntlet_mode", False) and self.game.state == GameState.GAME_OVER:
                if not hasattr(self, "_gauntlet_delay"):
                    self._gauntlet_delay = 90
                self._gauntlet_delay -= 1
                if self._gauntlet_delay <= 0:
                    del self._gauntlet_delay
                    if self.game.red_score > self.game.blue_score:
                        self.gauntlet_index = (self.gauntlet_index + 1) % len(GAUNTLET_ORDER)
                        next_bot = GAUNTLET_ORDER[self.gauntlet_index]
                        self.checkpoint_feedback = f"Vitória! Desafio {self.gauntlet_index + 1}/{len(GAUNTLET_ORDER)}: {next_bot.upper()}"
                    else:
                        self.checkpoint_feedback = f"Derrota! Tente novamente contra {self.active_bot_key.upper()}"
                    self.feedback_time = time.time()
                    self.active_bot_key = GAUNTLET_ORDER[self.gauntlet_index]
                    self.game.reset_match()

    def draw(self):
        self.screen.fill((26, 33, 42))

        stad = self.game.stadium
        p_top_left = self.world_to_screen(Vec2(-stad.width, stad.height))
        p_bottom_right = self.world_to_screen(Vec2(stad.width, -stad.height))
        pitch_rect = pygame.Rect(
            p_top_left[0],
            p_top_left[1],
            p_bottom_right[0] - p_top_left[0],
            p_bottom_right[1] - p_top_left[1]
        )

        pitch_color = hex_to_rgb(stad.color) if hasattr(stad, 'color') and stad.color else (45, 60, 75)
        pygame.draw.rect(self.screen, pitch_color, pitch_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), pitch_rect, width=2)

        # Center line & circle
        c_top = self.world_to_screen(Vec2(0, stad.height))
        c_bottom = self.world_to_screen(Vec2(0, -stad.height))
        pygame.draw.line(self.screen, (255, 255, 255), c_top, c_bottom, width=2)

        c_center = self.world_to_screen(Vec2(0, 0))
        c_radius = self.world_len_to_screen(60.0)
        pygame.draw.circle(self.screen, (255, 255, 255), c_center, c_radius, width=2)
        pygame.draw.circle(self.screen, (255, 255, 255), c_center, 4)

        # Segments & Goals
        for seg in stad.segments:
            if not seg.vis:
                continue
            p0 = self.world_to_screen(seg.p0)
            p1 = self.world_to_screen(seg.p1)
            c = getattr(seg, 'color_rgb', None) or (hex_to_rgb(seg.color) if hasattr(seg, 'color') and seg.color else (255, 255, 255))
            if getattr(seg, 'is_curved', False) or getattr(seg, 'curve', 0.0) != 0.0:
                self._draw_curved_segment(seg, c)
            else:
                pygame.draw.line(self.screen, c, p0, p1, width=2)

        for goal in stad.goals:
            p0 = self.world_to_screen(goal.p0)
            p1 = self.world_to_screen(goal.p1)
            g_c = (229, 110, 86) if goal.team == Team.RED else (86, 137, 229)
            pygame.draw.line(self.screen, g_c, p0, p1, width=4)

        # Discs & Ball
        discs_list = getattr(self.game, 'discs', getattr(self.game.physics, 'discs', []))
        for disc in discs_list:
            pos = self.world_to_screen(disc.pos)
            rad = self.world_len_to_screen(disc.radius)
            c = getattr(disc, 'color_rgb', None) or (hex_to_rgb(disc.color) if hasattr(disc, 'color') and disc.color else (255, 255, 255))
            pygame.draw.circle(self.screen, c, pos, rad)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, rad, width=2)

        # Players
        for player in self.game.players:
            pos = self.world_to_screen(player.pos)
            rad = self.world_len_to_screen(player.radius)
            c = (229, 110, 86) if player.team == Team.RED else (86, 137, 229)

            if getattr(player, 'kick', False) or getattr(player, 'is_kicking', False) or getattr(player, 'kick_flash', 0) > 0:
                pygame.draw.circle(self.screen, (255, 255, 255), pos, rad + 3, width=2)

            pygame.draw.circle(self.screen, c, pos, rad)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, rad, width=2)

            # Draw Real-Time Intent / Velocity Vector
            if player.speed.length_sq() > 0.05:
                vel_end = player.pos + player.speed.normalized() * (player.radius + 18.0)
                s_end = self.world_to_screen(vel_end)
                pygame.draw.line(self.screen, (255, 255, 100) if player.team == Team.BLUE else (255, 200, 80), pos, s_end, width=2)
                pygame.draw.circle(self.screen, (255, 255, 255), s_end, 3)

            num_str = str(player.player_id)
            num_surf = self.font_player.render(num_str, True, (255, 255, 255))
            self.screen.blit(num_surf, num_surf.get_rect(center=pos))

        # Overlays
        self._draw_scoreboard()
        self._draw_hud_overlay()
        self._draw_interaction_telemetry_hud()

        if self.play_mode == "self_play" and self.self_play_trainer:
            self._draw_self_play_banner()

        self._draw_bottom_dock()

        # Modals
        if self.show_stadium_modal:
            self._draw_stadium_modal()
        elif self.show_rl_modal:
            self._draw_rl_modal()
        elif self.show_help_modal:
            self._draw_help_modal()

        pygame.display.flip()

    def _draw_interaction_telemetry_hud(self):
        """Draws live real-time interaction metrics between human player and AI bots."""
        ball = self.game.ball
        if not ball:
            return

        blue_p = next((p for p in self.game.players if p.team == Team.BLUE), None)
        red_p = next((p for p in self.game.players if p.team == Team.RED), None)

        if not blue_p:
            return

        stad = self.game.stadium
        dist_to_ball = blue_p.pos.distance_to(ball.pos)

        to_ball = (ball.pos - blue_p.pos).normalized() if dist_to_ball > 1e-4 else Vec2(1, 0)
        opp_goal = Vec2(-stad.bg_width, 0.0)
        to_goal = (opp_goal - ball.pos).normalized()

        # Metrics calculation
        # 1. Ball Pursuit Index (% alignment with ball direction)
        if blue_p.speed.length_sq() > 0.05:
            move_dir = blue_p.speed.normalized()
            pursuit_cos = float(move_dir.dot(to_ball))
            pursuit_pct = int(max(0.0, pursuit_cos) * 100)
        else:
            pursuit_pct = 0

        # 2. Shot Alignment Index
        shot_cos = float(to_ball.dot(to_goal))
        shot_pct = int(max(0.0, shot_cos) * 100)

        # 3. Tactical State
        if dist_to_ball < (blue_p.radius + ball.radius + 10.0):
            tactical_state = "Controle / Chute"
            state_color = (255, 100, 100)
        elif pursuit_pct > 75:
            tactical_state = "Caça Ativa"
            state_color = (60, 210, 120)
        elif blue_p.pos.x > ball.pos.x:
            tactical_state = "Contorno Tático"
            state_color = (240, 190, 60)
        else:
            tactical_state = "Recomposição"
            state_color = (160, 180, 220)

        # Telemetry Card on Right Corner
        hud_w = 230
        hud_h = 100
        hud_x = self.width - hud_w - 20
        hud_y = 56

        card_r = pygame.Rect(hud_x, hud_y, hud_w, hud_h)
        pygame.draw.rect(self.screen, (24, 30, 38), card_r, border_radius=8)
        pygame.draw.rect(self.screen, (58, 142, 230), card_r, width=1, border_radius=8)

        t_title = self.font_telemetry.render("TELEMETRIA DA IA (TIME AZUL)", True, (100, 180, 255))
        self.screen.blit(t_title, (hud_x + 10, hud_y + 8))

        m1 = self.font_small.render(f"• Pressão na Bola: {pursuit_pct}%", True, (220, 230, 240))
        m2 = self.font_small.render(f"• Alinhamento ao Gol: {shot_pct}%", True, (220, 230, 240))
        m3 = self.font_small.render(f"• Distância da Bola: {int(dist_to_ball)} px", True, (220, 230, 240))
        m4 = self.font_bold.render(f"• Ação: {tactical_state}", True, state_color)

        self.screen.blit(m1, (hud_x + 10, hud_y + 26))
        self.screen.blit(m2, (hud_x + 10, hud_y + 44))
        self.screen.blit(m3, (hud_x + 10, hud_y + 62))
        self.screen.blit(m4, (hud_x + 10, hud_y + 80))

    def _draw_curved_segment(self, seg, color):
        center = getattr(seg, 'arc_center', getattr(seg, 'center', None))
        radius = getattr(seg, 'arc_radius', getattr(seg, 'radius', 0.0))
        if radius <= 0.0 or center is None:
            return
        start_a = getattr(seg, 'arc_start_angle', getattr(seg, 'start_angle', 0.0))
        span_a = getattr(seg, 'arc_span_angle', getattr(seg, 'span_angle', 0.0))
        steps = 16
        pts = []
        for i in range(steps + 1):
            t = i / steps
            ang = start_a + span_a * t
            pt = center + Vec2(math.cos(ang), math.sin(ang)) * radius
            pts.append(self.world_to_screen(pt))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, width=2)


    def _draw_scoreboard(self):
        sb_w = 260
        sb_h = 36
        sb_x = int(self.width / 2.0 - sb_w / 2.0)
        sb_y = 12

        red_box = pygame.Rect(sb_x, sb_y, 75, sb_h)
        pygame.draw.rect(self.screen, (229, 110, 86), red_box, border_top_left_radius=6, border_bottom_left_radius=6)
        r_txt = self.font_score.render(str(self.game.red_score), True, (255, 255, 255))
        self.screen.blit(r_txt, r_txt.get_rect(center=red_box.center))

        time_box = pygame.Rect(sb_x + 75, sb_y, 110, sb_h)
        pygame.draw.rect(self.screen, (34, 43, 53), time_box)
        t_txt = self.font_time.render(self.game.time_string, True, (240, 240, 240))
        self.screen.blit(t_txt, t_txt.get_rect(center=time_box.center))

        blue_box = pygame.Rect(sb_x + 185, sb_y, 75, sb_h)
        pygame.draw.rect(self.screen, (86, 137, 229), blue_box, border_top_right_radius=6, border_bottom_right_radius=6)
        b_txt = self.font_score.render(str(self.game.blue_score), True, (255, 255, 255))
        self.screen.blit(b_txt, b_txt.get_rect(center=blue_box.center))

    def _draw_hud_overlay(self):
        # Stadium pill on top-left
        stad_title = STADIUM_CATALOG[self.current_stadium_key]["title"]
        self.pill_stad_r = pygame.Rect(20, 14, 230, 34)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_stad_r, border_radius=17)
        pygame.draw.rect(self.screen, (55, 68, 85), self.pill_stad_r, width=1, border_radius=17)
        draw_icon_stadium(self.screen, (self.pill_stad_r.x + 18, self.pill_stad_r.centery), (100, 180, 255), size=12)
        s_txt = self.font_hud.render(stad_title[:20], True, (220, 230, 240))
        self.screen.blit(s_txt, (self.pill_stad_r.x + 32, self.pill_stad_r.centery - s_txt.get_height() // 2))

        # Active Model Pill
        self.pill_model_r = pygame.Rect(260, 14, 230, 34)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_model_r, border_radius=17)
        pygame.draw.rect(self.screen, (60, 179, 113), self.pill_model_r, width=1, border_radius=17)
        draw_icon_brain(self.screen, (self.pill_model_r.x + 18, self.pill_model_r.centery), (60, 210, 120), size=12)
        m_name = self.active_checkpoint_name.replace(".pt", "")
        if len(m_name) > 16:
            m_name = m_name[:14] + ".."
        m_txt = self.font_hud.render(f"IA: {m_name}", True, (60, 210, 120))
        self.screen.blit(m_txt, (self.pill_model_r.x + 32, self.pill_model_r.centery - m_txt.get_height() // 2))

        # Speed Multiplier Pill
        self.pill_speed_r = pygame.Rect(self.width - 290, 14, 130, 34)
        sp_c = (230, 140, 40) if self.speed_multiplier > 1 else (55, 68, 85)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_speed_r, border_radius=17)
        pygame.draw.rect(self.screen, sp_c, self.pill_speed_r, width=2 if self.speed_multiplier > 1 else 1, border_radius=17)
        draw_icon_lightning(self.screen, (self.pill_speed_r.x + 20, self.pill_speed_r.centery), (255, 210, 60), size=13)
        sp_txt = self.font_hud.render(f"Vel: {self.speed_multiplier}x", True, (255, 180, 50) if self.speed_multiplier > 1 else (200, 210, 225))
        self.screen.blit(sp_txt, (self.pill_speed_r.x + 34, self.pill_speed_r.centery - sp_txt.get_height() // 2))

        # Mode Pill
        self.pill_mode_r = pygame.Rect(self.width - 150, 14, 130, 34)
        is_self_play = (self.play_mode == "self_play")
        m_c = (60, 179, 113) if is_self_play else (58, 142, 230)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_mode_r, border_radius=17)
        pygame.draw.rect(self.screen, m_c, self.pill_mode_r, width=2, border_radius=17)
        if is_self_play:
            draw_icon_robot(self.screen, (self.pill_mode_r.x + 18, self.pill_mode_r.centery), m_c, size=12)
        else:
            draw_icon_user(self.screen, (self.pill_mode_r.x + 18, self.pill_mode_r.centery), m_c, size=12)
        m_label = "Self-Play" if is_self_play else "Humano"
        m_txt = self.font_hud.render(m_label, True, m_c)
        self.screen.blit(m_txt, (self.pill_mode_r.x + 32, self.pill_mode_r.centery - m_txt.get_height() // 2))

    def _draw_self_play_banner(self):
        stats = self.self_play_trainer.last_stats
        steps = stats.get("total_steps", 0)
        iters = stats.get("iteration", 0)
        rew = stats.get("mean_reward", 0.0)
        passes = stats.get("passes", 0)

        if steps < 5000:
            phase_name = "FASE 1: Exploracao Burra"
            phase_color = (220, 80, 80)
        elif steps < 25000:
            phase_name = "FASE 2: Perseguicao da Bola"
            phase_color = (235, 150, 40)
        elif steps < 70000:
            phase_name = "FASE 3: Alinhamento ao Gol"
            phase_color = (220, 200, 50)
        elif steps < 500000:
            phase_name = "FASE 4: Team-Play e Passes"
            phase_color = (60, 210, 120)
        else:
            phase_name = "FASE 5: Pro Master Consolidado"
            phase_color = (160, 120, 255)

        banner_w = 900
        banner_h = 32
        banner_x = int(self.width / 2.0 - banner_w / 2.0)
        banner_y = 54

        b_rect = pygame.Rect(banner_x, banner_y, banner_w, banner_h)
        pygame.draw.rect(self.screen, (22, 28, 36), b_rect, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 74), b_rect, width=1, border_radius=8)

        ph_rect = pygame.Rect(banner_x + 8, banner_y + 4, 250, 24)
        pygame.draw.rect(self.screen, (34, 44, 56), ph_rect, border_radius=5)
        ph_txt = self.font_telemetry.render(phase_name, True, phase_color)
        self.screen.blit(ph_txt, ph_txt.get_rect(center=ph_rect.center))

        met_str = f"Passos: {steps:,}  |  Iter: {iters}  |  Reward: {rew:+.2f}  |  Passes: {passes}"
        met_surf = self.font_telemetry.render(met_str, True, (210, 220, 235))
        self.screen.blit(met_surf, (banner_x + 270, banner_y + 8))

        self.btn_dumb_r = pygame.Rect(banner_x + banner_w - 145, banner_y + 4, 135, 24)
        pygame.draw.rect(self.screen, (60, 30, 35), self.btn_dumb_r, border_radius=5)
        pygame.draw.rect(self.screen, (200, 70, 70), self.btn_dumb_r, width=1, border_radius=5)
        draw_icon_reset(self.screen, (self.btn_dumb_r.x + 14, self.btn_dumb_r.centery), (240, 130, 130), size=11)
        d_txt = self.font_telemetry.render("Ficar Burro", True, (240, 130, 130))
        self.screen.blit(d_txt, (self.btn_dumb_r.x + 24, self.btn_dumb_r.centery - d_txt.get_height() // 2))

    def _draw_bottom_dock(self):
        dock_h = 58
        dock_y = self.height - dock_h
        pygame.draw.rect(self.screen, (22, 28, 35), (0, dock_y, self.width, dock_h))
        pygame.draw.line(self.screen, (45, 56, 70), (0, dock_y), (self.width, dock_y), width=1)

        # 1. Play / Pause
        self.btn_pause_r = pygame.Rect(12, dock_y + 10, 95, 38)
        p_txt = "Play" if self.is_paused else "Pausar"
        p_icon = "play" if self.is_paused else "pause"
        self._draw_btn(self.btn_pause_r, p_txt, (58, 142, 230), icon=p_icon)

        # 2. Reset match
        self.btn_reset_r = pygame.Rect(112, dock_y + 10, 85, 38)
        self._draw_btn(self.btn_reset_r, "Reset", (45, 55, 68), icon="reset")

        # 3. Choose Stadium
        self.btn_stadium_r = pygame.Rect(202, dock_y + 10, 115, 38)
        self._draw_btn(self.btn_stadium_r, "Estádios", (45, 55, 68), icon="stadium")

        # 4. Format 1v1 / 2v2 / 3v3 / 5v5
        self.btn_format_r = pygame.Rect(322, dock_y + 10, 110, 38)
        self._draw_btn(self.btn_format_r, f"{self.team_format}v{self.team_format}", (45, 55, 68), icon="user")

        # 5. Speed Multiplier
        self.btn_dock_speed_r = pygame.Rect(437, dock_y + 10, 110, 38)
        sp_c = (210, 120, 30) if self.speed_multiplier > 1 else (45, 55, 68)
        self._draw_btn(self.btn_dock_speed_r, f"Vel: {self.speed_multiplier}x", sp_c, icon="lightning")

        # 6. Mode Toggle (Self-Play vs Human)
        self.btn_dock_mode_r = pygame.Rect(552, dock_y + 10, 130, 38)
        is_sp = (self.play_mode == "self_play")
        m_txt = "Self-Play IA" if is_sp else "Humano"
        m_bg = (50, 140, 90) if is_sp else (58, 142, 230)
        m_icon = "robot" if is_sp else "user"
        self._draw_btn(self.btn_dock_mode_r, m_txt, m_bg, icon=m_icon)

        # 7. Opponent Bot Cycle
        self.btn_dock_bot_r = pygame.Rect(687, dock_y + 10, 155, 38)
        b_labels = {
            "press": "Bot: Pressing",
            "striker": "Bot: Striker",
            "bank": "Bot: Tabelas",
            "dribbler": "Bot: Dribbler",
            "counter": "Bot: Counter",
            "master": "Bot: Master Pro",
            "heuristic": "Bot: Clássico",
            "wall": "Bot: Rebound",
            "goalie": "Bot: Goleiro",
            "rl": "Bot: IA Treinada",
        }
        if getattr(self, "gauntlet_mode", False):
            b_txt = f"★ Gauntlet ({self.gauntlet_index + 1}/{len(GAUNTLET_ORDER)})"
        else:
            b_txt = b_labels.get(self.active_bot_key, f"Bot: {self.active_bot_key.title()}")
        self._draw_btn(self.btn_dock_bot_r, b_txt, (50, 60, 75), icon="robot")

        # 8. Checkpoints & RL Studio Modal
        self.btn_rl_r = pygame.Rect(847, dock_y + 10, 175, 38)
        rl_c = (60, 179, 113) if not self.training_active else (220, 70, 70)
        rl_t = "Modelos & RL Studio" if not self.training_active else "Treinando..."
        self._draw_btn(self.btn_rl_r, rl_t, rl_c, icon="brain")

        # 9. Help / Controls
        self.btn_help_r = pygame.Rect(self.width - 110, dock_y + 10, 95, 38)
        self._draw_btn(self.btn_help_r, "Teclas", (45, 55, 68), icon="help")

    def _draw_btn(self, rect: pygame.Rect, text: str, bg_color: Tuple[int, int, int], icon: Optional[str] = None):
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)
        c = (min(255, bg_color[0] + 25), min(255, bg_color[1] + 25), min(255, bg_color[2] + 25)) if hover else bg_color
        pygame.draw.rect(self.screen, c, rect, border_radius=6)
        pygame.draw.rect(self.screen, (55, 68, 85), rect, width=1, border_radius=6)

        txt = self.font_bold.render(text, True, (240, 245, 250))
        if icon and icon in ICON_DISPATCH:
            total_w = 14 + 8 + txt.get_width()
            start_x = rect.centerx - total_w // 2
            icon_center = (start_x + 7, rect.centery)
            ICON_DISPATCH[icon](self.screen, icon_center, (240, 245, 250), size=12)
            self.screen.blit(txt, (start_x + 18, rect.centery - txt.get_height() // 2))
        else:
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_stadium_modal(self):
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 780, 520
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (34, 43, 53), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (58, 142, 230), m_r, width=2, border_radius=12)

        t = self.font_title.render("Selecionar Estádio (.hbs) - Catálogo Oficial", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 20))

        keys = list(STADIUM_CATALOG.keys())
        self.stadium_cards = {}
        card_w = 345
        card_h = 82

        for i, key in enumerate(keys):
            col = i % 2
            row = i // 2
            cx = m_r.x + 30 + col * (card_w + 20)
            cy = m_r.y + 60 + row * (card_h + 12)

            card_r = pygame.Rect(cx, cy, card_w, card_h)
            self.stadium_cards[key] = card_r
            info = STADIUM_CATALOG[key]
            is_active = (key == self.current_stadium_key)

            c_bg = (45, 58, 74) if is_active else (40, 49, 62)
            pygame.draw.rect(self.screen, c_bg, card_r, border_radius=8)
            pygame.draw.rect(self.screen, (58, 142, 230) if is_active else (55, 68, 85), card_r, width=2 if is_active else 1, border_radius=8)

            t_card = self.font_bold.render(info["title"][:28], True, (255, 255, 255))
            self.screen.blit(t_card, (card_r.x + 12, cy + 10))

            d_card = self.font_regular.render(info["desc"][:42] + "...", True, (170, 180, 195))
            self.screen.blit(d_card, (card_r.x + 12, cy + 34))

            act_label = "● ATIVO" if is_active else "Escolher"
            lbl_surf = self.font_bold.render(act_label, True, (60, 179, 113) if is_active else (58, 142, 230))
            self.screen.blit(lbl_surf, (card_r.right - 75, cy + 54))

        self.btn_close_stad_r = pygame.Rect(m_r.right - 140, m_r.bottom - 44, 110, 32)
        self._draw_btn(self.btn_close_stad_r, "Fechar", (45, 55, 68))

    def _draw_rl_modal(self):
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 175))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 840, 560
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (30, 38, 48), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (60, 179, 113), m_r, width=2, border_radius=12)

        # Header
        t = self.font_title.render("Central de Modelos e Treinos de RL (End-to-End)", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 18))

        # Tabs Header
        self.tab_models_r = pygame.Rect(m_r.x + 30, m_r.y + 52, 230, 32)
        self.tab_train_r = pygame.Rect(m_r.x + 270, m_r.y + 52, 230, 32)
        self.tab_bots_r = pygame.Rect(m_r.x + 510, m_r.y + 52, 210, 32)

        def draw_tab(rect, label, is_active):
            bg = (45, 58, 74) if is_active else (34, 42, 52)
            bd = (60, 179, 113) if is_active else (50, 60, 72)
            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            pygame.draw.rect(self.screen, bd, rect, width=2 if is_active else 1, border_radius=6)
            txt = self.font_bold.render(label, True, (255, 255, 255) if is_active else (160, 175, 190))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        draw_tab(self.tab_models_r, "📁 Modelos & Checkpoints", self.rl_modal_tab == "models")
        draw_tab(self.tab_train_r, "⚙️ Treinador Background", self.rl_modal_tab == "training")
        draw_tab(self.tab_bots_r, "🤖 Escolher Oponente", self.rl_modal_tab == "bots")

        # Feedback Toast
        if self.checkpoint_feedback and (time.time() - self.feedback_time < 3.5):
            fb_surf = self.font_bold.render(f"✓ {self.checkpoint_feedback}", True, (60, 230, 130))
            self.screen.blit(fb_surf, (m_r.right - fb_surf.get_width() - 30, m_r.y + 22))

        # Tab 1: Checkpoints / Models
        if self.rl_modal_tab == "models":
            checkpoints = self.get_available_checkpoints()
            self.checkpoint_cards = {}

            card_w = 370
            card_h = 78
            for i, cp in enumerate(checkpoints[:6]):
                col = i % 2
                row = i // 2
                cx = m_r.x + 30 + col * (card_w + 20)
                cy = m_r.y + 98 + row * (card_h + 10)

                card_r = pygame.Rect(cx, cy, card_w, card_h)
                self.checkpoint_cards[cp["filename"]] = card_r
                is_active = (cp["filename"] == self.active_checkpoint_name)

                bg_c = (42, 54, 68) if is_active else (36, 45, 56)
                pygame.draw.rect(self.screen, bg_c, card_r, border_radius=8)
                pygame.draw.rect(self.screen, (60, 179, 113) if is_active else (55, 68, 85), card_r, width=2 if is_active else 1, border_radius=8)

                # Badge
                b_rect = pygame.Rect(cx + 10, cy + 10, 60, 20)
                pygame.draw.rect(self.screen, (28, 36, 45), b_rect, border_radius=4)
                pygame.draw.rect(self.screen, cp["color"], b_rect, width=1, border_radius=4)
                b_txt = self.font_player.render(cp["badge"], True, cp["color"])
                self.screen.blit(b_txt, b_txt.get_rect(center=b_rect.center))

                # Title
                t_card = self.font_bold.render(cp["title"][:24], True, (255, 255, 255))
                self.screen.blit(t_card, (cx + 78, cy + 12))

                # Desc
                d_card = self.font_small.render(cp["desc"][:42], True, (170, 185, 200))
                self.screen.blit(d_card, (cx + 10, cy + 38))

                # Action label
                status_txt = "● ATIVO" if is_active else "Carregar"
                status_col = (60, 210, 120) if is_active else (58, 142, 230)
                s_surf = self.font_bold.render(status_txt, True, status_col)
                self.screen.blit(s_surf, (card_r.right - s_surf.get_width() - 14, cy + 54))

            # Bottom Quick Actions in Tab 1
            self.btn_save_cp_r = pygame.Rect(m_r.x + 30, m_r.bottom - 50, 210, 36)
            self._draw_btn(self.btn_save_cp_r, "💾 Salvar Checkpoint", (45, 58, 74))

            self.btn_quick_train_r = pygame.Rect(m_r.x + 250, m_r.bottom - 50, 210, 36)
            self._draw_btn(self.btn_quick_train_r, "⚡ +25k Passos Rápidos", (210, 120, 30))

        # Tab 2: Training Config & Background Worker
        elif self.rl_modal_tab == "training":
            sub = self.font_regular.render(f"Status do Treinador: {self.training_status}", True, (60, 179, 113) if self.training_active else (170, 180, 195))
            self.screen.blit(sub, (m_r.x + 30, m_r.y + 95))

            self.btn_rl_algo_r = pygame.Rect(m_r.x + 30, m_r.y + 125, 170, 36)
            self._draw_btn(self.btn_rl_algo_r, f"Algoritmo: {self.training_algo}", (58, 142, 230))

            self.btn_rl_pol_r = pygame.Rect(m_r.x + 215, m_r.y + 125, 220, 36)
            pol_txt = "Rede: MLP Padrão" if self.training_policy == "mlp" else "Rede: Transformer Attention"
            self._draw_btn(self.btn_rl_pol_r, pol_txt, (45, 58, 74))

            met_r = pygame.Rect(m_r.x + 30, m_r.y + 175, m_w - 60, 180)
            pygame.draw.rect(self.screen, (24, 30, 38), met_r, border_radius=8)
            pygame.draw.rect(self.screen, (55, 68, 85), met_r, width=1, border_radius=8)

            m_head = self.font_bold.render("Métricas de Aprendizado em Segundo Plano:", True, (235, 240, 245))
            self.screen.blit(m_head, (met_r.x + 18, met_r.y + 16))

            m1 = self.font_regular.render(f"• Passos de Interação: {self.training_metrics['step']:,}", True, (220, 225, 235))
            m2 = self.font_regular.render(f"• Recompensa Média: {self.training_metrics['reward']:+.2f}", True, (220, 225, 235))
            m3 = self.font_regular.render(f"• Taxa de Vitória: {self.training_metrics['win_rate']:.1f}%", True, (220, 225, 235))
            m4 = self.font_regular.render(f"• Função de Perda (Loss): {self.training_metrics['loss']:.4f}", True, (220, 225, 235))

            self.screen.blit(m1, (met_r.x + 18, met_r.y + 55))
            self.screen.blit(m2, (met_r.x + 18, met_r.y + 90))
            self.screen.blit(m3, (met_r.x + 360, met_r.y + 55))
            self.screen.blit(m4, (met_r.x + 360, met_r.y + 90))

            self.btn_rl_action_r = pygame.Rect(m_r.x + 30, m_r.bottom - 50, 240, 38)
            act_bg = (220, 70, 70) if self.training_active else (60, 179, 113)
            act_txt = "⏹ Parar Treinamento" if self.training_active else "🚀 Iniciar Treinamento"
            self._draw_btn(self.btn_rl_action_r, act_txt, act_bg)

        # Tab 3: Bot Opponents
        elif self.rl_modal_tab == "bots":
            bot_options = [
                ("gauntlet", "🏆 Torneio Desafio (Gauntlet)", "Enfrente todos os 9 bots em sequência!", (255, 215, 0)),
                ("rl", "RL Bot (IA Treinada)", f"Usa o modelo ativo: {self.active_checkpoint_name}", (60, 210, 120)),
            ]
            for bkey, (cls, label, title, desc, col) in NPC_BOTS.items():
                bot_options.append((bkey, title, desc, col))

            self.bot_cards = {}
            card_w = 370
            card_h = 68
            for i, (bkey, btitle, bdesc, bcol) in enumerate(bot_options):
                col_idx = i % 2
                row_idx = i // 2
                cx = m_r.x + 30 + col_idx * 390
                cy = m_r.y + 92 + row_idx * 74
                c_rect = pygame.Rect(cx, cy, card_w, card_h)
                self.bot_cards[bkey] = c_rect
                is_cur = (self.gauntlet_mode if bkey == "gauntlet" else (not self.gauntlet_mode and self.active_bot_key == bkey))

                pygame.draw.rect(self.screen, (45, 58, 74) if is_cur else (36, 45, 56), c_rect, border_radius=6)
                pygame.draw.rect(self.screen, bcol if is_cur else (55, 68, 85), c_rect, width=2 if is_cur else 1, border_radius=6)

                t_surf = self.font_bold.render(btitle[:28], True, (255, 255, 255))
                self.screen.blit(t_surf, (cx + 12, cy + 8))

                d_surf = self.font_small.render(bdesc[:45], True, (170, 185, 200))
                self.screen.blit(d_surf, (cx + 12, cy + 28))

                lbl = "● ATIVO" if is_cur else "Selecionar"
                lbl_s = self.font_bold.render(lbl, True, (60, 210, 120) if is_cur else (58, 142, 230))
                self.screen.blit(lbl_s, (c_rect.right - lbl_s.get_width() - 12, cy + 46))

        self.btn_close_rl_r = pygame.Rect(m_r.right - 140, m_r.bottom - 46, 110, 32)
        self._draw_btn(self.btn_close_rl_r, "Fechar", (45, 55, 68))

    def _draw_help_modal(self):
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 720, 450
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (34, 43, 53), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (58, 142, 230), m_r, width=2, border_radius=12)

        t = self.font_title.render("Guia de Controles, Treino e Modelos Prontos", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 24))

        lines = [
            "• Controles: TECLAS WASD OU SETAS DO TECLADO funcionam simultaneamente!",
            "• Chute: BARRA DE ESPAÇO, TECLA X, TECLA C ou SHIFT.",
            "• Modelos Prontos: Acesse 'Modelos & RL' para carregar os checkpoints treinados (25k a 1M).",
            "• Treino ao Vivo (2x2 Self-Play): Veja a IA jogando e aprendendo recursivamente na tela!",
            "• Aceleração até 100x: Use o botão de velocidade (1x a 100x) para simular 6.000 passos/segundo.",
            "• 'Ficar Burro (Reset)': Clique para zerar os pesos e ver os modelos aprendendo do zero.",
            "• Troca de Oponentes: Alterne entre RL Bot, WallRebound (Tabelas), Heuristic e Goleiro."
        ]
        y = m_r.y + 70
        for l in lines:
            t_line = self.font_regular.render(l, True, (220, 230, 240))
            self.screen.blit(t_line, (m_r.x + 30, y))
            y += 32

        self.btn_close_help_r = pygame.Rect(m_r.right - 140, m_r.bottom - 46, 110, 34)
        self._draw_btn(self.btn_close_help_r, "Entendi!", (58, 142, 230))

    def cycle_speed(self):
        curr_idx = SPEED_LEVELS.index(self.speed_multiplier) if self.speed_multiplier in SPEED_LEVELS else 0
        nxt_idx = (curr_idx + 1) % len(SPEED_LEVELS)
        self.speed_multiplier = SPEED_LEVELS[nxt_idx]

    def toggle_play_mode(self):
        self.play_mode = "human" if self.play_mode == "self_play" else "self_play"

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.training_active = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_stadium_modal or self.show_rl_modal or self.show_help_modal:
                        self.show_stadium_modal = self.show_rl_modal = self.show_help_modal = False
                    else:
                        self.is_paused = not self.is_paused
                elif event.key == pygame.K_p:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_r:
                    self.game.reset_match()
                elif event.key == pygame.K_m:
                    self.toggle_play_mode()
                elif event.key == pygame.K_TAB:
                    self.cycle_speed()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                # Modal handling: Stadium
                if self.show_stadium_modal:
                    if self.btn_close_stad_r.collidepoint(pos):
                        self.show_stadium_modal = False
                    else:
                        for k, r in self.stadium_cards.items():
                            if r.collidepoint(pos):
                                self._init_game(k, STADIUM_CATALOG[k]["format"])
                                self.show_stadium_modal = False
                                break
                    continue

                # Modal handling: RL & Checkpoints
                if self.show_rl_modal:
                    if self.btn_close_rl_r.collidepoint(pos):
                        self.show_rl_modal = False
                    elif self.tab_models_r.collidepoint(pos):
                        self.rl_modal_tab = "models"
                    elif self.tab_train_r.collidepoint(pos):
                        self.rl_modal_tab = "training"
                    elif self.tab_bots_r.collidepoint(pos):
                        self.rl_modal_tab = "bots"

                    elif self.rl_modal_tab == "models":
                        if hasattr(self, "checkpoint_cards"):
                            for cp_file, c_rect in self.checkpoint_cards.items():
                                if c_rect.collidepoint(pos):
                                    self.load_checkpoint(cp_file)
                                    break
                        if hasattr(self, "btn_save_cp_r") and self.btn_save_cp_r.collidepoint(pos):
                            self.save_current_checkpoint()
                        elif hasattr(self, "btn_quick_train_r") and self.btn_quick_train_r.collidepoint(pos):
                            if self.self_play_trainer:
                                self.self_play_trainer.step_multistep(25000)
                                self.checkpoint_feedback = "Treinou +25k passos!"
                                self.feedback_time = time.time()

                    elif self.rl_modal_tab == "training":
                        if hasattr(self, "btn_rl_algo_r") and self.btn_rl_algo_r.collidepoint(pos):
                            self.training_algo = "DQN" if self.training_algo == "PPO" else "PPO"
                        elif hasattr(self, "btn_rl_pol_r") and self.btn_rl_pol_r.collidepoint(pos):
                            self.training_policy = "attention" if self.training_policy == "mlp" else "mlp"
                        elif hasattr(self, "btn_rl_action_r") and self.btn_rl_action_r.collidepoint(pos):
                            self._toggle_training()

                    elif self.rl_modal_tab == "bots":
                        if hasattr(self, "bot_cards"):
                            for bkey, brect in self.bot_cards.items():
                                if brect.collidepoint(pos):
                                    if bkey == "gauntlet":
                                        self.gauntlet_mode = True
                                        self.gauntlet_index = 0
                                        self.active_bot_key = GAUNTLET_ORDER[0]
                                        self.checkpoint_feedback = f"Gauntlet Iniciado! Desafio 1/{len(GAUNTLET_ORDER)}: {self.active_bot_key.upper()}"
                                    else:
                                        self.gauntlet_mode = False
                                        self.active_bot_key = bkey
                                        self.checkpoint_feedback = f"Oponente: {bkey.upper()}"
                                    self.feedback_time = time.time()
                                    break
                    continue

                # Modal handling: Help
                if self.show_help_modal:
                    if self.btn_close_help_r.collidepoint(pos):
                        self.show_help_modal = False
                    continue

                # Top HUD Pills clicks
                if self.pill_stad_r.collidepoint(pos):
                    self.show_stadium_modal = True
                    continue
                elif self.pill_model_r.collidepoint(pos):
                    self.show_rl_modal = True
                    self.rl_modal_tab = "models"
                    continue
                elif self.pill_speed_r.collidepoint(pos):
                    self.cycle_speed()
                    continue
                elif self.pill_mode_r.collidepoint(pos):
                    self.toggle_play_mode()
                    continue

                # Live Training Banner "Ficar Burro" Click
                if self.play_mode == "self_play" and hasattr(self, "btn_dumb_r") and self.btn_dumb_r.collidepoint(pos):
                    if self.self_play_trainer:
                        self.self_play_trainer.reset_policy_to_random()
                        self.active_checkpoint_name = "Aleatório (Burro)"
                    continue

                # Bottom Dock clicks
                if self.btn_pause_r.collidepoint(pos):
                    self.is_paused = not self.is_paused
                elif self.btn_reset_r.collidepoint(pos):
                    self.game.reset_match()
                elif self.btn_stadium_r.collidepoint(pos):
                    self.show_stadium_modal = True
                elif self.btn_format_r.collidepoint(pos):
                    nxt = 2 if self.team_format == 1 else (3 if self.team_format == 2 else (5 if self.team_format == 3 else 1))
                    self._init_game(self.current_stadium_key, nxt)
                elif self.btn_dock_speed_r.collidepoint(pos):
                    self.cycle_speed()
                elif self.btn_dock_mode_r.collidepoint(pos):
                    self.toggle_play_mode()
                elif self.btn_dock_bot_r.collidepoint(pos):
                    self.cycle_bot()
                elif self.btn_rl_r.collidepoint(pos):
                    self.show_rl_modal = True
                elif self.btn_help_r.collidepoint(pos):
                    self.show_help_modal = True

    def _toggle_training(self):
        if not self.training_active:
            self.training_active = True
            self.training_status = "Treinando em segundo plano..."
            self.training_thread = threading.Thread(target=self._run_training_worker, daemon=True)
            self.training_thread.start()
        else:
            self.training_active = False
            self.training_status = "Pausado"

    def _run_training_worker(self):
        map_path = STADIUM_CATALOG[self.current_stadium_key]["file"]
        opp_bot = self.bot_catalog[self.active_bot_key]
        env = HaxBallEnv(stadium_file=map_path, opponent_bot=opp_bot, max_steps=1200)

        if self.training_algo == "PPO":
            trainer = PPOTrainer(
                env=env,
                policy_type=self.training_policy,
                num_steps=1024,
                batch_size=64,
                update_epochs=6
            )
            iteration = 0
            while self.training_active:
                iteration += 1
                stats = trainer.train_iteration()
                self.training_metrics["step"] = iteration * 1024
                self.training_metrics["reward"] = stats["mean_reward"]
                self.training_metrics["win_rate"] = stats["win_rate"]
                self.training_metrics["loss"] = stats["loss"]
                self.training_status = f"PPO Iter {iteration} | WinRate {stats['win_rate']:.1f}%"
        else:
            trainer = DQNTrainer(env=env, batch_size=64, target_update_freq=500)
            step_count = 0
            while self.training_active:
                step_count += 500
                stats = trainer.train_step(num_env_steps=500)
                self.training_metrics["step"] = step_count
                self.training_metrics["reward"] = stats["mean_reward"]
                self.training_metrics["loss"] = stats["loss"]
                self.training_status = f"DQN Step {step_count} | Eps {stats['epsilon']:.2f}"

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

HaxBallGUI = HaxBallApp

def main():
    app = HaxBallApp()
    app.run()

if __name__ == "__main__":
    main()
