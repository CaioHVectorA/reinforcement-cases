"""
HaxBall AI Battle Studio - Modern Interactive UI Suite.
Features:
- Clean, intuitive Start Lobby (Game Mode, Opponent/Model, Stadium Selection).
- Direct 1v1 Human vs AI with instant .pt model loading & Drag-and-Drop.
- Official map catalog: Futsal 2v2, Micro 1v1, Futsal 3v3 GLH, Classic, Big Stadium.
- Real-time tactical telemetry & vector intent visualization.
- Keyboard controls: WASD or Arrow Keys + Space/X/C/Shift.
"""

from __future__ import annotations
import os
import sys
import time
import math
import shutil
import subprocess
import threading
from typing import Dict, Tuple, Optional, Any, List
import pygame
import torch

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, FPS
from haxball.core.disc import hex_to_rgb
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots import NPC_BOTS, GAUNTLET_ORDER, BaseBot, RLBot
from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer
from haxball.ui.widgets import (
    ICON_DISPATCH, draw_icon_stadium, draw_icon_lightning,
    draw_icon_robot, draw_icon_user, draw_icon_reset,
    draw_icon_play, draw_icon_pause, draw_icon_brain, draw_icon_help,
    draw_icon_upload
)

MAP_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")

STADIUM_CATALOG = {
    "futsal_2v2": {
        "title": "Futsal 2v2 Arena",
        "file": os.path.join(MAP_DIR, "futsal_2v2.hbs"),
        "desc": "Quadra balanceada (450x200). Paredes elásticas e condução ágil.",
        "format": 2
    },
    "micro_1v1": {
        "title": "Micro 1v1 Arena",
        "file": os.path.join(MAP_DIR, "micro_1v1.hbs"),
        "desc": "Arena compacta (340x160) com transições imediatas de 1x1.",
        "format": 1
    },
    "futsal_3v3": {
        "title": "Futsal 3v3 GLH (7899)",
        "file": os.path.join(MAP_DIR, "futsal_3v3.hbs"),
        "desc": "Mapa oficial mais jogado do mundo. Bola rápida e tabelas perfeitas.",
        "format": 3
    },
    "classic": {
        "title": "Classic Stadium Oficial",
        "file": os.path.join(MAP_DIR, "classic.hbs"),
        "desc": "Campo de grama padrão de HaxBall com traves arredondadas.",
        "format": 1
    },
    "big_stadium": {
        "title": "Big Stadium Oficial",
        "file": os.path.join(MAP_DIR, "big_stadium.hbs"),
        "desc": "Campo amplo de grama (840x400) para passes longos e cruzamentos.",
        "format": 3
    }
}

SPEED_LEVELS = [1, 2, 5, 10, 25, 50]

class HaxBallApp:
    def __init__(
        self,
        width: int = 1280,
        height: int = 768,
        mode: str = "human",
        model_path: Optional[str] = None,
        bot_key: str = "rl"
    ):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("HaxBall AI Battle Studio - 1v1 Arena & Model Hub")

        self.clock = pygame.time.Clock()
        self.running = True
        self.is_paused = False

        # App Screen State: "lobby" or "match"
        self.screen_mode = "lobby"

        # Modes: "human" (1v1 / NvN) or "self_play" or "gauntlet"
        self.play_mode = mode
        self.speed_multiplier = 1

        # Fonts
        self.font_title_huge = pygame.font.SysFont("Verdana", 26, bold=True)
        self.font_title = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font_score = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_time = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font_hud = pygame.font.SysFont("Verdana", 13, bold=True)
        self.font_regular = pygame.font.SysFont("Arial", 13)
        self.font_bold = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_player = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_telemetry = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 11)

        # Modals
        self.show_help_modal = False
        self.checkpoint_feedback = ""
        self.feedback_time = 0.0

        # Current Stadium & Match Configuration
        self.current_stadium_key = "futsal_2v2"
        self.team_format = 1  # Default to 1v1

        # Auto-detect initial model path
        default_model = model_path
        if default_model is None:
            for candidate in ["checkpoints/human_expert_bc.pt", "checkpoints/fase5_pro_master_1M.pt", "checkpoints/meu_haxball_bot.pt"]:
                if os.path.exists(candidate):
                    default_model = candidate
                    break

        self.active_checkpoint_name = os.path.basename(default_model) if default_model else "Nenhum"

        # Bots Catalog
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
            self.active_bot_key = "rl" if default_model else "press"

        self._init_game(self.current_stadium_key, self.team_format)

    def _init_game(self, stadium_key: str, players_count: int = 1):
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
            "human_expert_bc.pt": {"title": "★ Human Expert BC", "desc": "Imitação treinada de replays de humanos", "badge": "Expert", "color": (255, 215, 0)},
            "fase5_pro_master_1M.pt": {"title": "Fase 5: Pro Master (1M)", "desc": "RL com antecipação e posicionamento", "badge": "1M ★", "color": (60, 210, 120)},
            "fase4_veterano_500k.pt": {"title": "Fase 4: Veterano (500k)", "desc": "Passes, desarmes e tabelas", "badge": "500k", "color": (155, 100, 230)},
            "fase3_intermediario_200k.pt": {"title": "Fase 3: Intermediário", "desc": "Cobertura de trave e chutes fortes", "badge": "200k", "color": (58, 142, 230)},
            "fase2_amador_70k.pt": {"title": "Fase 2: Amador (70k)", "desc": "Alinhamento ao gol", "badge": "70k", "color": (220, 200, 50)},
            "fase1_iniciante_25k.pt": {"title": "Fase 1: Iniciante (25k)", "desc": "Perseguição básica da bola", "badge": "25k", "color": (235, 150, 40)},
            "meu_haxball_bot.pt": {"title": "Modelo Recente", "desc": "Último checkpoint salvo", "badge": "Recente", "color": (80, 190, 160)},
        }

        results = []
        ordered_keys = ["human_expert_bc.pt", "fase5_pro_master_1M.pt", "fase4_veterano_500k.pt", "fase3_intermediario_200k.pt", "fase2_amador_70k.pt", "fase1_iniciante_25k.pt", "meu_haxball_bot.pt"]

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
                    "desc": "Modelo customizado carregado",
                    "badge": "Custom",
                    "color": (160, 170, 185),
                    "size_kb": int(os.path.getsize(full_p) / 1024)
                })

        return results

    def plug_in_model_for_1v1(self, file_path: str, auto_start: bool = True):
        if not os.path.exists(file_path):
            self.checkpoint_feedback = f"Erro: Arquivo não encontrado!"
            self.feedback_time = time.time()
            return

        os.makedirs("checkpoints", exist_ok=True)
        base_name = os.path.basename(file_path)
        dest_path = os.path.join("checkpoints", base_name)
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            try:
                shutil.copy2(file_path, dest_path)
            except Exception:
                dest_path = file_path

        if "rl" not in self.bot_catalog:
            self.bot_catalog["rl"] = RLBot(model_path=dest_path, name="RL Bot (Trained)")
            success = True
        else:
            success = self.bot_catalog["rl"].load_model(dest_path)

        if success:
            self.active_checkpoint_name = base_name
            self.active_bot_key = "rl"
            self.gauntlet_mode = False
            self.play_mode = "human"
            self.team_format = 1
            self._init_game(self.current_stadium_key, 1)
            self.game.reset_match()
            if auto_start:
                self.screen_mode = "match"
            self.checkpoint_feedback = f"Modelo {base_name} PLUGADO! 1x1 Pronto!"
            self.feedback_time = time.time()
            print(f"[GUI] Modelo plugado: {dest_path}")
        else:
            self.checkpoint_feedback = f"Falha ao carregar {base_name}"
            self.feedback_time = time.time()

    def prompt_upload_model(self):
        zenity_path = shutil.which("zenity")
        selected_file = None

        if zenity_path:
            try:
                cmd = [
                    zenity_path,
                    "--file-selection",
                    "--title=Selecione o Modelo PyTorch (.pt) para Duelo 1x1",
                    "--file-filter=Modelos PyTorch (*.pt) | *.pt",
                    "--file-filter=Todos os Arquivos | *"
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                if res.returncode == 0 and res.stdout.strip():
                    selected_file = res.stdout.strip()
            except Exception as e:
                print(f"[GUI] Aviso zenity: {e}")

        if not selected_file:
            home_dl = os.path.expanduser("~/Downloads")
            candidates = []
            if os.path.exists(home_dl):
                candidates.extend([os.path.join(home_dl, f) for f in os.listdir(home_dl) if f.endswith(".pt")])
            if os.path.exists("checkpoints"):
                candidates.extend([os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pt")])

            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                selected_file = candidates[0]

        if selected_file and os.path.exists(selected_file):
            self.plug_in_model_for_1v1(selected_file, auto_start=True)
        else:
            self.checkpoint_feedback = "Arraste e solte o arquivo .pt na tela!"
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

        # Red Team (Player 0 is Human if play_mode != "self_play")
        red_players = [p for p in self.game.players if p.team == Team.RED]
        for i, p in enumerate(red_players):
            if i == 0 and self.play_mode != "self_play":
                inputs[p.player_id] = (mx, my, kick)
            else:
                bot = self.bot_catalog.get(self.active_bot_key, self.bot_catalog["rl"])
                inputs[p.player_id] = bot.act(self.game, p)

        # Blue Team (Opponent Bots)
        blue_players = [p for p in self.game.players if p.team == Team.BLUE]
        for i, p in enumerate(blue_players):
            bot = self.bot_catalog.get(self.active_bot_key, self.bot_catalog["rl"])
            inputs[p.player_id] = bot.act(self.game, p)

        return inputs

    def update(self):
        if self.screen_mode != "match" or self.is_paused or self.show_help_modal:
            return

        for _ in range(self.speed_multiplier):
            inputs = self.get_player_inputs()
            step_info = self.game.step(inputs)

        # Gauntlet transition
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
        if self.screen_mode == "lobby":
            self._draw_lobby()
        else:
            self._draw_match()

        if self.show_help_modal:
            self._draw_help_modal()

        pygame.display.flip()

    # =========================================================================
    # LOBBY / START SCREEN
    # =========================================================================
    def _draw_lobby(self):
        self.screen.fill((18, 24, 32))

        # Title Header
        title_surf = self.font_title_huge.render("⚽ HAXBALL AI BATTLE STUDIO", True, (255, 255, 255))
        self.screen.blit(title_surf, (40, 28))

        sub_surf = self.font_regular.render("Hub de Batalha: Escolha o Modo, Modelo de IA e Estádio para Jogar", True, (160, 175, 195))
        self.screen.blit(sub_surf, (42, 64))

        # Feedback Toast
        if self.checkpoint_feedback and (time.time() - self.feedback_time < 4.0):
            fb_surf = self.font_bold.render(f"✓ {self.checkpoint_feedback}", True, (60, 230, 130))
            self.screen.blit(fb_surf, (self.width - fb_surf.get_width() - 40, 36))

        # 3 Panels: Column 1 (Mode), Column 2 (Opponent / Model), Column 3 (Stadium)
        col_y = 105
        col_h = 540

        # ---------------------------------------------------------------------
        # COLUMN 1: GAME MODE (Width 360)
        # ---------------------------------------------------------------------
        col1_w = 360
        c1_rect = pygame.Rect(40, col_y, col1_w, col_h)
        pygame.draw.rect(self.screen, (26, 33, 44), c1_rect, border_radius=10)
        pygame.draw.rect(self.screen, (45, 58, 76), c1_rect, width=1, border_radius=10)

        h1 = self.font_title.render("1. MODO DE JOGO", True, (100, 180, 255))
        self.screen.blit(h1, (58, col_y + 16))

        mode_options = [
            ("human_1v1", "⚔️ Duelo 1v1 (Você vs IA)", "Confronto direto contra o modelo carregado", 1, "human"),
            ("human_2v2", "👥 Partida 2v2 (Futsal)", "Você + 1 Bot contra dupla de IAs", 2, "human"),
            ("human_3v3", "👥 Partida 3v3 (GLH)", "Trio completo em quadra oficial", 3, "human"),
            ("gauntlet", "🏆 Torneio Gauntlet", "Enfrente a escada com 9 arquétipos", 1, "gauntlet"),
            ("self_play", "🤖 Assistir / Auto-Confronto", "IAs jogando e aprendendo sozinhas", 2, "self_play"),
        ]

        self.lobby_mode_buttons = {}
        for i, (m_id, m_title, m_desc, m_format, p_mode) in enumerate(mode_options):
            m_rect = pygame.Rect(58, col_y + 55 + i * 92, col1_w - 36, 80)
            self.lobby_mode_buttons[m_id] = (m_rect, m_format, p_mode)

            is_selected = (
                (self.gauntlet_mode and m_id == "gauntlet") or
                (not self.gauntlet_mode and self.play_mode == "self_play" and m_id == "self_play") or
                (not self.gauntlet_mode and self.play_mode == "human" and self.team_format == m_format and m_id == f"human_{m_format}v{m_format}")
            )

            bg_col = (45, 60, 80) if is_selected else (32, 40, 52)
            bd_col = (58, 142, 230) if is_selected else (50, 62, 78)
            pygame.draw.rect(self.screen, bg_col, m_rect, border_radius=8)
            pygame.draw.rect(self.screen, bd_col, m_rect, width=2 if is_selected else 1, border_radius=8)

            t_surf = self.font_bold.render(m_title, True, (255, 255, 255))
            self.screen.blit(t_surf, (m_rect.x + 14, m_rect.y + 12))

            d_surf = self.font_small.render(m_desc, True, (170, 185, 200))
            self.screen.blit(d_surf, (m_rect.x + 14, m_rect.y + 36))

            badge_text = "● ATIVO" if is_selected else "Selecionar"
            badge_color = (60, 210, 120) if is_selected else (58, 142, 230)
            b_surf = self.font_bold.render(badge_text, True, badge_color)
            self.screen.blit(b_surf, (m_rect.right - b_surf.get_width() - 14, m_rect.y + 52))

        # ---------------------------------------------------------------------
        # COLUMN 2: OPPONENT & MODEL (.PT) (Width 440)
        # ---------------------------------------------------------------------
        col2_x = 420
        col2_w = 440
        c2_rect = pygame.Rect(col2_x, col_y, col2_w, col_h)
        pygame.draw.rect(self.screen, (26, 33, 44), c2_rect, border_radius=10)
        pygame.draw.rect(self.screen, (45, 58, 76), c2_rect, width=1, border_radius=10)

        h2 = self.font_title.render("2. OPONENTE & MODELO IA", True, (60, 210, 120))
        self.screen.blit(h2, (col2_x + 18, col_y + 16))

        # Upload Button in Column 2
        self.btn_lobby_upload_r = pygame.Rect(col2_x + 18, col_y + 52, col2_w - 36, 40)
        self._draw_btn(self.btn_lobby_upload_r, "📤 Carregar Modelo .PT do PC", (215, 135, 25), icon="upload")

        # Available Models & Bot Archetypes
        cps = self.get_available_checkpoints()
        self.lobby_model_buttons = {}
        
        y_cursor = col_y + 104
        for i, cp in enumerate(cps[:4]):
            cp_rect = pygame.Rect(col2_x + 18, y_cursor, col2_w - 36, 60)
            self.lobby_model_buttons[cp["filename"]] = cp_rect
            is_active_cp = (self.active_bot_key == "rl" and self.active_checkpoint_name == cp["filename"])

            bg_c = (45, 62, 78) if is_active_cp else (32, 40, 52)
            pygame.draw.rect(self.screen, bg_c, cp_rect, border_radius=6)
            pygame.draw.rect(self.screen, cp["color"] if is_active_cp else (50, 62, 78), cp_rect, width=2 if is_active_cp else 1, border_radius=6)

            t_c = self.font_bold.render(cp["title"][:28], True, (255, 255, 255))
            self.screen.blit(t_c, (cp_rect.x + 12, cp_rect.y + 10))

            d_c = self.font_small.render(cp["desc"][:45], True, (160, 175, 190))
            self.screen.blit(d_c, (cp_rect.x + 12, cp_rect.y + 32))

            status_txt = "● ATIVO" if is_active_cp else "Escolher"
            status_col = (60, 210, 120) if is_active_cp else (58, 142, 230)
            s_s = self.font_bold.render(status_txt, True, status_col)
            self.screen.blit(s_s, (cp_rect.right - s_s.get_width() - 12, cp_rect.y + 20))

            y_cursor += 68

        # NPC Archetype Quick Pick
        npc_label = self.font_telemetry.render("Ou enfrente Bots Programados:", True, (160, 175, 195))
        self.screen.blit(npc_label, (col2_x + 18, y_cursor + 4))
        y_cursor += 24

        npc_picks = [
            ("master", "Master Pro", (255, 100, 100)),
            ("press", "Pressing", (255, 165, 0)),
            ("bank", "Tabelas", (100, 200, 255)),
            ("dribbler", "Dribbler", (200, 100, 255)),
        ]
        self.lobby_npc_buttons = {}
        for idx, (n_key, n_name, n_color) in enumerate(npc_picks):
            col_i = idx % 2
            row_i = idx // 2
            n_r = pygame.Rect(col2_x + 18 + col_i * 205, y_cursor + row_i * 38, 195, 32)
            self.lobby_npc_buttons[n_key] = n_r
            is_active_npc = (not self.gauntlet_mode and self.active_bot_key == n_key)

            bg_n = (45, 58, 76) if is_active_npc else (32, 40, 52)
            pygame.draw.rect(self.screen, bg_n, n_r, border_radius=6)
            pygame.draw.rect(self.screen, n_color if is_active_npc else (50, 62, 78), n_r, width=2 if is_active_npc else 1, border_radius=6)

            n_txt = self.font_bold.render(f"Bot: {n_name}", True, (240, 245, 250))
            self.screen.blit(n_txt, n_txt.get_rect(center=n_r.center))

        # ---------------------------------------------------------------------
        # COLUMN 3: STADIUM / MAP (Width 380)
        # ---------------------------------------------------------------------
        col3_x = 880
        col3_w = 360
        c3_rect = pygame.Rect(col3_x, col_y, col3_w, col_h)
        pygame.draw.rect(self.screen, (26, 33, 44), c3_rect, border_radius=10)
        pygame.draw.rect(self.screen, (45, 58, 76), c3_rect, width=1, border_radius=10)

        h3 = self.font_title.render("3. ESTÁDIO / ARENA", True, (255, 180, 60))
        self.screen.blit(h3, (col3_x + 18, col_y + 16))

        self.lobby_stadium_buttons = {}
        stads = list(STADIUM_CATALOG.keys())
        for i, s_key in enumerate(stads):
            s_info = STADIUM_CATALOG[s_key]
            s_rect = pygame.Rect(col3_x + 18, col_y + 55 + i * 92, col3_w - 36, 80)
            self.lobby_stadium_buttons[s_key] = s_rect

            is_active_stad = (self.current_stadium_key == s_key)
            bg_s = (45, 60, 80) if is_active_stad else (32, 40, 52)
            bd_s = (255, 180, 60) if is_active_stad else (50, 62, 78)
            pygame.draw.rect(self.screen, bg_s, s_rect, border_radius=8)
            pygame.draw.rect(self.screen, bd_s, s_rect, width=2 if is_active_stad else 1, border_radius=8)

            t_s = self.font_bold.render(s_info["title"][:26], True, (255, 255, 255))
            self.screen.blit(t_s, (s_rect.x + 12, s_rect.y + 12))

            d_s = self.font_small.render(s_info["desc"][:42] + "...", True, (170, 185, 200))
            self.screen.blit(d_s, (s_rect.x + 12, s_rect.y + 36))

            status_txt = "● ATIVO" if is_active_stad else "Escolher"
            status_col = (255, 180, 60) if is_active_stad else (58, 142, 230)
            b_surf = self.font_bold.render(status_txt, True, status_col)
            self.screen.blit(b_surf, (s_rect.right - b_surf.get_width() - 12, s_rect.y + 52))

        # ---------------------------------------------------------------------
        # BOTTOM ACTION DOCK: START BUTTON & DRAG-AND-DROP HINT
        # ---------------------------------------------------------------------
        dock_y = self.height - 90
        dock_w = self.width - 80
        pygame.draw.rect(self.screen, (22, 28, 38), (40, dock_y, dock_w, 75), border_radius=12)
        pygame.draw.rect(self.screen, (45, 58, 76), (40, dock_y, dock_w, 75), width=1, border_radius=12)

        # Start Button
        self.btn_lobby_start_r = pygame.Rect(self.width // 2 - 160, dock_y + 12, 320, 50)
        self._draw_btn(self.btn_lobby_start_r, "▶  ENTRAR EM CAMPO (START)", (40, 180, 100), icon="play")

        # Summary text on left of dock
        mode_label = f"{self.team_format}v{self.team_format}" if not self.gauntlet_mode else "Gauntlet"
        bot_label = self.active_checkpoint_name if self.active_bot_key == "rl" else self.active_bot_key.upper()
        stad_label = STADIUM_CATALOG[self.current_stadium_key]["title"].split(" (")[0]

        summary_txt = f"Modo: {mode_label}  |  Oponente: {bot_label}  |  Arena: {stad_label}"
        sum_surf = self.font_bold.render(summary_txt, True, (220, 235, 250))
        self.screen.blit(sum_surf, (60, dock_y + 18))

        hint_surf = self.font_small.render("💡 Dica: Arraste e solte arquivos .pt na tela para carregar na hora!", True, (160, 180, 205))
        self.screen.blit(hint_surf, (60, dock_y + 44))

        # Help button on right of dock
        self.btn_lobby_help_r = pygame.Rect(self.width - 160, dock_y + 18, 90, 38)
        self._draw_btn(self.btn_lobby_help_r, "Teclas", (45, 58, 76), icon="help")

    # =========================================================================
    # IN-GAME MATCH SCREEN
    # =========================================================================
    def _draw_match(self):
        self.screen.fill((22, 28, 36))

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

            # Intent / Velocity Vector
            if player.speed.length_sq() > 0.05:
                vel_end = player.pos + player.speed.normalized() * (player.radius + 16.0)
                s_end = self.world_to_screen(vel_end)
                pygame.draw.line(self.screen, (255, 255, 100) if player.team == Team.BLUE else (255, 200, 80), pos, s_end, width=2)
                pygame.draw.circle(self.screen, (255, 255, 255), s_end, 3)

            num_str = str(player.player_id)
            num_surf = self.font_player.render(num_str, True, (255, 255, 255))
            self.screen.blit(num_surf, num_surf.get_rect(center=pos))

        # Overlays
        self._draw_scoreboard()
        self._draw_interaction_telemetry_hud()
        self._draw_in_game_dock()

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

    def _draw_interaction_telemetry_hud(self):
        ball = self.game.ball
        blue_p = next((p for p in self.game.players if p.team == Team.BLUE), None)
        if not ball or not blue_p:
            return

        stad = self.game.stadium
        dist_to_ball = blue_p.pos.distance_to(ball.pos)
        to_ball = (ball.pos - blue_p.pos).normalized() if dist_to_ball > 1e-4 else Vec2(1, 0)
        opp_goal = Vec2(-stad.bg_width, 0.0)
        to_goal = (opp_goal - ball.pos).normalized()

        if blue_p.speed.length_sq() > 0.05:
            pursuit_cos = float(blue_p.speed.normalized().dot(to_ball))
            pursuit_pct = int(max(0.0, pursuit_cos) * 100)
        else:
            pursuit_pct = 0

        shot_cos = float(to_ball.dot(to_goal))
        shot_pct = int(max(0.0, shot_cos) * 100)

        if dist_to_ball < (blue_p.radius + ball.radius + 10.0):
            tactical_state = "Controle / Chute"
            state_color = (255, 100, 100)
        elif pursuit_pct > 75:
            tactical_state = "Perseguição Ativa"
            state_color = (60, 210, 120)
        elif blue_p.pos.x > ball.pos.x:
            tactical_state = "Contorno Tático"
            state_color = (240, 190, 60)
        else:
            tactical_state = "Recomposição"
            state_color = (160, 180, 220)

        hud_w = 230
        hud_h = 100
        hud_x = self.width - hud_w - 20
        hud_y = 12

        card_r = pygame.Rect(hud_x, hud_y, hud_w, hud_h)
        pygame.draw.rect(self.screen, (24, 30, 38), card_r, border_radius=8)
        pygame.draw.rect(self.screen, (58, 142, 230), card_r, width=1, border_radius=8)

        t_title = self.font_telemetry.render("TELEMETRIA DA IA (TIME AZUL)", True, (100, 180, 255))
        self.screen.blit(t_title, (hud_x + 10, hud_y + 8))

        m1 = self.font_small.render(f"• Pressão na Bola: {pursuit_pct}%", True, (220, 230, 240))
        m2 = self.font_small.render(f"• Alinhamento ao Gol: {shot_pct}%", True, (220, 230, 240))
        m3 = self.font_small.render(f"• Distância da Bola: {int(dist_to_ball)} px", True, (220, 230, 240))
        m4 = self.font_bold.render(f"• Tática: {tactical_state}", True, state_color)

        self.screen.blit(m1, (hud_x + 10, hud_y + 26))
        self.screen.blit(m2, (hud_x + 10, hud_y + 44))
        self.screen.blit(m3, (hud_x + 10, hud_y + 62))
        self.screen.blit(m4, (hud_x + 10, hud_y + 80))

    def _draw_in_game_dock(self):
        dock_h = 58
        dock_y = self.height - dock_h
        pygame.draw.rect(self.screen, (22, 28, 35), (0, dock_y, self.width, dock_h))
        pygame.draw.line(self.screen, (45, 56, 70), (0, dock_y), (self.width, dock_y), width=1)

        # 1. Back to Lobby
        self.btn_back_lobby_r = pygame.Rect(12, dock_y + 10, 125, 38)
        self._draw_btn(self.btn_back_lobby_r, "🏠 Menu Principal", (50, 62, 78))

        # 2. Play / Pause
        self.btn_pause_r = pygame.Rect(145, dock_y + 10, 95, 38)
        p_txt = "Play" if self.is_paused else "Pausar"
        p_icon = "play" if self.is_paused else "pause"
        self._draw_btn(self.btn_pause_r, p_txt, (58, 142, 230), icon=p_icon)

        # 3. Reset match
        self.btn_reset_r = pygame.Rect(248, dock_y + 10, 85, 38)
        self._draw_btn(self.btn_reset_r, "Reset", (45, 55, 68), icon="reset")

        # 4. Speed Multiplier
        self.btn_dock_speed_r = pygame.Rect(341, dock_y + 10, 100, 38)
        sp_c = (210, 120, 30) if self.speed_multiplier > 1 else (45, 55, 68)
        self._draw_btn(self.btn_dock_speed_r, f"Vel: {self.speed_multiplier}x", sp_c, icon="lightning")

        # 5. Upload / Switch Model
        self.btn_match_upload_r = pygame.Rect(449, dock_y + 10, 165, 38)
        self._draw_btn(self.btn_match_upload_r, "📤 Trocar / Upload .PT", (215, 135, 25), icon="upload")

        # Match Status Tag in Middle
        status_tag = f"Modo: {self.team_format}v{self.team_format}  |  IA: {self.active_checkpoint_name}"
        st_surf = self.font_bold.render(status_tag, True, (200, 215, 235))
        self.screen.blit(st_surf, (635, dock_y + 20))

        # 6. Help / Controls
        self.btn_help_r = pygame.Rect(self.width - 100, dock_y + 10, 85, 38)
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

    def _draw_help_modal(self):
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 720, 420
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (34, 43, 53), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (58, 142, 230), m_r, width=2, border_radius=12)

        t = self.font_title.render("Guia de Controles e Duelo 1x1", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 24))

        lines = [
            "• Movimentação: TECLAS WASD OU SETAS DO TECLADO simultaneamente.",
            "• Chute: BARRA DE ESPAÇO, TECLA X, TECLA C ou SHIFT.",
            "• Upload de Modelos: Arraste e solte (Drag & Drop) qualquer .pt na tela do jogo!",
            "• Reiniciar partida: Tecla R.",
            "• Pausar partida: Tecla P ou ESC.",
            "• Velocidade acelerada: Tecla TAB (1x a 50x para testes ultrarrápidos).",
            "• Menu / Lobby: Clique em 'Menu Principal' para trocar modo, arena ou oponente."
        ]
        y = m_r.y + 70
        for l in lines:
            t_line = self.font_regular.render(l, True, (220, 230, 240))
            self.screen.blit(t_line, (m_r.x + 30, y))
            y += 36

        self.btn_close_help_r = pygame.Rect(m_r.right - 140, m_r.bottom - 46, 110, 34)
        self._draw_btn(self.btn_close_help_r, "Entendi!", (58, 142, 230))

    def cycle_speed(self):
        curr_idx = SPEED_LEVELS.index(self.speed_multiplier) if self.speed_multiplier in SPEED_LEVELS else 0
        nxt_idx = (curr_idx + 1) % len(SPEED_LEVELS)
        self.speed_multiplier = SPEED_LEVELS[nxt_idx]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_help_modal:
                        self.show_help_modal = False
                    elif self.screen_mode == "match":
                        self.screen_mode = "lobby"
                elif event.key == pygame.K_p:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_r:
                    self.game.reset_match()
                elif event.key == pygame.K_TAB:
                    self.cycle_speed()

            elif event.type == pygame.DROPFILE:
                dropped_file = event.file
                print(f"[GUI] Arquivo arrastado detectado: {dropped_file}")
                self.plug_in_model_for_1v1(dropped_file, auto_start=True)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if self.show_help_modal:
                    if hasattr(self, "btn_close_help_r") and self.btn_close_help_r.collidepoint(pos):
                        self.show_help_modal = False
                    continue

                # LOBBY INTERACTIONS
                if self.screen_mode == "lobby":
                    # Mode button clicks
                    if hasattr(self, "lobby_mode_buttons"):
                        for m_id, (m_rect, m_format, p_mode) in self.lobby_mode_buttons.items():
                            if m_rect.collidepoint(pos):
                                if p_mode == "gauntlet":
                                    self.gauntlet_mode = True
                                    self.gauntlet_index = 0
                                    self.active_bot_key = GAUNTLET_ORDER[0]
                                    self.team_format = 1
                                else:
                                    self.gauntlet_mode = False
                                    self.play_mode = p_mode
                                    self.team_format = m_format
                                self._init_game(self.current_stadium_key, self.team_format)
                                break

                    # Model button clicks
                    if hasattr(self, "lobby_model_buttons"):
                        for cp_name, cp_rect in self.lobby_model_buttons.items():
                            if cp_rect.collidepoint(pos):
                                self.plug_in_model_for_1v1(os.path.join("checkpoints", cp_name), auto_start=False)
                                break

                    # NPC button clicks
                    if hasattr(self, "lobby_npc_buttons"):
                        for n_key, n_rect in self.lobby_npc_buttons.items():
                            if n_rect.collidepoint(pos):
                                self.gauntlet_mode = False
                                self.active_bot_key = n_key
                                self.checkpoint_feedback = f"Oponente Selecionado: {n_key.upper()}"
                                self.feedback_time = time.time()
                                break

                    # Stadium button clicks
                    if hasattr(self, "lobby_stadium_buttons"):
                        for s_key, s_rect in self.lobby_stadium_buttons.items():
                            if s_rect.collidepoint(pos):
                                self.current_stadium_key = s_key
                                self._init_game(s_key, self.team_format)
                                break

                    # Upload button click
                    if hasattr(self, "btn_lobby_upload_r") and self.btn_lobby_upload_r.collidepoint(pos):
                        self.prompt_upload_model()

                    # Start button click
                    if hasattr(self, "btn_lobby_start_r") and self.btn_lobby_start_r.collidepoint(pos):
                        self.screen_mode = "match"
                        self.game.reset_match()

                    # Help button click
                    if hasattr(self, "btn_lobby_help_r") and self.btn_lobby_help_r.collidepoint(pos):
                        self.show_help_modal = True

                # MATCH INTERACTIONS
                else:
                    if hasattr(self, "btn_back_lobby_r") and self.btn_back_lobby_r.collidepoint(pos):
                        self.screen_mode = "lobby"
                    elif hasattr(self, "btn_pause_r") and self.btn_pause_r.collidepoint(pos):
                        self.is_paused = not self.is_paused
                    elif hasattr(self, "btn_reset_r") and self.btn_reset_r.collidepoint(pos):
                        self.game.reset_match()
                    elif hasattr(self, "btn_dock_speed_r") and self.btn_dock_speed_r.collidepoint(pos):
                        self.cycle_speed()
                    elif hasattr(self, "btn_match_upload_r") and self.btn_match_upload_r.collidepoint(pos):
                        self.prompt_upload_model()
                    elif hasattr(self, "btn_help_r") and self.btn_help_r.collidepoint(pos):
                        self.show_help_modal = True

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
