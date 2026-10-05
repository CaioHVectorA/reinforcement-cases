"""
HaxBall AI Battle Studio - Modern Interactive UI Suite.
Features:
- Full Model vs Model Arena: Put any AI (.pt) against any AI (.pt), bot, or human!
- Flexible Team Sizing: 1v1, 2v2, 3v3 in all official stadiums.
- Red Team Controller: [Human | AI Model A | Bot Script].
- Blue Team Controller: [AI Model B | Bot Script | Human 2].
- Instant .pt model loading with Drag-and-Drop and dynamic file scanner.
- High-Speed Simulation multiplier (1x to 50x) for AI vs AI fast-forward.
- Real-time dual-team tactical telemetry and intent vectors.
"""

from __future__ import annotations
import os
import sys
import time
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import pygame
import torch

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, FPS
from haxball.core.disc import hex_to_rgb
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots import NPC_BOTS, GAUNTLET_ORDER, BaseBot, RLBot
from haxball.ui.widgets import (
    ICON_DISPATCH, draw_icon_stadium, draw_icon_lightning,
    draw_icon_robot, draw_icon_user, draw_icon_reset,
    draw_icon_play, draw_icon_pause, draw_icon_brain, draw_icon_help,
    draw_icon_upload
)

MAP_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

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
        pygame.display.set_caption("HaxBall AI Battle Studio - Arena Multimodelos & Duelos")

        self.clock = pygame.time.Clock()
        self.running = True
        self.is_paused = False

        # App Screen State: "lobby" or "match"
        self.screen_mode = "lobby"

        # Team Setup
        self.players_per_team = 1  # 1v1, 2v2, 3v3
        self.speed_multiplier = 1

        # Red Team Controller: "human", "ai_model", or bot_key (e.g. "press")
        self.red_controller_type = "human"
        self.red_model_file = str(CHECKPOINT_DIR / "haxball_rl_best.pt")

        # Blue Team Controller: "ai_model", bot_key, or "human"
        self.blue_controller_type = "ai_model"
        self.blue_model_file = str(CHECKPOINT_DIR / "haxball_rl_best.pt")

        if model_path:
            self.blue_controller_type = "ai_model"
            self.blue_model_file = os.path.abspath(model_path)

        # Fonts
        self.font_title_huge = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_title = pygame.font.SysFont("Verdana", 17, bold=True)
        self.font_score = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_time = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font_hud = pygame.font.SysFont("Verdana", 13, bold=True)
        self.font_regular = pygame.font.SysFont("Arial", 13)
        self.font_bold = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_player = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_telemetry = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 11)

        # Modals & Feedback
        self.show_help_modal = False
        self.checkpoint_feedback = ""
        self.feedback_time = 0.0

        # Current Stadium
        self.current_stadium_key = "futsal_2v2"

        # Initialize Bot Instances
        self._init_bot_catalogs()
        if not model_path and bot_key in self.bot_instances:
            self.blue_controller_type = bot_key

        # Initialize Game World
        self._init_game(self.current_stadium_key, self.players_per_team)

    def _init_bot_catalogs(self):
        self.bot_instances: Dict[str, BaseBot] = {}
        for k, (cls, label, title, desc, col) in NPC_BOTS.items():
            self.bot_instances[k] = cls(name=title.split(" ")[0])

        # RL Bots Cache
        self.rl_bots: Dict[str, RLBot] = {}

    def get_rl_bot(self, model_file: str) -> RLBot:
        if model_file not in self.rl_bots:
            self.rl_bots[model_file] = RLBot(model_path=model_file, name=os.path.basename(model_file))
        return self.rl_bots[model_file]

    def _init_game(self, stadium_key: str, players_count: int = 1):
        self.current_stadium_key = stadium_key
        stadium_info = STADIUM_CATALOG[stadium_key]
        stadium = Stadium.load_from_file(stadium_info["file"])

        self.players_per_team = players_count
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

    def get_available_checkpoints(self) -> List[str]:
        cp_dir = CHECKPOINT_DIR
        if not cp_dir.exists():
            return []
        files = [str(path) for path in cp_dir.rglob("*.pt")]
        files.sort(key=os.path.getmtime, reverse=True)
        return files

    def plug_in_model(self, file_path: str, target_team: str = "blue", auto_start: bool = True):
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.isfile(file_path):
            self.checkpoint_feedback = "Erro: arquivo .pt não encontrado."
            self.feedback_time = time.time()
            return False

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        base_name = os.path.basename(file_path)
        dest_path = str(CHECKPOINT_DIR / base_name)
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            try:
                shutil.copy2(file_path, dest_path)
            except OSError as error:
                self.checkpoint_feedback = f"Erro ao copiar modelo: {error}"
                self.feedback_time = time.time()
                return False

        # Test load
        test_bot = RLBot(model_path=dest_path, name=base_name)
        if test_bot.policy is None:
            self.checkpoint_feedback = f"Erro: não foi possível carregar {base_name}."
            self.feedback_time = time.time()
            return False
        self.rl_bots[dest_path] = test_bot

        if target_team == "red":
            self.red_controller_type = "ai_model"
            self.red_model_file = dest_path
        else:
            self.blue_controller_type = "ai_model"
            self.blue_model_file = dest_path

        self.checkpoint_feedback = f"Modelo {base_name} carregado no Time {target_team.upper()}!"
        self.feedback_time = time.time()
        if auto_start:
            self.screen_mode = "match"
            self.game.reset_match()
        return True

    def prompt_upload_model(self, target_team: str = "blue"):
        selected_file = None

        # Windows has no zenity; use the native file picker instead of silently
        # selecting an unrelated checkpoint from Downloads.
        if sys.platform.startswith("win"):
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                selected_file = filedialog.askopenfilename(
                    title=f"Selecione o modelo PyTorch para o time {target_team.upper()}",
                    initialdir=str(CHECKPOINT_DIR if CHECKPOINT_DIR.exists() else PROJECT_ROOT),
                    filetypes=[("Modelos PyTorch", "*.pt"), ("Todos os arquivos", "*.*")],
                )
                root.destroy()
            except Exception as error:
                self.checkpoint_feedback = f"Seletor de arquivo indisponível: {error}"
                self.feedback_time = time.time()

        zenity_path = shutil.which("zenity")

        if not selected_file and zenity_path:
            try:
                cmd = [
                    zenity_path,
                    "--file-selection",
                    "--title=Selecione o Modelo PyTorch (.pt)",
                    "--file-filter=Modelos PyTorch (*.pt) | *.pt",
                    "--file-filter=Todos os Arquivos | *"
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                if res.returncode == 0 and res.stdout.strip():
                    selected_file = res.stdout.strip()
            except Exception as e:
                print(f"[GUI] Aviso zenity: {e}")

        if not selected_file:
            # Do not guess a file. The user can cancel and choose again.
            selected_file = None

        if selected_file and os.path.exists(selected_file):
            self.plug_in_model(selected_file, target_team=target_team, auto_start=False)

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

        # 1. Red Team Control
        red_players = [p for p in self.game.players if p.team == Team.RED]
        for i, p in enumerate(red_players):
            if self.red_controller_type == "human" and i == 0:
                inputs[p.player_id] = (mx, my, kick)
            elif self.red_controller_type == "ai_model":
                bot = self.get_rl_bot(self.red_model_file)
                inputs[p.player_id] = bot.act(self.game, p)
            elif self.red_controller_type in self.bot_instances:
                bot = self.bot_instances[self.red_controller_type]
                inputs[p.player_id] = bot.act(self.game, p)
            else:
                # Default heuristic
                inputs[p.player_id] = self.bot_instances["heuristic"].act(self.game, p)

        # 2. Blue Team Control
        blue_players = [p for p in self.game.players if p.team == Team.BLUE]
        for i, p in enumerate(blue_players):
            if self.blue_controller_type == "human" and i == 0:
                inputs[p.player_id] = (mx, my, kick)
            elif self.blue_controller_type == "ai_model":
                bot = self.get_rl_bot(self.blue_model_file)
                inputs[p.player_id] = bot.act(self.game, p)
            elif self.blue_controller_type in self.bot_instances:
                bot = self.bot_instances[self.blue_controller_type]
                inputs[p.player_id] = bot.act(self.game, p)
            else:
                inputs[p.player_id] = self.bot_instances["press"].act(self.game, p)

        return inputs

    def update(self):
        if self.screen_mode != "match" or self.is_paused or self.show_help_modal:
            return

        for _ in range(self.speed_multiplier):
            inputs = self.get_player_inputs()
            self.game.step(inputs)

    def draw(self):
        if self.screen_mode == "lobby":
            self._draw_lobby()
        else:
            self._draw_match()

        if self.show_help_modal:
            self._draw_help_modal()

        pygame.display.flip()

    # =========================================================================
    # LOBBY / CONFIGURATION HUB
    # =========================================================================
    def _draw_lobby(self):
        self.screen.fill((16, 22, 30))

        # Title Header
        title_surf = self.font_title_huge.render("⚽ HAXBALL AI BATTLE STUDIO - HUB DE CONFRONTO", True, (255, 255, 255))
        self.screen.blit(title_surf, (40, 24))

        sub_txt = "Configure qualquer time: Duelo Humano vs IA, IA vs IA (Modelo vs Modelo), 1v1, 2v2 ou 3v3!"
        sub_surf = self.font_regular.render(sub_txt, True, (160, 175, 195))
        self.screen.blit(sub_surf, (42, 58))

        # Feedback Toast
        if self.checkpoint_feedback and (time.time() - self.feedback_time < 4.0):
            fb_surf = self.font_bold.render(f"✓ {self.checkpoint_feedback}", True, (60, 230, 130))
            self.screen.blit(fb_surf, (self.width - fb_surf.get_width() - 40, 30))

        col_y = 95
        col_h = 555

        # ---------------------------------------------------------------------
        # COLUMN 1: TIME VERMELHO (Red Team) - Width 380
        # ---------------------------------------------------------------------
        c1_w = 380
        c1_r = pygame.Rect(40, col_y, c1_w, col_h)
        pygame.draw.rect(self.screen, (24, 30, 40), c1_r, border_radius=10)
        pygame.draw.rect(self.screen, (229, 110, 86), c1_r, width=2, border_radius=10)

        h1 = self.font_title.render("🔴 TIME VERMELHO", True, (255, 130, 110))
        self.screen.blit(h1, (58, col_y + 14))

        # Red Options
        self.red_ctrl_buttons = {}
        red_opts = [
            ("human", "👤 Humano (Você no Teclado)", "Controle manual via WASD / Setas"),
            ("ai_model", f"🤖 Modelo IA: {os.path.basename(self.red_model_file)[:22]}", "Rede neural carregada (.pt)"),
            ("master", "👑 Bot: Master Pro (Mestre)", "Tabelas, fintas e chutes nos cantos"),
            ("press", "⚡ Bot: Pressão Total", "Marcação sob pressão alta"),
            ("wall", "🧱 Bot: Tabelador de Parede", "Especialista em rebotes"),
        ]

        for i, (k, title, desc) in enumerate(red_opts):
            btn_r = pygame.Rect(58, col_y + 48 + i * 58, c1_w - 36, 52)
            self.red_ctrl_buttons[k] = btn_r
            is_active = (self.red_controller_type == k)

            bg_c = (55, 38, 38) if is_active else (30, 36, 46)
            bd_c = (240, 90, 80) if is_active else (48, 58, 72)
            pygame.draw.rect(self.screen, bg_c, btn_r, border_radius=6)
            pygame.draw.rect(self.screen, bd_c, btn_r, width=2 if is_active else 1, border_radius=6)

            t_s = self.font_bold.render(title, True, (255, 255, 255))
            self.screen.blit(t_s, (btn_r.x + 10, btn_r.y + 8))
            d_s = self.font_small.render(desc, True, (180, 190, 205))
            self.screen.blit(d_s, (btn_r.x + 10, btn_r.y + 28))

            tag = "● ATIVO" if is_active else "Escolher"
            tag_c = (255, 120, 100) if is_active else (70, 150, 230)
            tag_s = self.font_bold.render(tag, True, tag_c)
            self.screen.blit(tag_s, (btn_r.right - tag_s.get_width() - 10, btn_r.y + 16))

        # Checkpoints Picker for Red
        self.btn_red_upload_r = pygame.Rect(58, col_y + 350, c1_w - 36, 36)
        self._draw_btn(self.btn_red_upload_r, "📤 Trocar .PT Vermelho", (215, 115, 35), icon="upload")

        # Quick model buttons for Red
        all_cps = self.get_available_checkpoints()
        self.red_quick_cp_buttons = {}
        for idx, cp_p in enumerate(all_cps[:3]):
            cp_name = os.path.basename(cp_p)
            cp_btn_r = pygame.Rect(58, col_y + 396 + idx * 42, c1_w - 36, 36)
            self.red_quick_cp_buttons[cp_p] = cp_btn_r
            is_cur = (self.red_controller_type == "ai_model" and self.red_model_file == cp_p)
            self._draw_btn(cp_btn_r, f"► {cp_name[:24]}", (45, 55, 70) if not is_cur else (160, 50, 50))

        # ---------------------------------------------------------------------
        # COLUMN 2: TIME AZUL (Blue Team) - Width 380
        # ---------------------------------------------------------------------
        c2_x = 450
        c2_w = 380
        c2_r = pygame.Rect(c2_x, col_y, c2_w, col_h)
        pygame.draw.rect(self.screen, (24, 30, 40), c2_r, border_radius=10)
        pygame.draw.rect(self.screen, (86, 137, 229), c2_r, width=2, border_radius=10)

        h2 = self.font_title.render("🔵 TIME AZUL", True, (110, 170, 255))
        self.screen.blit(h2, (c2_x + 18, col_y + 14))

        # Blue Options
        self.blue_ctrl_buttons = {}
        blue_opts = [
            ("ai_model", f"🤖 Modelo IA: {os.path.basename(self.blue_model_file)[:22]}", "Rede neural carregada (.pt)"),
            ("human", "👤 Humano 2 (WASD / Setas)", "Controle manual secundário"),
            ("master", "👑 Bot: Master Pro (Mestre)", "Tabelas, fintas e chutes nos cantos"),
            ("press", "⚡ Bot: Pressão Total", "Marcação sob pressão alta"),
            ("heuristic", "🎯 Bot: Heurístico Clássico", "Perseguição e chute em linha reta"),
        ]

        for i, (k, title, desc) in enumerate(blue_opts):
            btn_r = pygame.Rect(c2_x + 18, col_y + 48 + i * 58, c2_w - 36, 52)
            self.blue_ctrl_buttons[k] = btn_r
            is_active = (self.blue_controller_type == k)

            bg_c = (35, 48, 68) if is_active else (30, 36, 46)
            bd_c = (86, 137, 229) if is_active else (48, 58, 72)
            pygame.draw.rect(self.screen, bg_c, btn_r, border_radius=6)
            pygame.draw.rect(self.screen, bd_c, btn_r, width=2 if is_active else 1, border_radius=6)

            t_s = self.font_bold.render(title, True, (255, 255, 255))
            self.screen.blit(t_s, (btn_r.x + 10, btn_r.y + 8))
            d_s = self.font_small.render(desc, True, (180, 190, 205))
            self.screen.blit(d_s, (btn_r.x + 10, btn_r.y + 28))

            tag = "● ATIVO" if is_active else "Escolher"
            tag_c = (100, 180, 255) if is_active else (70, 150, 230)
            tag_s = self.font_bold.render(tag, True, tag_c)
            self.screen.blit(tag_s, (btn_r.right - tag_s.get_width() - 10, btn_r.y + 16))

        # Checkpoints Picker for Blue
        self.btn_blue_upload_r = pygame.Rect(c2_x + 18, col_y + 350, c2_w - 36, 36)
        self._draw_btn(self.btn_blue_upload_r, "📤 Trocar .PT Azul", (215, 115, 35), icon="upload")

        self.blue_quick_cp_buttons = {}
        for idx, cp_p in enumerate(all_cps[:3]):
            cp_name = os.path.basename(cp_p)
            cp_btn_r = pygame.Rect(c2_x + 18, col_y + 396 + idx * 42, c2_w - 36, 36)
            self.blue_quick_cp_buttons[cp_p] = cp_btn_r
            is_cur = (self.blue_controller_type == "ai_model" and self.blue_model_file == cp_p)
            self._draw_btn(cp_btn_r, f"► {cp_name[:24]}", (45, 55, 70) if not is_cur else (50, 90, 160))

        # ---------------------------------------------------------------------
        # COLUMN 3: MAPA & FORMATO (1v1, 2v2, 3v3) - Width 370
        # ---------------------------------------------------------------------
        c3_x = 860
        c3_w = 380
        c3_r = pygame.Rect(c3_x, col_y, c3_w, col_h)
        pygame.draw.rect(self.screen, (24, 30, 40), c3_r, border_radius=10)
        pygame.draw.rect(self.screen, (60, 210, 120), c3_r, width=2, border_radius=10)

        h3 = self.font_title.render("⚙️ FORMATO & ESTÁDIO", True, (100, 230, 150))
        self.screen.blit(h3, (c3_x + 18, col_y + 14))

        # Player Count Selector (1v1, 2v2, 3v3)
        self.format_buttons = {}
        f_opts = [(1, "1v1 Duelo"), (2, "2v2 Futsal"), (3, "3v3 GLH")]
        for idx, (f_num, f_label) in enumerate(f_opts):
            f_r = pygame.Rect(c3_x + 18 + idx * 116, col_y + 48, 110, 40)
            self.format_buttons[f_num] = f_r
            is_f_act = (self.players_per_team == f_num)
            f_bg = (40, 90, 60) if is_f_act else (32, 40, 52)
            self._draw_btn(f_r, f_label, f_bg)

        # Stadium list
        self.lobby_stadium_buttons = {}
        stads = list(STADIUM_CATALOG.keys())
        for i, s_key in enumerate(stads):
            s_info = STADIUM_CATALOG[s_key]
            s_rect = pygame.Rect(c3_x + 18, col_y + 104 + i * 88, c3_w - 36, 78)
            self.lobby_stadium_buttons[s_key] = s_rect

            is_active_stad = (self.current_stadium_key == s_key)
            bg_s = (35, 55, 50) if is_active_stad else (30, 36, 46)
            bd_s = (60, 210, 120) if is_active_stad else (48, 58, 72)
            pygame.draw.rect(self.screen, bg_s, s_rect, border_radius=8)
            pygame.draw.rect(self.screen, bd_s, s_rect, width=2 if is_active_stad else 1, border_radius=8)

            t_s = self.font_bold.render(s_info["title"][:26], True, (255, 255, 255))
            self.screen.blit(t_s, (s_rect.x + 12, s_rect.y + 12))
            d_s = self.font_small.render(s_info["desc"][:42] + "...", True, (170, 185, 200))
            self.screen.blit(d_s, (s_rect.x + 12, s_rect.y + 34))

            status_txt = "● ATIVO" if is_active_stad else "Escolher"
            status_col = (60, 210, 120) if is_active_stad else (70, 150, 230)
            b_surf = self.font_bold.render(status_txt, True, status_col)
            self.screen.blit(b_surf, (s_rect.right - b_surf.get_width() - 12, s_rect.y + 50))

        # ---------------------------------------------------------------------
        # BOTTOM ACTION DOCK: START & QUICK SUMMARIES
        # ---------------------------------------------------------------------
        dock_y = self.height - 90
        dock_w = self.width - 80
        pygame.draw.rect(self.screen, (22, 28, 38), (40, dock_y, dock_w, 75), border_radius=12)
        pygame.draw.rect(self.screen, (45, 58, 76), (40, dock_y, dock_w, 75), width=1, border_radius=12)

        # Start Button
        self.btn_lobby_start_r = pygame.Rect(self.width // 2 - 160, dock_y + 12, 320, 50)
        self._draw_btn(self.btn_lobby_start_r, "▶  INICIAR PARTIDA (START)", (40, 180, 100), icon="play")

        # Summary text on left
        red_desc = "Humano" if self.red_controller_type == "human" else (os.path.basename(self.red_model_file)[:16] if self.red_controller_type == "ai_model" else self.red_controller_type.upper())
        blue_desc = "Humano" if self.blue_controller_type == "human" else (os.path.basename(self.blue_model_file)[:16] if self.blue_controller_type == "ai_model" else self.blue_controller_type.upper())
        summary_txt = f"🔴 {red_desc}  VS  🔵 {blue_desc}  ({self.players_per_team}v{self.players_per_team})"
        sum_surf = self.font_bold.render(summary_txt, True, (240, 245, 255))
        self.screen.blit(sum_surf, (60, dock_y + 16))

        hint_surf = self.font_small.render("💡 Arraste e solte qualquer .pt para carregar como IA Vermelha ou Azul!", True, (160, 180, 205))
        self.screen.blit(hint_surf, (60, dock_y + 44))

        # Help button
        self.btn_lobby_help_r = pygame.Rect(self.width - 160, dock_y + 18, 90, 38)
        self._draw_btn(self.btn_lobby_help_r, "Teclas", (45, 58, 76), icon="help")

    # =========================================================================
    # IN-GAME MATCH SCREEN
    # =========================================================================
    def _draw_match(self):
        self.screen.fill((20, 26, 34))

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
                vec_col = (255, 200, 80) if player.team == Team.RED else (255, 255, 100)
                pygame.draw.line(self.screen, vec_col, pos, s_end, width=2)
                pygame.draw.circle(self.screen, (255, 255, 255), s_end, 3)

            num_str = str(player.player_id)
            num_surf = self.font_player.render(num_str, True, (255, 255, 255))
            self.screen.blit(num_surf, num_surf.get_rect(center=pos))

        # Scoreboard & Overlays
        self._draw_scoreboard()
        self._draw_dual_telemetry_hud()
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

    def _draw_dual_telemetry_hud(self):
        ball = self.game.ball
        if not ball:
            return

        red_p = next((p for p in self.game.players if p.team == Team.RED), None)
        blue_p = next((p for p in self.game.players if p.team == Team.BLUE), None)

        hud_w = 210
        hud_h = 92
        hud_y = 12

        # 1. Red Team Telemetry Card (Left)
        if red_p:
            dist_r = red_p.pos.distance_to(ball.pos)
            red_ctrl = "Humano" if self.red_controller_type == "human" else (os.path.basename(self.red_model_file)[:14] if self.red_controller_type == "ai_model" else self.red_controller_type.upper())
            r_card = pygame.Rect(20, hud_y, hud_w, hud_h)
            pygame.draw.rect(self.screen, (26, 20, 20), r_card, border_radius=8)
            pygame.draw.rect(self.screen, (229, 110, 86), r_card, width=1, border_radius=8)
            r_t1 = self.font_telemetry.render(f"🔴 RED: {red_ctrl}", True, (255, 140, 120))
            r_t2 = self.font_small.render(f"• Distância Bola: {int(dist_r)} px", True, (220, 230, 240))
            r_t3 = self.font_small.render(f"• Velocidade: {red_p.speed.length():.1f} px/f", True, (220, 230, 240))
            self.screen.blit(r_t1, (30, hud_y + 8))
            self.screen.blit(r_t2, (30, hud_y + 32))
            self.screen.blit(r_t3, (30, hud_y + 54))

        # 2. Blue Team Telemetry Card (Right)
        if blue_p:
            dist_b = blue_p.pos.distance_to(ball.pos)
            blue_ctrl = "Humano" if self.blue_controller_type == "human" else (os.path.basename(self.blue_model_file)[:14] if self.blue_controller_type == "ai_model" else self.blue_controller_type.upper())
            b_card = pygame.Rect(self.width - hud_w - 20, hud_y, hud_w, hud_h)
            pygame.draw.rect(self.screen, (20, 26, 36), b_card, border_radius=8)
            pygame.draw.rect(self.screen, (86, 137, 229), b_card, width=1, border_radius=8)
            b_t1 = self.font_telemetry.render(f"🔵 BLUE: {blue_ctrl}", True, (120, 180, 255))
            b_t2 = self.font_small.render(f"• Distância Bola: {int(dist_b)} px", True, (220, 230, 240))
            b_t3 = self.font_small.render(f"• Velocidade: {blue_p.speed.length():.1f} px/f", True, (220, 230, 240))
            self.screen.blit(b_t1, (self.width - hud_w - 10, hud_y + 8))
            self.screen.blit(b_t2, (self.width - hud_w - 10, hud_y + 32))
            self.screen.blit(b_t3, (self.width - hud_w - 10, hud_y + 54))

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
        self._draw_btn(self.btn_match_upload_r, "📤 Carregar .PT", (215, 135, 25), icon="upload")

        # Status Tag in Middle
        red_label = "Humano" if self.red_controller_type == "human" else os.path.basename(self.red_model_file)[:14]
        blue_label = "Humano" if self.blue_controller_type == "human" else os.path.basename(self.blue_model_file)[:14]
        status_tag = f"{self.players_per_team}v{self.players_per_team} Arena: 🔴 {red_label} vs 🔵 {blue_label}"
        st_surf = self.font_bold.render(status_tag, True, (210, 225, 240))
        self.screen.blit(st_surf, (625, dock_y + 20))

        # Help / Controls
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

        t = self.font_title.render("Guia de Controles e Duelo Multimodelos", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 24))

        lines = [
            "• Movimentação: TECLAS WASD OU SETAS DO TECLADO simultaneamente.",
            "• Chute: BARRA DE ESPAÇO, TECLA X, TECLA C ou SHIFT.",
            "• Modo Modelo vs Modelo: Escolha uma IA para o Vermelho e outra para o Azul!",
            "• Upload de Modelos: Arraste e solte (Drag & Drop) qualquer .pt na tela a qualquer hora!",
            "• Reiniciar partida: Tecla R.",
            "• Pausar partida: Tecla P ou ESC.",
            "• Velocidade acelerada: Tecla TAB (1x a 50x para partidas aceleradas de IA vs IA).",
            "• Menu / Lobby: Clique em 'Menu Principal' para trocar modo, arena ou escalação."
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
                # Plug as blue opponent by default
                self.plug_in_model(dropped_file, target_team="blue", auto_start=True)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                if self.show_help_modal:
                    if hasattr(self, "btn_close_help_r") and self.btn_close_help_r.collidepoint(pos):
                        self.show_help_modal = False
                    continue

                # LOBBY INTERACTIONS
                if self.screen_mode == "lobby":
                    # Red controller button clicks
                    if hasattr(self, "red_ctrl_buttons"):
                        for k, btn_r in self.red_ctrl_buttons.items():
                            if btn_r.collidepoint(pos):
                                self.red_controller_type = k
                                break

                    if hasattr(self, "red_quick_cp_buttons"):
                        for cp_p, btn_r in self.red_quick_cp_buttons.items():
                            if btn_r.collidepoint(pos):
                                self.red_controller_type = "ai_model"
                                self.red_model_file = cp_p
                                break

                    if hasattr(self, "btn_red_upload_r") and self.btn_red_upload_r.collidepoint(pos):
                        self.prompt_upload_model(target_team="red")

                    # Blue controller button clicks
                    if hasattr(self, "blue_ctrl_buttons"):
                        for k, btn_r in self.blue_ctrl_buttons.items():
                            if btn_r.collidepoint(pos):
                                self.blue_controller_type = k
                                break

                    if hasattr(self, "blue_quick_cp_buttons"):
                        for cp_p, btn_r in self.blue_quick_cp_buttons.items():
                            if btn_r.collidepoint(pos):
                                self.blue_controller_type = "ai_model"
                                self.blue_model_file = cp_p
                                break

                    if hasattr(self, "btn_blue_upload_r") and self.btn_blue_upload_r.collidepoint(pos):
                        self.prompt_upload_model(target_team="blue")

                    # Player Count Formats (1v1, 2v2, 3v3)
                    if hasattr(self, "format_buttons"):
                        for f_num, f_r in self.format_buttons.items():
                            if f_r.collidepoint(pos):
                                self.players_per_team = f_num
                                self._init_game(self.current_stadium_key, f_num)
                                break

                    # Stadium button clicks
                    if hasattr(self, "lobby_stadium_buttons"):
                        for s_key, s_rect in self.lobby_stadium_buttons.items():
                            if s_rect.collidepoint(pos):
                                self.current_stadium_key = s_key
                                self._init_game(s_key, self.players_per_team)
                                break

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
                        self.prompt_upload_model(target_team="blue")
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
