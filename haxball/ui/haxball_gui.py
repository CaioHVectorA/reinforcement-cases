"""
Authentic HaxBall GUI Suite & Recursive Self-Play Training Center.
Faithful to original HaxBall game look-and-feel:
- Real-time 2v2 (and NvN) live recursive self-play training directly on pitch.
- Speed acceleration up to 100x (run 100 physics & policy steps per frame!).
- Cognitive progression phases: from "burro" (random actions) to ball seeking,
  spacing, pass completion, and defensive covering.
- "Ficar Burro (Reset)" button to reset policy weights back to zero on the fly.
- Expanded official map catalog: 2v2 Futsal, 3v3 Futsal (7899), 5v5 Futsal (9362),
  Micro 1v1, Big Stadium, Small Classic, Classic, and Dodgeball.
- Full hybrid controls: WASD or Arrow Keys + Space/X/C/Shift.
"""

from __future__ import annotations
import os
import sys
import math
import threading
import pygame
from typing import Dict, Tuple, Optional, Any, List

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, FPS
from haxball.core.disc import hex_to_rgb
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots import HeuristicBot, WallReboundBot, GoalieBot, BaseBot
from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.rl.algorithms.ppo.ppo_trainer import PPOTrainer
from haxball.rl.algorithms.standard_rl.dqn_trainer import DQNTrainer
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer

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
    def __init__(self, width: int = 1280, height: int = 768):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("HaxBall - Official Futsal Suite & Recursive Self-Play Training")

        self.clock = pygame.time.Clock()
        self.running = True
        self.is_paused = False

        # Modes: "self_play" or "human"
        self.play_mode = "self_play"
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

        # Modals
        self.show_stadium_modal = False
        self.show_rl_modal = False
        self.show_help_modal = False

        # Current Stadium & Match
        self.current_stadium_key = "futsal_2v2"
        self.team_format = 2
        self.self_play_trainer: Optional[SelfPlay2v2Trainer] = None
        self._init_game(self.current_stadium_key, self.team_format)

        # Bots
        self.bot_catalog = {
            "wall": WallReboundBot("WallReboundBot"),
            "heuristic": HeuristicBot("HeuristicBot"),
            "goalie": GoalieBot("GoalieBot")
        }
        self.active_bot_key = "wall"

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
            # Preserve learned weights across stadium or format changes!
            try:
                self.self_play_trainer.policy.load_state_dict(old_policy.state_dict())
            except Exception:
                pass

        self._calc_camera()

    def _calc_camera(self):
        # Calculate pitch scale and center
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

    def get_player_inputs(self) -> Dict[int, Tuple[float, float, bool]]:
        keys = pygame.key.get_pressed()
        mx = 0.0
        my = 0.0

        # Move: WASD or Arrow keys!
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            mx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            mx += 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            my += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            my -= 1.0

        # Kick: Space, X, C, Shift
        kick = (
            keys[pygame.K_SPACE] or
            keys[pygame.K_x] or
            keys[pygame.K_c] or
            keys[pygame.K_LSHIFT] or
            keys[pygame.K_RSHIFT]
        )

        inputs: Dict[int, Tuple[float, float, bool]] = {}

        # Red Team
        red_players = [p for p in self.game.players if p.team == Team.RED]
        for i, p in enumerate(red_players):
            if i == 0:
                inputs[p.player_id] = (mx, my, kick)
            else:
                bot = self.bot_catalog["heuristic"]
                inputs[p.player_id] = bot.act(self.game, p)

        # Blue Team
        blue_players = [p for p in self.game.players if p.team == Team.BLUE]
        for i, p in enumerate(blue_players):
            bot = self.bot_catalog[self.active_bot_key]
            inputs[p.player_id] = bot.act(self.game, p)

        return inputs

    def update(self):
        if self.is_paused or self.show_stadium_modal or self.show_rl_modal or self.show_help_modal:
            return

        if self.play_mode == "self_play":
            # Live Multi-Agent Recursive Self-Play Training
            if self.self_play_trainer:
                self.self_play_trainer.step_multistep(self.speed_multiplier)
        else:
            # Human Play mode with speed multiplier
            for _ in range(self.speed_multiplier):
                inputs = self.get_player_inputs()
                self.game.step(inputs)

    def draw(self):
        # Authentic HaxBall dark theme background
        self.screen.fill((28, 36, 43))

        # 1. Pitch & Field markings
        self._draw_pitch()

        # 2. Authentic Top Scoreboard
        self._draw_scoreboard()

        # 3. HUD Overlays (Stadium pill, Mode, Speed)
        self._draw_hud_overlay()

        # 4. Live Training Telemetry Banner (in self-play mode)
        if self.play_mode == "self_play" and self.self_play_trainer:
            self._draw_self_play_banner()

        # 5. Bottom Dock
        self._draw_bottom_dock()

        # 6. Modals (if active)
        if self.show_stadium_modal:
            self._draw_stadium_modal()
        elif self.show_rl_modal:
            self._draw_rl_modal()
        elif self.show_help_modal:
            self._draw_help_modal()

        pygame.display.flip()

    def _draw_pitch(self):
        stad = self.game.stadium

        # Pitch floor
        bg_rgb = hex_to_rgb(stad.bg_color)
        bg_w = self.world_len_to_screen(stad.bg_width * 2.0)
        bg_h = self.world_len_to_screen(stad.bg_height * 2.0)

        pitch_rect = pygame.Rect(
            int(self.center_x - bg_w / 2.0),
            int(self.center_y - bg_h / 2.0),
            bg_w,
            bg_h
        )
        pygame.draw.rect(self.screen, bg_rgb, pitch_rect)
        pygame.draw.rect(self.screen, (245, 245, 245), pitch_rect, width=2)

        # Center line and kickoff circle
        c_top = (int(self.center_x), int(self.center_y - bg_h / 2.0))
        c_bottom = (int(self.center_x), int(self.center_y + bg_h / 2.0))
        pygame.draw.line(self.screen, (255, 255, 255, 200), c_top, c_bottom, width=2)

        ko_rad = self.world_len_to_screen(stad.bg_kickoff_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.center_x), int(self.center_y)), ko_rad, width=2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.center_x), int(self.center_y)), 4)

        # Stadium Segments (Walls & lines)
        for seg in self.game.physics.segments:
            if not seg.vis:
                continue

            color = seg.color_rgb
            if seg.is_curved:
                self._draw_curved_segment(seg, color)
            else:
                p0_s = self.world_to_screen(seg.p0)
                p1_s = self.world_to_screen(seg.p1)
                pygame.draw.line(self.screen, color, p0_s, p1_s, width=2)

        # Goal posts (Static discs)
        for d in self.game.physics.discs:
            if d.is_static:
                c_s = self.world_to_screen(d.pos)
                r_s = self.world_len_to_screen(d.radius)
                pygame.draw.circle(self.screen, d.color_rgb, c_s, r_s)
                pygame.draw.circle(self.screen, (30, 30, 30), c_s, r_s, width=2)

        # Ball
        if self.game.ball:
            b_s = self.world_to_screen(self.game.ball.pos)
            b_r = self.world_len_to_screen(self.game.ball.radius)
            pygame.draw.circle(self.screen, (15, 20, 25), (b_s[0] + 1, b_s[1] + 1), b_r)
            pygame.draw.circle(self.screen, self.game.ball.color_rgb, b_s, b_r)
            pygame.draw.circle(self.screen, (25, 25, 25), b_s, b_r, width=2)
            pygame.draw.circle(self.screen, (60, 60, 60), b_s, max(1, b_r // 3))

        # Players
        for p in self.game.players:
            p_s = self.world_to_screen(p.pos)
            p_r = self.world_len_to_screen(p.radius)

            # Kick flash outer glow
            if p.kick_flash > 0:
                pygame.draw.circle(self.screen, (255, 255, 255), p_s, p_r + 4, width=3)

            # Body circle
            pygame.draw.circle(self.screen, p.color_rgb, p_s, p_r)
            pygame.draw.circle(self.screen, (25, 25, 25), p_s, p_r, width=2)
            inner_r = max(1, p_r - 4)
            pygame.draw.circle(self.screen, (255, 255, 255), p_s, inner_r, width=1)

            # Player number
            num_surf = self.font_player.render(str(p.player_number), True, (255, 255, 255))
            self.screen.blit(num_surf, num_surf.get_rect(center=p_s))

        # Goal celebration overlay
        if self.game.state == GameState.GOAL_CELEBRATION:
            team_str = "RED" if self.game.last_goal_team == Team.RED else "BLUE"
            color = (229, 110, 86) if self.game.last_goal_team == Team.RED else (86, 137, 229)
            txt = self.font_title.render(f"GOL! {team_str} MARCOU!", True, color)
            box = txt.get_rect(center=(self.center_x, self.center_y - 45)).inflate(50, 20)
            pygame.draw.rect(self.screen, (20, 24, 30), box, border_radius=8)
            pygame.draw.rect(self.screen, color, box, width=2, border_radius=8)
            self.screen.blit(txt, txt.get_rect(center=box.center))

    def _draw_curved_segment(self, seg, color):
        steps = 14
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

    def _draw_scoreboard(self):
        # Authentic HaxBall Top Scoreboard
        sb_w = 260
        sb_h = 36
        sb_x = int(self.width / 2.0 - sb_w / 2.0)
        sb_y = 12

        # Red score box
        red_box = pygame.Rect(sb_x, sb_y, 75, sb_h)
        pygame.draw.rect(self.screen, (229, 110, 86), red_box, border_top_left_radius=6, border_bottom_left_radius=6)
        r_txt = self.font_score.render(str(self.game.red_score), True, (255, 255, 255))
        self.screen.blit(r_txt, r_txt.get_rect(center=red_box.center))

        # Time box
        time_box = pygame.Rect(sb_x + 75, sb_y, 110, sb_h)
        pygame.draw.rect(self.screen, (34, 43, 53), time_box)
        t_txt = self.font_time.render(self.game.time_string, True, (240, 240, 240))
        self.screen.blit(t_txt, t_txt.get_rect(center=time_box.center))

        # Blue score box
        blue_box = pygame.Rect(sb_x + 185, sb_y, 75, sb_h)
        pygame.draw.rect(self.screen, (86, 137, 229), blue_box, border_top_right_radius=6, border_bottom_right_radius=6)
        b_txt = self.font_score.render(str(self.game.blue_score), True, (255, 255, 255))
        self.screen.blit(b_txt, b_txt.get_rect(center=blue_box.center))

    def _draw_hud_overlay(self):
        # Stadium pill on top-left
        stad_title = STADIUM_CATALOG[self.current_stadium_key]["title"]
        self.pill_stad_r = pygame.Rect(20, 14, 250, 34)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_stad_r, border_radius=17)
        pygame.draw.rect(self.screen, (55, 68, 85), self.pill_stad_r, width=1, border_radius=17)
        s_txt = self.font_hud.render(f"⚽ {stad_title[:24]}", True, (220, 230, 240))
        self.screen.blit(s_txt, s_txt.get_rect(center=self.pill_stad_r.center))

        # Speed Multiplier Pill (Clickable)
        self.pill_speed_r = pygame.Rect(self.width - 290, 14, 130, 34)
        sp_c = (230, 140, 40) if self.speed_multiplier > 1 else (55, 68, 85)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_speed_r, border_radius=17)
        pygame.draw.rect(self.screen, sp_c, self.pill_speed_r, width=2 if self.speed_multiplier > 1 else 1, border_radius=17)
        sp_txt = self.font_hud.render(f"⚡ Vel: {self.speed_multiplier}x", True, (255, 180, 50) if self.speed_multiplier > 1 else (200, 210, 225))
        self.screen.blit(sp_txt, sp_txt.get_rect(center=self.pill_speed_r.center))

        # Mode Pill (Clickable)
        self.pill_mode_r = pygame.Rect(self.width - 150, 14, 130, 34)
        is_self_play = (self.play_mode == "self_play")
        m_c = (60, 179, 113) if is_self_play else (58, 142, 230)
        pygame.draw.rect(self.screen, (36, 46, 56), self.pill_mode_r, border_radius=17)
        pygame.draw.rect(self.screen, m_c, self.pill_mode_r, width=2, border_radius=17)
        m_label = "🤖 Self-Play" if is_self_play else "👤 Humano"
        m_txt = self.font_hud.render(m_label, True, m_c)
        self.screen.blit(m_txt, m_txt.get_rect(center=self.pill_mode_r.center))

    def _draw_self_play_banner(self):
        stats = self.self_play_trainer.last_stats
        steps = stats.get("total_steps", 0)
        iters = stats.get("iteration", 0)
        rew = stats.get("mean_reward", 0.0)
        passes = stats.get("passes", 0)

        # Cognitive phase identification
        if steps < 5000:
            phase_name = "FASE 1: Exploração Burra (Aleatório)"
            phase_color = (220, 80, 80)
        elif steps < 25000:
            phase_name = "FASE 2: Perseguição de Bola"
            phase_color = (235, 150, 40)
        elif steps < 70000:
            phase_name = "FASE 3: Alinhamento ao Gol & Espaçamento"
            phase_color = (220, 200, 50)
        else:
            phase_name = "FASE 4: Team-Play & Passes Coordenados"
            phase_color = (60, 210, 120)

        banner_w = 900
        banner_h = 32
        banner_x = int(self.width / 2.0 - banner_w / 2.0)
        banner_y = 54

        b_rect = pygame.Rect(banner_x, banner_y, banner_w, banner_h)
        pygame.draw.rect(self.screen, (22, 28, 36), b_rect, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 74), b_rect, width=1, border_radius=8)

        # Phase badge
        ph_rect = pygame.Rect(banner_x + 8, banner_y + 4, 270, 24)
        pygame.draw.rect(self.screen, (34, 44, 56), ph_rect, border_radius=5)
        ph_txt = self.font_telemetry.render(phase_name, True, phase_color)
        self.screen.blit(ph_txt, ph_txt.get_rect(center=ph_rect.center))

        # Metrics text
        met_str = f"Passos: {steps:,}  |  Iter: {iters}  |  Reward: {rew:+.2f}  |  Passes: {passes}"
        met_surf = self.font_telemetry.render(met_str, True, (210, 220, 235))
        self.screen.blit(met_surf, (banner_x + 290, banner_y + 8))

        # "Ficar Burro" Reset button
        self.btn_dumb_r = pygame.Rect(banner_x + banner_w - 145, banner_y + 4, 135, 24)
        pygame.draw.rect(self.screen, (60, 30, 35), self.btn_dumb_r, border_radius=5)
        pygame.draw.rect(self.screen, (200, 70, 70), self.btn_dumb_r, width=1, border_radius=5)
        d_txt = self.font_telemetry.render("🔄 Ficar Burro (Reset)", True, (240, 130, 130))
        self.screen.blit(d_txt, d_txt.get_rect(center=self.btn_dumb_r.center))

    def _draw_bottom_dock(self):
        dock_h = 58
        dock_y = self.height - dock_h
        pygame.draw.rect(self.screen, (22, 28, 35), (0, dock_y, self.width, dock_h))
        pygame.draw.line(self.screen, (45, 56, 70), (0, dock_y), (self.width, dock_y), width=1)

        # 1. Play / Pause
        self.btn_pause_r = pygame.Rect(15, dock_y + 10, 110, 38)
        p_txt = "▶ Play" if self.is_paused else "⏸ Pausar"
        self._draw_btn(self.btn_pause_r, p_txt, (58, 142, 230))

        # 2. Reset match
        self.btn_reset_r = pygame.Rect(132, dock_y + 10, 95, 38)
        self._draw_btn(self.btn_reset_r, "🔄 Reset", (45, 55, 68))

        # 3. Choose Stadium
        self.btn_stadium_r = pygame.Rect(234, dock_y + 10, 135, 38)
        self._draw_btn(self.btn_stadium_r, "🗺 Estádio", (45, 55, 68))

        # 4. Format 1v1 / 2v2 / 3v3 / 5v5
        self.btn_format_r = pygame.Rect(376, dock_y + 10, 130, 38)
        self._draw_btn(self.btn_format_r, f"👥 {self.team_format}v{self.team_format}", (45, 55, 68))

        # 5. Speed Multiplier (Cycle 1x -> 100x)
        self.btn_dock_speed_r = pygame.Rect(513, dock_y + 10, 130, 38)
        sp_c = (210, 120, 30) if self.speed_multiplier > 1 else (45, 55, 68)
        self._draw_btn(self.btn_dock_speed_r, f"⚡ {self.speed_multiplier}x Acelerar", sp_c)

        # 6. Mode Toggle (Self-Play vs Human)
        self.btn_dock_mode_r = pygame.Rect(650, dock_y + 10, 150, 38)
        is_sp = (self.play_mode == "self_play")
        m_txt = "🤖 Self-Play IA" if is_sp else "👤 Jogo Humano"
        m_bg = (50, 140, 90) if is_sp else (58, 142, 230)
        self._draw_btn(self.btn_dock_mode_r, m_txt, m_bg)

        # 7. RL Training Center Modal
        self.btn_rl_r = pygame.Rect(807, dock_y + 10, 150, 38)
        rl_c = (60, 179, 113) if not self.training_active else (220, 70, 70)
        rl_t = "🧠 Central RL" if not self.training_active else "⚡ Treinando RL..."
        self._draw_btn(self.btn_rl_r, rl_t, rl_c)

        # 8. Help / Controls
        self.btn_help_r = pygame.Rect(self.width - 120, dock_y + 10, 105, 38)
        self._draw_btn(self.btn_help_r, "❓ Teclas", (45, 55, 68))

    def _draw_btn(self, rect: pygame.Rect, text: str, bg_color: Tuple[int, int, int]):
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)
        c = (min(255, bg_color[0] + 25), min(255, bg_color[1] + 25), min(255, bg_color[2] + 25)) if hover else bg_color
        pygame.draw.rect(self.screen, c, rect, border_radius=6)
        pygame.draw.rect(self.screen, (55, 68, 85), rect, width=1, border_radius=6)
        txt = self.font_bold.render(text, True, (240, 245, 250))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_stadium_modal(self):
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 760, 520
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (34, 43, 53), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (58, 142, 230), m_r, width=2, border_radius=12)

        t = self.font_title.render("Selecionar Estádio (.hbs) - Catálogo Expandido", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 20))

        # 2 Columns of 4 Cards
        keys = list(STADIUM_CATALOG.keys())
        self.stadium_cards = {}
        card_w = 335
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
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 720, 500
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (34, 43, 53), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (60, 179, 113), m_r, width=2, border_radius=12)

        t = self.font_title.render("Central de Treinamento de RL (HaxBall)", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 24))

        sub = self.font_regular.render(f"Status do Treinador: {self.training_status}", True, (60, 179, 113) if self.training_active else (170, 180, 195))
        self.screen.blit(sub, (m_r.x + 30, m_r.y + 60))

        # Toggle Algo
        self.btn_rl_algo_r = pygame.Rect(m_r.x + 30, m_r.y + 95, 170, 36)
        self._draw_btn(self.btn_rl_algo_r, f"Algoritmo: {self.training_algo}", (58, 142, 230))

        # Toggle Policy
        self.btn_rl_pol_r = pygame.Rect(m_r.x + 215, m_r.y + 95, 220, 36)
        pol_txt = "Rede: MLP Padrão" if self.training_policy == "mlp" else "Rede: Transformer Attention"
        self._draw_btn(self.btn_rl_pol_r, pol_txt, (45, 58, 74))

        # Metrics box
        met_r = pygame.Rect(m_r.x + 30, m_r.y + 150, m_w - 60, 180)
        pygame.draw.rect(self.screen, (26, 33, 42), met_r, border_radius=8)
        pygame.draw.rect(self.screen, (55, 68, 85), met_r, width=1, border_radius=8)

        m_head = self.font_bold.render("Métricas de Aprendizado em Segundo Plano:", True, (235, 240, 245))
        self.screen.blit(m_head, (met_r.x + 18, met_r.y + 16))

        m1 = self.font_regular.render(f"• Passos de Interação: {self.training_metrics['step']:,}", True, (220, 225, 235))
        m2 = self.font_regular.render(f"• Recompensa Média: {self.training_metrics['reward']:+.2f}", True, (220, 225, 235))
        m3 = self.font_regular.render(f"• Taxa de Vitória (WinRate): {self.training_metrics['win_rate']:.1f}%", True, (220, 225, 235))
        m4 = self.font_regular.render(f"• Erro da Função de Perda (Loss): {self.training_metrics['loss']:.4f}", True, (220, 225, 235))

        self.screen.blit(m1, (met_r.x + 18, met_r.y + 55))
        self.screen.blit(m2, (met_r.x + 18, met_r.y + 90))
        self.screen.blit(m3, (met_r.x + 330, met_r.y + 55))
        self.screen.blit(m4, (met_r.x + 330, met_r.y + 90))

        # Big Action Button
        self.btn_rl_action_r = pygame.Rect(m_r.x + 30, m_r.y + 355, 220, 44)
        act_bg = (220, 70, 70) if self.training_active else (60, 179, 113)
        act_txt = "⏹ Parar Treinamento" if self.training_active else "🚀 Iniciar Treinamento"
        self._draw_btn(self.btn_rl_action_r, act_txt, act_bg)

        self.btn_close_rl_r = pygame.Rect(m_r.right - 140, m_r.bottom - 46, 110, 34)
        self._draw_btn(self.btn_close_rl_r, "Fechar", (45, 55, 68))

    def _draw_help_modal(self):
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))

        m_w, m_h = 680, 430
        m_r = pygame.Rect(int((self.width - m_w) / 2.0), int((self.height - m_h) / 2.0), m_w, m_h)
        pygame.draw.rect(self.screen, (34, 43, 53), m_r, border_radius=12)
        pygame.draw.rect(self.screen, (58, 142, 230), m_r, width=2, border_radius=12)

        t = self.font_title.render("Guia de Controles, Treino e Team-Play", True, (245, 245, 245))
        self.screen.blit(t, (m_r.x + 30, m_r.y + 24))

        lines = [
            "• Controles: TECLAS WASD OU SETAS DO TECLADO funcionam simultaneamente!",
            "• Chute: BARRA DE ESPAÇO, TECLA X, TECLA C ou SHIFT.",
            "• Treino ao Vivo (2x2 Self-Play): Veja a IA jogando e aprendendo recursivamente na tela!",
            "• Aceleração até 100x: Use o botão de velocidade (1x a 100x) para simular 6.000 passos/segundo.",
            "• 'Ficar Burro (Reset)': Clique para zerar os pesos e ver os modelos aprendendo do zero.",
            "• Como surge o Team-Play: O Reward Engine pune aglomeração mútua (spacing), bonifica passes",
            "  completos (+3.0) e assistências de gol (+4.0), forçando divisão de papéis (ataque e âncora)."
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

                # Modal handling
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

                if self.show_rl_modal:
                    if self.btn_close_rl_r.collidepoint(pos):
                        self.show_rl_modal = False
                    elif self.btn_rl_algo_r.collidepoint(pos):
                        self.training_algo = "DQN" if self.training_algo == "PPO" else "PPO"
                    elif self.btn_rl_pol_r.collidepoint(pos):
                        self.training_policy = "attention" if self.training_policy == "mlp" else "mlp"
                    elif self.btn_rl_action_r.collidepoint(pos):
                        self._toggle_training()
                    continue

                if self.show_help_modal:
                    if self.btn_close_help_r.collidepoint(pos):
                        self.show_help_modal = False
                    continue

                # Top HUD Pills clicks
                if self.pill_stad_r.collidepoint(pos):
                    self.show_stadium_modal = True
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
                    continue

                # Bottom Dock clicks
                if self.btn_pause_r.collidepoint(pos):
                    self.is_paused = not self.is_paused
                elif self.btn_reset_r.collidepoint(pos):
                    self.game.reset_match()
                elif self.btn_stadium_r.collidepoint(pos):
                    self.show_stadium_modal = True
                elif self.btn_format_r.collidepoint(pos):
                    # Cycle format: 1 -> 2 -> 3 -> 5 -> 1
                    nxt = 2 if self.team_format == 1 else (3 if self.team_format == 2 else (5 if self.team_format == 3 else 1))
                    self._init_game(self.current_stadium_key, nxt)
                elif self.btn_dock_speed_r.collidepoint(pos):
                    self.cycle_speed()
                elif self.btn_dock_mode_r.collidepoint(pos):
                    self.toggle_play_mode()
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
