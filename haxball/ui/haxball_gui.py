"""
HaxBall Futsal Studio - Modern Interactive UI Suite.
Features:
- Pure Futsal Competitive Engine (Futsal 3v3 GLH, Futsal 2v2, Futsal 1v1).
- Authentic Matte Gray Futsal Court (#3C3F43) with crisp white markings and diamond mesh goal nets.
- Granular Per-Player Customization: Click any player slot (Red 1-3, Blue 1-3) to toggle:
  [HUMANO | IA BC (Calibrada) | IA RL (PPO) | BOT FIXO | BOT ALA | BOT PRESS | BOT TABELAS | IDLE].
- Calibrated Behavioral Cloning inference (active goal shooting & clearing).
- Real-time High-Speed Simulation multiplier (0.5x to 10x).
- Procedural Audio Engine (kicks, post bounces, referee whistles, goal celebration horns).
- Live Telemetry: Real-time Possession bar, Shots on goal, and on-field role badges.
"""

from __future__ import annotations
import os
import sys
import time
import math
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
from haxball.renderer.sound_effects import SoundManager

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
        "desc": "Quadra Cinza Equilibrada (450x200). Espaçamento tático e transições dinâmicas.",
        "default_format": 2
    },
    "futsal_1v1": {
        "title": "Futsal 1v1 Rápido",
        "file": str(MAP_DIR / "futsal.hbs"),
        "desc": "Duelo individual (380x180). Ataque e defesa imediatos.",
        "default_format": 1
    }
}

CONTROLLER_TYPES = [
    ("human", "HUMANO", (255, 235, 50)),
    ("bc_ai", "IA BC", (255, 120, 120)),
    ("ppo_rl", "IA RL", (120, 225, 150)),
    ("bot_fixo", "BOT FIXO", (100, 185, 255)),
    ("bot_ala", "BOT ALA", (160, 215, 255)),
    ("bot_press", "BOT PRESS", (255, 145, 80)),
    ("bot_wall", "BOT TABELA", (195, 145, 255)),
    ("idle", "IDLE", (130, 135, 145)),
]

SPEED_OPTIONS = [0.5, 1.0, 2.0, 5.0, 10.0]


class HaxBallStudioApp:
    def __init__(self, width: int = 1280, height: int = 768):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("HaxBall Futsal Studio - Arena Interativa & Benchmarks")

        self.clock = pygame.time.Clock()
        self.running = True
        self.is_paused = False
        self.speed_idx = 1  # 1.0x by default
        self.sound_enabled = True

        # Fonts
        self.font_logo = pygame.font.SysFont("Trebuchet MS", 18, bold=True)
        self.font_score_badge = pygame.font.SysFont("Trebuchet MS", 26, bold=True)
        self.font_timer = pygame.font.SysFont("Lucida Console", 22, bold=True)
        self.font_banner = pygame.font.SysFont("Trebuchet MS", 32, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_slot = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_player = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_tag = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_telemetry = pygame.font.SysFont("Arial", 12, bold=True)

        self.sound = SoundManager.get_instance()
        self.kick_ripples: List[Dict[str, Any]] = []
        self.last_state: Optional[GameState] = None

        # Game Format & Stadium
        self.current_stadium_key = "futsal_3v3"
        self.players_per_team = 3  # 1v1, 2v2, 3v3

        # Per-Slot Controller Configuration
        # Default: Red 1 is Human, Red 2 & 3 are Calibrated BC AI
        # Blue 1, 2, 3 are Coordinated Futsal Bots (Fixo, Press, Ala)
        self.slot_controllers = {
            Team.RED: ["human", "bc_ai", "bc_ai"],
            Team.BLUE: ["bot_fixo", "bot_press", "bot_ala"]
        }

        # Observation and Action Handlers
        self.obs_builder = DecoupledObservationBuilder()
        self.action_handler = ActionHandler()

        # Cached AI Models & Bots
        self._init_models_and_bots()

        # Telemetry Stats
        self.red_possession_ticks = 0
        self.blue_possession_ticks = 0
        self.red_shots = 0
        self.blue_shots = 0

        # UI Clickable Hitboxes
        self.ui_buttons: List[Dict[str, Any]] = []

        # Initialize Game World
        self._init_game()

    def _init_models_and_bots(self):
        # 1. Behavioral Cloning Model (Entity Attention)
        self.bc_model = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True)
        bc_path = CHECKPOINT_DIR / "bc_futsal_3v3.pt"
        if bc_path.exists():
            try:
                self.bc_model.load_state_dict(torch.load(str(bc_path), map_location="cpu"))
                print(f"[Studio] Modelo BC carregado com sucesso: {bc_path}")
            except Exception as e:
                print(f"[Studio] Aviso ao carregar BC: {e}")
        self.bc_model.eval()

        # 2. PPO Reinforcement Learning Model (MLP Discrete 18-action)
        self.rl_model = ActorCriticMLP(obs_dim=61, act_dim=18, is_discrete=True)
        rl_path = PROJECT_ROOT / "checkpoints" / "haxball_rl_best.pt"
        if rl_path.exists():
            try:
                self.rl_model.load_state_dict(torch.load(str(rl_path), map_location="cpu"))
                print(f"[Studio] Modelo RL carregado com sucesso: {rl_path}")
            except Exception as e:
                print(f"[Studio] Aviso ao carregar RL: {e}")
        self.rl_model.eval()

        # 3. Analytical Coordinated Bots
        self.red_coord = Futsal3v3Coordinator(Team.RED)
        self.blue_coord = Futsal3v3Coordinator(Team.BLUE)
        self.red_futsal_bots = [Futsal3v3Bot(f"Red_{i}", self.red_coord) for i in range(3)]
        self.blue_futsal_bots = [Futsal3v3Bot(f"Blue_{i}", self.blue_coord) for i in range(3)]

        # 4. Specialist Standalone Bots
        self.heuristic_bot = HeuristicBot(name="Heuristic")
        self.wall_rebound_bot = WallReboundBot(name="WallRebound")

    def _init_game(self):
        stadium_info = STADIUM_CATALOG[self.current_stadium_key]
        stadium = Stadium.load_from_file(stadium_info["file"])
        self.game = HaxBallGame(
            stadium=stadium,
            score_limit=5,
            time_limit_secs=180,
            red_players_count=self.players_per_team,
            blue_players_count=self.players_per_team
        )
        self.red_possession_ticks = 0
        self.blue_possession_ticks = 0
        self.red_shots = 0
        self.blue_shots = 0
        self._calc_camera()
        if self.sound_enabled:
            self.sound.play_whistle()

    def _calc_camera(self):
        header_h = 56.0
        footer_h = 92.0
        margin_x = 60.0
        margin_y = 35.0

        avail_w = self.width - margin_x * 2.0
        avail_h = self.height - header_h - footer_h - margin_y * 2.0

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

    def cycle_slot_controller(self, team: Team, slot_idx: int):
        current = self.slot_controllers[team][slot_idx]
        all_keys = [t[0] for t in CONTROLLER_TYPES]
        curr_idx = all_keys.index(current) if current in all_keys else 0
        next_key = all_keys[(curr_idx + 1) % len(all_keys)]
        self.slot_controllers[team][slot_idx] = next_key

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

        # 1. Capture Human Inputs (WASD / Arrows)
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

        # 2. Assign action for each player based on its individual slot controller
        for p_idx, p in enumerate(red_players):
            ctrl = self.slot_controllers[Team.RED][p_idx] if p_idx < len(self.slot_controllers[Team.RED]) else "idle"
            inputs_dict[p.player_id] = self._get_action_for_player(p, ctrl, mx, my, kick, Team.RED, p_idx)

        for p_idx, p in enumerate(blue_players):
            ctrl = self.slot_controllers[Team.BLUE][p_idx] if p_idx < len(self.slot_controllers[Team.BLUE]) else "idle"
            inputs_dict[p.player_id] = self._get_action_for_player(p, ctrl, mx, my, kick, Team.BLUE, p_idx)

        # 3. Advance Physics
        step_info = self.game.step(inputs_dict)

        # 4. Telemetry Tracking
        events = step_info.get("events", {})
        kicks = events.get("kicks", [])
        bounces = events.get("bounces", [])

        if kicks:
            if self.sound_enabled:
                self.sound.play_kick()
            for k in kicks:
                kx, ky = self.world_to_screen(Vec2.from_iterable(k["pos"]))
                self.kick_ripples.append({"x": kx, "y": ky, "radius": 14, "alpha": 255})
                if k["team"] == Team.RED:
                    self.red_shots += 1
                else:
                    self.blue_shots += 1

        if bounces and self.sound_enabled:
            self.sound.play_bounce()

        if step_info.get("goal_scored", False) and self.sound_enabled:
            self.sound.play_goal()

        curr_state = step_info.get("state")
        if curr_state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE) and self.last_state == GameState.GOAL_CELEBRATION:
            if self.sound_enabled:
                self.sound.play_whistle()
        self.last_state = curr_state

        # Possession
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
            # Calibrated Behavioral Cloning Inference (prevents argmax kick collapse)
            with torch.no_grad():
                obs = self.obs_builder.build_observation(self.game, player)
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                logits = self.bc_model.actor(self.bc_model.forward_repr(obs_t))[0]
                probs = torch.softmax(logits, dim=-1)

                kick_prob = probs[9:].sum().item()
                # Calibrated threshold: if kick intent is over 22%, trigger kick direction!
                if kick_prob > 0.22:
                    act_idx = 9 + torch.argmax(probs[9:]).item()
                else:
                    act_idx = torch.argmax(probs[:9]).item()

                return self.action_handler.decode_discrete(act_idx)

        elif ctrl == "ppo_rl":
            with torch.no_grad():
                obs = self.obs_builder.build_observation(self.game, player)
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                logits = self.rl_model.actor(obs_t)
                act_idx = torch.argmax(logits, dim=-1).item()
                return self.action_handler.decode_discrete(act_idx)

        elif ctrl in ("bot_fixo", "bot_ala", "bot_press"):
            coord = self.red_coord if team == Team.RED else self.blue_coord
            bots = self.red_futsal_bots if team == Team.RED else self.blue_futsal_bots
            bot_obj = bots[slot_idx] if slot_idx < len(bots) else bots[0]
            return bot_obj.act(self.game, player)

        elif ctrl == "bot_wall":
            return self.wall_rebound_bot.act(self.game, player)

        # "idle"
        return (0.0, 0.0, False)

    def render(self):
        self.ui_buttons.clear()

        # 1. Dark Outer Arena Background
        self.screen.fill((20, 24, 32))

        stad = self.game.stadium

        # 2. Authentic Gray Futsal Pitch Surface
        bg_w = self.world_len_to_screen(stad.bg_width * 2.0)
        bg_h = self.world_len_to_screen(stad.bg_height * 2.0)
        pitch_rect = pygame.Rect(
            int(self.center_x - bg_w / 2.0),
            int(self.center_y - bg_h / 2.0),
            bg_w,
            bg_h
        )

        court_gray = hex_to_rgb(stad.bg_color) if hasattr(stad, "bg_color") and stad.bg_color else (60, 63, 67)
        outer_court = (max(0, court_gray[0] - 16), max(0, court_gray[1] - 16), max(0, court_gray[2] - 16))

        # Perimeter buffer court
        court_buffer_rect = pitch_rect.inflate(self.world_len_to_screen(35), self.world_len_to_screen(35))
        pygame.draw.rect(self.screen, outer_court, court_buffer_rect, border_radius=6)
        pygame.draw.rect(self.screen, court_gray, pitch_rect, border_radius=4)

        line_color = (250, 250, 250)
        pygame.draw.rect(self.screen, line_color, pitch_rect, width=2, border_radius=4)

        # 3. Goal Netting (Cross-hatch diamond mesh)
        self._draw_goal_nets(pitch_rect)

        # 4. Field Markings
        c_top = (int(self.center_x), pitch_rect.top)
        c_bottom = (int(self.center_x), pitch_rect.bottom)
        pygame.draw.line(self.screen, line_color, c_top, c_bottom, width=2)

        ko_rad = self.world_len_to_screen(stad.bg_kickoff_radius)
        center_pt = (int(self.center_x), int(self.center_y))
        pygame.draw.circle(self.screen, line_color, center_pt, ko_rad, width=2)
        pygame.draw.circle(self.screen, line_color, center_pt, 4)

        # Goal areas (Futsal penalty arcs)
        area_rad = self.world_len_to_screen(75.0)
        pygame.draw.circle(self.screen, line_color, (pitch_rect.left, int(self.center_y)), area_rad, width=2)
        pygame.draw.circle(self.screen, line_color, (pitch_rect.right, int(self.center_y)), area_rad, width=2)

        # 5. Expanding Kick Ripples
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

        # 6. Segments
        for seg in self.game.physics.segments:
            if not seg.vis or seg.trait == "goalNet":
                continue
            color = seg.color_rgb
            if seg.is_curved:
                self._draw_curved_segment(seg, color)
            else:
                p0_s = self.world_to_screen(seg.p0)
                p1_s = self.world_to_screen(seg.p1)
                pygame.draw.line(self.screen, color, p0_s, p1_s, width=3)

        # 7. Static Discs (Posts)
        for d in self.game.physics.discs:
            if d.is_static:
                self._draw_disc(d)

        # 8. Dynamic Discs (Ball & Players)
        if self.game.ball:
            self._draw_disc(self.game.ball)

        for p in self.game.players:
            self._draw_disc(p)
            self._draw_player_badge(p)

        # 9. Scoreboard Overlay
        self._draw_scoreboard()

        # 10. Top Header Navigation Bar
        self._draw_top_bar()

        # 11. Bottom Team Customization Matrix
        self._draw_bottom_customizer()

        pygame.display.flip()

    def _draw_goal_nets(self, pitch_rect: pygame.Rect):
        gw = self.world_len_to_screen(40.0)
        gh = self.world_len_to_screen(160.0)

        # Left Net
        lx = pitch_rect.left - gw
        ly = int(self.center_y - gh / 2.0)
        l_rect = pygame.Rect(lx, ly, gw, gh)
        pygame.draw.rect(self.screen, (16, 20, 26), l_rect)
        for x in range(lx, lx + gw + 8, 8):
            pygame.draw.line(self.screen, (55, 65, 80), (x, ly), (x + 10, ly + gh), width=1)
        for y in range(ly, ly + gh + 8, 8):
            pygame.draw.line(self.screen, (55, 65, 80), (lx, y), (lx + gw, y + 6), width=1)
        pygame.draw.rect(self.screen, (220, 220, 220), l_rect, width=2)

        # Right Net
        rx = pitch_rect.right
        ry = int(self.center_y - gh / 2.0)
        r_rect = pygame.Rect(rx, ry, gw, gh)
        pygame.draw.rect(self.screen, (16, 20, 26), r_rect)
        for x in range(rx, rx + gw + 8, 8):
            pygame.draw.line(self.screen, (55, 65, 80), (x, ry), (x - 10, ry + gh), width=1)
        for y in range(ry, ry + gh + 8, 8):
            pygame.draw.line(self.screen, (55, 65, 80), (rx, y), (rx + gw, y - 6), width=1)
        pygame.draw.rect(self.screen, (220, 220, 220), r_rect, width=2)

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
            pygame.draw.lines(self.screen, color, False, pts, width=3)

    def _draw_disc(self, disc: Disc):
        center_s = self.world_to_screen(disc.pos)
        rad_s = self.world_len_to_screen(disc.radius)

        if disc.is_player:
            # Kick flash white ring
            if disc.is_kicking or disc.kick_flash > 0:
                pygame.draw.circle(self.screen, (255, 255, 255), center_s, rad_s + 4, width=3)

            # Player body
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            pygame.draw.circle(self.screen, (20, 22, 28), center_s, rad_s, width=2)
            # Inner white circle
            pygame.draw.circle(self.screen, (255, 255, 255), center_s, max(2, rad_s - 4), width=1)

            # Number
            num_surf = self.font_player.render(str(disc.player_number), True, (255, 255, 255))
            self.screen.blit(num_surf, num_surf.get_rect(center=center_s))

        elif disc.name == "Ball":
            # Shadow
            pygame.draw.circle(self.screen, (15, 20, 28, 110), (center_s[0] + 2, center_s[1] + 2), rad_s)
            # Body (Yellow)
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            pygame.draw.circle(self.screen, (25, 25, 30), center_s, rad_s, width=2)
            # Center core dot
            pygame.draw.circle(self.screen, (60, 60, 60), center_s, max(1, rad_s // 3))
            # Specular highlight
            pygame.draw.circle(self.screen, (255, 255, 255), (center_s[0] - max(1, rad_s // 3), center_s[1] - max(1, rad_s // 3)), max(1, rad_s // 5))

        else:
            # Goal Posts
            pygame.draw.circle(self.screen, (255, 255, 255), center_s, rad_s)
            pygame.draw.circle(self.screen, (40, 45, 55), center_s, rad_s, width=2)
            pygame.draw.circle(self.screen, (180, 180, 180), center_s, max(1, rad_s - 3), width=1)

    def _draw_player_badge(self, p: Disc):
        cx, cy = self.world_to_screen(p.pos)
        rad_s = self.world_len_to_screen(p.radius)

        slot_idx = p.player_number - 1
        ctrl_key = self.slot_controllers[p.team][slot_idx] if slot_idx < len(self.slot_controllers[p.team]) else "idle"

        if ctrl_key == "human":
            # Pulsing yellow chevron
            pts = [(cx, cy - rad_s - 8), (cx - 7, cy - rad_s - 18), (cx + 7, cy - rad_s - 18)]
            pygame.draw.polygon(self.screen, (255, 230, 40), pts)
            lbl = self.font_tag.render("VOCÊ", True, (255, 230, 40))
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - rad_s - 29))
        else:
            # Role tag
            tag_text = next((t[1] for t in CONTROLLER_TYPES if t[0] == ctrl_key), ctrl_key.upper())
            color = next((t[2] for t in CONTROLLER_TYPES if t[0] == ctrl_key), (200, 200, 200))
            lbl = self.font_tag.render(tag_text, True, color)
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - rad_s - 18))

    def _draw_scoreboard(self):
        # Digital Timer in Center
        timer_surf = self.font_timer.render(self.game.time_string, True, (255, 255, 255))
        self.screen.blit(timer_surf, timer_surf.get_rect(center=(int(self.center_x), 27)))

        # Score Badges
        red_box = pygame.Rect(int(self.center_x - 170), 10, 105, 34)
        pygame.draw.rect(self.screen, (229, 110, 86), red_box, border_radius=4)
        r_txt = self.font_score_badge.render(f"RED  {self.game.red_score}", True, (255, 255, 255))
        self.screen.blit(r_txt, r_txt.get_rect(center=red_box.center))

        blue_box = pygame.Rect(int(self.center_x + 65), 10, 105, 34)
        pygame.draw.rect(self.screen, (86, 137, 229), blue_box, border_radius=4)
        b_txt = self.font_score_badge.render(f"{self.game.blue_score}  BLUE", True, (255, 255, 255))
        self.screen.blit(b_txt, b_txt.get_rect(center=blue_box.center))

        # Goal and Match End Notifications
        if self.game.state == GameState.GOAL_CELEBRATION:
            team_str = "RED" if self.game.last_goal_team == Team.RED else "BLUE"
            color = (229, 110, 86) if self.game.last_goal_team == Team.RED else (86, 137, 229)
            banner = self.font_banner.render(f"GOAL! {team_str} SCORED!", True, color)
            b_rect = banner.get_rect(center=(int(self.center_x), int(self.center_y - 75)))
            box = b_rect.inflate(40, 18)
            pygame.draw.rect(self.screen, (12, 16, 22), box, border_radius=8)
            pygame.draw.rect(self.screen, color, box, width=3, border_radius=8)
            self.screen.blit(banner, b_rect)
        elif self.game.state == GameState.GAME_OVER:
            w_str = "RED VENCEU!" if self.game.red_score > self.game.blue_score else "BLUE VENCEU!"
            banner = self.font_banner.render(f"FIM DE JOGO - {w_str}", True, (255, 215, 0))
            b_rect = banner.get_rect(center=(int(self.center_x), int(self.center_y - 75)))
            box = b_rect.inflate(40, 18)
            pygame.draw.rect(self.screen, (12, 16, 22), box, border_radius=8)
            pygame.draw.rect(self.screen, (255, 215, 0), box, width=3, border_radius=8)
            self.screen.blit(banner, b_rect)
        elif self.game.state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE):
            ko_str = "SAÍDA RED" if self.game.state == GameState.KICKOFF_RED else "SAÍDA BLUE"
            ko_color = (229, 110, 86) if self.game.state == GameState.KICKOFF_RED else (86, 137, 229)
            ko_surf = self.font_btn.render(ko_str, True, ko_color)
            self.screen.blit(ko_surf, ko_surf.get_rect(center=(int(self.center_x), 66)))

    def _draw_top_bar(self):
        bar_h = 54
        bar_rect = pygame.Rect(0, 0, self.width, bar_h)
        pygame.draw.rect(self.screen, (15, 18, 25), bar_rect)
        pygame.draw.line(self.screen, (32, 40, 54), (0, bar_h), (self.width, bar_h), width=2)

        # Title & Stadium Mode
        logo = self.font_logo.render("HAXBALL FUTSAL", True, (255, 255, 255))
        self.screen.blit(logo, (20, 16))

        # Stadium Selector Buttons
        cur_x = 205
        for s_key in ["futsal_3v3", "futsal_2v2", "futsal_1v1"]:
            info = STADIUM_CATALOG[s_key]
            is_active = (s_key == self.current_stadium_key)
            label = info["title"].split(" ")[1]  # "3v3", "2v2", "1v1"
            btn_w = 46
            btn_rect = pygame.Rect(cur_x, 12, btn_w, 28)
            bg_col = (45, 95, 175) if is_active else (28, 35, 48)
            pygame.draw.rect(self.screen, bg_col, btn_rect, border_radius=4)
            pygame.draw.rect(self.screen, (70, 90, 120), btn_rect, width=1, border_radius=4)
            txt = self.font_btn.render(label, True, (255, 255, 255) if is_active else (160, 175, 195))
            self.screen.blit(txt, txt.get_rect(center=btn_rect.center))
            self.ui_buttons.append({"rect": btn_rect, "action": "set_stadium", "val": s_key})
            cur_x += btn_w + 6

        # Format Switcher Buttons (1v1, 2v2, 3v3)
        cur_x += 16
        for fmt in [1, 2, 3]:
            is_active = (fmt == self.players_per_team)
            btn_w = 42
            btn_rect = pygame.Rect(cur_x, 12, btn_w, 28)
            bg_col = (35, 135, 80) if is_active else (28, 35, 48)
            pygame.draw.rect(self.screen, bg_col, btn_rect, border_radius=4)
            pygame.draw.rect(self.screen, (70, 90, 120), btn_rect, width=1, border_radius=4)
            txt = self.font_btn.render(f"{fmt}x{fmt}", True, (255, 255, 255) if is_active else (160, 175, 195))
            self.screen.blit(txt, txt.get_rect(center=btn_rect.center))
            self.ui_buttons.append({"rect": btn_rect, "action": "set_format", "val": fmt})
            cur_x += btn_w + 6

        # Right Controls: Speed, Pause, Reset, Sound
        r_x = self.width - 20

        # Sound Button
        r_x -= 65
        snd_rect = pygame.Rect(r_x, 12, 65, 28)
        snd_bg = (35, 110, 80) if self.sound_enabled else (70, 35, 35)
        pygame.draw.rect(self.screen, snd_bg, snd_rect, border_radius=4)
        snd_txt = self.font_btn.render("SOM: ON" if self.sound_enabled else "MUDO", True, (255, 255, 255))
        self.screen.blit(snd_txt, snd_txt.get_rect(center=snd_rect.center))
        self.ui_buttons.append({"rect": snd_rect, "action": "toggle_sound"})

        # Speed Multiplier Button
        r_x -= 65
        spd_rect = pygame.Rect(r_x, 12, 60, 28)
        pygame.draw.rect(self.screen, (36, 45, 62), spd_rect, border_radius=4)
        spd_val = SPEED_OPTIONS[self.speed_idx]
        spd_txt = self.font_btn.render(f"{spd_val}x", True, (255, 215, 60))
        self.screen.blit(spd_txt, spd_txt.get_rect(center=spd_rect.center))
        self.ui_buttons.append({"rect": spd_rect, "action": "cycle_speed"})

        # Reset Round (R)
        r_x -= 70
        rst_rect = pygame.Rect(r_x, 12, 65, 28)
        pygame.draw.rect(self.screen, (40, 48, 65), rst_rect, border_radius=4)
        rst_txt = self.font_btn.render("RESET (R)", True, (210, 220, 235))
        self.screen.blit(rst_txt, rst_txt.get_rect(center=rst_rect.center))
        self.ui_buttons.append({"rect": rst_rect, "action": "reset_round"})

        # Pause / Play
        r_x -= 75
        p_rect = pygame.Rect(r_x, 12, 70, 28)
        p_bg = (180, 50, 50) if self.is_paused else (40, 100, 180)
        pygame.draw.rect(self.screen, p_bg, p_rect, border_radius=4)
        p_txt = self.font_btn.render("RESUMIR" if self.is_paused else "PAUSAR", True, (255, 255, 255))
        self.screen.blit(p_txt, p_txt.get_rect(center=p_rect.center))
        self.ui_buttons.append({"rect": p_rect, "action": "toggle_pause"})

    def _draw_bottom_customizer(self):
        bot_h = 88
        bot_rect = pygame.Rect(0, self.height - bot_h, self.width, bot_h)
        pygame.draw.rect(self.screen, (15, 18, 25), bot_rect)
        pygame.draw.line(self.screen, (32, 40, 54), (0, self.height - bot_h), (self.width, self.height - bot_h), width=2)

        # Team Red Slots (Left Side)
        cur_x = 25
        card_w = 145
        card_h = 64
        card_y = self.height - bot_h + 12

        lbl_red = self.font_btn.render("TIME VERMELHO (Clique p/ Trocar):", True, (229, 110, 86))
        self.screen.blit(lbl_red, (cur_x, card_y - 10))

        for idx in range(self.players_per_team):
            ctrl_key = self.slot_controllers[Team.RED][idx]
            tag_name, tag_color = next(((t[1], t[2]) for t in CONTROLLER_TYPES if t[0] == ctrl_key), ("IDLE", (150, 150, 150)))

            slot_rect = pygame.Rect(cur_x, card_y + 10, card_w, 48)
            pygame.draw.rect(self.screen, (28, 34, 46), slot_rect, border_radius=6)
            border_col = (229, 110, 86) if ctrl_key != "idle" else (60, 70, 85)
            pygame.draw.rect(self.screen, border_col, slot_rect, width=2, border_radius=6)

            p_title = self.font_slot.render(f"JOGADOR #{idx+1} [RED]", True, (200, 210, 225))
            c_title = self.font_slot.render(tag_name, True, tag_color)
            self.screen.blit(p_title, (slot_rect.x + 8, slot_rect.y + 7))
            self.screen.blit(c_title, (slot_rect.x + 8, slot_rect.y + 26))

            self.ui_buttons.append({"rect": slot_rect, "action": "cycle_slot", "team": Team.RED, "idx": idx})
            cur_x += card_w + 10

        # Center Telemetry Bar
        center_w = 260
        cx = int(self.center_x - center_w / 2.0)
        cy = card_y + 12

        tot_poss = max(1, self.red_possession_ticks + self.blue_possession_ticks)
        r_pct = 100.0 * self.red_possession_ticks / tot_poss
        b_pct = 100.0 * self.blue_possession_ticks / tot_poss

        poss_lbl = self.font_telemetry.render(f"Posse: RED {r_pct:.0f}%  |  BLUE {b_pct:.0f}%", True, (220, 225, 235))
        self.screen.blit(poss_lbl, (cx + (center_w - poss_lbl.get_width()) // 2, cy - 8))

        # Possession Bar
        bar_w = 240
        bar_h = 10
        bx = cx + (center_w - bar_w) // 2
        by = cy + 14
        r_w = int(bar_w * (r_pct / 100.0))
        pygame.draw.rect(self.screen, (229, 110, 86), (bx, by, r_w, bar_h), border_top_left_radius=3, border_bottom_left_radius=3)
        pygame.draw.rect(self.screen, (86, 137, 229), (bx + r_w, by, bar_w - r_w, bar_h), border_top_right_radius=3, border_bottom_right_radius=3)

        shots_lbl = self.font_slot.render(f"Finalizações: Red {self.red_shots}  |  Blue {self.blue_shots}", True, (160, 175, 195))
        self.screen.blit(shots_lbl, (cx + (center_w - shots_lbl.get_width()) // 2, by + 16))

        # Team Blue Slots (Right Side)
        rx = self.width - 25 - (card_w + 10) * self.players_per_team
        lbl_blue = self.font_btn.render("TIME AZUL (Clique p/ Trocar):", True, (86, 137, 229))
        self.screen.blit(lbl_blue, (rx, card_y - 10))

        for idx in range(self.players_per_team):
            ctrl_key = self.slot_controllers[Team.BLUE][idx]
            tag_name, tag_color = next(((t[1], t[2]) for t in CONTROLLER_TYPES if t[0] == ctrl_key), ("IDLE", (150, 150, 150)))

            slot_rect = pygame.Rect(rx, card_y + 10, card_w, 48)
            pygame.draw.rect(self.screen, (28, 34, 46), slot_rect, border_radius=6)
            border_col = (86, 137, 229) if ctrl_key != "idle" else (60, 70, 85)
            pygame.draw.rect(self.screen, border_col, slot_rect, width=2, border_radius=6)

            p_title = self.font_slot.render(f"JOGADOR #{idx+1} [BLUE]", True, (200, 210, 225))
            c_title = self.font_slot.render(tag_name, True, tag_color)
            self.screen.blit(p_title, (slot_rect.x + 8, slot_rect.y + 7))
            self.screen.blit(c_title, (slot_rect.x + 8, slot_rect.y + 26))

            self.ui_buttons.append({"rect": slot_rect, "action": "cycle_slot", "team": Team.BLUE, "idx": idx})
            rx += card_w + 10

    def handle_click(self, pos: Tuple[int, int]):
        for btn in self.ui_buttons:
            if btn["rect"].collidepoint(pos):
                act = btn["action"]
                if act == "set_stadium":
                    self.set_stadium(btn["val"])
                elif act == "set_format":
                    self.set_format(btn["val"])
                elif act == "cycle_speed":
                    self.speed_idx = (self.speed_idx + 1) % len(SPEED_OPTIONS)
                elif act == "toggle_sound":
                    self.sound_enabled = not self.sound_enabled
                elif act == "toggle_pause":
                    self.is_paused = not self.is_paused
                elif act == "reset_round":
                    self.game.reset_round()
                elif act == "cycle_slot":
                    self.cycle_slot_controller(btn["team"], btn["idx"])
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
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.game.reset_round()
                    elif event.key == pygame.K_n:
                        self._init_game()
                    elif event.key in (pygame.K_p, pygame.K_PAUSE):
                        self.is_paused = not self.is_paused
                    elif event.key == pygame.K_m:
                        self.sound_enabled = not self.sound_enabled
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
