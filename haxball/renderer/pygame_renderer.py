"""
Pygame Renderer for HaxBall with Authentic Futsal UI, Aesthetics & Procedural Audio.
Features:
- Authentic Futsal parquet pitch styling with crisp markings and goal net cross-hatching
- Faithful player discs (number, inner accent ring, official team colors, kick flash ring)
- Player highlight indicators (chevron for human, labels for AI and Bots)
- Smooth expanding kick impact ripple particles
- Official HaxBall scoreboard header (Red/Blue score badges, digital match timer, status banners)
- Fully integrated procedural audio via SoundManager (kicks, post bounces, referee whistle, goal celebrations)
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Dict, Any, List
import pygame
from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.disc import Disc, hex_to_rgb
from haxball.core.segment import Segment
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.renderer.sound_effects import SoundManager


class PygameRenderer:
    def __init__(self, game: HaxBallGame, width: int = 1100, height: int = 620):
        self.game = game
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"HaxBall Futsal - {game.stadium.name}")

        # Authentic HaxBall UI Fonts
        self.font_score_badge = pygame.font.SysFont("Trebuchet MS", 26, bold=True)
        self.font_timer = pygame.font.SysFont("Lucida Console", 22, bold=True)
        self.font_banner = pygame.font.SysFont("Trebuchet MS", 34, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_player = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_tag = pygame.font.SysFont("Arial", 11, bold=True)

        self.sound = SoundManager.get_instance()
        self.kick_ripples: List[Dict[str, Any]] = []
        self.last_game_state: Optional[GameState] = None

        self._calc_transform()

    def _calc_transform(self):
        """Calculates scale and screen center offset to fit the stadium."""
        stad = self.game.stadium
        margin_x = 55.0
        margin_y = 50.0
        header_height = 54.0

        scale_x = (self.width - margin_x * 2.0) / (stad.width * 2.0)
        scale_y = (self.height - header_height - margin_y * 2.0) / (stad.height * 2.0)
        self.scale = min(scale_x, scale_y)

        self.center_x = self.width / 2.0
        self.center_y = header_height + (self.height - header_height) / 2.0

    def world_to_screen(self, vec: Vec2) -> Tuple[int, int]:
        sx = int(self.center_x + vec.x * self.scale)
        sy = int(self.center_y - vec.y * self.scale)  # Flip Y so up is positive
        return (sx, sy)

    def world_len_to_screen(self, length: float) -> int:
        return max(1, int(round(length * self.scale)))

    def render(self, step_info: Optional[Dict[str, Any]] = None, human_player_id: Optional[int] = None):
        """Draws one complete frame of the game and handles audio/visual triggers."""
        # 1. Process Audio and Particle Events
        if step_info:
            events = step_info.get("events", {})
            kicks = events.get("kicks", [])
            bounces = events.get("bounces", [])

            if kicks:
                self.sound.play_kick()
                for k in kicks:
                    kx, ky = self.world_to_screen(Vec2.from_iterable(k["pos"]))
                    self.kick_ripples.append({"x": kx, "y": ky, "radius": 14, "alpha": 255})

            if bounces:
                self.sound.play_bounce()

            if step_info.get("goal_scored", False):
                self.sound.play_goal()

            # State transition to kickoff
            curr_state = step_info.get("state")
            if curr_state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE) and self.last_game_state == GameState.GOAL_CELEBRATION:
                self.sound.play_whistle()
            self.last_game_state = curr_state

        # 2. Outer Stadium Background (Dark modern HaxBall arena border)
        self.screen.fill((22, 28, 38))

        stad = self.game.stadium

        # 3. Authentic Futsal Pitch Surface
        bg_w = self.world_len_to_screen(stad.bg_width * 2.0)
        bg_h = self.world_len_to_screen(stad.bg_height * 2.0)
        pitch_rect = pygame.Rect(
            int(self.center_x - bg_w / 2.0),
            int(self.center_y - bg_h / 2.0),
            bg_w,
            bg_h
        )

        # Stadium Court Colors (Authentic competitive gray futsal court)
        court_color = hex_to_rgb(stad.bg_color) if hasattr(stad, "bg_color") and stad.bg_color else (60, 63, 67)
        border_court_color = (max(0, court_color[0] - 18), max(0, court_color[1] - 18), max(0, court_color[2] - 18))
        
        # Perimeter court buffer
        court_buffer_rect = pitch_rect.inflate(self.world_len_to_screen(35), self.world_len_to_screen(35))
        pygame.draw.rect(self.screen, border_court_color, court_buffer_rect, border_radius=6)
        pygame.draw.rect(self.screen, court_color, pitch_rect, border_radius=4)

        # Subtle court lines pattern
        line_color = (248, 248, 248)
        pygame.draw.rect(self.screen, line_color, pitch_rect, width=2, border_radius=4)

        # 4. Goal Net Cross-Hatching (behind goal lines)
        self._draw_goal_nets()

        # 5. Field Markings (Center line, Center circle, Spot)
        c_top = (int(self.center_x), pitch_rect.top)
        c_bottom = (int(self.center_x), pitch_rect.bottom)
        pygame.draw.line(self.screen, line_color, c_top, c_bottom, width=2)

        ko_rad = self.world_len_to_screen(stad.bg_kickoff_radius)
        center_pt = (int(self.center_x), int(self.center_y))
        pygame.draw.circle(self.screen, line_color, center_pt, ko_rad, width=2)
        pygame.draw.circle(self.screen, line_color, center_pt, 4)

        # Penalty spots and goal area markings if standard
        self._draw_pitch_markings(pitch_rect)

        # 6. Kick Ripple Particles
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

        # 7. Segments (Walls and barriers)
        for seg in self.game.physics.segments:
            if not seg.vis:
                continue
            color = seg.color_rgb
            if seg.trait == "goalNet":
                continue  # Drawn with authentic net mesh
            if seg.is_curved:
                self._draw_curved_segment(seg, color)
            else:
                p0_s = self.world_to_screen(seg.p0)
                p1_s = self.world_to_screen(seg.p1)
                pygame.draw.line(self.screen, color, p0_s, p1_s, width=3)

        # 8. Static Obstacles (Goal Posts)
        for d in self.game.physics.discs:
            if d.is_static:
                self._draw_disc(d)

        # 9. Dynamic Entities (Ball and Players)
        if self.game.ball:
            self._draw_disc(self.game.ball)

        for p in self.game.players:
            self._draw_disc(p)
            self._draw_player_tags(p, human_player_id)

        # 10. Official HaxBall Top Scoreboard
        self._draw_scoreboard()

        pygame.display.flip()

    def _draw_goal_nets(self):
        """Draws realistic cross-hatched goal netting."""
        stad = self.game.stadium
        gw = self.world_len_to_screen(40.0)
        gh = self.world_len_to_screen(160.0)
        pitch_w = self.world_len_to_screen(stad.bg_width * 2.0)

        # Left Net (Behind Red goal)
        lx = int(self.center_x - pitch_w / 2.0 - gw)
        ly = int(self.center_y - gh / 2.0)
        left_net_rect = pygame.Rect(lx, ly, gw, gh)
        pygame.draw.rect(self.screen, (15, 20, 26), left_net_rect)

        # Draw diamond mesh
        step = 8
        for x in range(lx, lx + gw + step, step):
            pygame.draw.line(self.screen, (60, 75, 90), (x, ly), (x + 12, ly + gh), width=1)
        for y in range(ly, ly + gh + step, step):
            pygame.draw.line(self.screen, (60, 75, 90), (lx, y), (lx + gw, y + 6), width=1)
        pygame.draw.rect(self.screen, (220, 220, 220), left_net_rect, width=2)

        # Right Net (Behind Blue goal)
        rx = int(self.center_x + pitch_w / 2.0)
        ry = int(self.center_y - gh / 2.0)
        right_net_rect = pygame.Rect(rx, ry, gw, gh)
        pygame.draw.rect(self.screen, (15, 20, 26), right_net_rect)
        for x in range(rx, rx + gw + step, step):
            pygame.draw.line(self.screen, (60, 75, 90), (x, ry), (x - 12, ry + gh), width=1)
        for y in range(ly, ly + gh + step, step):
            pygame.draw.line(self.screen, (60, 75, 90), (rx, y), (rx + gw, y - 6), width=1)
        pygame.draw.rect(self.screen, (220, 220, 220), right_net_rect, width=2)

    def _draw_pitch_markings(self, pitch_rect: pygame.Rect):
        """Draws subtle futsal goal area lines."""
        line_color = (248, 248, 248, 140)
        # Left penalty area arc approximation
        area_rad = self.world_len_to_screen(75.0)
        p_left = (pitch_rect.left, int(self.center_y))
        p_right = (pitch_rect.right, int(self.center_y))
        pygame.draw.circle(self.screen, (248, 248, 248), p_left, area_rad, width=2)
        pygame.draw.circle(self.screen, (248, 248, 248), p_right, area_rad, width=2)

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
            # Player outer flash ring when kicking (Official HaxBall White Ring)
            if disc.is_kicking or disc.kick_flash > 0:
                flash_rad = rad_s + 4
                pygame.draw.circle(self.screen, (255, 255, 255), center_s, flash_rad, width=3)

            # Solid player circle body
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            # Outer dark rim
            pygame.draw.circle(self.screen, (20, 22, 28), center_s, rad_s, width=2)
            # Inner white accent ring
            inner_rad = max(2, rad_s - 4)
            pygame.draw.circle(self.screen, (255, 255, 255), center_s, inner_rad, width=1)

            # Centered player number
            num_surf = self.font_player.render(str(disc.player_number), True, (255, 255, 255))
            num_rect = num_surf.get_rect(center=center_s)
            self.screen.blit(num_surf, num_rect)

        elif disc.name == "Ball":
            # Ball drop shadow
            shadow_pos = (center_s[0] + 2, center_s[1] + 2)
            pygame.draw.circle(self.screen, (15, 20, 28, 120), shadow_pos, rad_s)
            # Ball body
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            # Ball dark outer border
            pygame.draw.circle(self.screen, (25, 25, 30), center_s, rad_s, width=2)
            # Authentic center dot
            dot_rad = max(1, rad_s // 3)
            pygame.draw.circle(self.screen, (60, 60, 60), center_s, dot_rad)
            # 3D specular highlight dot
            highlight_pos = (center_s[0] - max(1, rad_s // 3), center_s[1] - max(1, rad_s // 3))
            pygame.draw.circle(self.screen, (255, 255, 255), highlight_pos, max(1, rad_s // 5))

        else:
            # Static Obstacles / Goal Posts
            pygame.draw.circle(self.screen, (255, 255, 255), center_s, rad_s)
            pygame.draw.circle(self.screen, (40, 45, 55), center_s, rad_s, width=2)
            pygame.draw.circle(self.screen, (180, 180, 180), center_s, max(1, rad_s - 3), width=1)

    def _draw_player_tags(self, p: Disc, human_player_id: Optional[int]):
        """Draws clear identification badges above players."""
        cx, cy = self.world_to_screen(p.pos)
        rad_s = self.world_len_to_screen(p.radius)

        if p.player_id == human_player_id or (human_player_id is None and p.team == Team.RED and p.player_number == 1):
            # Glowing yellow chevron arrow indicator for human
            pts = [(cx, cy - rad_s - 8), (cx - 7, cy - rad_s - 18), (cx + 7, cy - rad_s - 18)]
            pygame.draw.polygon(self.screen, (255, 220, 0), pts)
            pygame.draw.polygon(self.screen, (40, 30, 0), pts, width=1)
            lbl = self.font_tag.render("VOCÊ", True, (255, 230, 40))
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - rad_s - 29))
        elif p.team == Team.RED:
            lbl = self.font_tag.render(f"IA {p.player_number}", True, (255, 170, 170))
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - rad_s - 18))
        elif p.team == Team.BLUE:
            role_map = {1: "FIXO", 2: "ALA", 3: "PRESS"}
            role_str = role_map.get(p.player_number, f"BOT {p.player_number}")
            lbl = self.font_tag.render(role_str, True, (170, 205, 255))
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - rad_s - 18))

    def _draw_scoreboard(self):
        """Draws official HaxBall top scoreboard with clean badges."""
        bar_height = 50
        bar_rect = pygame.Rect(0, 0, self.width, bar_height)
        pygame.draw.rect(self.screen, (16, 20, 28), bar_rect)
        pygame.draw.line(self.screen, (34, 42, 58), (0, bar_height), (self.width, bar_height), width=2)

        # RED Score Badge Box
        red_box = pygame.Rect(int(self.center_x - 170), 8, 110, 34)
        pygame.draw.rect(self.screen, (229, 110, 86), red_box, border_radius=4)
        red_txt = self.font_score_badge.render(f"RED  {self.game.red_score}", True, (255, 255, 255))
        self.screen.blit(red_txt, red_txt.get_rect(center=red_box.center))

        # BLUE Score Badge Box
        blue_box = pygame.Rect(int(self.center_x + 60), 8, 110, 34)
        pygame.draw.rect(self.screen, (86, 137, 229), blue_box, border_radius=4)
        blue_txt = self.font_score_badge.render(f"{self.game.blue_score}  BLUE", True, (255, 255, 255))
        self.screen.blit(blue_txt, blue_txt.get_rect(center=blue_box.center))

        # Digital Timer in center
        timer_txt = self.font_timer.render(self.game.time_string, True, (255, 255, 255))
        self.screen.blit(timer_txt, timer_txt.get_rect(center=(int(self.center_x), bar_height // 2)))

        # Stadium Name and Mode (Top Left)
        mode_surf = self.font_sub.render(f"Futsal 3v3  |  {self.game.stadium.name}", True, (160, 175, 195))
        self.screen.blit(mode_surf, (20, 16))

        # Match Info / Controls (Top Right)
        ctrl_surf = self.font_sub.render("WASD = Mover | Espaco = Chutar | R = Reset", True, (140, 155, 175))
        self.screen.blit(ctrl_surf, (self.width - ctrl_surf.get_width() - 20, 16))

        # Goal and Match End Notification Banners
        if self.game.state == GameState.GOAL_CELEBRATION:
            team_str = "RED" if self.game.last_goal_team == Team.RED else "BLUE"
            color = (229, 110, 86) if self.game.last_goal_team == Team.RED else (86, 137, 229)
            banner = self.font_banner.render(f"GOAL! {team_str} SCORED!", True, color)
            b_rect = banner.get_rect(center=(int(self.center_x), int(self.center_y - 75)))
            box = b_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, (12, 16, 22), box, border_radius=8)
            pygame.draw.rect(self.screen, color, box, width=3, border_radius=8)
            self.screen.blit(banner, b_rect)

        elif self.game.state == GameState.GAME_OVER:
            w_str = "RED WINS!" if self.game.red_score > self.game.blue_score else "BLUE WINS!"
            banner = self.font_banner.render(f"MATCH OVER - {w_str}", True, (255, 215, 0))
            b_rect = banner.get_rect(center=(int(self.center_x), int(self.center_y - 75)))
            box = b_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, (12, 16, 22), box, border_radius=8)
            pygame.draw.rect(self.screen, (255, 215, 0), box, width=3, border_radius=8)
            self.screen.blit(banner, b_rect)

        elif self.game.state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE):
            ko_str = "RED KICK OFF" if self.game.state == GameState.KICKOFF_RED else "BLUE KICK OFF"
            ko_color = (229, 110, 86) if self.game.state == GameState.KICKOFF_RED else (86, 137, 229)
            ko_surf = self.font_sub.render(ko_str, True, ko_color)
            self.screen.blit(ko_surf, ko_surf.get_rect(center=(int(self.center_x), 68)))
