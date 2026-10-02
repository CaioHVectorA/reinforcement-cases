"""
Pygame Renderer for HaxBall.
Renders pitch, center circle, goal areas, curved walls, player discs,
ball, kick animations, score board, and match timers.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import pygame
from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.disc import Disc, hex_to_rgb
from haxball.core.segment import Segment
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame

class PygameRenderer:
    def __init__(self, game: HaxBallGame, width: int = 1000, height: int = 560):
        self.game = game
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"HaxBall Simulator - {game.stadium.name}")

        self.font_score = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_timer = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_banner = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_sub = pygame.font.SysFont("Arial", 16)
        self.font_player = pygame.font.SysFont("Arial", 14, bold=True)

        self._calc_transform()

    def _calc_transform(self):
        """Calculates scale and screen center offset to fit the stadium."""
        stad = self.game.stadium
        margin = 40.0
        scale_x = (self.width - margin * 2) / (stad.width * 2.0)
        scale_y = (self.height - margin * 2 - 50.0) / (stad.height * 2.0)
        self.scale = min(scale_x, scale_y)

        self.center_x = self.width / 2.0
        self.center_y = (self.height + 50.0) / 2.0  # Leave top space for scoreboard

    def world_to_screen(self, vec: Vec2) -> Tuple[int, int]:
        sx = int(self.center_x + vec.x * self.scale)
        sy = int(self.center_y - vec.y * self.scale)  # Flip Y so up is positive
        return (sx, sy)

    def world_len_to_screen(self, length: float) -> int:
        return max(1, int(round(length * self.scale)))

    def render(self):
        """Draws one frame of the game."""
        self.screen.fill((20, 24, 30))  # Modern dark background around stadium

        stad = self.game.stadium

        # 1. Pitch background
        bg_rgb = hex_to_rgb(stad.bg_color)
        bg_w = self.world_len_to_screen(stad.bg_width * 2.0)
        bg_h = self.world_len_to_screen(stad.bg_height * 2.0)
        bg_rect = pygame.Rect(
            int(self.center_x - bg_w / 2.0),
            int(self.center_y - bg_h / 2.0),
            bg_w,
            bg_h
        )
        pygame.draw.rect(self.screen, bg_rgb, bg_rect)
        pygame.draw.rect(self.screen, (240, 240, 240), bg_rect, width=2)

        # 2. Field markings (center line, center circle, kickoff spot)
        c_top = (int(self.center_x), int(self.center_y - bg_h / 2.0))
        c_bottom = (int(self.center_x), int(self.center_y + bg_h / 2.0))
        pygame.draw.line(self.screen, (255, 255, 255, 180), c_top, c_bottom, width=2)

        ko_rad = self.world_len_to_screen(stad.bg_kickoff_radius)
        pygame.draw.circle(
            self.screen, (255, 255, 255),
            (int(self.center_x), int(self.center_y)),
            ko_rad, width=2
        )
        pygame.draw.circle(
            self.screen, (255, 255, 255),
            (int(self.center_x), int(self.center_y)),
            4
        )

        # 3. Segments (walls, goal nets)
        for seg in self.game.physics.segments:
            if not seg.vis:
                continue

            color = seg.color_rgb
            if seg.is_curved:
                self._draw_curved_segment(seg, color)
            else:
                p0_s = self.world_to_screen(seg.p0)
                p1_s = self.world_to_screen(seg.p1)
                pygame.draw.line(self.screen, color, p0_s, p1_s, width=3)

        # 4. Static Discs (Goal posts)
        for d in self.game.physics.discs:
            if d.is_static:
                self._draw_disc(d)

        # 5. Dynamic Entities (Ball and Players)
        if self.game.ball:
            self._draw_disc(self.game.ball)

        for p in self.game.players:
            self._draw_disc(p)

        # 6. Scoreboard and UI
        self._draw_scoreboard()

        pygame.display.flip()

    def _draw_curved_segment(self, seg: Segment, color: Tuple[int, int, int]):
        """Approximates and draws an arc segment using line segments."""
        steps = 16
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
            # Player outer glow if kicking
            if disc.kick_flash > 0:
                pygame.draw.circle(self.screen, (255, 255, 255), center_s, rad_s + 4, width=3)

            # Player base circle
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            # Black border
            pygame.draw.circle(self.screen, (20, 20, 20), center_s, rad_s, width=2)
            # Inner circle aesthetic
            inner_rad = max(1, rad_s - 5)
            pygame.draw.circle(self.screen, (255, 255, 255), center_s, inner_rad, width=1)

            # Player number
            num_surf = self.font_player.render(str(disc.player_number), True, (255, 255, 255))
            num_rect = num_surf.get_rect(center=center_s)
            self.screen.blit(num_surf, num_rect)
        elif disc.name == "Ball":
            # Ball shadow
            shadow_pos = (center_s[0] + 2, center_s[1] + 2)
            pygame.draw.circle(self.screen, (0, 0, 0, 80), shadow_pos, rad_s)
            # Ball body
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            # Ball border
            pygame.draw.circle(self.screen, (20, 20, 20), center_s, rad_s, width=2)
            # Authentic center dot
            pygame.draw.circle(self.screen, (60, 60, 60), center_s, max(1, rad_s // 3))
        else:
            # Static obstacles / goal posts
            pygame.draw.circle(self.screen, disc.color_rgb, center_s, rad_s)
            pygame.draw.circle(self.screen, (40, 40, 40), center_s, rad_s, width=2)

    def _draw_scoreboard(self):
        # Top bar background
        bar_rect = pygame.Rect(0, 0, self.width, 48)
        pygame.draw.rect(self.screen, (15, 18, 24), bar_rect)
        pygame.draw.line(self.screen, (35, 42, 54), (0, 48), (self.width, 48), width=2)

        # Team Red Score
        red_surf = self.font_score.render(f"RED  {self.game.red_score}", True, (229, 110, 86))
        self.screen.blit(red_surf, (self.width / 2 - 160, 8))

        # Divider
        vs_surf = self.font_score.render("-", True, (200, 200, 200))
        self.screen.blit(vs_surf, (self.width / 2 - 8, 8))

        # Team Blue Score
        blue_surf = self.font_score.render(f"{self.game.blue_score}  BLUE", True, (86, 137, 229))
        self.screen.blit(blue_surf, (self.width / 2 + 30, 8))

        # Match Timer
        timer_surf = self.font_timer.render(self.game.time_string, True, (220, 220, 220))
        self.screen.blit(timer_surf, (self.width - 90, 12))

        # Stadium Name
        stad_surf = self.font_sub.render(self.game.stadium.name, True, (160, 160, 160))
        self.screen.blit(stad_surf, (20, 16))

        # Central game status banners
        if self.game.state == GameState.GOAL_CELEBRATION:
            team_name = "RED" if self.game.last_goal_team == Team.RED else "BLUE"
            team_color = (229, 110, 86) if self.game.last_goal_team == Team.RED else (86, 137, 229)
            banner = self.font_banner.render(f"GOAL! {team_name} SCORED!", True, team_color)
            b_rect = banner.get_rect(center=(self.width / 2, self.center_y - 60))
            # Background box for readability
            bg_box = b_rect.inflate(30, 16)
            pygame.draw.rect(self.screen, (10, 10, 10), bg_box, border_radius=8)
            pygame.draw.rect(self.screen, team_color, bg_box, width=2, border_radius=8)
            self.screen.blit(banner, b_rect)

        elif self.game.state == GameState.GAME_OVER:
            winner_str = "RED WINS!" if self.game.red_score > self.game.blue_score else "BLUE WINS!"
            banner = self.font_banner.render(f"MATCH OVER - {winner_str}", True, (255, 215, 0))
            b_rect = banner.get_rect(center=(self.width / 2, self.center_y - 60))
            bg_box = b_rect.inflate(30, 16)
            pygame.draw.rect(self.screen, (10, 10, 10), bg_box, border_radius=8)
            pygame.draw.rect(self.screen, (255, 215, 0), bg_box, width=2, border_radius=8)
            self.screen.blit(banner, b_rect)

        elif self.game.state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE):
            ko_str = "RED KICK OFF" if self.game.state == GameState.KICKOFF_RED else "BLUE KICK OFF"
            ko_color = (229, 110, 86) if self.game.state == GameState.KICKOFF_RED else (86, 137, 229)
            ko_surf = self.font_sub.render(ko_str, True, ko_color)
            k_rect = ko_surf.get_rect(center=(self.width / 2, 70))
            self.screen.blit(ko_surf, k_rect)
