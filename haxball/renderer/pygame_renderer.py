"""
Pygame Renderer for HaxBall Futsal - Pixel-Perfect Match to Official HaxBall Futsal Court.
Faithful reproduction of competitive GLH / Bazinga HaxBall Futsal:
- Matte dark slate-gray futsal court (#3E464D) with clean white court markings (#FFFFFF)
- Official goal posts: Pink/Red posts (#FF6B6B) on Left, Blue posts (#4FA3FF) on Right with blue guide rails
- 4 Yellow corner dots (#FFC107) and dual white penalty spots along the center axis
- Out-of-bounds boundary tick marks along touchlines
- Small orange/amber futsal ball (#FFA000) with subtle dark outline
- Authentic player avatars with jersey numbers, team styling, and player nicknames rendered below each avatar
- Completely silent (all audio/sound effects disabled as requested)
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


# Default nicknames matching authentic competitive HaxBall Futsal rooms
DEFAULT_NICKNAMES = {
    Team.RED: ["Umbabaraum", "özil", "lucasfera15"],
    Team.BLUE: ["alex atacante", "gimendez", "falcao12"]
}


class PygameRenderer:
    def __init__(self, game: HaxBallGame, width: int = 1200, height: int = 620):
        self.game = game
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("HaxBall Futsal - Futsal x3 GLH")

        # Fonts
        self.font_nick = pygame.font.SysFont("Verdana", 11, bold=False)
        self.font_num = pygame.font.SysFont("Verdana", 11, bold=True)
        self.font_score = pygame.font.SysFont("Trebuchet MS", 22, bold=True)
        self.font_timer = pygame.font.SysFont("Lucida Console", 18, bold=True)
        self.font_banner = pygame.font.SysFont("Trebuchet MS", 28, bold=True)

        self.nicknames: Dict[int, str] = {}
        self._init_nicknames()

        self._calc_transform()

    def _init_nicknames(self):
        red_players = [p for p in self.game.players if p.team == Team.RED]
        blue_players = [p for p in self.game.players if p.team == Team.BLUE]
        for i, p in enumerate(red_players):
            self.nicknames[p.player_id] = DEFAULT_NICKNAMES[Team.RED][i % len(DEFAULT_NICKNAMES[Team.RED])]
        for i, p in enumerate(blue_players):
            self.nicknames[p.player_id] = DEFAULT_NICKNAMES[Team.BLUE][i % len(DEFAULT_NICKNAMES[Team.BLUE])]

    def set_player_nickname(self, player_id: int, nickname: str):
        self.nicknames[player_id] = nickname

    def _calc_transform(self):
        stad = self.game.stadium
        margin_x = 35.0
        margin_y = 22.0
        header_h = 24.0

        scale_x = (self.width - margin_x * 2.0) / (stad.width * 2.0)
        scale_y = (self.height - header_h - margin_y * 2.0) / (stad.height * 2.0)
        self.scale = min(scale_x, scale_y)

        self.center_x = self.width / 2.0
        self.center_y = header_h + (self.height - header_h) / 2.0

    def world_to_screen(self, vec: Vec2) -> Tuple[int, int]:
        sx = int(self.center_x + vec.x * self.scale)
        sy = int(self.center_y - vec.y * self.scale)  # Flip Y so up is positive
        return (sx, sy)

    def world_len_to_screen(self, length: float) -> int:
        return max(1, int(round(length * self.scale)))

    def render(self, step_info: Optional[Dict[str, Any]] = None, human_player_id: Optional[int] = None):
        """Draws one frame matching the authentic HaxBall Futsal visual specification."""
        # 1. Outer Arena Floor (Uniform dark slate-gray #424D55 / rgb(66, 77, 85))
        futsal_gray = (66, 77, 85)
        self.screen.fill(futsal_gray)

        stad = self.game.stadium

        # 2. Main Pitch Rectangle
        pw = self.world_len_to_screen(stad.bg_width * 2.0)
        ph = self.world_len_to_screen(stad.bg_height * 2.0)
        pitch_rect = pygame.Rect(
            int(self.center_x - pw / 2.0),
            int(self.center_y - ph / 2.0),
            pw,
            ph
        )

        court_color = futsal_gray
        pygame.draw.rect(self.screen, court_color, pitch_rect)

        # 3. Outer Touchlines and Boundaries (Solid clean white #FFFFFF, width 2)
        line_color = (255, 255, 255)
        pygame.draw.rect(self.screen, line_color, pitch_rect, width=2)

        # 4. Out-of-bounds boundary tick marks along touchlines
        self._draw_boundary_ticks(pitch_rect)

        # 5. Field Markings
        # Center Line
        c_top = (int(self.center_x), pitch_rect.top)
        c_bottom = (int(self.center_x), pitch_rect.bottom)
        pygame.draw.line(self.screen, line_color, c_top, c_bottom, width=2)

        # Center Circle
        ko_rad = self.world_len_to_screen(stad.bg_kickoff_radius)
        center_pt = (int(self.center_x), int(self.center_y))
        pygame.draw.circle(self.screen, line_color, center_pt, ko_rad, width=2)

        # Penalty Arcs (Goal areas)
        self._draw_futsal_penalty_arcs(pitch_rect)

        # Penalty Spots: 2 white dots on each half along the center axis
        self._draw_penalty_spots()

        # 6. Blue Goal Rails (on right goal extending outward)
        self._draw_goal_rails(pitch_rect)

        # 7. Four Corner Dots (Yellow #FFCC00)
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

        # 8. Segments (Walls, barriers)
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

        # 9. Goal Posts (Red/Pink on Left, Blue on Right)
        self._draw_goal_posts()

        # 10. Dynamic Discs: Ball & Players
        if self.game.ball:
            self._draw_futsal_ball(self.game.ball)

        for p in self.game.players:
            self._draw_futsal_player(p, human_player_id)

        # 11. Minimalist Top Scoreboard (Subtle, non-intrusive)
        self._draw_minimal_scoreboard()

        pygame.display.flip()

    def _draw_boundary_ticks(self, rect: pygame.Rect):
        """Draws the subtle white tick marks outside touchlines as seen in the screenshot."""
        tick_len = 10
        # Center top and bottom ticks
        pygame.draw.line(self.screen, (255, 255, 255), (int(self.center_x), rect.top - tick_len), (int(self.center_x), rect.top), width=2)
        pygame.draw.line(self.screen, (255, 255, 255), (int(self.center_x), rect.bottom), (int(self.center_x), rect.bottom + tick_len), width=2)

        # Left / Right side ticks
        for offset_x in [-self.world_len_to_screen(180), self.world_len_to_screen(180)]:
            x = int(self.center_x + offset_x)
            pygame.draw.line(self.screen, (255, 255, 255), (x, rect.top - tick_len), (x, rect.top), width=2)
            pygame.draw.line(self.screen, (255, 255, 255), (x, rect.bottom), (x, rect.bottom + tick_len), width=2)

        # Goal boundary sideline ticks
        y_mid = int(self.center_y)
        pygame.draw.line(self.screen, (255, 255, 255), (rect.left - tick_len, y_mid), (rect.left, y_mid), width=2)
        pygame.draw.line(self.screen, (255, 255, 255), (rect.right, y_mid), (rect.right + tick_len, y_mid), width=2)

    def _draw_futsal_penalty_arcs(self, rect: pygame.Rect):
        """Draws the authentic curved futsal penalty areas."""
        line_color = (255, 255, 255)
        # Goal arc spans from posts curving into pitch
        arc_w = self.world_len_to_screen(125.0)
        arc_h = self.world_len_to_screen(160.0)

        # Left arc
        l_box = pygame.Rect(rect.left - arc_w, int(self.center_y - arc_h), arc_w * 2, arc_h * 2)
        pygame.draw.arc(self.screen, line_color, l_box, -math.pi / 2, math.pi / 2, width=2)

        # Right arc
        r_box = pygame.Rect(rect.right - arc_w, int(self.center_y - arc_h), arc_w * 2, arc_h * 2)
        pygame.draw.arc(self.screen, line_color, r_box, math.pi / 2, 3 * math.pi / 2, width=2)

    def _draw_penalty_spots(self):
        """Draws the 2 penalty spots on each half along the center axis."""
        p_color = (255, 255, 255)
        spot_rad = 3

        # Red side spots
        s1 = self.world_to_screen(Vec2(-380.0, 0.0))
        s2 = self.world_to_screen(Vec2(-250.0, 0.0))
        pygame.draw.circle(self.screen, p_color, s1, spot_rad)
        pygame.draw.circle(self.screen, p_color, s2, spot_rad)

        # Blue side spots
        s3 = self.world_to_screen(Vec2(250.0, 0.0))
        s4 = self.world_to_screen(Vec2(380.0, 0.0))
        pygame.draw.circle(self.screen, p_color, s3, spot_rad)
        pygame.draw.circle(self.screen, p_color, s4, spot_rad)

    def _draw_goal_rails(self, rect: pygame.Rect):
        """Draws the blue guide lines extending outward from the blue goal mouth as seen in the image."""
        blue_rail_col = (74, 163, 255)
        rail_len = self.world_len_to_screen(45.0)
        y_top = self.world_to_screen(Vec2(550.0, 80.0))[1]
        y_bot = self.world_to_screen(Vec2(550.0, -80.0))[1]

        pygame.draw.line(self.screen, blue_rail_col, (rect.right, y_top), (rect.right + rail_len, y_top), width=2)
        pygame.draw.line(self.screen, blue_rail_col, (rect.right, y_bot), (rect.right + rail_len, y_bot), width=2)

    def _draw_goal_posts(self):
        """Draws the authentic goal posts: Red/Pink on Left, Blue on Right."""
        post_rad = 5

        # Left Goal Posts (Pink/Red #FF6B6B)
        red_post_col = (255, 115, 115)
        p_l_top = self.world_to_screen(Vec2(-550.0, 80.0))
        p_l_bot = self.world_to_screen(Vec2(-550.0, -80.0))
        pygame.draw.circle(self.screen, red_post_col, p_l_top, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_l_top, post_rad, width=1)
        pygame.draw.circle(self.screen, red_post_col, p_l_bot, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_l_bot, post_rad, width=1)

        # Right Goal Posts (Sky Blue #4DA3FF)
        blue_post_col = (77, 163, 255)
        p_r_top = self.world_to_screen(Vec2(550.0, 80.0))
        p_r_bot = self.world_to_screen(Vec2(550.0, -80.0))
        pygame.draw.circle(self.screen, blue_post_col, p_r_top, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_r_top, post_rad, width=1)
        pygame.draw.circle(self.screen, blue_post_col, p_r_bot, post_rad)
        pygame.draw.circle(self.screen, (20, 20, 20), p_r_bot, post_rad, width=1)

    def _draw_futsal_ball(self, ball: Disc):
        """Draws the small orange/amber futsal ball with crisp dark rim."""
        center_s = self.world_to_screen(ball.pos)
        rad_s = self.world_len_to_screen(ball.radius)

        # Orange futsal ball (#FF9900)
        ball_col = (255, 153, 0)
        pygame.draw.circle(self.screen, ball_col, center_s, rad_s)
        # Dark rim
        pygame.draw.circle(self.screen, (20, 22, 25), center_s, rad_s, width=2)
        # Subtle core
        pygame.draw.circle(self.screen, (230, 130, 0), center_s, max(1, rad_s // 3))

    def _draw_futsal_player(self, p: Disc, human_player_id: Optional[int]):
        """Draws player discs with authentic HaxBall avatars and nicknames below."""
        cx, cy = self.world_to_screen(p.pos)
        rad_s = self.world_len_to_screen(p.radius)

        # Kick flash white ring (authentic HaxBall mechanic)
        if p.is_kicking or p.kick_flash > 0:
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), rad_s + 4, width=3)

        # Disc Body
        if p.team == Team.RED:
            # Red/Black styling
            pygame.draw.circle(self.screen, (211, 47, 47), (cx, cy), rad_s)
            pygame.draw.circle(self.screen, (20, 20, 20), (cx, cy), rad_s, width=2)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, rad_s - 4), width=1)
        else:
            # Blue with yellow crescent/half top (matching screenshot)
            pygame.draw.circle(self.screen, (21, 101, 192), (cx, cy), rad_s)
            # Yellow top crescent
            y_box = pygame.Rect(cx - rad_s, cy - rad_s, rad_s * 2, rad_s)
            pygame.draw.arc(self.screen, (255, 214, 0), y_box, 0, math.pi, width=3)
            pygame.draw.circle(self.screen, (20, 20, 20), (cx, cy), rad_s, width=2)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, rad_s - 4), width=1)

        # Jersey Number
        num_str = str(p.player_number)
        num_surf = self.font_num.render(num_str, True, (255, 255, 255))
        self.screen.blit(num_surf, num_surf.get_rect(center=(cx, cy)))

        # Nickname rendered cleanly below avatar (matching the screenshot!)
        nick = self.nicknames.get(p.player_id, f"player_{p.player_number}")
        # If this is the human player, highlight nickname
        is_human = (p.player_id == human_player_id) or (human_player_id is None and p.team == Team.RED and p.player_number == 1)
        text_col = (255, 235, 60) if is_human else (240, 240, 240)

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
        """Draws subtle, non-intrusive score and timer at top."""
        # Clean top scoreboard
        r_score = str(self.game.red_score)
        b_score = str(self.game.blue_score)
        timer_str = self.game.time_string

        # Center pill
        pill_w = 210
        pill_h = 32
        pill_rect = pygame.Rect(int(self.center_x - pill_w / 2.0), 6, pill_w, pill_h)
        pygame.draw.rect(self.screen, (32, 38, 44), pill_rect, border_radius=6)
        pygame.draw.rect(self.screen, (55, 65, 75), pill_rect, width=1, border_radius=6)

        # Red score
        r_txt = self.font_score.render(r_score, True, (255, 110, 110))
        self.screen.blit(r_txt, (pill_rect.left + 24, pill_rect.centery - r_txt.get_height() // 2))

        # Timer
        t_txt = self.font_timer.render(timer_str, True, (240, 240, 240))
        self.screen.blit(t_txt, t_txt.get_rect(center=pill_rect.center))

        # Blue score
        b_txt = self.font_score.render(b_score, True, (110, 180, 255))
        self.screen.blit(b_txt, (pill_rect.right - 24 - b_txt.get_width(), pill_rect.centery - b_txt.get_height() // 2))

        # Notifications
        if self.game.state == GameState.GOAL_CELEBRATION:
            t_str = "RED" if self.game.last_goal_team == Team.RED else "BLUE"
            col = (255, 110, 110) if self.game.last_goal_team == Team.RED else (110, 180, 255)
            banner = self.font_banner.render(f"GOL! {t_str} MARCOU!", True, col)
            b_r = banner.get_rect(center=(int(self.center_x), int(self.center_y - 70)))
            pygame.draw.rect(self.screen, (24, 28, 34), b_r.inflate(36, 16), border_radius=6)
            pygame.draw.rect(self.screen, col, b_r.inflate(36, 16), width=2, border_radius=6)
            self.screen.blit(banner, b_r)
