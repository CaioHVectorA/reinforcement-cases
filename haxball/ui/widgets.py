"""
UI Widget components and procedural vector icon renderer for HaxBall.
Eliminates emoji font dependencies by drawing crisp, antialiased vector icons directly in Pygame.
"""

from __future__ import annotations
import math
from typing import Tuple, Callable, Optional, List
import pygame

COLOR_BG = (26, 33, 42)
COLOR_PANEL = (35, 43, 53)
COLOR_PANEL_LIGHT = (45, 55, 68)
COLOR_PRIMARY = (58, 142, 230)
COLOR_PRIMARY_HOVER = (75, 160, 248)
COLOR_SUCCESS = (60, 179, 113)
COLOR_SUCCESS_HOVER = (75, 200, 130)
COLOR_DANGER = (220, 70, 70)
COLOR_WARNING = (235, 150, 40)
COLOR_TEXT = (235, 240, 245)
COLOR_TEXT_MUTED = (160, 170, 185)
COLOR_BORDER = (55, 68, 85)

# ==========================================
# Procedural Vector Icons (Zero Font Dependency)
# ==========================================

def draw_icon_play(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255), size: int = 12):
    cx, cy = center
    hs = size // 2
    pts = [
        (cx - hs + 1, cy - hs),
        (cx + hs + 1, cy),
        (cx - hs + 1, cy + hs)
    ]
    pygame.draw.polygon(surface, color, pts)

def draw_icon_pause(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255), size: int = 12):
    cx, cy = center
    bar_w = max(2, size // 4)
    bar_h = size
    gap = 2
    r1 = pygame.Rect(cx - bar_w - gap, cy - bar_h // 2, bar_w, bar_h)
    r2 = pygame.Rect(cx + gap, cy - bar_h // 2, bar_w, bar_h)
    pygame.draw.rect(surface, color, r1, border_radius=1)
    pygame.draw.rect(surface, color, r2, border_radius=1)

def draw_icon_reset(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255), size: int = 12):
    cx, cy = center
    rad = size // 2
    rect = pygame.Rect(cx - rad, cy - rad, rad * 2, rad * 2)
    # 270 degree arc
    pygame.draw.arc(surface, color, rect, 0.4, 5.8, width=2)
    # Arrow head at the end of the arc
    tip_x = cx + rad - 1
    tip_y = cy - 2
    pts = [
        (tip_x - 3, tip_y - 4),
        (tip_x + 3, tip_y),
        (tip_x - 4, tip_y + 3)
    ]
    pygame.draw.polygon(surface, color, pts)

def draw_icon_lightning(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (255, 210, 60), size: int = 14):
    cx, cy = center
    pts = [
        (cx + 1, cy - 7),
        (cx - 5, cy + 0),
        (cx - 1, cy + 0),
        (cx - 3, cy + 7),
        (cx + 5, cy - 1),
        (cx + 1, cy - 1)
    ]
    pygame.draw.polygon(surface, color, pts)

def draw_icon_ball(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255), size: int = 12):
    cx, cy = center
    rad = size // 2
    pygame.draw.circle(surface, color, (cx, cy), rad)
    pygame.draw.circle(surface, (25, 30, 38), (cx, cy), rad, width=1)
    # Center pentagon dot
    pygame.draw.circle(surface, (30, 35, 45), (cx, cy), max(1, rad // 3))

def draw_icon_stadium(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (200, 220, 240), size: int = 14):
    cx, cy = center
    w, h = size + 4, size
    rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
    pygame.draw.rect(surface, color, rect, width=1, border_radius=2)
    pygame.draw.line(surface, color, (cx, cy - h // 2), (cx, cy + h // 2), width=1)
    pygame.draw.circle(surface, color, (cx, cy), max(2, h // 4), width=1)

def draw_icon_user(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (100, 180, 255), size: int = 12):
    cx, cy = center
    # Head
    pygame.draw.circle(surface, color, (cx, cy - 3), 3)
    # Shoulders
    rect = pygame.Rect(cx - 5, cy + 1, 10, 6)
    pygame.draw.arc(surface, color, rect, 0.0, math.pi, width=2)

def draw_icon_robot(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (80, 220, 140), size: int = 13):
    cx, cy = center
    # Antenna
    pygame.draw.line(surface, color, (cx, cy - 5), (cx, cy - 7), width=1)
    pygame.draw.circle(surface, color, (cx, cy - 8), 1)
    # Head box
    rect = pygame.Rect(cx - 5, cy - 4, 10, 8)
    pygame.draw.rect(surface, color, rect, width=1, border_radius=2)
    # Eyes
    pygame.draw.rect(surface, color, (cx - 3, cy - 2, 2, 2))
    pygame.draw.rect(surface, color, (cx + 1, cy - 2, 2, 2))

def draw_icon_brain(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (160, 130, 255), size: int = 13):
    cx, cy = center
    # Neural network nodes
    n1 = (cx - 4, cy - 4)
    n2 = (cx - 4, cy + 4)
    n3 = (cx + 4, cy - 4)
    n4 = (cx + 4, cy + 4)
    c_node = (cx, cy)
    # Connections
    for n in [n1, n2]:
        pygame.draw.line(surface, (color[0] // 2, color[1] // 2, color[2] // 2), n, c_node, width=1)
    for n in [n3, n4]:
        pygame.draw.line(surface, (color[0] // 2, color[1] // 2, color[2] // 2), c_node, n, width=1)
    # Dots
    for pt in [n1, n2, n3, n4, c_node]:
        pygame.draw.circle(surface, color, pt, 2)

def draw_icon_help(surface: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int] = (200, 210, 225), size: int = 12):
    cx, cy = center
    rad = size // 2
    pygame.draw.circle(surface, color, (cx, cy), rad, width=1)
    font = pygame.font.SysFont("Verdana", 9, bold=True)
    q = font.render("?", True, color)
    surface.blit(q, q.get_rect(center=(cx, cy)))

ICON_DISPATCH = {
    "play": draw_icon_play,
    "pause": draw_icon_pause,
    "reset": draw_icon_reset,
    "speed": draw_icon_lightning,
    "lightning": draw_icon_lightning,
    "ball": draw_icon_ball,
    "stadium": draw_icon_stadium,
    "map": draw_icon_stadium,
    "user": draw_icon_user,
    "human": draw_icon_user,
    "robot": draw_icon_robot,
    "bot": draw_icon_robot,
    "brain": draw_icon_brain,
    "rl": draw_icon_brain,
    "help": draw_icon_help
}

class UIButton:
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        icon: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        bg_color: Tuple[int, int, int] = COLOR_PANEL_LIGHT,
        hover_color: Tuple[int, int, int] = COLOR_PRIMARY_HOVER,
        text_color: Tuple[int, int, int] = COLOR_TEXT,
        font_size: int = 13,
        border_radius: int = 6
    ):
        self.rect = rect
        self.text = text
        self.icon = icon
        self.on_click = on_click
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def draw(self, surface: pygame.Surface):
        color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(surface, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=self.border_radius)

        txt_surf = self.font.render(self.text, True, self.text_color)

        if self.icon and self.icon in ICON_DISPATCH:
            icon_fn = ICON_DISPATCH[self.icon]
            total_w = 16 + 6 + txt_surf.get_width()
            start_x = self.rect.centerx - total_w // 2
            icon_center = (start_x + 8, self.rect.centery)
            icon_fn(surface, icon_center, self.text_color, size=12)
            surface.blit(txt_surf, (start_x + 20, self.rect.centery - txt_surf.get_height() // 2))
        else:
            txt_rect = txt_surf.get_rect(center=self.rect.center)
            surface.blit(txt_surf, txt_rect)
