"""
UI Widget components for HaxBall: buttons, panels, tabs, modals, and indicators.
Styled with authentic HaxBall dark theme aesthetics.
"""

from __future__ import annotations
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
COLOR_TEXT = (235, 240, 245)
COLOR_TEXT_MUTED = (160, 170, 185)
COLOR_BORDER = (55, 68, 85)

class UIButton:
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        on_click: Optional[Callable[[], None]] = None,
        bg_color: Tuple[int, int, int] = COLOR_PANEL_LIGHT,
        hover_color: Tuple[int, int, int] = COLOR_PRIMARY_HOVER,
        text_color: Tuple[int, int, int] = COLOR_TEXT,
        font_size: int = 15,
        border_radius: int = 6
    ):
        self.rect = rect
        self.text = text
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
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)

class UITabBar:
    def __init__(
        self,
        rect: pygame.Rect,
        tabs: List[str],
        on_change: Optional[Callable[[int], None]] = None,
        active_idx: int = 0
    ):
        self.rect = rect
        self.tabs = tabs
        self.on_change = on_change
        self.active_idx = active_idx
        self.font = pygame.font.SysFont("Arial", 15, bold=True)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                tab_w = self.rect.width / len(self.tabs)
                idx = int((event.pos[0] - self.rect.x) // tab_w)
                if 0 <= idx < len(self.tabs) and idx != self.active_idx:
                    self.active_idx = idx
                    if self.on_change:
                        self.on_change(idx)
                    return True
        return False

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, COLOR_PANEL, self.rect, border_radius=8)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, width=1, border_radius=8)

        tab_w = self.rect.width / len(self.tabs)
        for i, tab in enumerate(self.tabs):
            tab_r = pygame.Rect(int(self.rect.x + i * tab_w), self.rect.y, int(tab_w), self.rect.height)
            is_active = (i == self.active_idx)

            if is_active:
                pygame.draw.rect(surface, COLOR_PRIMARY, tab_r, border_radius=8)
                color = (255, 255, 255)
            else:
                color = COLOR_TEXT_MUTED

            txt = self.font.render(tab, True, color)
            txt_r = txt.get_rect(center=tab_r.center)
            surface.blit(txt, txt_r)
