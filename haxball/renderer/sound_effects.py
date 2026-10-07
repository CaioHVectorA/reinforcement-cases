"""
Authentic HaxBall Procedural Sound Engine.
Synthesizes classic 16-bit PCM sound effects using NumPy and PyGame Mixer:
- Kick thud/pop: punchy low-frequency kick impulse
- Wall / Post bounce: crisp metallic click
- Goal whistle & celebration: double-tone referee whistle + crowd horn
- Kickoff whistle: referee start whistle
100% self-contained, lag-free, zero external audio asset files needed.
"""

from __future__ import annotations
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class SoundManager:
    _instance = None

    def __init__(self):
        # Sound permanently disabled as requested by user
        self.enabled = False
        self.sounds = {}

    @classmethod
    def get_instance(cls) -> SoundManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _generate_sounds(self):
        sr = 44100

        # 1. Kick Sound: Punchy pitch drop (190Hz -> 45Hz) with quick exponential decay
        t_kick = np.linspace(0, 0.08, int(sr * 0.08), False)
        f_kick = 190.0 * np.exp(-t_kick * 35.0) + 45.0
        phase_kick = 2 * np.pi * np.cumsum(f_kick) / sr
        wave_kick = np.sin(phase_kick) * np.exp(-t_kick * 32.0)
        # Add slight click transient at onset
        click = np.random.uniform(-0.15, 0.15, len(t_kick)) * np.exp(-t_kick * 120.0)
        kick_pcm = ((wave_kick * 0.85 + click) * 32767).clip(-32768, 32767).astype(np.int16)
        stereo_kick = np.column_stack((kick_pcm, kick_pcm))
        self.sounds["kick"] = pygame.sndarray.make_sound(stereo_kick)

        # 2. Bounce / Post Sound: Crisp short tap (850Hz with sharp decay)
        t_bounce = np.linspace(0, 0.04, int(sr * 0.04), False)
        wave_bounce = (
            np.sin(2 * np.pi * 850 * t_bounce) * 0.6 +
            np.sin(2 * np.pi * 1700 * t_bounce) * 0.4
        ) * np.exp(-t_bounce * 90.0)
        bounce_pcm = (wave_bounce * 28000).clip(-32768, 32767).astype(np.int16)
        stereo_bounce = np.column_stack((bounce_pcm, bounce_pcm))
        self.sounds["bounce"] = pygame.sndarray.make_sound(stereo_bounce)

        # 3. Whistle Sound: Dual-tone referee whistle (2450Hz & 2700Hz with vibrato)
        t_whistle = np.linspace(0, 0.22, int(sr * 0.22), False)
        vib = 1.0 + 0.04 * np.sin(2 * np.pi * 28 * t_whistle)
        wave_w = (
            np.sin(2 * np.pi * 2450 * vib * t_whistle) * 0.5 +
            np.sin(2 * np.pi * 2720 * vib * t_whistle) * 0.5
        ) * np.exp(-t_whistle * 4.0)
        whistle_pcm = (wave_w * 24000).clip(-32768, 32767).astype(np.int16)
        stereo_whistle = np.column_stack((whistle_pcm, whistle_pcm))
        self.sounds["whistle"] = pygame.sndarray.make_sound(stereo_whistle)

        # 4. Goal Celebration: Whistle blast followed by deep stadium horn
        t_goal = np.linspace(0, 0.75, int(sr * 0.75), False)
        # Whistle part (first 0.25s)
        w_part = np.where(
            t_goal < 0.25,
            (np.sin(2 * np.pi * 2500 * t_goal) + np.sin(2 * np.pi * 2800 * t_goal)) * 0.4,
            0.0
        )
        # Horn part (resonant 220Hz + 440Hz stadium organ/horn)
        horn_sin = np.clip(np.sin(np.pi * (t_goal - 0.15) / 0.60), 0.0, 1.0)
        horn_env = np.where(t_goal >= 0.15, horn_sin ** 0.5, 0.0)
        horn = (
            np.sin(2 * np.pi * 220 * t_goal) * 0.45 +
            np.sin(2 * np.pi * 330 * t_goal) * 0.35 +
            np.sin(2 * np.pi * 440 * t_goal) * 0.20
        ) * horn_env
        goal_pcm = ((w_part + horn) * 29000).clip(-32768, 32767).astype(np.int16)
        stereo_goal = np.column_stack((goal_pcm, goal_pcm))
        self.sounds["goal"] = pygame.sndarray.make_sound(stereo_goal)

    def play_kick(self):
        if self.enabled and "kick" in self.sounds:
            self.sounds["kick"].play()

    def play_bounce(self):
        if self.enabled and "bounce" in self.sounds:
            self.sounds["bounce"].play()

    def play_goal(self):
        if self.enabled and "goal" in self.sounds:
            self.sounds["goal"].play()

    def play_whistle(self):
        if self.enabled and "whistle" in self.sounds:
            self.sounds["whistle"].play()
