"""
NPC archetypes. They all share the `NPCBot` brain (haxball/bots/npc.py) and differ only by profile.

  PressBot     "Pressão Total"  - never stops pressing, shoots on free lanes, banks around you.
  StrikerBot   "Artilheiro"     - shoots early and hard from long range at the corners.
  BankBot      "Tabelador"      - wall specialist: bank shots and bank-pass fintas.
  DribblerBot  "Fintador"       - keeps the ball, swerves around you, shoots only when wide open.
  CounterBot   "Muralha"        - sits on the guard line, clears with walls, counter-attacks.
  MasterBot    "Mestre"         - everything on: press + free-lane shots + banks + fintas + team play.
  HeuristicBot "Clássico"       - baseline: presses and shoots, no walls, no fintas.
  WallReboundBot "Tabelas"      - baseline wall bot (kept for compatibility).
"""

from __future__ import annotations
from haxball.bots.npc import NPCBot, NPCProfile


class PressBot(NPCBot):
    def __init__(self, name: str = "PressBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=12.0, shoot_range=520.0, aim_tol_deg=14.0,
            bank_shots=True, bank_bonus=-25.0,
            finta=True, finta_trigger=100.0, finta_ahead=140.0, finta_cooldown=55,
            dribble_swerve_deg=55.0,
        ))


class StrikerBot(NPCBot):
    def __init__(self, name: str = "StrikerBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=3.0, shoot_range=760.0, aim_tol_deg=20.0, far_post_bias=0.35,
            bank_shots=False, finta=False,
            dribble_swerve_deg=30.0, dribble_avoid_radius=110.0,
        ))


class BankBot(NPCBot):
    def __init__(self, name: str = "BankBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=22.0, shoot_range=480.0, aim_tol_deg=12.0,
            bank_shots=True, bank_bonus=18.0,
            finta=True, finta_trigger=135.0, finta_ahead=170.0, finta_cooldown=40,
        ))


class DribblerBot(NPCBot):
    def __init__(self, name: str = "DribblerBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=26.0, shoot_range=400.0, aim_tol_deg=12.0,
            bank_shots=True, bank_bonus=-30.0,
            finta=True, finta_trigger=110.0, finta_ahead=120.0, finta_cooldown=35,
            dribble_swerve_deg=85.0, dribble_avoid_radius=185.0,
        ))


class CounterBot(NPCBot):
    def __init__(self, name: str = "CounterBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=10.0, shoot_range=640.0, aim_tol_deg=16.0, far_post_bias=0.25,
            bank_shots=True, bank_bonus=-5.0,
            finta=True, finta_trigger=90.0, finta_ahead=190.0,
            defensive=True, defend_depth=125.0, engage_radius=115.0,
        ))


class MasterBot(NPCBot):
    def __init__(self, name: str = "MasterBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=14.0, shoot_range=620.0, aim_tol_deg=10.0, far_post_bias=0.25,
            bank_shots=True, bank_bonus=-12.0,
            finta=True, finta_trigger=115.0, finta_ahead=150.0, finta_cooldown=45,
            dribble_swerve_deg=70.0, dribble_avoid_radius=170.0,
            team_play=True,
        ))


class HeuristicBot(NPCBot):
    """Classic baseline: presses, shoots when a lane is free, never uses walls."""

    def __init__(self, name: str = "HeuristicBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=6.0, shoot_range=520.0, aim_tol_deg=18.0,
            bank_shots=False, finta=False,
            dribble_swerve_deg=40.0, dribble_avoid_radius=130.0,
        ))


class WallReboundBot(NPCBot):
    """Baseline wall bot: banks whenever the direct lane is blocked."""

    def __init__(self, name: str = "WallReboundBot"):
        super().__init__(name, NPCProfile(
            shoot_clear=14.0, shoot_range=500.0, aim_tol_deg=14.0,
            bank_shots=True, bank_bonus=10.0,
            finta=False,
        ))
