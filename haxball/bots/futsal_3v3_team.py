"""
Futsal 3v3 Team Coordination and Tactical Specialization Framework.

Implements high-fidelity, competitive 3v3 Futsal doctrine:
1. Dynamic Role Allocation (Presser, Fixo/Anchor, Ala/Winger):
   - The teammate closest to the ball is dynamically designated as PRESSER / BALL HANDLER.
   - The teammate positioned deepest toward own goal becomes FIXO (Anchor / Defensive Guardian).
   - The third teammate becomes ALA (Support Winger / Far-Post Target).
2. Tactical Triangulation & Anti-Clustering:
   - Solves the 'kindergarten soccer' dilemma: teammates never chase the same ball.
   - Ala opens on the opposite flank, creating diagonal passing angles.
   - Fixo covers the defensive corridor, preventing breakaways and empty-net counter-attacks.
3. Offensive Passing & Rebounds:
   - When the direct shooting lane is blocked, the Presser seeks passes to the open Ala or executes wall banks.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot
from haxball.bots.npc import Field, _Plan


class Futsal3v3Coordinator:
    """
    Coordinates tactical decision-making for a 3-player Futsal team.
    Synchronizes assignments across players in the same simulation tick.
    Includes humanized action smoothing, commitment inertia, and role specialization.
    """
    def __init__(self, team: Team):
        self.team = team
        self._last_tick: int = -1
        self._cached_actions: Dict[int, Tuple[float, float, bool]] = {}
        self._field_cache: Optional[Field] = None

        # Humanization: action commitment and exponential momentum smoothing
        self._prev_moves: Dict[int, Tuple[float, float]] = {}
        self._commit_ticks: Dict[int, int] = {}
        self._kick_cooldown: Dict[int, int] = {}

        # Role hysteresis (prevents tick-by-tick role swapping)
        self._assigned_presser_id: Optional[int] = None
        self._assigned_fixo_id: Optional[int] = None
        self._assigned_ala_id: Optional[int] = None

    def get_action(self, game: HaxBallGame, player: Disc, role: Optional[str] = None) -> Tuple[float, float, bool]:
        curr_tick = game.time_ticks
        if curr_tick != self._last_tick:
            self._update_team_plan(game)
            self._last_tick = curr_tick

        # If a fixed role is explicitly requested for this player
        if role is not None and role in ("fixo", "ala", "press", "pivo"):
            raw_act = self._compute_specialized_action(game, player, role)
        else:
            raw_act = self._cached_actions.get(player.player_id, (0.0, 0.0, False))

        # Apply Humanized Inertia & Smoothing Filter
        return self._apply_human_smoothing(player.player_id, raw_act)

    def _apply_human_smoothing(self, pid: int, raw_act: Tuple[float, float, bool]) -> Tuple[float, float, bool]:
        rx, ry, rkick = raw_act

        # 1. Kick smoothing: human tap lasts 2-3 ticks, then brief cooldown
        prev_move = self._prev_moves.get(pid, (0.0, 0.0))
        commit = self._commit_ticks.get(pid, 0)
        cooldown = self._kick_cooldown.get(pid, 0)

        kick_out = False
        if rkick and cooldown <= 0:
            kick_out = True
            self._kick_cooldown[pid] = 4
        elif cooldown > 0:
            self._kick_cooldown[pid] = cooldown - 1
            if cooldown >= 2:
                kick_out = True

        # 2. Movement smoothing with human reaction inertia
        # If in a committed sprint and target direction is similar, keep momentum
        target_len = math.hypot(rx, ry)
        if target_len < 0.1:
            # Player is stopping
            smooth_x = prev_move[0] * 0.7
            smooth_y = prev_move[1] * 0.7
            self._commit_ticks[pid] = 0
        else:
            # Exponential moving average filter (alpha = 0.35)
            # Simulates human finger holding key rather than instant 60Hz discrete flutter
            alpha = 0.35
            smooth_x = prev_move[0] * (1.0 - alpha) + rx * alpha
            smooth_y = prev_move[1] * (1.0 - alpha) + ry * alpha

            # Normalize to clean unit vector if running
            sm_len = math.hypot(smooth_x, smooth_y)
            if sm_len > 0.15:
                smooth_x /= sm_len
                smooth_y /= sm_len

        self._prev_moves[pid] = (smooth_x, smooth_y)
        return (smooth_x, smooth_y, kick_out)

    def _compute_specialized_action(self, game: HaxBallGame, me: Disc, role: str) -> Tuple[float, float, bool]:
        if self._field_cache is None or self._field_cache.half_w != float(game.stadium.bg_width):
            self._field_cache = Field(game)
        f = self._field_cache
        ball = game.ball
        attack_dir = 1.0 if self.team == Team.RED else -1.0
        own_goal = f.own_goal_center(self.team)
        enemy_goal_x, gy0, gy1 = f.goal_to_score(self.team)
        enemy_goal_y = (gy0 + gy1) * 0.5
        opponents = [p for p in game.players if p.team != self.team and p.team != Team.NONE]
        teammates = [p for p in game.players if p.team == self.team and p.player_id != me.player_id]

        if role == "fixo":
            presser = min(teammates, key=lambda p: p.pos.distance_to(ball.pos)) if teammates else me
            return self._compute_fixo_action(f, me, ball, presser, opponents, attack_dir, own_goal)
        elif role == "ala":
            presser = min(teammates, key=lambda p: p.pos.distance_to(ball.pos)) if teammates else me
            fixo = max(teammates, key=lambda p: -p.pos.x * attack_dir) if teammates else me
            return self._compute_ala_action(f, me, ball, presser, fixo, opponents, attack_dir, enemy_goal_x, enemy_goal_y)
        else:
            # Atacante / Presser
            ala = teammates[0] if teammates else None
            return self._compute_presser_action(f, me, ball, opponents, ala, attack_dir, enemy_goal_x, enemy_goal_y)

    def _update_team_plan(self, game: HaxBallGame):
        self._cached_actions.clear()
        if self._field_cache is None or self._field_cache.half_w != float(game.stadium.bg_width):
            self._field_cache = Field(game)

        f = self._field_cache
        ball = game.ball
        attack_dir = 1.0 if self.team == Team.RED else -1.0
        own_goal = f.own_goal_center(self.team)
        enemy_goal_x, gy0, gy1 = f.goal_to_score(self.team)
        enemy_goal_y = (gy0 + gy1) * 0.5

        teammates = [p for p in game.players if p.team == self.team]
        opponents = [p for p in game.players if p.team != self.team and p.team != Team.NONE]

        if not teammates:
            return

        # Role Assignment with Hysteresis (prevents jitter when 2 players are close to ball)
        teammates_by_dist = sorted(teammates, key=lambda p: p.pos.distance_to(ball.pos))
        closest_p = teammates_by_dist[0]

        # Keep current presser unless another teammate is significantly closer (threshold 40px)
        curr_presser = next((p for p in teammates if p.player_id == self._assigned_presser_id), None)
        if curr_presser is not None:
            curr_dist = curr_presser.pos.distance_to(ball.pos)
            best_dist = closest_p.pos.distance_to(ball.pos)
            if best_dist < curr_dist - 40.0:
                presser = closest_p
            else:
                presser = curr_presser
        else:
            presser = closest_p

        self._assigned_presser_id = presser.player_id

        remaining = [p for p in teammates if p.player_id != presser.player_id]
        fixo = None
        ala = None

        if len(remaining) == 1:
            fixo = remaining[0]
        elif len(remaining) >= 2:
            if self.team == Team.RED:
                fixo = min(remaining, key=lambda p: p.pos.x)
            else:
                fixo = max(remaining, key=lambda p: p.pos.x)
            ala = [p for p in remaining if p.player_id != fixo.player_id][0]

        # 2. Compute Individual Actions
        # A) PRESSER / BALL HANDLER ACTION
        presser_act = self._compute_presser_action(f, presser, ball, opponents, ala, attack_dir, enemy_goal_x, enemy_goal_y)
        self._cached_actions[presser.player_id] = presser_act

        # B) FIXO (ANCHOR / DEFENSIVE COVERAGE)
        if fixo is not None:
            fixo_act = self._compute_fixo_action(f, fixo, ball, presser, opponents, attack_dir, own_goal)
            self._cached_actions[fixo.player_id] = fixo_act

        # C) ALA (WING SUPPORT / FAR-POST TARGET)
        if ala is not None:
            ala_act = self._compute_ala_action(f, ala, ball, presser, fixo, opponents, attack_dir, enemy_goal_x, enemy_goal_y)
            self._cached_actions[ala.player_id] = ala_act


    def _compute_presser_action(
        self, f: Field, me: Disc, ball: Disc, opponents: List[Disc], ala: Optional[Disc],
        attack: float, gx: float, gy: float
    ) -> Tuple[float, float, bool]:
        b = ball.pos
        p = me.pos
        dist_to_ball = p.distance_to(b)
        reach = me.radius + ball.radius + me.kick_margin

        # Check if direct shot to goal is clear
        target_goal = Vec2(gx, gy)
        to_goal = (target_goal - b).normalized()
        shot_clear = True
        for opp in opponents:
            # Check projection onto shot line
            rel = opp.pos - b
            proj = rel.dot(to_goal)
            if 0 < proj < (target_goal - b).length():
                perp = (rel - to_goal * proj).length()
                if perp < opp.radius + ball.radius + 15.0:
                    shot_clear = False
                    break

        # If aligned and in reach, shoot!
        to_ball = (b - p).normalized()
        aligned = to_ball.dot(to_goal) >= math.cos(math.radians(18.0))
        if shot_clear and dist_to_ball <= reach and aligned:
            return (to_ball.x, to_ball.y, True)

        # Passing lane to Ala if Ala is open forward
        if ala is not None and not shot_clear:
            to_ala = (ala.pos - b).normalized()
            pass_dist = b.distance_to(ala.pos)
            if pass_dist > 80.0 and (ala.pos.x - b.x) * attack > 15.0:
                pass_clear = True
                for opp in opponents:
                    rel_opp = opp.pos - b
                    proj = rel_opp.dot(to_ala)
                    if 0 < proj < pass_dist:
                        perp = (rel_opp - to_ala * proj).length()
                        if perp < opp.radius + ball.radius + 18.0:
                            pass_clear = False
                            break
                if pass_clear and dist_to_ball <= reach and to_ball.dot(to_ala) >= math.cos(math.radians(22.0)):
                    return (to_ball.x, to_ball.y, True)

        # Wall bank ("tabela na parede") if direct shot blocked
        lim_y = f.half_h - f.ball_r
        near_wall_y = lim_y - 20.0 if b.y > 0 else -lim_y + 20.0
        bank_dir = Vec2(attack * 0.8, (1.0 if b.y < 0 else -1.0) * 0.6).normalized()
        if dist_to_ball <= reach and to_ball.dot(bank_dir) >= math.cos(math.radians(20.0)):
            return (to_ball.x, to_ball.y, True)

        # Intercept / Orbit behind ball toward attack
        desired_heading = to_goal if shot_clear else bank_dir
        contact_point = b - desired_heading * (me.radius + ball.radius + 4.0)

        # Check if already behind the ball
        rel_contact = contact_point - p
        if rel_contact.length() > 6.0:
            mv = rel_contact.normalized()
            return (mv.x, mv.y, False)

        drive = (b - p).normalized()
        return (drive.x, drive.y, False)

    def _compute_fixo_action(
        self, f: Field, me: Disc, ball: Disc, presser: Disc, opponents: List[Disc],
        attack: float, own_goal: Vec2
    ) -> Tuple[float, float, bool]:
        b = ball.pos
        p = me.pos

        # Determine target defensive anchor position
        # Bisector between ball and own goal
        v_ball = b - own_goal
        dist_ball_to_goal = v_ball.length()
        dir_to_ball = v_ball.normalized() if dist_ball_to_goal > 1e-4 else Vec2(attack, 0)

        # Anchor depth: stays comfortably in front of own goal
        # If ball is deep in enemy half, anchor steps up to mid-field (~ -0.2 * half_w)
        ball_in_attack = b.x * attack > 0
        if ball_in_attack:
            anchor_dist = min(f.half_w * 0.65, dist_ball_to_goal * 0.45)
        else:
            anchor_dist = min(f.half_w * 0.35, dist_ball_to_goal * 0.35)

        target_anchor = own_goal + dir_to_ball * anchor_dist
        # Clamp within corridor
        lim_y = f.goal_half_h + 30.0
        target_anchor = Vec2(target_anchor.x, max(-lim_y, min(lim_y, target_anchor.y)))

        # If ball is extremely close and loose, fixo clears firmly!
        dist_to_ball = p.distance_to(b)
        reach = me.radius + ball.radius + me.kick_margin
        if dist_to_ball < 65.0 and dist_to_ball < presser.pos.distance_to(b):
            # Clear away down the flank
            clear_dir = Vec2(attack * 0.85, (1.0 if b.y < 0 else -1.0) * 0.5).normalized()
            to_ball = (b - p).normalized()
            if dist_to_ball <= reach:
                return (to_ball.x, to_ball.y, True)
            mv = (b - p).normalized()
            return (mv.x, mv.y, False)

        # Move to target anchor position
        diff = target_anchor - p
        if diff.length() < 6.0:
            # Brake
            stop = -me.speed
            n = stop.normalized() if stop.length_sq() > 0.04 else Vec2(0, 0)
            return (n.x, n.y, False)

        mv = diff.normalized()
        return (mv.x, mv.y, False)

    def _compute_ala_action(
        self, f: Field, me: Disc, ball: Disc, presser: Disc, fixo: Disc,
        opponents: List[Disc], attack: float, gx: float, gy: float
    ) -> Tuple[float, float, bool]:
        b = ball.pos
        p = me.pos

        # Ala operates on the opposite flank of the presser to provide width
        presser_flank = 1.0 if presser.pos.y > 0 else -1.0
        ala_target_y = -presser_flank * (f.half_h * 0.55)

        # In attack: push forward into the attacking half anticipating cross / far-post
        ball_in_attack = b.x * attack > 0
        if ball_in_attack:
            ala_target_x = b.x + attack * min(150.0, f.half_w * 0.25)
        else:
            # In defense: support midfield
            ala_target_x = max(-f.half_w * 0.35, min(f.half_w * 0.35, b.x - attack * 60.0))

        ala_target_x = max(-f.half_w + 60.0, min(f.half_w - 60.0, ala_target_x))
        target_pos = Vec2(ala_target_x, ala_target_y)

        # If ball rebounds directly to Ala, shoot or pass!
        dist_to_ball = p.distance_to(b)
        reach = me.radius + ball.radius + me.kick_margin
        if dist_to_ball < 60.0:
            to_goal = (Vec2(gx, gy) - b).normalized()
            to_ball = (b - p).normalized()
            if dist_to_ball <= reach:
                return (to_ball.x, to_ball.y, True)
            return (to_ball.x, to_ball.y, False)

        # Move to support position
        diff = target_pos - p
        if diff.length() < 8.0:
            stop = -me.speed
            n = stop.normalized() if stop.length_sq() > 0.04 else Vec2(0, 0)
            return (n.x, n.y, False)

        mv = diff.normalized()
        return (mv.x, mv.y, False)


class Futsal3v3Bot(BaseBot):
    """
    Individual bot participant in a coordinated Futsal 3v3 team.
    Shares a tactical coordinator instance with its teammates.
    Can be assigned an explicit role ("fixo", "ala", "press") or dynamically coordinated.
    """
    def __init__(self, name: str = "Futsal3v3Bot", coordinator: Optional[Futsal3v3Coordinator] = None, role: Optional[str] = None):
        super().__init__(name)
        self.coordinator = coordinator
        self.role = role

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        if self.coordinator is None:
            # Fallback coordinator for player's team
            self.coordinator = Futsal3v3Coordinator(player.team)
        return self.coordinator.get_action(game, player, role=self.role)

    def reset(self):
        if self.coordinator is not None:
            self.coordinator._last_tick = -1
            self.coordinator._cached_actions.clear()
            self.coordinator._prev_moves.clear()
            self.coordinator._commit_ticks.clear()
