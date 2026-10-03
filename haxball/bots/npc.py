"""
Unified NPC framework for HaxBall bots.

Every scripted bot is the *same* brain (`NPCBot`) driven by a different `NPCProfile`.
The brain implements the 1v1 doctrine:

  1. PRESS the ball all the time (intercept prediction, orbiting around the ball
     to get "behind" it instead of pushing it into the own goal).
  2. SHOOT only when there is a free lane (no opponent disc on the segment ball -> goal).
     The kick direction in HaxBall is always (ball - player), so the bot first gets
     behind the ball along the desired direction and only then taps the kick.
  3. With the ball and no free lane: DRIBBLE (carry the ball with the body, swerving
     away from the opponent) or play a WALL BANK ("tabela"): a shot/pass that rebounds
     on the top/bottom wall and goes around the opponent.

Bank shots use the *real* engine restitution (`e = ball.bCoef * wall.bCoef`): on a wall
hit the normal speed is scaled by `e` while the tangential speed is preserved, so the
rebound is NOT a perfect mirror. `Field.bank_point` solves the exact bounce point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.disc import Disc
from haxball.core.game import HaxBallGame
from haxball.bots.base_bot import BaseBot


# ----------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class NPCProfile:
    """Tunable personality of an NPC. All distances are in world pixels."""

    # --- shooting ---
    shoot_clear: float = 12.0        # min free-lane clearance (px) required to shoot
    shoot_range: float = 520.0       # max distance ball -> goal for a direct shot
    aim_tol_deg: float = 14.0        # max angular error between kick direction and plan
    far_post_bias: float = 0.15      # prefers corners of the goal over the center
    # --- walls ("tabelas") ---
    bank_shots: bool = True          # consider bank shots at goal
    bank_bonus: float = -25.0        # added to bank score (positive = prefers bank)
    finta: bool = True               # bank-pass around a nearby opponent
    finta_trigger: float = 100.0     # opponent closer than this triggers the finta
    finta_ahead: float = 140.0       # how far ahead the finta target is
    finta_cooldown: int = 55         # ticks between fintas
    # --- dribbling ---
    dribble_swerve_deg: float = 55.0
    dribble_avoid_radius: float = 150.0
    # --- positioning ---
    defensive: bool = False          # sit on the guard line until the ball is engaged
    defend_depth: float = 120.0
    engage_radius: float = 110.0
    team_play: bool = True           # only the closest teammate presses, others support


@dataclass
class _Plan:
    kind: str                        # "shoot" | "bank" | "finta" | "clear" | "dribble"
    direction: Vec2                  # desired ball direction right after the kick / carry
    aim: Vec2                        # point the direction is aiming at (for debug / hysteresis)
    score: float = 0.0
    kicks: bool = True               # False for dribble


# ----------------------------------------------------------------------------
# Geometry helpers (cached per stadium)
# ----------------------------------------------------------------------------
class Field:
    def __init__(self, game: HaxBallGame):
        stad = game.stadium
        ball = game.ball
        self.half_w: float = float(stad.bg_width)
        self.half_h: float = float(stad.bg_height)
        self.ball_r: float = float(ball.radius)
        wall_e = 1.0
        xlim = self.half_w
        for seg in stad.segments:
            if seg.is_curved:
                continue
            if abs(seg.p0.y - seg.p1.y) < 1e-6 and abs(abs(seg.p0.y) - self.half_h) < 1.0:
                wall_e = float(seg.bCoef)
                xlim = max(abs(seg.p0.x), abs(seg.p1.x))
                break
        self.wall_xlim: float = xlim
        self.e: float = float(ball.bCoef) * wall_e
        self.goal_half_h: float = max((abs(g.p0.y) for g in stad.goals), default=70.0)
        self.goals = {g.scoring_team: g for g in stad.goals}

    # --- goal geometry ---
    def goal_to_score(self, team: Team) -> Tuple[float, float, float]:
        """Returns (x, y_min, y_max) of the goal `team` scores in."""
        g = self.goals[team]
        return (g.p0.x + g.p1.x) * 0.5, min(g.p0.y, g.p1.y), max(g.p0.y, g.p1.y)

    def own_goal_center(self, team: Team) -> Vec2:
        other = Team.BLUE if team == Team.RED else Team.RED
        x, y0, y1 = self.goal_to_score(other)
        return Vec2(x, (y0 + y1) * 0.5)

    # --- ball prediction ---
    def predict_ball(self, ball: Disc, n: int = 70) -> List[Tuple[float, float]]:
        px, py = ball.pos.x, ball.pos.y
        vx, vy = ball.speed.x, ball.speed.y
        damp = ball.damping
        e = self.e
        lim = self.half_h - self.ball_r
        half_w = self.half_w
        gh = self.goal_half_h
        traj: List[Tuple[float, float]] = []
        for _ in range(n):
            vx *= damp
            vy *= damp
            px += vx
            py += vy
            if py > lim:
                py = 2 * lim - py
                vy = -vy * e
            elif py < -lim:
                py = -2 * lim - py
                vy = -vy * e
            if abs(px) > half_w and abs(py) > gh:
                px = math.copysign(2 * half_w - abs(px), px)
                vx = -vx * e
            traj.append((px, py))
        return traj

    @staticmethod
    def intercept(player: Disc, ball: Disc, traj: List[Tuple[float, float]]) -> Tuple[int, Vec2]:
        """First tick at which the player can reach the ball (conservative terminal speed)."""
        vmax = player.acceleration * player.damping / max(1e-6, 1.0 - player.damping) * 0.85
        reach = player.radius + ball.radius
        x0, y0 = player.pos.x, player.pos.y
        if math.hypot(x0 - ball.pos.x, y0 - ball.pos.y) - reach <= 0.0:
            return 0, ball.pos.copy()
        t = 0
        for qx, qy in traj:
            t += 1
            if math.hypot(x0 - qx, y0 - qy) - reach <= vmax * t:
                return t, Vec2(qx, qy)
        qx, qy = traj[-1]
        return len(traj), Vec2(qx, qy)

    # --- lanes ---
    @staticmethod
    def _seg_dist(a: Vec2, b: Vec2, p: Vec2) -> float:
        ab = b - a
        l2 = ab.length_sq()
        if l2 < 1e-9:
            return a.distance_to(p)
        t = max(0.0, min(1.0, (p - a).dot(ab) / l2))
        return (a + ab * t).distance_to(p)

    def lane_clearance(self, a: Vec2, b: Vec2, opps: List[Disc]) -> float:
        """Free space (px) between the ball path a->b and the nearest opponent (now and ~10 ticks ahead)."""
        best = 1e9
        for o in opps:
            r = o.radius + self.ball_r
            for pos in (o.pos, o.pos + o.speed * 10.0):
                best = min(best, self._seg_dist(a, b, pos) - r)
        return best

    # --- wall bank ---
    def bank_point(self, b: Vec2, t: Vec2, wall_sign: int) -> Optional[Vec2]:
        """
        Exact bounce point P on the wall (y = wall_sign * (H - ball_r)) so that a ball shot from `b`
        towards P rebounds (normal speed * e, tangential preserved) and passes through `t`.
        """
        w = wall_sign * (self.half_h - self.ball_r)
        db = (w - b.y) * wall_sign
        dt = (w - t.y) * wall_sign
        if db <= 1.0 or dt <= 1.0:
            return None
        k = self.e * db / dt
        px = (b.x + k * t.x) / (1.0 + k)
        if abs(px) > self.wall_xlim - 20.0:
            return None
        return Vec2(px, w)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ----------------------------------------------------------------------------
# The brain
# ----------------------------------------------------------------------------
class NPCBot(BaseBot):
    profile: NPCProfile = NPCProfile()

    def __init__(self, name: str = "NPCBot", profile: Optional[NPCProfile] = None):
        super().__init__(name=name)
        if profile is not None:
            self.profile = profile
        self._mem: Dict[int, dict] = {}
        self._fields: Dict[int, Field] = {}
        self.debug: Dict[int, dict] = {}

    # ---- BaseBot ----
    def reset(self):
        self._mem.clear()

    def act(self, game: HaxBallGame, player: Disc) -> Tuple[float, float, bool]:
        ball = game.ball
        if ball is None:
            return (0.0, 0.0, False)

        field = self._fields.get(id(game.stadium))
        if field is None or field.ball_r != float(ball.radius):
            field = Field(game)
            self._fields[id(game.stadium)] = field

        st = self._mem.setdefault(player.player_id, {"tick": 0, "finta_cd": 0, "aim": None, "kind": None})
        st["tick"] += 1
        if st["finta_cd"] > 0:
            st["finta_cd"] -= 1

        prof = self.profile
        attack = 1.0 if player.team == Team.RED else -1.0
        opps = [p for p in game.players if p.team != player.team]
        mates = [p for p in game.players if p.team == player.team and p is not player]

        traj = field.predict_ball(ball)
        t_me, q_me = field.intercept(player, ball, traj)
        own_goal = field.own_goal_center(player.team)

        # --- kickoff for the other team: hold a defensive spot (barrier blocks us anyway) ---
        if game.state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE):
            red_turn = (game.state == GameState.KICKOFF_RED)
            if red_turn != (player.team == Team.RED):
                spot = Vec2(own_goal.x * 0.55, 0.0)
                return self._goto(player, spot, brake=6.0) + (False,)

        # --- team role: only the quickest teammate presses ---
        if mates and prof.team_play:
            t_mates = [field.intercept(m, ball, traj)[0] for m in mates]
            if min(t_mates) + 3 < t_me:
                return self._support(game, field, player, ball, opps, attack, own_goal) + (False,)

        # --- defensive profiles hold the guard line until the ball is engaged ---
        in_own_half = (ball.pos.x * attack) < 0
        if prof.defensive:
            t_opp = min((field.intercept(o, ball, traj)[0] for o in opps), default=999)
            near = player.pos.distance_to(ball.pos) < prof.engage_radius
            danger_zone = (ball.pos.x - own_goal.x) * attack < field.half_w * 0.7
            if not (near or danger_zone or t_me + 4 < t_opp):
                guard = self._guard_point(field, own_goal, ball, prof.defend_depth)
                return self._goto(player, guard, brake=8.0) + (False,)

        # --- ATTACK / PRESS: plan, then execute ---
        contact_ball = q_me if t_me <= 40 else ball.pos
        plan = self._plan(field, player, ball, contact_ball, opps, attack, st, prof, defending=prof.defensive and in_own_half)
        move, kick = self._execute(field, player, ball, contact_ball, plan, prof)
        self.debug[player.player_id] = {"kind": plan.kind, "aim": plan.aim.to_tuple()}
        return (move[0], move[1], kick)

    # ------------------------------------------------------------------ planning
    def _plan(self, f: Field, me: Disc, ball: Disc, b: Vec2, opps: List[Disc], attack: float,
              st: dict, prof: NPCProfile, defending: bool) -> _Plan:
        team = me.team
        gx, gy0, gy1 = f.goal_to_score(team)
        cy = (gy0 + gy1) * 0.5
        ymin = gy0 + f.ball_r + 9.0
        ymax = gy1 - f.ball_r - 9.0
        goal_pts = [Vec2(gx, ymin + (ymax - ymin) * i / 6.0) for i in range(7)]
        dist_goal = b.distance_to(Vec2(gx, cy))

        cands: List[_Plan] = []

        # 1) direct shots
        if dist_goal <= prof.shoot_range:
            for t in goal_pts:
                d = (t - b).normalized()
                if d.x * attack < 0.15:
                    continue
                clr = f.lane_clearance(b, t, opps)
                if clr >= prof.shoot_clear:
                    cands.append(_Plan("shoot", d, t, clr + prof.far_post_bias * abs(t.y - cy)))

        # 2) bank shots at goal
        if prof.bank_shots and dist_goal <= prof.shoot_range * 1.35:
            for ws in (1, -1):
                for t in goal_pts[1:-1:2]:
                    p = f.bank_point(b, t, ws)
                    if p is None:
                        continue
                    d = (p - b).normalized()
                    if d.x * attack < -0.2 or b.distance_to(p) < 40.0:
                        continue
                    clr = min(f.lane_clearance(b, p, opps), f.lane_clearance(p, t, opps))
                    if clr >= prof.shoot_clear * 0.8:
                        length = b.distance_to(p) + p.distance_to(t)
                        cands.append(_Plan("bank", d, p, clr + prof.far_post_bias * abs(t.y - cy)
                                           + prof.bank_bonus - 0.02 * length))

        # 3) finta: bank-pass around a close opponent (or a clearance when defending)
        close_opp = any(o.pos.distance_to(b) < prof.finta_trigger or o.pos.distance_to(me.pos) < prof.finta_trigger
                        for o in opps)
        if (prof.finta and close_opp and st["finta_cd"] == 0) or defending:
            ahead_x = max(-f.half_w + 80.0, min(f.half_w - 80.0, b.x + attack * prof.finta_ahead))
            best = None
            for ws in (1, -1):
                for ty in (-0.5, -0.25, 0.0, 0.25, 0.5):
                    t = Vec2(ahead_x, ty * f.half_h)
                    p = f.bank_point(b, t, ws)
                    if p is None or b.distance_to(p) < 40.0:
                        continue
                    d = (p - b).normalized()
                    if d.x * attack < -0.35:
                        continue
                    clr = min(f.lane_clearance(b, p, opps), f.lane_clearance(p, t, opps))
                    if clr < 18.0:
                        continue
                    end_gap = min((o.pos.distance_to(t) for o in opps), default=200.0)
                    score = clr + 0.1 * end_gap - 0.03 * (b.distance_to(p) + p.distance_to(t))
                    if best is None or score > best.score:
                        best = _Plan("finta", d, p, score)
            if best is not None:
                best.score -= 5.0  # a real shot is always preferred over a pass
                cands.append(best)

        # 4) pick best candidate with hysteresis (avoid flip-flopping every tick)
        if cands:
            best = max(cands, key=lambda c: c.score)
            last = st.get("aim")
            if last is not None:
                for c in cands:
                    if c.kind == st.get("kind") and c.aim.distance_to(last) < 30.0 and c.score >= best.score - 20.0:
                        best = c
                        break
            st["aim"], st["kind"] = best.aim, best.kind
            return best

        st["aim"], st["kind"] = None, None

        # 5) fallbacks
        if defending:
            side = -1.0 if (opps and opps[0].pos.y > b.y) else 1.0
            d = Vec2(attack * 0.85, side * 0.5).normalized()
            return _Plan("clear", d, b + d * 200.0)

        # dribble: carry the ball toward goal swerving away from opponents
        heading = (Vec2(gx, cy) - b).normalized()
        for o in opps:
            rel = o.pos - b
            dist = rel.length()
            if dist < prof.dribble_avoid_radius and rel.dot(heading) > -10.0:
                side = -1.0 if heading.cross(rel) > 0 else 1.0
                ang = math.radians(prof.dribble_swerve_deg) * (1.0 - dist / prof.dribble_avoid_radius)
                heading = heading.rotate(side * ang)
        # slide along a wall instead of pushing the ball into it
        lim = f.half_h - f.ball_r - 18.0
        if (b.y > lim and heading.y > 0) or (b.y < -lim and heading.y < 0):
            heading = Vec2(heading.x, -heading.y * 0.25).normalized()
        if heading.x * attack < 0.1:
            heading = Vec2(attack * 0.1, heading.y).normalized()
        return _Plan("dribble", heading, b + heading * 100.0, kicks=False)

    # ------------------------------------------------------------------ execution
    def _execute(self, f: Field, me: Disc, ball: Disc, b: Vec2, plan: _Plan, prof: NPCProfile
                 ) -> Tuple[Tuple[float, float], bool]:
        d = plan.direction
        contact = b - d * (me.radius + ball.radius + 2.0)
        clear = me.radius + ball.radius + 14.0
        wp = self._waypoint(me.pos, contact, b, clear)

        dist_ball = me.pos.distance_to(ball.pos)
        reach = me.radius + ball.radius + me.kick_margin
        to_ball = (ball.pos - me.pos).normalized()
        aligned = to_ball.dot(d) >= math.cos(math.radians(prof.aim_tol_deg))

        if plan.kicks and dist_ball <= reach - 0.5 and aligned:
            return (to_ball.x, to_ball.y), True

        if wp is contact and me.pos.distance_to(contact) < 28.0:
            drive = (b + d * 16.0 - me.pos).normalized()
            return (drive.x, drive.y), False

        mv = (wp - me.pos).normalized()
        return (mv.x, mv.y), False

    @staticmethod
    def _waypoint(p: Vec2, contact: Vec2, b: Vec2, clear: float) -> Vec2:
        """Straight to the contact point if we are behind the ball, else orbit around the ball."""
        rp = p - b
        dist = rp.length()
        if dist < 1e-6:
            return contact
        ap = math.atan2(rp.y, rp.x)
        ac = math.atan2((contact - b).y, (contact - b).x)
        diff = _wrap(ac - ap)
        if abs(diff) < math.radians(28.0):
            return contact
        step = math.copysign(min(abs(diff), math.radians(55.0)), diff)
        r = clear + 8.0
        return b + Vec2(math.cos(ap + step), math.sin(ap + step)) * r

    @staticmethod
    def _goto(me: Disc, target: Vec2, brake: float = 6.0) -> Tuple[float, float]:
        v = target - me.pos
        if v.length() < brake:
            # actively brake: oppose current velocity
            stop = -me.speed
            n = stop.normalized() if stop.length_sq() > 0.04 else Vec2(0, 0)
            return (n.x, n.y)
        n = v.normalized()
        return (n.x, n.y)

    @staticmethod
    def _guard_point(f: Field, own_goal: Vec2, ball: Disc, depth: float) -> Vec2:
        v = (ball.pos - own_goal)
        dist = v.length()
        d = v.normalized() if dist > 1e-6 else Vec2(1, 0)
        dd = min(depth, dist * 0.55)
        p = own_goal + d * dd
        lim = f.goal_half_h + 35.0
        return Vec2(p.x, max(-lim, min(lim, p.y)))

    def _support(self, game, f: Field, me: Disc, ball: Disc, opps, attack: float, own_goal: Vec2):
        if ball.pos.x * attack > 0:
            side = -1.0 if ball.pos.y > 0 else 1.0
            target = Vec2(ball.pos.x - attack * 90.0, side * f.half_h * 0.45)
        else:
            target = self._guard_point(f, own_goal, ball, 150.0)
        return self._goto(me, target, brake=8.0)
