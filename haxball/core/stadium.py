"""
Stadium parser and data model for HaxBall (.hbs files).
Supports JSON5 (comments, trailing commas), traits resolution, custom physics,
goals, segments, discs, and planes.
Faithful to official HaxBall stadium specifications.
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Any, Optional
from haxball.core.vector import Vec2
from haxball.core.constants import (
    CollisionMask, Team, parse_collision_flags,
    DEFAULT_PLAYER_PHYSICS, DEFAULT_BALL_PHYSICS
)
from haxball.core.disc import Disc
from haxball.core.segment import Vertex, Segment

def strip_json_comments_and_trailing_commas(text: str) -> str:
    """Cleans JSON5-style Haxball stadium text to standard JSON."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text

class Goal:
    __slots__ = ("p0", "p1", "team", "scoring_team")

    def __init__(self, p0: Vec2, p1: Vec2, team_str: str):
        self.p0 = p0
        self.p1 = p1
        t_lower = team_str.lower()
        # In HaxBall standard convention:
        # A goal assigned team="red" belongs to the Red team (defended by Red).
        # Therefore, when the ball enters it, the BLUE team scores!
        # Vice-versa: goal assigned team="blue" is defended by Blue, so RED scores!
        if t_lower == "red":
            self.team = Team.RED
            self.scoring_team = Team.BLUE
        elif t_lower == "blue":
            self.team = Team.BLUE
            self.scoring_team = Team.RED
        else:
            self.team = Team.NONE
            # Fallback based on X coordinate
            mid_x = (p0.x + p1.x) * 0.5
            if mid_x < 0:
                self.team = Team.RED
                self.scoring_team = Team.BLUE
            else:
                self.team = Team.BLUE
                self.scoring_team = Team.RED

    def check_crossed(self, prev_pos: Vec2, curr_pos: Vec2) -> bool:
        """
        Checks if line segment (prev_pos -> curr_pos) intersects the goal line (p0 -> p1).
        Uses robust 2D cross product line-segment intersection.
        """
        d_ball = curr_pos - prev_pos
        d_goal = self.p1 - self.p0

        denom = d_ball.cross(d_goal)
        if abs(denom) < 1e-9:
            return False

        diff = self.p0 - prev_pos
        t_ball = diff.cross(d_goal) / denom
        t_goal = diff.cross(d_ball) / denom

        return (0.0 <= t_ball <= 1.0) and (0.0 <= t_goal <= 1.0)

class Plane:
    __slots__ = ("normal", "dist", "bCoef", "cMask", "cGroup")

    def __init__(
        self,
        normal: Vec2,
        dist: float,
        bCoef: float = 1.0,
        cMask: int = CollisionMask.ALL,
        cGroup: int = CollisionMask.WALL
    ):
        self.normal = normal.normalized()
        self.dist = float(dist)
        self.bCoef = float(bCoef)
        self.cMask = cMask
        self.cGroup = cGroup

class Stadium:
    def __init__(self):
        self.name: str = "Classic"
        self.width: float = 550.0
        self.height: float = 240.0
        self.spawn_distance: float = 200.0

        self.bg_type: str = "grass"
        self.bg_width: float = 420.0
        self.bg_height: float = 200.0
        self.bg_kickoff_radius: float = 75.0
        self.bg_corner_radius: float = 0.0
        self.bg_color: str = "718C5A"

        self.player_physics = dict(DEFAULT_PLAYER_PHYSICS)
        self.ball_physics = dict(DEFAULT_BALL_PHYSICS)

        self.traits: Dict[str, Dict[str, Any]] = {}
        self.vertexes: List[Vertex] = []
        self.segments: List[Segment] = []
        self.goals: List[Goal] = []
        self.discs: List[Disc] = []
        self.planes: List[Plane] = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stadium:
        stad = cls()
        stad.name = data.get("name", "Custom Stadium")
        stad.width = float(data.get("width", 550))
        stad.height = float(data.get("height", 240))
        stad.spawn_distance = float(data.get("spawnDistance", 200))

        bg = data.get("bg", {})
        stad.bg_type = bg.get("type", "grass")
        stad.bg_width = float(bg.get("width", 420))
        stad.bg_height = float(bg.get("height", 200))
        stad.bg_kickoff_radius = float(bg.get("kickOffRadius", 75))
        stad.bg_corner_radius = float(bg.get("cornerRadius", 0))
        stad.bg_color = bg.get("color", "718C5A" if stad.bg_type == "grass" else "C49458")

        stad.traits = data.get("traits", {})

        if "playerPhysics" in data:
            for k, v in data["playerPhysics"].items():
                stad.player_physics[k] = v

        if "ballPhysics" in data:
            for k, v in data["ballPhysics"].items():
                stad.ball_physics[k] = v

        def apply_trait(item: Dict[str, Any]) -> Dict[str, Any]:
            trait_name = item.get("trait")
            merged = {}
            if trait_name and trait_name in stad.traits:
                merged.update(stad.traits[trait_name])
            merged.update(item)
            return merged

        # Vertexes
        raw_vertexes = data.get("vertexes", [])
        for raw_v in raw_vertexes:
            v_data = apply_trait(raw_v)
            pos = Vec2(v_data.get("x", 0.0), v_data.get("y", 0.0))
            b_coef = float(v_data.get("bCoef", 1.0))
            c_mask = parse_collision_flags(v_data.get("cMask", ["all"]))
            c_group = parse_collision_flags(v_data.get("cGroup", ["wall"]))
            stad.vertexes.append(Vertex(pos, b_coef, c_mask, c_group, trait=raw_v.get("trait", "")))

        # Segments
        raw_segments = data.get("segments", [])
        for raw_s in raw_segments:
            s_data = apply_trait(raw_s)
            v0_idx = s_data.get("v0", 0)
            v1_idx = s_data.get("v1", 0)
            if v0_idx < len(stad.vertexes) and v1_idx < len(stad.vertexes):
                p0 = stad.vertexes[v0_idx].pos
                p1 = stad.vertexes[v1_idx].pos
                b_coef = float(s_data.get("bCoef", 1.0))
                c_mask = parse_collision_flags(s_data.get("cMask", ["all"]))
                c_group = parse_collision_flags(s_data.get("cGroup", ["wall"]))
                curve = float(s_data.get("curve", 0.0))
                bias = float(s_data.get("bias", 0.0))
                vis = bool(s_data.get("vis", True))
                color = s_data.get("color", "C7E59B")
                stad.segments.append(
                    Segment(p0, p1, b_coef, c_mask, c_group, curve, bias, vis, color, trait=raw_s.get("trait", ""))
                )

        # Goals
        raw_goals = data.get("goals", [])
        for g in raw_goals:
            p0 = Vec2.from_iterable(g["p0"])
            p1 = Vec2.from_iterable(g["p1"])
            team_str = g.get("team", "")
            stad.goals.append(Goal(p0, p1, team_str))

        # Discs
        raw_discs = data.get("discs", [])
        for raw_d in raw_discs:
            d_data = apply_trait(raw_d)
            pos = Vec2.from_iterable(d_data.get("pos", [0, 0]))
            speed = Vec2.from_iterable(d_data.get("speed", [0, 0]))
            radius = float(d_data.get("radius", 10.0))
            b_coef = float(d_data.get("bCoef", 0.5))
            inv_mass = float(d_data.get("invMass", 0.0))
            damping = float(d_data.get("damping", 0.99))
            color = d_data.get("color", "FFFFFF")
            c_mask = parse_collision_flags(d_data.get("cMask", ["all"]))
            c_group = parse_collision_flags(d_data.get("cGroup", ["wall"]))
            stad.discs.append(
                Disc(
                    pos=pos, speed=speed, radius=radius, bCoef=b_coef,
                    invMass=inv_mass, damping=damping, color=color,
                    cMask=c_mask, cGroup=c_group, trait=raw_d.get("trait", "")
                )
            )

        # Planes
        raw_planes = data.get("planes", [])
        for raw_p in raw_planes:
            p_data = apply_trait(raw_p)
            normal = Vec2.from_iterable(p_data.get("normal", [0, 1]))
            dist = float(p_data.get("dist", 0.0))
            b_coef = float(p_data.get("bCoef", 1.0))
            c_mask = parse_collision_flags(p_data.get("cMask", ["all"]))
            c_group = parse_collision_flags(p_data.get("cGroup", ["wall"]))
            stad.planes.append(Plane(normal, dist, b_coef, c_mask, c_group))

        return stad

    @classmethod
    def load_from_file(cls, filepath: str) -> Stadium:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        cleaned = strip_json_comments_and_trailing_commas(content)
        data = json.loads(cleaned)
        return cls.from_dict(data)

    def create_ball(self, pos: Optional[Vec2] = None) -> Disc:
        bp = self.ball_physics
        p = pos.copy() if pos else Vec2(0, 0)
        c_mask = parse_collision_flags(bp.get("cMask", ["all"]))
        c_group = parse_collision_flags(bp.get("cGroup", ["ball"]))
        return Disc(
            pos=p,
            speed=Vec2(0, 0),
            radius=float(bp.get("radius", 10.0)),
            bCoef=float(bp.get("bCoef", 0.5)),
            invMass=float(bp.get("invMass", 1.0)),
            damping=float(bp.get("damping", 0.99)),
            color=bp.get("color", "FFFFFF"),
            cMask=c_mask,
            cGroup=c_group,
            name="Ball"
        )

    def create_player(
        self,
        team: Team,
        player_id: int,
        player_number: int = 1,
        pos: Optional[Vec2] = None
    ) -> Disc:
        pp = self.player_physics
        if pos is None:
            x_offset = -self.spawn_distance if team == Team.RED else self.spawn_distance
            pos = Vec2(x_offset, 0.0)

        color = "E56E56" if team == Team.RED else "5689E5"
        team_group = CollisionMask.RED if team == Team.RED else CollisionMask.BLUE
        team_mask = CollisionMask.ALL

        return Disc(
            pos=pos,
            speed=Vec2(0, 0),
            radius=float(pp.get("radius", 15.0)),
            bCoef=float(pp.get("bCoef", 0.0 if "bCoef" in pp else 0.5)),
            invMass=float(pp.get("invMass", 0.5)),
            damping=float(pp.get("damping", 0.96)),
            color=color,
            cMask=team_mask,
            cGroup=team_group,
            is_player=True,
            team=team,
            player_id=player_id,
            player_number=player_number,
            acceleration=float(pp.get("acceleration", 0.11)),
            kicking_acceleration=float(pp.get("kickingAcceleration", 0.083)),
            kicking_damping=float(pp.get("kickingDamping", 0.96)),
            kick_strength=float(pp.get("kickStrength", 5.0)),
            kick_margin=float(pp.get("kickMargin", 4.0)),
            kick_back=float(pp.get("kickBack", 0.0)),
            name=f"Player_{team.name}_{player_number}"
        )
