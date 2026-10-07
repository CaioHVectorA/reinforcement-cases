"""
Stadium loader and definition for HaxBall Dodgeball arenas.
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Any, Optional
from .vector import Vec2
from .constants import (
    CollisionMask, parse_collision_flags,
    DEFAULT_PLAYER_PHYSICS, DEFAULT_BALL_PHYSICS, Team
)
from .disc import Disc
from .segment import Vertex, Segment

def remove_comments(json_str: str) -> str:
    pattern = r"//.*?$|/\*.*?\*/"
    return re.sub(pattern, "", json_str, flags=re.MULTILINE | re.DOTALL)

class Stadium:
    def __init__(self):
        self.name: str = "Dodgeball Arena"
        self.width: float = 420.0
        self.height: float = 220.0
        self.spawn_distance: float = 180.0

        self.bg_width: float = 360.0
        self.bg_height: float = 180.0
        self.bg_color: str = "222831"

        self.player_physics = dict(DEFAULT_PLAYER_PHYSICS)
        self.ball_physics = dict(DEFAULT_BALL_PHYSICS)

        self.traits: Dict[str, Dict[str, Any]] = {}
        self.vertexes: List[Vertex] = []
        self.segments: List[Segment] = []
        self.discs: List[Disc] = []

    @classmethod
    def load_from_file(cls, path: str) -> Stadium:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        clean = remove_comments(content)
        data = json.loads(clean)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stadium:
        stad = cls()
        stad.name = data.get("name", "Dodgeball Arena")
        stad.width = float(data.get("width", 420))
        stad.height = float(data.get("height", 220))
        stad.spawn_distance = float(data.get("spawnDistance", 180))

        bg = data.get("bg", {})
        stad.bg_width = float(bg.get("width", 360))
        stad.bg_height = float(bg.get("height", 180))
        stad.bg_color = bg.get("color", "222831")

        stad.traits = data.get("traits", {})

        if "playerPhysics" in data:
            stad.player_physics.update(data["playerPhysics"])
        if "ballPhysics" in data:
            stad.ball_physics.update(data["ballPhysics"])

        def apply_trait(item: Dict[str, Any]) -> Dict[str, Any]:
            trait_name = item.get("trait")
            merged = {}
            if trait_name and trait_name in stad.traits:
                merged.update(stad.traits[trait_name])
            merged.update(item)
            return merged

        # Vertexes
        for raw_v in data.get("vertexes", []):
            v_data = apply_trait(raw_v)
            pos = Vec2(v_data.get("x", 0.0), v_data.get("y", 0.0))
            b_coef = float(v_data.get("bCoef", 0.75))
            c_mask = parse_collision_flags(v_data.get("cMask", ["all"]))
            c_group = parse_collision_flags(v_data.get("cGroup", ["wall"]))
            stad.vertexes.append(Vertex(pos, b_coef, c_mask, c_group, trait=raw_v.get("trait", "")))

        # Segments
        for raw_s in data.get("segments", []):
            s_data = apply_trait(raw_s)
            v0_idx = s_data.get("v0", 0)
            v1_idx = s_data.get("v1", 0)
            if v0_idx < len(stad.vertexes) and v1_idx < len(stad.vertexes):
                p0 = stad.vertexes[v0_idx].pos
                p1 = stad.vertexes[v1_idx].pos
                b_coef = float(s_data.get("bCoef", 0.75))
                c_mask = parse_collision_flags(s_data.get("cMask", ["all"]))
                c_group = parse_collision_flags(s_data.get("cGroup", ["wall"]))
                vis = bool(s_data.get("vis", True))
                color = s_data.get("color", "FFFFFF")
                stad.segments.append(Segment(p0, p1, b_coef, c_mask, c_group, vis, color, trait=raw_s.get("trait", "")))

        return stad

    def create_ball(self) -> Disc:
        bp = self.ball_physics
        return Disc(
            pos=Vec2(0.0, 0.0),
            speed=Vec2(0.0, 0.0),
            radius=float(bp.get("radius", 9.0)),
            bCoef=float(bp.get("bCoef", 0.72)),
            invMass=float(bp.get("invMass", 1.1)),
            damping=float(bp.get("damping", 0.988)),
            color=bp.get("color", "FF4466"),
            cMask=parse_collision_flags(bp.get("cMask", ["all"])),
            cGroup=parse_collision_flags(bp.get("cGroup", ["ball"])),
            name="Ball"
        )

    def create_player(self, team: Team, player_id: int, player_number: int = 1) -> Disc:
        pp = self.player_physics
        color = "E56E56" if team == Team.RED else "5689E5"
        c_group = CollisionMask.RED if team == Team.RED else CollisionMask.BLUE
        c_mask = CollisionMask.ALL

        p = Disc(
            pos=Vec2(0.0, 0.0),
            speed=Vec2(0.0, 0.0),
            radius=float(pp.get("radius", 15.0)),
            bCoef=float(pp.get("bCoef", 0.5)),
            invMass=float(pp.get("invMass", 0.5)),
            damping=float(pp.get("damping", 0.96)),
            color=color,
            cMask=c_mask,
            cGroup=c_group,
            name=f"Player_{player_id}"
        )
        p.is_player = True
        p.player_id = player_id
        p.team = team
        p.player_number = player_number
        p.acceleration = float(pp.get("acceleration", 0.12))
        p.kicking_acceleration = float(pp.get("kickingAcceleration", 0.09))
        p.kicking_damping = float(pp.get("kickingDamping", 0.96))
        p.kick_strength = float(pp.get("kickStrength", 6.5))
        p.kick_margin = float(pp.get("kickMargin", 4.0))
        p.kick_back = float(pp.get("kickBack", 0.0))
        return p
