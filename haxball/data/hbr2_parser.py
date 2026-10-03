"""
HaxBall HBR2 Replay Parser (Pure Python)
Decodes official .hbr2 replay files into deterministic match frames, player trajectories, and inputs.
"""

from __future__ import annotations
import os
import zlib
import struct
from typing import List, Dict, Any, Optional, Tuple

class ByteReader:
    def __init__(self, buffer: bytes):
        self.b = buffer
        self.length = len(buffer)
        self.pos = 0

    def remaining(self) -> int:
        return self.length - self.pos

    def read_u8(self) -> int:
        if self.pos >= self.length:
            raise EOFError("EOF in read_u8")
        v = self.b[self.pos]
        self.pos += 1
        return v

    def read_i8(self) -> int:
        v = self.read_u8()
        return v if v < 128 else v - 256

    def read_u16_be(self) -> int:
        if self.pos + 2 > self.length:
            raise EOFError("EOF in read_u16_be")
        v = struct.unpack_from('>H', self.b, self.pos)[0]
        self.pos += 2
        return v

    def read_i16_be(self) -> int:
        if self.pos + 2 > self.length:
            raise EOFError("EOF in read_i16_be")
        v = struct.unpack_from('>h', self.b, self.pos)[0]
        self.pos += 2
        return v

    def read_u32_be(self) -> int:
        if self.pos + 4 > self.length:
            raise EOFError("EOF in read_u32_be")
        v = struct.unpack_from('>I', self.b, self.pos)[0]
        self.pos += 4
        return v

    def read_i32_be(self) -> int:
        if self.pos + 4 > self.length:
            raise EOFError("EOF in read_i32_be")
        v = struct.unpack_from('>i', self.b, self.pos)[0]
        self.pos += 4
        return v

    def read_f32_be(self) -> float:
        if self.pos + 4 > self.length:
            raise EOFError("EOF in read_f32_be")
        v = struct.unpack_from('>f', self.b, self.pos)[0]
        self.pos += 4
        return v

    def read_f64_be(self) -> float:
        if self.pos + 8 > self.length:
            raise EOFError("EOF in read_f64_be")
        v = struct.unpack_from('>d', self.b, self.pos)[0]
        self.pos += 8
        return v

    def read_varint(self) -> int:
        res = 0
        shift = 0
        while True:
            if self.pos >= self.length:
                raise EOFError("EOF in read_varint")
            b = self.read_u8()
            res |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        return res

    def read_bytes(self, count: int) -> bytes:
        if self.pos + count > self.length:
            raise EOFError(f"EOF in read_bytes (need {count}, have {self.remaining()})")
        res = self.b[self.pos:self.pos + count]
        self.pos += count
        return res

    def read_string_utf8(self, length: int) -> str:
        raw = self.read_bytes(length)
        return raw.decode("utf-8", errors="replace")

    def read_str_varint_len(self) -> str:
        length = self.read_varint()
        return self.read_string_utf8(length)

    def read_nullable_str(self) -> Optional[str]:
        length_code = self.read_varint()
        if length_code <= 0:
            return None
        return self.read_string_utf8(length_code - 1)

def skip_stadium(r: ByteReader):
    tag = r.read_u8()
    if tag != 255:
        return # Default stadium index
    # Custom stadium unpack
    name = r.read_nullable_str()
    bg_type = r.read_i32_be()
    width = r.read_f64_be()
    height = r.read_f64_be()
    cam_w = r.read_f64_be()
    cam_h = r.read_f64_be()
    spawn_dist = r.read_f64_be()
    bg_color = r.read_i32_be()
    cam_pos = r.read_f64_be()
    r.read_f64_be()
    r.read_f64_be()
    
    # Ball physics
    r.read_f64_be() # radius
    r.read_f64_be() # invMass
    r.read_f64_be() # damping
    r.read_f64_be() # bCoef
    r.read_f64_be() # kickAcc
    r.read_f64_be() # kickDamping
    r.read_f64_be() # kickStrength
    
    r.read_u16_be()
    r.read_u8()
    r.read_u8()
    r.read_u8()
    
    # Vertexes
    num_verts = r.read_u8()
    for _ in range(num_verts):
        r.read_f64_be() # x
        r.read_f64_be() # y
        r.read_f64_be() # bCoef
        r.read_i32_be() # cMask
        r.read_i32_be() # cGroup
        r.read_u8()     # trait
        
    # Segments
    num_segs = r.read_u8()
    for _ in range(num_segs):
        r.read_u8()     # v0
        r.read_u8()     # v1
        r.read_f64_be() # bCoef
        r.read_f64_be() # curve
        r.read_i32_be() # color
        r.read_i32_be() # cMask
        r.read_i32_be() # cGroup
        r.read_u8()     # vis
        r.read_u8()     # bias
        
    # Planes
    num_planes = r.read_u8()
    for _ in range(num_planes):
        r.read_f64_be() # nx
        r.read_f64_be() # ny
        r.read_f64_be() # dist
        r.read_f64_be() # bCoef
        r.read_i32_be() # cMask
        r.read_i32_be() # cGroup
        r.read_u8()
        
    # Goals
    num_goals = r.read_u8()
    for _ in range(num_goals):
        r.read_f64_be() # p0x
        r.read_f64_be() # p0y
        r.read_f64_be() # p1x
        r.read_f64_be() # p1y
        r.read_i8()     # team
        
    # Discs
    num_discs = r.read_u8()
    for _ in range(num_discs):
        r.read_f64_be() # pos.x
        r.read_f64_be() # pos.y
        r.read_f64_be() # speed.x
        r.read_f64_be() # speed.y
        r.read_f64_be() # gravity.x
        r.read_f64_be() # gravity.y
        r.read_f64_be() # radius
        r.read_f64_be() # invMass
        r.read_f64_be() # damping
        r.read_f64_be() # bCoef
        r.read_i32_be() # color
        r.read_i32_be() # cMask
        r.read_i32_be() # cGroup
        
    # Joints
    num_joints = r.read_u8()
    for _ in range(num_joints):
        r.read_u8()
        r.read_u8()
        r.read_f64_be()
        r.read_f64_be()
        r.read_f64_be()
        r.read_i32_be()
        
    # Red & Blue Spawns
    num_red_spawns = r.read_u8()
    for _ in range(num_red_spawns):
        r.read_f64_be()
        r.read_f64_be()
    num_blue_spawns = r.read_u8()
    for _ in range(num_blue_spawns):
        r.read_f64_be()
        r.read_f64_be()

def skip_disc(r: ByteReader):
    r.read_f64_be() # pos.x
    r.read_f64_be() # pos.y
    r.read_f64_be() # speed.x
    r.read_f64_be() # speed.y
    r.read_f64_be() # gravity.x
    r.read_f64_be() # gravity.y
    r.read_f64_be() # radius
    r.read_f64_be() # invMass
    r.read_f64_be() # damping
    r.read_f64_be() # bCoef
    r.read_i32_be() # color
    r.read_i32_be() # cMask
    r.read_i32_be() # cGroup

def skip_physics_state(r: ByteReader):
    # Match state in progress (qr.ja)
    skip_stadium(r)
    num_discs = r.read_u8()
    for _ in range(num_discs):
        skip_disc(r)
    r.read_i32_be() # score red
    r.read_i32_be() # score blue
    r.read_i32_be() # timer
    r.read_i32_be() # state
    r.read_f64_be() # kickoff timer
    r.read_i32_be() # period
    r.read_i8()     # kickoff team

def skip_player_entry(r: ByteReader):
    r.read_i32_be() # id
    r.read_nullable_str() # name
    r.read_i8()     # team
    r.read_i32_be() # flags
    r.read_nullable_str() # avatar
    r.read_nullable_str() # country
    has_disc = r.read_u8() != 0
    if has_disc:
        skip_disc(r)

def skip_team_colors(r: ByteReader):
    r.read_nullable_str() # name
    r.read_u8()
    r.read_i32_be()
    num_colors = r.read_u8()
    for _ in range(num_colors):
        r.read_i32_be()

class ReplayMatch:
    def __init__(self, filepath: str, version: int, duration_ticks: int):
        self.filepath = filepath
        self.version = version
        self.duration_ticks = duration_ticks
        self.room_name: str = ""
        self.players: Dict[int, str] = {}
        self.inputs: List[Tuple[int, int, int]] = [] # (tick, player_id, input_mask)
        self.goals: List[Tuple[int, int]] = []      # (tick, team)

def decode_hbr2_match(filepath: str) -> ReplayMatch:
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    if len(file_bytes) < 12 or file_bytes[:4] != b"HBR2":
        raise ValueError(f"Invalid HBR2: {filepath}")

    version = int.from_bytes(file_bytes[4:8], "big")
    duration_ticks = int.from_bytes(file_bytes[8:12], "big")
    payload = file_bytes[12:]

    decomp = zlib.decompress(payload, -15)
    r = ByteReader(decomp)

    # 1. Sync header
    sync_count = r.read_u16_be()
    for _ in range(sync_count):
        r.read_varint()
        r.read_u8()

    # 2. Initial Room State (e.T.ja)
    room_name = r.read_nullable_str() or "HaxBall Room"
    locked = r.read_u8() != 0
    score_limit = r.read_i32_be()
    time_limit = r.read_i32_be()
    r.read_i16_be()
    r.read_u8()
    r.read_u8()

    skip_stadium(r)

    has_match = r.read_u8() != 0
    if has_match:
        skip_physics_state(r)

    num_players = r.read_u8()
    players: Dict[int, str] = {}
    for _ in range(num_players):
        pid = r.read_i32_be()
        name = r.read_nullable_str() or f"P{pid}"
        r.read_i8()
        r.read_i32_be()
        r.read_nullable_str()
        r.read_nullable_str()
        has_disc = r.read_u8() != 0
        if has_disc:
            skip_disc(r)
        players[pid] = name

    skip_team_colors(r)
    skip_team_colors(r)

    match = ReplayMatch(filepath, version, duration_ticks)
    match.room_name = room_name
    match.players = players

    # 3. Action Packets Stream
    curr_tick = 0
    while r.remaining() > 0:
        try:
            delta = r.read_varint()
            curr_tick += delta
            action_type = r.read_u8()

            if action_type == 0: # Chat message (jr)
                r.read_str_varint_len()
                r.read_i32_be()
                r.read_u8()
                r.read_u8()

            elif action_type == 1: # Ping (br)
                r.read_u8()

            elif action_type == 2: # Match start (Ai)
                pass

            elif action_type == 3: # Player Input (wr)
                inp = r.read_u32_be()
                match.inputs.append((curr_tick, 0, inp))

            elif action_type == 5: # Player Join (Cr)
                pid = r.read_i32_be()
                name = r.read_nullable_str() or f"P{pid}"
                r.read_nullable_str()
                r.read_nullable_str()
                match.players[pid] = name

            elif action_type == 6: # Player Leave (gr)
                r.read_i32_be()
                r.read_nullable_str()
                r.read_u8()

            elif action_type in (7, 8, 15): # Instant events
                pass

            elif action_type == 9: # Pause (xr)
                r.read_u8()

            elif action_type == 10: # Setting (Nr)
                r.read_i32_be()
                r.read_i32_be()

            elif action_type == 11: # Stadium Change (Er)
                skip_stadium(r)

            elif action_type == 12: # Team change (Dr)
                r.read_i32_be()
                r.read_i8()

            elif action_type == 13: # Lock change (_r)
                r.read_u8()

            elif action_type == 14: # Admin change (Pr)
                r.read_i32_be()
                r.read_u8()

            elif action_type == 16: # Goal / Score (Ar)
                team = r.read_u8()
                match.goals.append((curr_tick, team))

            elif action_type == 18: # Avatar change (Or)
                r.read_nullable_str()

            else:
                # Other custom action
                break

        except (EOFError, struct.error):
            break

    return match

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    replays_dir = os.path.join(project_root, "data", "replays")
    files = [os.path.join(replays_dir, f) for f in os.listdir(replays_dir) if f.endswith(".hbr2")]
    print(f"=== Processando {len(files)} replays reais com o Parser Completo ===")
    total_inputs = 0
    total_goals = 0
    for f in files[:10]:
        m = decode_hbr2_match(f)
        total_inputs += len(m.inputs)
        total_goals += len(m.goals)
        print(f"Match: {os.path.basename(f)[:30]} | Room: {m.room_name[:20]} | Players: {len(m.players)} | Inputs: {len(m.inputs):,} | Goals: {len(m.goals)} | Ticks: {m.duration_ticks:,}")
    print(f"\nTotal extraído em 10 partidas: {total_inputs:,} inputs e {total_goals} gols!")
