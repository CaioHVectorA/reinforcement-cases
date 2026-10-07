"""
Clusterizador e Classificador Científico de Replays HaxBall (.hbr2).
Desempacota o cabeçalho binário oficial HBR2 de acordo com a especificação técnica:
- Extrai nome do estádio (stadium_name) decodificado dos bytes
- Extrai dimensões da quadra (width, height)
- Extrai duração em ticks e tamanho
- Descarta partidas vazias / gravações automáticas de bot sem gameplay (< 15s ou < 20 inputs)
- Agrupa os replays válidos em pastas dedicadas para cada modalidade/case:
    1. data/clustered_replays/futsal_3v3/
    2. data/clustered_replays/futsal_1v1/
    3. data/clustered_replays/real_soccer/
    4. data/clustered_replays/big_and_5v5/
    5. data/clustered_replays/other_modes/
"""

from __future__ import annotations
import os
import sys
import zlib
import glob
import shutil
import json
from typing import Dict, Any, Optional, Tuple

class ByteReader:
    def __init__(self, buffer: bytes):
        self.b = buffer
        self.length = len(buffer)
        self.pos = 0

    def remaining(self) -> int:
        return self.length - self.pos

    def read_u8(self) -> int:
        if self.pos >= self.length:
            raise EOFError("EOF")
        v = self.b[self.pos]
        self.pos += 1
        return v

    def read_i8(self) -> int:
        v = self.read_u8()
        return v if v < 128 else v - 256

    def read_u16_be(self) -> int:
        if self.pos + 2 > self.length:
            raise EOFError("EOF")
        v = (self.b[self.pos] << 8) | self.b[self.pos + 1]
        self.pos += 2
        return v

    def read_i16_be(self) -> int:
        v = self.read_u16_be()
        return v if v < 32768 else v - 65536

    def read_u32_be(self) -> int:
        if self.pos + 4 > self.length:
            raise EOFError("EOF")
        v = (self.b[self.pos] << 24) | (self.b[self.pos + 1] << 16) | (self.b[self.pos + 2] << 8) | self.b[self.pos + 3]
        self.pos += 4
        return v

    def read_i32_be(self) -> int:
        v = self.read_u32_be()
        return v if v < 2147483648 else v - 4294967296

    def read_f64_be(self) -> float:
        import struct
        if self.pos + 8 > self.length:
            raise EOFError("EOF")
        v = struct.unpack_from(">d", self.b, self.pos)[0]
        self.pos += 8
        return v

    def read_varint(self) -> int:
        res = 0
        shift = 0
        while True:
            b = self.read_u8()
            res |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7
        return res

    def read_str_varint_len(self) -> str:
        length = self.read_varint()
        if self.pos + length > self.length:
            raise EOFError("EOF")
        s = self.b[self.pos:self.pos + length].decode("utf-8", errors="replace")
        self.pos += length
        return s

    def read_nullable_str(self) -> Optional[str]:
        length_code = self.read_varint()
        if length_code <= 0:
            return None
        length = length_code - 1
        if self.pos + length > self.length:
            raise EOFError("EOF")
        s = self.b[self.pos:self.pos + length].decode("utf-8", errors="replace")
        self.pos += length
        return s


def parse_hbr2_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Decodifica com segurança o cabeçalho HBR2 para extrair estádio, dimensões e metadados.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read()

        if len(data) < 12 or data[:4] != b"HBR2":
            return None

        version = int.from_bytes(data[4:8], "big")
        duration_ticks = int.from_bytes(data[8:12], "big")
        payload = data[12:]

        try:
            decomp = zlib.decompress(payload, -15)
        except Exception:
            decomp = zlib.decompress(payload)

        r = ByteReader(decomp)

        # 1. Sync header
        sync_count = r.read_u16_be()
        for _ in range(sync_count):
            r.read_varint()
            r.read_u8()

        # 2. Room State
        room_name = r.read_nullable_str() or ""
        locked = r.read_u8() != 0
        score_limit = r.read_i32_be()
        time_limit = r.read_i32_be()
        r.read_i16_be()
        r.read_u8()
        r.read_u8()

        # 3. Stadium Unpack
        tag = r.read_u8()
        stadium_name = ""
        width = 0.0
        height = 0.0

        if tag == 255:
            stadium_name = r.read_nullable_str() or "Custom"
            bg_type = r.read_i32_be()
            width = r.read_f64_be()
            height = r.read_f64_be()
        else:
            defaults = {
                0: "Classic", 1: "Easy", 2: "Small", 3: "Big", 4: "Rounded",
                5: "Hockey", 6: "Big Hockey", 7: "Big Easy", 8: "Big Rounded", 9: "Huge"
            }
            stadium_name = defaults.get(tag, f"Default_{tag}")
            if tag == 0: width, height = 420.0, 200.0
            elif tag in (2,): width, height = 320.0, 150.0
            elif tag in (3, 6, 7, 8): width, height = 600.0, 270.0
            elif tag == 9: width, height = 750.0, 350.0

        # Scan for active inputs (action_type == 3)
        # In a compressed byte stream, quick scan of action_type == 3 count
        input_count = 0
        actions_seen = 0
        while r.remaining() > 0 and actions_seen < 3000:
            try:
                r.read_varint()
                atype = r.read_u8()
                actions_seen += 1
                if atype == 3:
                    r.read_u32_be()
                    input_count += 1
                elif atype == 0:
                    r.read_str_varint_len()
                    r.read_i32_be()
                    r.read_u8()
                    r.read_u8()
                elif atype == 1:
                    r.read_u8()
                elif atype == 5:
                    r.read_i32_be()
                    r.read_nullable_str()
                    r.read_nullable_str()
                    r.read_nullable_str()
                elif atype == 6:
                    r.read_i32_be()
                    r.read_nullable_str()
                    r.read_u8()
                elif atype == 12:
                    r.read_i32_be()
                    r.read_i8()
                elif atype == 16:
                    r.read_u8()
                elif atype in (2, 7, 8, 15):
                    pass
                elif atype == 9:
                    r.read_u8()
                elif atype == 10:
                    r.read_i32_be()
                    r.read_i32_be()
                elif atype == 18:
                    r.read_nullable_str()
                else:
                    break
            except Exception:
                break

        return {
            "room_name": room_name,
            "stadium_name": stadium_name,
            "width": width,
            "height": height,
            "duration_ticks": duration_ticks,
            "duration_sec": duration_ticks / 60.0,
            "file_size_kb": len(data) / 1024.0,
            "input_count": input_count
        }

    except Exception:
        return None


def classify_replay(meta: Dict[str, Any], filename: str) -> Tuple[str, str]:
    """
    Classifica a partida no cluster correto baseado no estádio oficial desempacotado.
    Retorna (categoria, justificativa).
    """
    stadium = meta["stadium_name"].lower()
    room = meta["room_name"].lower()
    fname = filename.lower()
    combined = f"{stadium} {room} {fname}"

    # Regras de descarte: partidas sem duração mínima ou sem inputs
    if meta["duration_sec"] < 15.0:
        return "empty_discard", "Duração insignificante (< 15 segundos)"
    if meta["input_count"] < 20 and meta["file_size_kb"] < 15.0:
        return "empty_discard", "Gravação automática de bot vazia (< 20 inputs e < 15 KB)"

    import re

    # 1. Modos especiais explícitos (Voley, Dodgeball, Hockey)
    if any(k in combined for k in ["voley", "volley", "queimada", "dodgeball", "hockey"]):
        return "other_modes", "Vôlei / Dodgeball / Hockey"

    # 2. Formatos gigantes (x10, x11, 11v11, 10v10, x8) vão para real_soccer
    if any(k in combined for k in ["x10", "x11", "x12", "11v11", "10v10", "rsx11", "x8"]):
        return "real_soccer", "Futebol de Campo Expandido (x10/x11/RS)"

    # 3. Real Soccer (RS, Futebol de campo grande, 4v4/7v7/11v11)
    if any(k in stadium for k in ["real soccer", "rs-", "rs ", "haxeleven", "revolution", "efrs", "lanzapiedras"]):
        return "real_soccer", "Real Soccer (Campo Aberto 4v4+)"
    if any(k in fname for k in ["realsoccer", "haxeleven", "rs-"]):
        return "real_soccer", "Real Soccer por metadados"

    # 4. Futsal 1v1 Estrito (usando limites de palavras para não casar com x10, x11 ou 2x1 de placar)
    # Requer que o estádio contenha explicitamente o termo isolado "x1", "1v1" ou "1x1"
    is_1v1_word = bool(re.search(r'\b(1v1|1x1|x1)\b', stadium, re.IGNORECASE))
    is_multi_word = bool(re.search(r'\b(2v2|x2|2x2|3v3|x3|3x3|4v4|x4|4x4|5v5|x5|5x5|7v7|x7|big)\b', stadium, re.IGNORECASE))
    if is_1v1_word and not is_multi_word:
        return "futsal_1v1", "Futsal 1v1 Exclusivo"

    # 5. Big e 5v5 (Campos grandes de Futsal ou Big Soccer)
    if any(k in stadium for k in ["big", "5v5", "x5", "5x5", "bff", "mar 5v5", "7v7", "x7"]):
        return "big_and_5v5", "Futsal Big / 5v5 / 7v7"

    # 6. Futsal 3v3 (O coração competitivo: Futsal x3, 3x3, GLH, Bazinga, FBF)
    if any(k in stadium for k in ["futsal x3", "futsal 3x3", "futsal 3v3", "futsal x1 and x2", "glh", "bazinga", "fbf"]):
        return "futsal_3v3", "Futsal 3v3 Competitivo"
    if "futsal" in stadium and (450.0 <= meta["width"] <= 750.0):
        return "futsal_3v3", "Futsal Padrão (Dimensões 3v3)"

    # 7. Fallback Classic / Outros
    if "classic" in stadium or stadium.startswith("default"):
        return "other_modes", "Mapa Clássico / Arcade"

    return "other_modes", f"Outro formato ('{meta['stadium_name']}')"



def run_clustering():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, "data", "replays")
    base_out_dir = os.path.join(project_root, "data", "clustered_replays")

    if os.path.exists(base_out_dir):
        shutil.rmtree(base_out_dir)

    categories = ["futsal_3v3", "futsal_1v1", "real_soccer", "big_and_5v5", "other_modes"]
    for cat in categories:
        os.makedirs(os.path.join(base_out_dir, cat), exist_ok=True)


    replays = glob.glob(os.path.join(source_dir, "*.hbr2"))
    print(f"=== INICIANDO CLUSTERIZAÇÃO DE {len(replays)} REPLAYS HBR2 ===")
    print(f"Origem: {source_dir}")
    print(f"Destino: {base_out_dir}\n")

    stats: Dict[str, int] = {cat: 0 for cat in categories}
    stats["empty_discard"] = 0
    stats["corrupt"] = 0

    stadium_sample: Dict[str, list] = {cat: [] for cat in categories}

    for i, rpath in enumerate(replays):
        fname = os.path.basename(rpath)
        meta = parse_hbr2_metadata(rpath)

        if not meta:
            stats["corrupt"] += 1
            continue

        cat, reason = classify_replay(meta, fname)

        if cat == "empty_discard":
            stats["empty_discard"] += 1
            continue

        stats[cat] += 1
        dest_path = os.path.join(base_out_dir, cat, fname)
        shutil.copy2(rpath, dest_path)

        if len(stadium_sample[cat]) < 5:
            stadium_sample[cat].append(f"{meta['stadium_name']} ({meta['file_size_kb']:.1f} KB, {meta['duration_sec']:.0f}s)")

        if (i + 1) % 500 == 0 or (i + 1) == len(replays):
            print(f"Processados {i+1}/{len(replays)} replays...")

    report_path = os.path.join(base_out_dir, "clustering_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_processados": len(replays),
            "stats": stats,
            "amostras_estadios": stadium_sample
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("=== RESULTADOS DA CLUSTERIZACAO ===")
    print("="*50)
    print(f"Total de replays brutos processados: {len(replays)}")
    print(f"Salas vazias descartadas (sem inputs/bot log): {stats['empty_discard']}")
    print(f"Arquivos corrompidos/invalidos: {stats['corrupt']}")
    print("-" * 50)
    print(f"[Futsal 3v3] (Pronto para BC de Equipe): {stats['futsal_3v3']} partidas")
    print(f"[Futsal 1v1] (Duelos 1v1 Reais): {stats['futsal_1v1']} partidas")
    print(f"[Real Soccer] (Campo Aberto 4v4/7v7): {stats['real_soccer']} partidas")
    print(f"[Big & 5v5] (Campo Expandido): {stats['big_and_5v5']} partidas")
    print(f"[Outros Modos] (Volei / Dodgeball / Classic): {stats['other_modes']} partidas")
    print("="*50)
    print(f"Relatorio salvo em: {report_path}")

if __name__ == "__main__":
    run_clustering()

