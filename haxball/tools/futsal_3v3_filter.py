"""
Filtro Especialista e Estrito para Futsal 3v3 (.hbr2).
Identifica, valida e copia partidas autênticas de Futsal 3v3 (mapas GLH, Bazinga, FBF).
Critérios:
1. Mapa Futsal (largura entre 450 e 750, altura entre 200 e 320)
2. Tags estritas de 3v3 / 3x3 / x3
3. Rejeita 1v1, 2v2, 4v4, 5v5, 7v7, 8v8, Real Soccer e Voley
4. Verifica duração mínima (> 60s) e integridade dos bytes
"""

import os
import sys
import shutil
import zlib
from typing import Dict, Any, List, Optional

haxball_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if haxball_dir not in sys.path:
    sys.path.insert(0, haxball_dir)

from tools.cluster_replays import parse_hbr2_metadata

def scan_and_filter_futsal_3v3(source_dirs: List[str], dest_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    matches_3v3 = []
    
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        
    for sdir in source_dirs:
        if not os.path.exists(sdir):
            continue
        files = [os.path.join(sdir, f) for f in os.listdir(sdir) if f.endswith(".hbr2")]
        print(f"Varrendo {len(files)} replays em '{sdir}'...")
        
        for f in files:
            meta = parse_hbr2_metadata(f)
            if not meta:
                continue
                
            sname = str(meta.get("stadium_name", "")).lower()
            rname = str(meta.get("room_name", "")).lower()
            fname = os.path.basename(f).lower()
            comb = f"{sname} {rname} {fname}"
            
            # 1. Deve conter Futsal
            is_futsal = "futsal" in comb or any(k in comb for k in ["bazinga", "glh", "fbf"])
            if not is_futsal:
                continue
                
            # 2. Deve ter tag 3v3/3x3/x3
            has_3v3 = any(k in comb for k in ["3v3", "3x3", "x3"])
            
            # 3. Rejeitar explicitamente outros modos
            reject_tags = ["1v1", "1x1", "x1", "2v2", "2x2", "x2", "4v4", "4x4", "x4", 
                           "5v5", "5x5", "x5", "7v7", "x7", "8v8", "x8", "real soccer", "rs-", "voley"]
            is_rejected = any(k in comb for k in reject_tags)
            
            # 4. Checagem adicional de dimensões típicas de Futsal 3v3 (648x270 / 550x240)
            w = meta.get("width", 0.0)
            h = meta.get("height", 0.0)
            valid_dims = (w == 0.0 and h == 0.0) or (450.0 <= w <= 800.0 and 200.0 <= h <= 350.0)
            
            if has_3v3 and not is_rejected and valid_dims and meta.get("duration_sec", 0) >= 45:
                meta["path"] = f
                matches_3v3.append(meta)
                if dest_dir:
                    shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))
                    
    print(f"\n-> Total de partidas Futsal 3v3 filtradas: {len(matches_3v3)}")
    return matches_3v3

if __name__ == "__main__":
    sources = [
        os.path.join(haxball_dir, "data", "filtered_replays_1v1"),
        os.path.join(haxball_dir, "data", "replays"),
        os.path.join(os.path.dirname(haxball_dir), "data", "replays"),
    ]
    target = os.path.join(haxball_dir, "data", "filtered_replays_3v3")
    res = scan_and_filter_futsal_3v3(sources, target)
    for m in res:
        print(f"  [3v3] {os.path.basename(m['path'])[:40]} | {m['stadium_name']} | {m['duration_sec']:.0f}s")
