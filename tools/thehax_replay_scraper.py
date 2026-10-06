"""
TheHax.pl Replay Scraper (.hbr2)
Downloads replays directly from replay.thehax.pl API into data/replays.
"""

from __future__ import annotations
import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")

def download_file(url: str, dest_path: str) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    try:
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def scrape_thehax_replays(max_files: int = 200, batch_size: int = 100) -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_api_url = "https://replay.thehax.pl/api/lastReplays"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://replay.thehax.pl/lastReplays"
    }

    offset = 0
    downloaded_count = 0

    print(f"=== Iniciando Scraping de Replays de replay.thehax.pl ===")
    print(f"Destino: {OUTPUT_DIR}")
    print(f"Meta de arquivos: {max_files}\n")

    while downloaded_count < max_files:
        limit = min(batch_size, max_files - downloaded_count)
        req_url = f"{base_api_url}?offset={offset}&limit={limit}"
        req = urllib.request.Request(req_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8")
                json_obj = json.loads(data)
                rows = json_obj.get("rows", [])
        except Exception as e:
            print(f"[Erro de Conexão] {e}")
            break

        if not rows:
            print("Não há mais replays disponíveis na API de replay.thehax.pl.")
            break

        for item in rows:
            url_uuid = item.get("url")
            final_name = item.get("finalName", "replay")
            if not url_uuid:
                continue

            dl_url = f"https://replay.thehax.pl/{url_uuid}/download"
            safe_name = "".join(c for c in final_name if c.isalnum() or c in ("-", "_", ".")).strip()
            if not safe_name.endswith(".hbr2"):
                safe_name += ".hbr2"

            target_fname = f"thehax_{url_uuid}_{safe_name}"
            dest = os.path.join(OUTPUT_DIR, target_fname)

            ok = download_file(dl_url, dest)
            if ok:
                downloaded_count += 1
                size_kb = os.path.getsize(dest) / 1024.0
                try:
                    print(f"[{downloaded_count}/{max_files}] Baixado (TheHax): {safe_name} ({size_kb:.1f} KB)")
                except Exception:
                    print(f"[{downloaded_count}/{max_files}] Baixado (TheHax): replay_{downloaded_count}.hbr2 ({size_kb:.1f} KB)")
                if downloaded_count >= max_files:
                    break

        offset += len(rows)
        time.sleep(0.3)

    print(f"\n=== Scraping TheHax Concluído! Total baixado: {downloaded_count} replays. ===")
    return downloaded_count

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    scrape_thehax_replays(max_files=count)
