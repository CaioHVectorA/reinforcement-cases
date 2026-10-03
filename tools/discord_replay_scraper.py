"""
Discord Replay Scraper for HaxBall (.hbr2 / .hbr)
Safely paginates channel messages and downloads replay attachments into data/replays.
"""

from __future__ import annotations
import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

CHANNEL_ID = "1451234201511923766"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")

def get_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    auth_token = token or os.environ.get("DISCORD_TOKEN")
    if not auth_token:
        # Check local config or prompt
        token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".discord_token")
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                auth_token = f.read().strip()
        else:
            auth_token = input("Digite o seu token do Discord: ").strip()

    return {
        "authorization": auth_token,
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "referer": f"https://discord.com/channels/1439766646897115169/{CHANNEL_ID}",
        "x-discord-locale": "pt-BR",
        "x-discord-timezone": "America/Sao_Paulo"
    }

def download_file(url: str, dest_path: str) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": HEADERS["user-agent"]})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def scrape_replays(channel_id: str = CHANNEL_ID, max_files: int = 100, batch_size: int = 50, token: Optional[str] = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_url = f"https://discord.com/api/v9/channels/{channel_id}/messages"
    headers = get_auth_headers(token)
    
    last_id: Optional[str] = None
    downloaded_count = 0
    total_messages_seen = 0
    
    print(f"=== Iniciando Scraping de Replays do Canal {channel_id} ===")
    print(f"Destino: {OUTPUT_DIR}")
    print(f"Meta de arquivos: {max_files}\n")

    while downloaded_count < max_files:
        query_str = f"limit={batch_size}"
        if last_id:
            query_str += f"&before={last_id}"
        req_url = f"{base_url}?{query_str}"
        
        req = urllib.request.Request(req_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8")
                messages = json.loads(data)
        except urllib.error.HTTPError as he:
            if he.code == 429:
                err_data = json.loads(he.read().decode("utf-8"))
                retry_after = float(err_data.get("retry_after", 5.0))
                print(f"[Rate Limit] Aguardando {retry_after:.1f}s...")
                time.sleep(retry_after)
                continue
            else:
                print(f"[Erro API] Status {he.code}: {he.read().decode('utf-8')}")
                break
        except Exception as e:
            print(f"[Erro de Conexão] {e}")
            break
            
        if not messages:
            print("Não há mais mensagens no canal.")
            break
            
        total_messages_seen += len(messages)
        
        for msg in messages:
            last_id = msg.get("id")
            attachments = msg.get("attachments", [])
            for att in attachments:
                fname = att.get("filename", "")
                url = att.get("url")
                if fname.endswith(".hbr2") or fname.endswith(".hbr"):
                    target_fname = f"{msg['id']}_{fname}"
                    dest = os.path.join(OUTPUT_DIR, target_fname)
                    ok = download_file(url, dest)
                    if ok:
                        downloaded_count += 1
                        size_kb = os.path.getsize(dest) / 1024.0
                        print(f"[{downloaded_count}/{max_files}] Baixado: {fname} ({size_kb:.1f} KB)")
                        if downloaded_count >= max_files:
                            break
            if downloaded_count >= max_files:
                break
                
        time.sleep(0.5) # Gentle request throttling
        
    print(f"\n=== Scraping Concluído! Total baixado: {downloaded_count} replays. ===")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    scrape_replays(max_files=count)
