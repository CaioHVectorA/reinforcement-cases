"""
Launcher rápido para jogar HaxBall contra Bots.
Uso direto:
    python play.py         # Joga Dodgeball 1v1 contra 1 Bot
    python play.py 2       # Joga Dodgeball contra 2 Bots (Desafio)
    python play.py 2v2     # Joga Dodgeball 2v2 em equipe (Você + Aliado vs 2 Bots)
    python play.py futsal  # Joga partida 3v3 Futsal
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    args = sys.argv[1:]
    
    if args and args[0].lower() in ("dodgeball", "dodge", "queimada"):
        script = PROJECT_ROOT / "haxball" / "dodgeball" / "play.py"
        forward_args = args[1:]
    elif args and args[0].lower() in ("match", "quick", "3v3"):
        script = PROJECT_ROOT / "haxball" / "scripts" / "play_3v3_human_vs_bots.py"
        forward_args = args[1:]

    else:
        # Default: Launch the full Interactive HaxBall Futsal Studio!
        script = PROJECT_ROOT / "main.py"
        forward_args = args

    cmd = [sys.executable, str(script)] + forward_args
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
