"""
Launcher Oficial para HaxBall Dodgeball isolado.
Redireciona para haxball-dodgeball/play.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    target = Path(__file__).resolve().parent / "haxball-dodgeball" / "play.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
