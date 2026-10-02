"""
Interactive HaxBall game launcher.
Launches the full HaxBall GUI Control Center by default, or runs command-line modes.
"""

from __future__ import annotations
import sys
import argparse
from haxball.ui.haxball_gui import HaxBallApp

def main():
    parser = argparse.ArgumentParser(description="HaxBall Suite Launcher")
    parser.add_argument("--cli", action="store_true", help="Launch in simple CLI window mode")
    parser.add_argument("--map", choices=["futsal_3v3", "futsal_5v5", "classic", "dodgeball"], default="futsal_3v3")
    args = parser.parse_args()

    app = HaxBallApp()
    if args.map != "futsal_3v3":
        app._init_game(args.map)
    app.run()

if __name__ == "__main__":
    main()
