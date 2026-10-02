"""
Interactive HaxBall game launcher.
Launches the full HaxBall GUI Control Center by default, or runs matches against trained RL bots.
"""

from __future__ import annotations
import sys
import argparse
from haxball.ui.haxball_gui import HaxBallApp, STADIUM_CATALOG

def main():
    parser = argparse.ArgumentParser(description="HaxBall RL Suite Launcher & Battle Arena")
    parser.add_argument("--map", choices=list(STADIUM_CATALOG.keys()), default="futsal_2v2", help="Map to play on")
    parser.add_argument("--mode", choices=["human", "self_play"], default="human", help="Play as human or run self-play")
    parser.add_argument("--bot", choices=["rl", "wall", "heuristic", "goalie"], default="rl", help="Opponent bot type")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .pt model weights")
    args = parser.parse_args()

    bot_key = "rl" if (args.model or args.bot == "rl") else args.bot
    app = HaxBallApp(mode=args.mode, model_path=args.model, bot_key=bot_key)
    if args.map != "futsal_2v2":
        app._init_game(args.map)
    app.run()

if __name__ == "__main__":
    main()
