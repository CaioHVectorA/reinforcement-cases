"""
Interactive HaxBall game launcher.
Launches the full HaxBall GUI Control Center by default, or runs matches against trained RL bots.
"""

from __future__ import annotations
import sys
import argparse
from haxball.ui.haxball_gui import HaxBallApp, STADIUM_CATALOG
from haxball.bots import NPC_BOTS, GAUNTLET_ORDER

ALL_BOT_CHOICES = list(NPC_BOTS.keys()) + ["heuristic", "wall", "goalie", "rl", "gauntlet"]

def main():
    parser = argparse.ArgumentParser(description="HaxBall RL Suite Launcher & Battle Arena")
    parser.add_argument("--map", choices=list(STADIUM_CATALOG.keys()), default="small_classic", help="Map to play on")
    parser.add_argument("--mode", choices=["human", "self_play"], default="human", help="Play as human or run self-play")
    parser.add_argument("--format", type=int, choices=[1, 2, 3, 5], default=1, help="Team format: 1 (1v1), 2 (2v2), 3 (3v3), 5 (5v5)")
    parser.add_argument("--bot", choices=ALL_BOT_CHOICES, default="press", help="Opponent bot type")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .pt model weights")
    args = parser.parse_args()

    bot_key = "rl" if (args.model or args.bot == "rl") else args.bot
    app = HaxBallApp(mode=args.mode, model_path=args.model, bot_key=bot_key if bot_key != "gauntlet" else GAUNTLET_ORDER[0])
    if args.bot == "gauntlet":
        app.gauntlet_mode = True
        app.gauntlet_index = 0
        app.active_bot_key = GAUNTLET_ORDER[0]
    app._init_game(args.map, players_count=args.format)
    app.run()

if __name__ == "__main__":
    main()
