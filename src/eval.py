"""Headless evaluation and benchmarking across multiple bots and stadiums.

    uv run src/eval.py                              # Current NeuralBot vs ChaseBot
    uv run src/eval.py --benchmark                  # Comprehensive benchmark: Random vs Chase, Neural vs Chase, Neural vs Random
    uv run src/eval.py --stadium futsal-classic     # Custom stadium evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bot import ChaseBot, NeuralBot, RandomBot
from haxballgym import BLUE, RED, make_default_env


def play_matchup(red_bot, blue_bot, stadium: str | None, n_envs: int, steps: int):
    env = make_default_env(n_envs, 1, 1, stadium=stadium)
    obs = env.reset()
    rg = bg = 0
    for _ in range(steps):
        # Red action
        a_red = red_bot.act(obs[:, 0:1]).reshape(n_envs, 1, 3)

        # Blue action
        if isinstance(blue_bot, ChaseBot):
            a_blue = blue_bot.act(env.prev_state, [1])
        else:
            a_blue = blue_bot.act(obs[:, 1:2]).reshape(n_envs, 1, 3)

        obs, rew, term, trunc = env.step(np.concatenate([a_red, a_blue], axis=1))
        sc = env.prev_state.scored
        rg += int((sc == BLUE).sum())  # blue conceded -> red scored
        bg += int((sc == RED).sum())  # red conceded -> blue scored
    return rg, bg


def main():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    meta_path = models_dir / "meta.json"
    default_stadium = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            default_stadium = meta.get("stadium")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="parallel games")
    ap.add_argument("--steps", type=int, default=600, help="decisions per game")
    ap.add_argument("--stadium", default=default_stadium, help="stadium name or .hbs path")
    ap.add_argument("--benchmark", action="store_true", help="run full 3-way comparative benchmark")
    ap.add_argument("--self", action="store_true", help="NeuralBot vs NeuralBot")
    args = ap.parse_args()

    stadium_name = args.stadium or "futsal-classic"

    if args.benchmark:
        print(f"\n{'='*65}")
        print(f"HAXBALL BENCHMARK — Arena: {stadium_name} ({args.n} games x {args.steps} steps)")
        print(f"{'='*65}\n")

        matchups = [
            ("RandomBot", RandomBot(seed=42), "ChaseBot", ChaseBot(seed=0)),
            ("NeuralBot (Trained)", NeuralBot(greedy=True), "ChaseBot", ChaseBot(seed=0)),
            ("NeuralBot (Trained)", NeuralBot(greedy=True), "RandomBot", RandomBot(seed=123)),
        ]

        for r_name, r_bot, b_name, b_bot in matchups:
            rg, bg = play_matchup(r_bot, b_bot, stadium=stadium_name, n_envs=args.n, steps=args.steps)
            net = rg - bg
            total = rg + bg
            winrate = (rg / total * 100) if total > 0 else 50.0
            print(f"[{r_name}] vs [{b_name}]:")
            print(f"  Gols: {rg} - {bg} (Saldo: {net:+d}) | Winrate estimada: {winrate:.1f}%\n")
        print(f"{'='*65}\n")
    else:
        red_bot = NeuralBot(greedy=True)
        blue_bot = NeuralBot(greedy=True) if args.self else ChaseBot(seed=0)
        b_name = "NeuralBot (Self)" if args.self else "ChaseBot"
        rg, bg = play_matchup(red_bot, blue_bot, stadium=stadium_name, n_envs=args.n, steps=args.steps)
        print(f"NeuralBot {rg} - {bg} {b_name} | Saldo: {rg - bg:+d} (em {args.n} jogos x {args.steps} steps no {stadium_name})")


if __name__ == "__main__":
    main()
