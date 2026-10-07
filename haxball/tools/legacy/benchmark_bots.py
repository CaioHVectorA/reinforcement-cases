"""
Headless 1v1 benchmark between scripted bots (round-robin) + behaviour telemetry.

Usage:
    PYTHONPATH=. .venv/bin/python benchmark_bots.py [--frames 3600] [--map futsal_2v2] [--bots a,b,c]
"""
from __future__ import annotations
import argparse
import itertools
import os
import time

from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.core.constants import Team, GameState
from haxball.bots import NPC_BOTS


def make_game(map_name: str, score_limit: int = 0):
    stad = Stadium.load_from_file(os.path.join("haxball", "maps", f"{map_name}.hbs"))
    return HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1,
                       score_limit=score_limit, time_limit_secs=0)


def play(map_name: str, red, blue, frames: int):
    game = make_game(map_name)
    red.reset(); blue.reset()
    stats = {"red_kicks": 0, "blue_kicks": 0, "red_dist": 0.0, "blue_dist": 0.0}
    for _ in range(frames):
        pr, pb = game.players[0], game.players[1]
        ir = red.act(game, pr)
        ib = blue.act(game, pb)
        info = game.step({pr.player_id: ir, pb.player_id: ib})
        for k in info.get("events", {}).get("kicks", []):
            stats["red_kicks" if k["team"] == Team.RED else "blue_kicks"] += 1
        stats["red_dist"] += pr.pos.distance_to(game.ball.pos)
        stats["blue_dist"] += pb.pos.distance_to(game.ball.pos)
    stats["red_dist"] /= frames
    stats["blue_dist"] /= frames
    return game.red_score, game.blue_score, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3600)
    ap.add_argument("--map", default="futsal_2v2")
    ap.add_argument("--bots", default=",".join(k for k in NPC_BOTS if k != "goalie"))
    args = ap.parse_args()

    keys = [k for k in args.bots.split(",") if k in NPC_BOTS]
    pts = {k: 0 for k in keys}
    gf = {k: 0 for k in keys}
    ga = {k: 0 for k in keys}
    t0 = time.time()
    print(f"{'RED':>10} x {'BLUE':<10} | score | kicks R/B | avg dist R/B")
    for a, b in itertools.permutations(keys, 2):
        red = NPC_BOTS[a][0](a)
        blue = NPC_BOTS[b][0](b)
        r, bl, st = play(args.map, red, blue, args.frames)
        print(f"{a:>10} x {b:<10} | {r}-{bl}   | {st['red_kicks']:>3}/{st['blue_kicks']:<3}   | {st['red_dist']:.0f}/{st['blue_dist']:.0f}")
        gf[a] += r; ga[a] += bl; gf[b] += bl; ga[b] += r
        if r > bl: pts[a] += 3
        elif bl > r: pts[b] += 3
        else: pts[a] += 1; pts[b] += 1
    print("\nRANKING")
    for k in sorted(keys, key=lambda x: (pts[x], gf[x] - ga[x]), reverse=True):
        print(f"{k:>10}  pts={pts[k]:>3}  GF={gf[k]:>2}  GA={ga[k]:>2}")
    print(f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
