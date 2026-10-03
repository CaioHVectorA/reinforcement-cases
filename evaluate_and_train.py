"""
Automated Training, Evaluation and Convergence Analysis Suite for HaxBall RL.
Trains policy iteratively, benchmarks against baselines, analyzes gameplay metrics,
and reports convergence indicators.
"""

import os
import time
import math
import torch
import numpy as np

# Thermal safety: limit PyTorch CPU threads
torch.set_num_threads(2)

from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.core.constants import Team
from haxball.core.vector import Vec2
from haxball.bots.rl_bot import RLBot
from haxball.bots.heuristic_bot import HeuristicBot
from haxball.bots.goalie_bot import GoalieBot
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer

def evaluate_bot_vs_heuristic(model_state_dict, num_matches: int = 5, match_steps: int = 800):
    """
    Evaluates policy in matches against HeuristicBot as opponent.
    Returns metrics dict.
    """
    map_path = os.path.join("haxball", "maps", "futsal_2v2.hbs")
    stadium = Stadium.load_from_file(map_path)
    
    rl_bot = RLBot()
    rl_bot.policy.load_state_dict(model_state_dict)
    rl_bot.policy.eval()

    opp_bot = HeuristicBot()

    total_goals_for = 0
    total_goals_against = 0
    total_ball_touches = 0
    total_shots = 0
    total_shots_on_target = 0
    dist_to_ball_samples = []

    for m in range(num_matches):
        game = HaxBallGame(stadium=stadium, red_players_count=1, blue_players_count=1, score_limit=0)
        p_red = next(p for p in game.players if p.team == Team.RED)
        p_blue = next(p for p in game.players if p.team == Team.BLUE)

        for step in range(match_steps):
            cmd_red = rl_bot.act(game, p_red)
            cmd_blue = opp_bot.act(game, p_blue)

            step_info = game.step({p_red.player_id: cmd_red, p_blue.player_id: cmd_blue})

            if game.ball:
                d = p_red.pos.distance_to(game.ball.pos)
                dist_to_ball_samples.append(d)

                # Check if kicking near ball
                if cmd_red[2] and d <= (p_red.radius + game.ball.radius + p_red.kick_margin + 5.0):
                    total_shots += 1
                    to_goal = Vec2(stadium.bg_width - game.ball.pos.x, -game.ball.pos.y).normalized()
                    to_ball = (game.ball.pos - p_red.pos).normalized()
                    if to_ball.dot(to_goal) > 0.4:
                        total_shots_on_target += 1

                if d <= (p_red.radius + game.ball.radius + 4.0):
                    total_ball_touches += 1

            if step_info.get("goal_scored", False):
                if step_info.get("scoring_team") == Team.RED:
                    total_goals_for += 1
                elif step_info.get("scoring_team") == Team.BLUE:
                    total_goals_against += 1

    mean_dist = float(np.mean(dist_to_ball_samples)) if dist_to_ball_samples else 999.0
    shot_accuracy = (total_shots_on_target / max(1, total_shots)) * 100.0

    return {
        "matches": num_matches,
        "goals_for": total_goals_for,
        "goals_against": total_goals_against,
        "goal_diff": total_goals_for - total_goals_against,
        "touches": total_ball_touches,
        "shots": total_shots,
        "shots_on_target": total_shots_on_target,
        "shot_accuracy": shot_accuracy,
        "mean_dist_to_ball": mean_dist
    }

def main():
    map_path = os.path.join("haxball", "maps", "futsal_2v2.hbs")
    stadium = Stadium.load_from_file(map_path)
    game = HaxBallGame(stadium=stadium, red_players_count=2, blue_players_count=2)

    trainer = SelfPlay2v2Trainer(game=game, rollout_steps=512)
    os.makedirs("checkpoints", exist_ok=True)

    print("=" * 70)
    print(" PIPELINE DE TREINAMENTO, TESTES E ANALISE DE CONVERGENCIA")
    print("=" * 70)

    # Initial baseline eval before training (random weights)
    print("\n[EVAL 0k] Avaliando modelo inicial (Pesos Aleatorios / Burro)...")
    init_eval = evaluate_bot_vs_heuristic(trainer.policy.state_dict(), num_matches=3, match_steps=600)
    print(f" -> Distancia Media da Bola: {init_eval['mean_dist_to_ball']:.1f} px | Toques: {init_eval['touches']} | Gols: {init_eval['goals_for']}x{init_eval['goals_against']}")

    target_steps = 300000
    chunk_steps = 50000
    num_chunks = target_steps // chunk_steps

    milestones = {
        25000: "checkpoints/fase1_iniciante_25k.pt",
        70000: "checkpoints/fase2_amador_70k.pt",
        200000: "checkpoints/fase3_intermediario_200k.pt",
        300000: "checkpoints/fase4_veterano_500k.pt",
    }
    milestone_keys = sorted(milestones.keys())
    next_ms_idx = 0

    total_start = time.time()

    for chunk in range(1, num_chunks + 1):
        print(f"\n--- Treinando Bloco {chunk}/{num_chunks} ({chunk_steps:,} passos) ---")
        chunk_start = time.time()
        
        # Train chunk in increments of 500
        for _ in range(chunk_steps // 500):
            stats = trainer.step_multistep(500)
            cur_s = stats.get("total_steps", 0)
            while next_ms_idx < len(milestone_keys) and cur_s >= milestone_keys[next_ms_idx]:
                ms_k = milestone_keys[next_ms_idx]
                torch.save(trainer.policy.state_dict(), milestones[ms_k])
                torch.save(trainer.policy.state_dict(), "checkpoints/meu_haxball_bot.pt")
                print(f"[CHECKPOINT SALVO] {ms_k:,} passos -> {milestones[ms_k]}")
                next_ms_idx += 1

        dt = time.time() - chunk_start
        total_steps = stats["total_steps"]
        print(f"Bloco {chunk} concluido em {dt:.1f}s ({chunk_steps/dt:.1f} passos/s) | Total: {total_steps:,} passos")

        # Run intermediate evaluation
        print(f"[EVAL {total_steps//1000}k] Testando contra HeuristicBot...")
        ev = evaluate_bot_vs_heuristic(trainer.policy.state_dict(), num_matches=4, match_steps=700)
        print(
            f" -> Dist. Bola: {ev['mean_dist_to_ball']:.1f} px | "
            f"Toques: {ev['touches']} | "
            f"Chutes ao Gol: {ev['shots_on_target']}/{ev['shots']} ({ev['shot_accuracy']:.0f}%) | "
            f"Placar: {ev['goals_for']}x{ev['goals_against']}"
        )

    # Save final
    final_path = "checkpoints/meu_haxball_bot.pt"
    torch.save(trainer.policy.state_dict(), final_path)
    torch.save(trainer.policy.state_dict(), "checkpoints/fase3_intermediario_200k.pt")
    
    print("\n" + "=" * 70)
    print(f"[OK] TREINAMENTO E ANALISE FINALIZADOS EM {time.time() - total_start:.1f}s!")
    print("=" * 70)

if __name__ == "__main__":
    main()
