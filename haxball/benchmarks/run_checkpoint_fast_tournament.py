"""
Optimized High-Speed Checkpoint Tournament & Winrate Analysis
Focuses on key generational milestones and final models:
1. Fase 1 (25k) - Early baseline
2. Fase 5 (1M) - Fully trained continuous model
3. Gen 100 - Early self-play discrete
4. Gen 1000 - Mid self-play discrete
5. Final Best - Peak convergent model
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

haxball_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(haxball_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if haxball_dir not in sys.path:
    sys.path.insert(0, haxball_dir)

from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots.rl_bot import RLBot

FRAME_SKIP = 3

def run_fast_tournament(models_config, n_matches=100, max_ticks=300):
    map_path = os.path.join(haxball_dir, "maps", "classic.hbs")
    stadium = Stadium.load_from_file(map_path)
    
    print(f"=== TORNEIO DE CHECKPOINTS ({len(models_config)} MODELOS CHAVE) ===")
    print(f"Partidas por confronto: {n_matches} | Frame Skip: {FRAME_SKIP} | Max Ticks: {max_ticks}")
    
    bots = {}
    for name, path in models_config:
        full_path = os.path.join(haxball_dir, path)
        bots[name] = RLBot(model_path=full_path)
        
    names = [m[0] for m in models_config]
    N = len(names)
    
    win_matrix = np.zeros((N, N), dtype=float)
    draw_matrix = np.zeros((N, N), dtype=float)
    inactivity_matrix = np.zeros((N, N), dtype=float)
    goals_matrix = np.zeros((N, N), dtype=float)
    
    t_start = time.time()
    total_confrontos = N * (N - 1) // 2
    idx = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            idx += 1
            t_match_start = time.time()
            name1, name2 = names[i], names[j]
            bot1, bot2 = bots[name1], bots[name2]
            
            w1, w2, draws, inacts = 0, 0, 0, 0
            g1, g2 = 0, 0
            
            for m in range(n_matches):
                flip = (m % 2 == 1)
                r_bot = bot2 if flip else bot1
                b_bot = bot1 if flip else bot2
                
                game = HaxBallGame(stadium=stadium, red_players_count=1, blue_players_count=1)
                p_red = game.players[0]
                p_blue = game.players[1]
                
                touches = 0
                r_moves, b_moves = 0, 0
                
                act_r = (0.0, 0.0, False)
                act_b = (0.0, 0.0, False)
                
                for tick in range(max_ticks):
                    if tick % FRAME_SKIP == 0:
                        act_r = r_bot.act(game, p_red)
                        act_b = b_bot.act(game, p_blue)
                        if abs(act_r[0]) > 0.05 or abs(act_r[1]) > 0.05: r_moves += 1
                        if abs(act_b[0]) > 0.05 or abs(act_b[1]) > 0.05: b_moves += 1
                        
                    info = game.step({p_red.player_id: act_r, p_blue.player_id: act_b})
                    touches += len(info.get("events", {}).get("disc_ball_collisions", []))
                    
                    if game.red_score > 0 or game.blue_score > 0:
                        break
                        
                score1 = game.blue_score if flip else game.red_score
                score2 = game.red_score if flip else game.blue_score
                g1 += score1
                g2 += score2
                
                is_afk = (touches == 0 and (r_moves < 10 or b_moves < 10))
                if is_afk:
                    inacts += 1
                    draws += 1
                elif score1 > score2:
                    w1 += 1
                elif score2 > score1:
                    w2 += 1
                else:
                    draws += 1
                    
            win_matrix[i, j] = w1 / n_matches
            win_matrix[j, i] = w2 / n_matches
            draw_matrix[i, j] = draws / n_matches
            draw_matrix[j, i] = draws / n_matches
            inactivity_matrix[i, j] = inacts / n_matches
            inactivity_matrix[j, i] = inacts / n_matches
            goals_matrix[i, j] = g1
            goals_matrix[j, i] = g2
            
            dt = time.time() - t_match_start
            print(f"[{idx:02d}/{total_confrontos}] {name1} vs {name2} ({dt:.1f}s) -> {name1}: {w1}W | {name2}: {w2}W | Empates: {draws} (Inativos: {inacts})")
            
    print(f"\nTorneio total concluído em {time.time() - t_start:.2f}s!")
    return {
        "names": names,
        "win_matrix": win_matrix.tolist(),
        "draw_matrix": draw_matrix.tolist(),
        "inactivity_matrix": inactivity_matrix.tolist(),
        "goals_matrix": goals_matrix.tolist()
    }

def generate_charts(results, output_png="benchmarks/checkpoint_tournament_winrate.png"):
    names = results["names"]
    win_mat = np.array(results["win_matrix"])
    draw_mat = np.array(results["draw_matrix"])
    inact_mat = np.array(results["inactivity_matrix"])
    N = len(names)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)
    
    # 1. Heatmap
    im0 = axes[0].imshow(win_mat * 100, cmap="viridis", vmin=0, vmax=100)
    axes[0].set_title("Taxa de Vitória (%) Head-to-Head (100 Partidas/Duelo)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(N))
    axes[0].set_yticks(range(N))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[0].set_yticklabels(names, fontsize=10)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                axes[0].text(j, i, "-", ha="center", va="center", color="gray", fontweight="bold")
            else:
                val = win_mat[i, j] * 100
                axes[0].text(j, i, f"{val:.0f}%", ha="center", va="center", 
                             color="white" if val < 50 else "black", fontweight="bold")
                             
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Winrate (%)")
    
    # 2. Bar Chart
    avg_wins, avg_draws, avg_inacts = [], [], []
    for i in range(N):
        opp_idxs = [j for j in range(N) if j != i]
        avg_wins.append(np.mean([win_mat[i, j] for j in opp_idxs]) * 100)
        avg_draws.append(np.mean([draw_mat[i, j] for j in opp_idxs]) * 100)
        avg_inacts.append(np.mean([inact_mat[i, j] for j in opp_idxs]) * 100)
        
    x = np.arange(N)
    width = 0.28
    
    axes[1].bar(x - width, avg_wins, width, label="Vitórias Médias (%)", color="#2ecc71")
    axes[1].bar(x, avg_draws, width, label="Empates / Sem Gol (%)", color="#3498db")
    axes[1].bar(x + width, avg_inacts, width, label="Inatividade / AFK (%)", color="#e74c3c")
    
    axes[1].set_title("Convergência e Taxa de Inatividade (AFK)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[1].set_ylabel("Porcentagem (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"Gráfico salvo em: {output_png}")

if __name__ == "__main__":
    checkpoints = [
        ("Fase 1 (25k)", "checkpoints/fase1_iniciante_25k.pt"),
        ("Fase 5 (1M)", "checkpoints/fase5_pro_master_1M.pt"),
        ("Gen 100", "checkpoints/interactive_local/gen_iter_100.pt"),
        ("Gen 1000", "checkpoints/interactive_local/gen_iter_1000.pt"),
        ("Final Best", "checkpoints/haxball_rl_best.pt"),
    ]
    
    res = run_fast_tournament(checkpoints, n_matches=100, max_ticks=300)
    with open("benchmarks/checkpoint_tournament_results.json", "w") as f:
        json.dump(res, f, indent=2)
    generate_charts(res)
