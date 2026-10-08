"""
Checkpoint Tournament & Winrate Matrix
Evaluates RL checkpoints in 100-match head-to-head simulations.
Tracks Wins, Draws, Losses, Goals, and Inactivity (AFK/no play).
Generates a graphical winrate matrix and convergence analysis report.
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

def run_tournament(models_config, n_matches=100, max_ticks=600):
    map_path = os.path.join(haxball_dir, "maps", "classic.hbs")
    stadium = Stadium.load_from_file(map_path)
    
    print(f"=== INICIANDO TORNEIO DE CHECKPOINTS ({len(models_config)} MODELOS) ===")
    print(f"Partidas por confronto: {n_matches} | Ticks máximos por partida: {max_ticks} (~{max_ticks/60:.1f}s)")
    
    # Load all models once
    bots = {}
    for name, path in models_config:
        full_path = os.path.join(haxball_dir, path)
        print(f"Carregando {name} de {path}...")
        bots[name] = RLBot(model_path=full_path)
        
    names = [m[0] for m in models_config]
    N = len(names)
    
    # Metrics
    win_matrix = np.zeros((N, N), dtype=float)
    draw_matrix = np.zeros((N, N), dtype=float)
    inactivity_matrix = np.zeros((N, N), dtype=float)
    goals_matrix = np.zeros((N, N), dtype=float)
    
    match_count = 0
    total_confrontos = N * (N - 1) // 2
    confronto_idx = 0
    
    t_start = time.time()
    
    for i in range(N):
        for j in range(i + 1, N):
            confronto_idx += 1
            name1, name2 = names[i], names[j]
            bot1, bot2 = bots[name1], bots[name2]
            
            w1, w2, draws, inacts = 0, 0, 0, 0
            g1, g2 = 0, 0
            
            # Run n_matches: alternate sides (Red/Blue) 50% each
            for m in range(n_matches):
                flip = (m % 2 == 1)
                r_bot = bot2 if flip else bot1
                b_bot = bot1 if flip else bot2
                
                game = HaxBallGame(stadium=stadium, red_players_count=1, blue_players_count=1)
                p_red = game.players[0]
                p_blue = game.players[1]
                
                touches = 0
                r_moves = 0
                b_moves = 0
                
                for tick in range(max_ticks):
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
                
                # Check for inactivity (both AFK / no touch & virtually no movement)
                is_afk = (touches == 0 and (r_moves < 20 or b_moves < 20))
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
            
            print(f"[{confronto_idx}/{total_confrontos}] {name1} vs {name2} -> {name1}: {w1}W | {name2}: {w2}W | Empates: {draws} (Inativos: {inacts})")
            
    total_time = time.time() - t_start
    print(f"\nTorneio concluído em {total_time:.2f}s!")
    
    return {
        "names": names,
        "win_matrix": win_matrix.tolist(),
        "draw_matrix": draw_matrix.tolist(),
        "inactivity_matrix": inactivity_matrix.tolist(),
        "goals_matrix": goals_matrix.tolist(),
        "elapsed_sec": total_time
    }

def generate_charts_and_report(results, output_png="benchmarks/checkpoint_tournament_winrate.png"):
    names = results["names"]
    win_mat = np.array(results["win_matrix"])
    draw_mat = np.array(results["draw_matrix"])
    inact_mat = np.array(results["inactivity_matrix"])
    N = len(names)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    # 1. Heatmap Winrate
    im0 = axes[0].imshow(win_mat * 100, cmap="viridis", vmin=0, vmax=100)
    axes[0].set_title("Taxa de Vitória (%) Head-to-Head (100 Partidas/Duelo)", fontsize=13, fontweight="bold")
    axes[0].set_xticks(range(N))
    axes[0].set_yticks(range(N))
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=10)
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
    
    # 2. Bar Chart Performance & Inactivity / No-play
    # Calculate overall winrate and inactive rate for each model
    avg_wins = []
    avg_draws = []
    avg_inacts = []
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
    
    axes[1].set_title("Desempenho Geral e Taxa de Inatividade (AFK)", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    axes[1].set_ylabel("Porcentagem (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"Gráfico salvo com sucesso em: {output_png}")

if __name__ == "__main__":
    checkpoints = [
        ("Fase 1 (25k)", "checkpoints/fase1_iniciante_25k.pt"),
        ("Fase 3 (200k)", "checkpoints/fase3_intermediario_200k.pt"),
        ("Fase 5 (1M)", "checkpoints/fase5_pro_master_1M.pt"),
        ("Gen 100", "checkpoints/interactive_local/gen_iter_100.pt"),
        ("Gen 600", "checkpoints/interactive_local/gen_iter_600.pt"),
        ("Gen 1000", "checkpoints/interactive_local/gen_iter_1000.pt"),
        ("Gen 2000", "checkpoints/interactive_local/gen_iter_2000.pt"),
        ("Final Best", "checkpoints/haxball_rl_best.pt"),
    ]
    
    results = run_tournament(checkpoints, n_matches=100, max_ticks=500)
    
    with open("benchmarks/checkpoint_tournament_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    generate_charts_and_report(results)
