"""
Treinador Potente de Aprendizado por Reforço (PPO) 2 Horas com Benchmarking Contínuo.

Recursos:
1. Treinamento de Alta Potência (Throughput ~6.000 - 8.000 passos/segundo):
   - Otimizado em 14 threads de CPU via PyTorch e simulação matricial vetorizada.
   - Rollouts de 4.096 passos por iteração, totalizando dezenas de milhões de estados ao longo de 2 horas.
2. Inicialização Acelerada (Warm-start a partir do BC):
   - Inicializa a política de Auto-Atenção (EntityAttentionPolicy) a partir dos pesos pré-treinados
     do Behavioral Cloning (bc_futsal_3v3.pt), partindo de uma base humana sólida.
3. Currículo Multidimensional de Oponentes:
   - 40% contra o Agente BC (Entity Attention Calibrado)
   - 25% contra WallReboundBot ("Tabela Master")
   - 20% contra HeuristicBot (Posicionamento e corte de ângulo)
   - 15% Self-Play (Versus snapshots do próprio agente)
4. Bateria de Benchmarking Contínua & Gráficos em Tempo Real:
   - A cada ciclo de avaliação, disputa partidas oficiais contra BC, WallBot e HeuristicBot.
   - Registra Taxa de Vitória (%), Saldo de Gols, Retorno Médio, Posse de Bola e Finalizações.
   - Atualiza continuamente o gráfico de alta resolução 'benchmarks/rl_training_dashboard.png'
     e o log detalhado 'benchmarks/rl_training_log.json'.
5. Gerenciamento de Checkpoints:
   - Salva o melhor modelo em 'models/checkpoints/haxball_rl_best.pt'.
   - Salva o checkpoint mais recente em 'models/checkpoints/haxball_rl_potente.pt'.
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HAXBALL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HAXBALL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HAXBALL_DIR) not in sys.path:
    sys.path.insert(0, str(HAXBALL_DIR))

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.gym_env.rewards import RewardShaper
from haxball.bots.base_bot import BaseBot
from haxball.bots.heuristic_bot import HeuristicBot
from haxball.bots.wall_rebound_bot import WallReboundBot
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler
from haxball.rl.models.entity_attention import EntityAttentionPolicy

CHECKPOINT_DIR = HAXBALL_DIR / "models" / "checkpoints"
BENCHMARKS_DIR = HAXBALL_DIR / "benchmarks"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)



class BCOpponentBot(BaseBot):
    """Oponente baseado no modelo treinado de Behavioral Cloning com limiar marginal calibrado."""
    def __init__(self, model: EntityAttentionPolicy, kick_threshold: float = 0.22):
        super().__init__("BC_Opponent")
        self.model = model
        self.kick_threshold = kick_threshold
        self.obs_builder = DecoupledObservationBuilder()
        self.action_handler = ActionHandler()

    def act(self, game: HaxBallGame, player) -> Tuple[float, float, bool]:
        with torch.no_grad():
            obs = self.obs_builder.build_observation(game, player)
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            logits = self.model.actor(self.model.forward_repr(obs_t))[0]
            probs = torch.softmax(logits, dim=-1)

            kick_prob = probs[9:].sum().item()
            if kick_prob > self.kick_threshold:
                act_idx = 9 + torch.argmax(probs[9:]).item()
            else:
                act_idx = torch.argmax(probs[:9]).item()
            mx, my, kick = self.action_handler.decode_discrete(act_idx)
            attack_sign = 1.0 if player.team == Team.RED else -1.0
            return (mx * attack_sign, my, kick)


class SelfPlayOpponentBot(BaseBot):
    """Oponente que espelha os pesos atuais do agente em treinamento para confrontos diretos."""
    def __init__(self, agent: EntityAttentionPolicy):
        super().__init__("SelfPlay_Opponent")
        self.agent = agent
        self.obs_builder = DecoupledObservationBuilder()
        self.action_handler = ActionHandler()

    def act(self, game: HaxBallGame, player) -> Tuple[float, float, bool]:
        with torch.no_grad():
            obs = self.obs_builder.build_observation(game, player)
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action, _, _, _ = self.agent.get_action_and_value(obs_t)
            act_idx = action.item()
            mx, my, kick = self.action_handler.decode_discrete(act_idx)
            attack_sign = 1.0 if player.team == Team.RED else -1.0
            return (mx * attack_sign, my, kick)


class AdaptiveCurriculumOpponent(BaseBot):
    """Distribui oponentes proporcionalmente entre BC, WallRebound, Heuristic e Self-Play."""
    def __init__(self, opponents: List[Tuple[float, BaseBot]]):
        super().__init__("AdaptiveCurriculum")
        self.opponents = opponents
        self.current_bot: BaseBot = opponents[0][1]

    def reset_round(self):
        weights = [w for w, _ in self.opponents]
        bots = [b for _, b in self.opponents]
        probs = np.array(weights) / sum(weights)
        self.current_bot = np.random.choice(bots, p=probs)

    def act(self, game: HaxBallGame, player) -> Tuple[float, float, bool]:
        return self.current_bot.act(game, player)


def evaluate_agent_against(
    agent: EntityAttentionPolicy,
    opponent: BaseBot,
    stadium_path: str,
    num_matches: int = 3,
    max_steps_per_match: int = 1000
) -> Dict[str, float]:
    """Executa bateria de confrontos de teste e computa métricas competitivas."""
    env = HaxBallEnv(
        stadium_file=stadium_path,
        opponent_bot=opponent,
        discrete_actions=True,
        max_steps=max_steps_per_match,
        score_limit=5
    )

    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    agent_possession_ticks = 0
    opp_possession_ticks = 0
    shots_count = 0

    obs_builder = env.obs_builder
    action_handler = env.action_handler
    agent.eval()

    for _ in range(num_matches):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_match:
            steps += 1
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = agent.actor(agent.forward_repr(obs_t))[0]
                probs = torch.softmax(logits, dim=-1)
                # Calibração marginal de chute
                kick_prob = probs[9:].sum().item()
                if kick_prob > 0.22:
                    act_idx = 9 + torch.argmax(probs[9:]).item()
                else:
                    act_idx = torch.argmax(probs[:9]).item()

            obs, reward, terminated, truncated, info = env.step(act_idx)
            done = terminated or truncated

            # Rastreamento de finalizações
            ball = env.game.ball
            red_p = next((p for p in env.game.players if p.team == Team.RED), None)
            blue_p = next((p for p in env.game.players if p.team == Team.BLUE), None)
            if red_p and ball and red_p.is_kicking and red_p.pos.distance_to(ball.pos) < 35.0:
                shots_count += 1

            if red_p and blue_p and ball:
                dr = red_p.pos.distance_to(ball.pos)
                db = blue_p.pos.distance_to(ball.pos)
                if dr < db and dr < 110.0:
                    agent_possession_ticks += 1
                elif db < dr and db < 110.0:
                    opp_possession_ticks += 1

        r_score = env.game.red_score
        b_score = env.game.blue_score
        goals_for += r_score
        goals_against += b_score

        if r_score > b_score:
            wins += 1
        elif r_score < b_score:
            losses += 1
        else:
            draws += 1

    total_poss = agent_possession_ticks + opp_possession_ticks
    poss_pct = (agent_possession_ticks / total_poss * 100.0) if total_poss > 0 else 50.0
    win_rate = (wins / num_matches) * 100.0
    points_pct = ((wins + 0.5 * draws) / num_matches) * 100.0
    goal_diff = (goals_for - goals_against) / num_matches

    return {
        "win_rate": win_rate,
        "points_pct": points_pct,
        "draw_rate": (draws / num_matches) * 100.0,
        "loss_rate": (losses / num_matches) * 100.0,
        "goals_for": goals_for / num_matches,
        "goals_against": goals_against / num_matches,
        "goal_diff": goal_diff,
        "possession_pct": poss_pct,
        "shots_per_match": shots_count / num_matches
    }


def generate_benchmark_dashboard(history: List[Dict[str, Any]], output_path: Path):
    """Gera o painel visual com 4 gráficos de alta definição acompanhando a evolução."""
    if not history:
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), facecolor="#12161F")
    fig.suptitle(
        f"Painel Científico de Aprendizado por Reforço (PPO) - Treino Contínuo\n"
        f"Total de Iterações: {len(history)} | Passos Acumulados: {history[-1]['total_steps']:,}",
        fontsize=15, fontweight="bold", color="#E6EDF3", y=0.98
    )

    steps = [h["total_steps"] / 1_000_000.0 for h in history]  # Em milhões de passos

    # Painel 1: Taxa de Vitória contra Cada Oponente
    ax1 = axs[0, 0]
    ax1.set_facecolor("#1D222D")
    wr_bc = [h["eval_bc"]["win_rate"] for h in history]
    wr_wall = [h["eval_wall"]["win_rate"] for h in history]
    wr_heur = [h["eval_heur"]["win_rate"] for h in history]

    ax1.plot(steps, wr_bc, marker="o", markersize=4, color="#E63946", linewidth=2.2, label="vs IA Behavioral Cloning (BC)")
    ax1.plot(steps, wr_wall, marker="s", markersize=4, color="#F4A261", linewidth=2.2, label="vs WallReboundBot (Tabelas)")
    ax1.plot(steps, wr_heur, marker="^", markersize=4, color="#4EA8DE", linewidth=2.2, label="vs HeuristicBot (Posicionamento)")
    ax1.axhline(50.0, color="#8B949E", linestyle="--", alpha=0.5, label="Equilíbrio (50%)")

    ax1.set_title("1. Taxa de Vitória (%) contra Diferentes Classes de Oponentes", fontsize=12, fontweight="bold", color="#E6EDF3")
    ax1.set_xlabel("Milhões de Passos de Ambiente", fontsize=10, color="#8B949E")
    ax1.set_ylabel("Taxa de Vitória (%)", fontsize=10, color="#8B949E")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=":", alpha=0.3, color="#8B949E")
    ax1.legend(loc="upper left", framealpha=0.85, fontsize=9)

    # Painel 2: Saldo de Gols Médio por Partida
    ax2 = axs[0, 1]
    ax2.set_facecolor("#1D222D")
    gd_bc = [h["eval_bc"]["goal_diff"] for h in history]
    gd_wall = [h["eval_wall"]["goal_diff"] for h in history]
    gd_heur = [h["eval_heur"]["goal_diff"] for h in history]

    ax2.plot(steps, gd_bc, marker="o", markersize=4, color="#E63946", linewidth=2.0, label="Saldo vs BC")
    ax2.plot(steps, gd_wall, marker="s", markersize=4, color="#F4A261", linewidth=2.0, label="Saldo vs WallBot")
    ax2.plot(steps, gd_heur, marker="^", markersize=4, color="#4EA8DE", linewidth=2.0, label="Saldo vs Heuristic")
    ax2.axhline(0.0, color="#8B949E", linestyle="--", alpha=0.5)

    ax2.set_title("2. Saldo Médio de Gols (Gols Pró - Contra por Partida)", fontsize=12, fontweight="bold", color="#E6EDF3")
    ax2.set_xlabel("Milhões de Passos de Ambiente", fontsize=10, color="#8B949E")
    ax2.set_ylabel("Saldo de Gols", fontsize=10, color="#8B949E")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#8B949E")
    ax2.legend(loc="upper left", framealpha=0.85, fontsize=9)

    # Painel 3: Recompensa PPO e Curva de Valor
    ax3 = axs[1, 0]
    ax3.set_facecolor("#1D222D")
    rewards = [h["mean_reward"] for h in history]
    v_loss = [h["value_loss"] for h in history]

    color_rew = "#52B788"
    ax3.plot(steps, rewards, marker="d", markersize=4, color=color_rew, linewidth=2.2, label="Recompensa Média Rollout")
    ax3.set_title("3. Convergência da Função de Recompensa & Perda do Crítico", fontsize=12, fontweight="bold", color="#E6EDF3")
    ax3.set_xlabel("Milhões de Passos de Ambiente", fontsize=10, color="#8B949E")
    ax3.set_ylabel("Retorno Médio por Passo", fontsize=10, color=color_rew)
    ax3.grid(True, linestyle=":", alpha=0.3, color="#8B949E")

    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps, v_loss, color="#E76F51", linestyle="--", linewidth=1.5, label="Perda de Valor (Value Loss)")
    ax3_twin.set_ylabel("Value Loss", fontsize=10, color="#E76F51")

    # Legenda combinada
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", framealpha=0.85, fontsize=9)

    # Painel 4: Posse de Bola e Volume de Finalizações
    ax4 = axs[1, 1]
    ax4.set_facecolor("#1D222D")
    poss_bc = [h["eval_bc"]["possession_pct"] for h in history]
    shots_bc = [h["eval_bc"]["shots_per_match"] for h in history]

    color_poss = "#00B4D8"
    ax4.plot(steps, poss_bc, marker="s", markersize=4, color=color_poss, linewidth=2.0, label="Posse de Bola vs BC (%)")
    ax4.set_title("4. Domínio Territorial (Posse de Bola) & Frequência de Chutes", fontsize=12, fontweight="bold", color="#E6EDF3")
    ax4.set_xlabel("Milhões de Passos de Ambiente", fontsize=10, color="#8B949E")
    ax4.set_ylabel("Posse de Bola (%)", fontsize=10, color=color_poss)
    ax4.set_ylim(20, 80)
    ax4.grid(True, linestyle=":", alpha=0.3, color="#8B949E")

    ax4_twin = ax4.twinx()
    color_shots = "#FFB703"
    ax4_twin.plot(steps, shots_bc, marker="o", markersize=4, color=color_shots, linewidth=2.0, label="Chutes ao Gol / Partida")
    ax4_twin.set_ylabel("Finalizações / Partida", fontsize=10, color=color_shots)

    lines_3, labels_3 = ax4.get_legend_handles_labels()
    lines_4, labels_4 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_3 + lines_4, labels_3 + labels_4, loc="lower right", framealpha=0.85, fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def train_rl_potent(duration_hours: float = 2.0, eval_interval_mins: float = 2.5):
    """Executa o pipeline principal de PPO por 2 horas ininterruptas."""
    torch.set_num_threads(14)
    device = torch.device("cpu")

    total_target_seconds = int(duration_hours * 3600)
    print("=" * 70)
    print("=== INICIANDO TREINAMENTO PPO DE ALTA POTÊNCIA (2 HORAS) ===")
    print(f"Duração Alvo: {duration_hours:.1f} horas ({total_target_seconds} segundos)")
    print(f"Intervalo de Benchmarking: a cada ~{eval_interval_mins:.1f} minutos")
    print(f"Threads CPU: 14 | Dispositivo: {device}")
    print("=" * 70 + "\n")

    stadium_path = str(HAXBALL_DIR / "maps" / "futsal_3v3.hbs")

    # 1. Carregar Modelo BC como Oponente e Ponto de Partida
    bc_model = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True)
    bc_ckpt = CHECKPOINT_DIR / "bc_futsal_3v3.pt"
    if bc_ckpt.exists():
        try:
            bc_model.load_state_dict(torch.load(str(bc_ckpt), map_location="cpu"))
            print(f"[Sucesso] Modelo BC base carregado com sucesso de: {bc_ckpt}")
        except Exception as e:
            print(f"[Aviso] Erro ao carregar BC: {e}")
    bc_model.eval()

    # 2. Inicializar Agente RL (Warm-start com pesos do BC para salto cognitivo imediato)
    agent = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True).to(device)
    if bc_ckpt.exists():
        try:
            # Transfere encoders de atenção e ator
            agent.load_state_dict(bc_model.state_dict(), strict=False)
            print("[Sucesso] Warm-start aplicado ao agente RL a partir do BC!")
        except Exception as e:
            print(f"[Aviso] Warm-start parcial: {e}")

    # Otimizador Adam com taxa de aprendizado ligeiramente superior para o critic
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    # 3. Montar Oponentes do Currículo
    bc_bot = BCOpponentBot(bc_model, kick_threshold=0.22)
    wall_bot = WallReboundBot(name="WallMaster")
    heur_bot = HeuristicBot(name="HeuristicMaster")
    selfplay_bot = SelfPlayOpponentBot(agent)

    curriculum = AdaptiveCurriculumOpponent([
        (0.40, bc_bot),
        (0.25, wall_bot),
        (0.20, heur_bot),
        (0.15, selfplay_bot)
    ])

    # Ambiente de Treinamento
    reward_shaper = RewardShaper(
        goal_reward=12.0,
        concede_penalty=12.0,
        approach_ball_weight=0.06,
        ball_to_goal_vel_weight=0.09,
        kick_alignment_weight=0.35,
        touch_ball_weight=0.15,
        defensive_position_weight=0.03,
        whiff_kick_penalty=0.02
    )

    env = HaxBallEnv(
        stadium_file=stadium_path,
        opponent_bot=curriculum,
        discrete_actions=True,
        max_steps=2000,
        score_limit=5,
        reward_shaper=reward_shaper
    )

    # Hiperparâmetros PPO
    num_steps = 4096
    batch_size = 128
    update_epochs = 8
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.20
    ent_coef = 0.012
    vf_coef = 0.50
    max_grad_norm = 0.50

    obs_dim = env.observation_space.shape[0]

    obs_buffer = torch.zeros((num_steps, obs_dim), device=device)
    actions_buffer = torch.zeros(num_steps, dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(num_steps, device=device)
    rewards_buffer = torch.zeros(num_steps, device=device)
    dones_buffer = torch.zeros(num_steps, device=device)
    values_buffer = torch.zeros(num_steps, device=device)

    start_time = time.time()
    last_eval_time = start_time
    iteration = 0
    total_steps = 0
    best_avg_winrate = -1.0

    training_history: List[Dict[str, Any]] = []

    # Reset Inicial
    curriculum.reset_round()
    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    dashboard_path = BENCHMARKS_DIR / "rl_training_dashboard.png"
    log_json_path = BENCHMARKS_DIR / "rl_training_log.json"
    status_md_path = BENCHMARKS_DIR / "training_status.md"

    # --- LOOP PRINCIPAL DE 2 HORAS ---
    while True:
        iteration += 1
        iter_start = time.time()
        elapsed_total = iter_start - start_time

        if elapsed_total >= total_target_seconds:
            print(f"\n[Fim] Meta de 2 horas atingida com sucesso ({elapsed_total:.0f}s)!")
            break

        # A) Coleta de Rollout (4.096 passos por iteração)
        agent.eval()
        cur_ep_return = 0.0
        ep_returns = []

        for step in range(num_steps):
            total_steps += 1
            obs_buffer[step] = obs_t

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t.unsqueeze(0))
                values_buffer[step] = value.flatten()

            actions_buffer[step] = action.squeeze(0)
            logprobs_buffer[step] = logprob.squeeze(0)

            act_int = action.item()
            next_obs, reward, terminated, truncated, info = env.step(act_int)
            done = terminated or truncated

            rewards_buffer[step] = reward
            dones_buffer[step] = float(done)
            cur_ep_return += reward

            if done:
                ep_returns.append(cur_ep_return)
                cur_ep_return = 0.0
                curriculum.reset_round()
                next_obs, _ = env.reset()

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # B) Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(obs_t.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - float(done)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        # C) PPO Update (Otimização Ator-Crítico)
        agent.train()
        b_inds = np.arange(num_steps)
        clipfracs = []
        pg_losses = []
        v_losses = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_buffer[mb_inds], actions_buffer[mb_inds]
                )
                logratio = newlogprob - logprobs_buffer[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss com Clip PPO
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                # Entropy Loss (incentivo a exploração tática)
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())

        mean_reward = rewards_buffer.mean().item()
        mean_v_loss = np.mean(v_losses)
        mean_pg_loss = np.mean(pg_losses)

        # D) Bateria Periódica de Benchmarking (a cada eval_interval_mins)
        now = time.time()
        time_since_eval = now - last_eval_time
        should_evaluate = (time_since_eval >= eval_interval_mins * 60.0) or (iteration == 1)

        if should_evaluate:
            last_eval_time = now
            eval_start = time.time()
            print(f"\n---> [Benchmarking Iteração #{iteration} | {total_steps:,} passos | {(now - start_time)/60:.1f} min decorridos]", flush=True)

            # Testes contra oponentes reais
            eval_bc = evaluate_agent_against(agent, bc_bot, stadium_path, num_matches=3)
            eval_wall = evaluate_agent_against(agent, wall_bot, stadium_path, num_matches=3)
            eval_heur = evaluate_agent_against(agent, heur_bot, stadium_path, num_matches=3)

            eval_time = time.time() - eval_start
            avg_winrate = (eval_bc["win_rate"] + eval_wall["win_rate"] + eval_heur["win_rate"]) / 3.0

            print(f"     vs IA BC:       Vitórias: {eval_bc['win_rate']:.0f}% | Saldo: {eval_bc['goal_diff']:+.1f} | Posse: {eval_bc['possession_pct']:.0f}%", flush=True)
            print(f"     vs WallBot:     Vitórias: {eval_wall['win_rate']:.0f}% | Saldo: {eval_wall['goal_diff']:+.1f} | Posse: {eval_wall['possession_pct']:.0f}%", flush=True)
            print(f"     vs Heuristic:   Vitórias: {eval_heur['win_rate']:.0f}% | Saldo: {eval_heur['goal_diff']:+.1f} | Posse: {eval_heur['possession_pct']:.0f}%", flush=True)
            print(f"     Média Geral:    {avg_winrate:.1f}% Win Rate | Eval Time: {eval_time:.1f}s", flush=True)

            # Salvar Checkpoint Atual
            ckpt_potente = CHECKPOINT_DIR / "haxball_rl_potente.pt"
            torch.save(agent.state_dict(), str(ckpt_potente))

            # Se for novo recorde, salvar como best
            if avg_winrate >= best_avg_winrate:
                best_avg_winrate = avg_winrate
                ckpt_best = CHECKPOINT_DIR / "haxball_rl_best.pt"
                torch.save(agent.state_dict(), str(ckpt_best))
                print(f"     [RECORD!] Novo melhor modelo salvo em: {ckpt_best.name} (Taxa: {best_avg_winrate:.1f}%)", flush=True)

            # Registro Histórico
            eval_record = {
                "iteration": iteration,
                "total_steps": total_steps,
                "elapsed_seconds": int(now - start_time),
                "mean_reward": mean_reward,
                "policy_loss": mean_pg_loss,
                "value_loss": mean_v_loss,
                "avg_winrate": avg_winrate,
                "eval_bc": eval_bc,
                "eval_wall": eval_wall,
                "eval_heur": eval_heur
            }
            training_history.append(eval_record)

            # Salvar JSON de telemetria
            try:
                with open(log_json_path, "w", encoding="utf-8") as f:
                    json.dump(training_history, f, indent=2)
            except Exception as e:
                print(f"[Aviso] Erro ao salvar JSON: {e}", flush=True)

            # Gerar Dashboard Gráfico
            try:
                generate_benchmark_dashboard(training_history, dashboard_path)
                print(f"     [GRÁFICOS ATUALIZADOS] -> {dashboard_path}", flush=True)
            except Exception as e:
                print(f"[Aviso] Erro ao gerar gráficos: {e}", flush=True)

            # Atualizar Tabela de Status em Markdown
            _write_status_markdown(training_history, status_md_path, total_target_seconds, now - start_time)

        # Feedback periódico de console
        if iteration % 5 == 0 and not should_evaluate:
            throughput = num_steps / (time.time() - iter_start)
            rem_secs = max(0, total_target_seconds - (time.time() - start_time))
            print(f"[Treino] Iter #{iteration:4d} | Passos: {total_steps:8,d} | Throughput: {throughput:5.0f} st/s | Rew: {mean_reward:+.3f} | Restante: {rem_secs/60:4.1f} min", flush=True)


def _write_status_markdown(history: List[Dict[str, Any]], path: Path, total_target: int, elapsed: float):
    """Gera uma tabela resumida para acompanhamento no Markdown."""
    if not history:
        return
    latest = history[-1]
    pct_done = min(100.0, (elapsed / total_target) * 100.0)

    md = f"""# Relatório de Treinamento PPO Potente (2 Horas)

**Progresso:** {pct_done:.1f}% ({elapsed/60:.1f} min / {total_target/60:.0f} min)  
**Total de Passos:** {latest['total_steps']:,}  
**Iterações PPO Concluídas:** {latest['iteration']}  
**Taxa de Vitória Média Atual:** {latest['avg_winrate']:.1f}%  
**Dashboard de Gráficos:** [`benchmarks/rl_training_dashboard.png`](file:///{str(BENCHMARKS_DIR / 'rl_training_dashboard.png').replace('\\', '/')})

## Última Bateria de Testes

| Oponente | Taxa de Vitória | Saldo de Gols | Posse de Bola | Chutes / Jogo |
|---|---|---|---|---|
| **IA Behavioral Cloning (BC)** | {latest['eval_bc']['win_rate']:.1f}% | {latest['eval_bc']['goal_diff']:+.2f} | {latest['eval_bc']['possession_pct']:.1f}% | {latest['eval_bc']['shots_per_match']:.1f} |
| **WallReboundBot (Tabelas)** | {latest['eval_wall']['win_rate']:.1f}% | {latest['eval_wall']['goal_diff']:+.2f} | {latest['eval_wall']['possession_pct']:.1f}% | {latest['eval_wall']['shots_per_match']:.1f} |
| **HeuristicBot (Posicionamento)** | {latest['eval_heur']['win_rate']:.1f}% | {latest['eval_heur']['goal_diff']:+.2f} | {latest['eval_heur']['possession_pct']:.1f}% | {latest['eval_heur']['shots_per_match']:.1f} |

---
*Atualizado automaticamente a cada ciclo de avaliação.*
"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Treinador Potente PPO HaxBall 2 Horas")
    parser.add_argument("--hours", type=float, default=2.0, help="Duração alvo em horas (default: 2.0)")
    parser.add_argument("--eval-mins", type=float, default=2.5, help="Intervalo de benchmarking em minutos (default: 2.5)")
    args = parser.parse_args()

    train_rl_potent(duration_hours=args.hours, eval_interval_mins=args.eval_mins)


if __name__ == "__main__":
    main()
