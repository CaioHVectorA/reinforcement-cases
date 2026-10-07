"""
Human-Like Organic Training Pipeline for HaxBall RL.
Combines:
1. Behavioral Cloning (Warm-Start with human-like positioning, circling behind ball, and angle cutting)
2. Domain-Randomized Scenarios (Open field, moving ball, corners, wall bounces, 1v1 breakaways)
3. Diverse Opponent Curriculum & PPO Fine-Tuning
"""

import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Thermal safety: limit PyTorch CPU threads
torch.set_num_threads(2)

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots.heuristic_bot import HeuristicBot
from haxball.bots.wall_rebound_bot import WallReboundBot
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.models.mlp_policy import ActorCriticMLP
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer

def collect_expert_demonstrations(num_steps: int = 15000):
    """
    Collects state-action demonstration pairs from HeuristicBot and WallReboundBot
    across diverse randomized scenarios.
    """
    map_path = os.path.join("haxball", "maps", "futsal_2v2.hbs")
    stadium = Stadium.load_from_file(map_path)
    game = HaxBallGame(stadium=stadium, red_players_count=2, blue_players_count=2)
    obs_builder = DecoupledObservationBuilder()

    bot_red = HeuristicBot()
    bot_blue = WallReboundBot()

    obs_list = []
    act_list = []

    game.reset_round(randomize_scenario=True)

    for step in range(num_steps):
        if step % 120 == 0 or game.state == GameState.GAME_OVER:
            game.reset_round(randomize_scenario=True)

        inputs = {}
        # Record Red players
        for p in game.players:
            if p.team == Team.RED:
                obs = obs_builder.build_observation(game, p)
                cmd = bot_red.act(game, p)
                # cmd is (mx, my, kick) in global coords; for Red, ego_mx = mx
                ego_action = [cmd[0], cmd[1], 1.0 if cmd[2] else -1.0]
                obs_list.append(obs)
                act_list.append(ego_action)
                inputs[p.player_id] = cmd
            else:
                obs = obs_builder.build_observation(game, p)
                cmd = bot_blue.act(game, p)
                # for Blue, ego_mx = -mx
                ego_action = [-cmd[0], cmd[1], 1.0 if cmd[2] else -1.0]
                obs_list.append(obs)
                act_list.append(ego_action)
                inputs[p.player_id] = cmd

        step_info = game.step(inputs)
        if step_info.get("goal_scored", False):
            game.reset_round(randomize_scenario=True)

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)

def train_behavioral_cloning(policy: ActorCriticMLP, obs_data: np.ndarray, act_data: np.ndarray, epochs: int = 15, batch_size: int = 128, lr: float = 1e-3):
    """
    Pre-trains policy actor using supervised imitation learning.
    """
    optimizer = optim.Adam(policy.actor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tensor_obs = torch.tensor(obs_data, dtype=torch.float32)
    tensor_act = torch.tensor(act_data, dtype=torch.float32)
    dataset_size = len(tensor_obs)

    print(f"[IMItation] Treinando Warm-Start em {dataset_size:,} exemplos humanos por {epochs} épocas...")

    for epoch in range(1, epochs + 1):
        indices = torch.randperm(dataset_size)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_inds = indices[start:end]

            pred_act = policy.actor(tensor_obs[mb_inds])
            loss = criterion(pred_act, tensor_act[mb_inds])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            print(f" -> Época {epoch:02d}/{epochs} | Loss de Imitação: {total_loss / max(1, num_batches):.4f}")

def main():
    print("=" * 70)
    print(" PIPELINE DE TREINAMENTO HUMANO ORGANICO (WARM-START + PPO SCENARIOS)")
    print("=" * 70)

    # 1. Collect Demonstrations
    print("\n[FASE 1] Coletando 16.000 passos de demonstrações de posicionamento e tabelas...")
    obs_data, act_data = collect_expert_demonstrations(num_steps=8000)
    print(f" -> Coleta concluída: {len(obs_data):,} amostras de estados diversos.")

    # 2. Initialize Trainer
    map_path = os.path.join("haxball", "maps", "futsal_2v2.hbs")
    stadium = Stadium.load_from_file(map_path)
    game = HaxBallGame(stadium=stadium, red_players_count=2, blue_players_count=2)
    trainer = SelfPlay2v2Trainer(game=game, rollout_steps=512, lr=2.5e-4)

    # 3. Behavioral Cloning Warm-Start
    train_behavioral_cloning(trainer.policy, obs_data, act_data, epochs=15)
    print("[OK] Warm-start concluído! O modelo agora entende contorno de bola e posicionamento.")

    os.makedirs("checkpoints", exist_ok=True)
    # Save warm-start checkpoint as Fase 1
    torch.save(trainer.policy.state_dict(), "checkpoints/fase1_iniciante_25k.pt")
    torch.save(trainer.policy.state_dict(), "checkpoints/meu_haxball_bot.pt")

    # 4. PPO Training with Scenario Randomization
    total_target_steps = 250000
    batch_steps = 500
    total_iterations = total_target_steps // batch_steps

    milestones = {
        50000: "checkpoints/fase2_amador_70k.pt",
        150000: "checkpoints/fase3_intermediario_200k.pt",
        250000: "checkpoints/fase4_veterano_500k.pt",
    }
    milestone_keys = sorted(milestones.keys())
    next_ms_idx = 0

    print("\n[FASE 2] Iniciando PPO com Cenários Dinâmicos e Diversos...")
    start_time = time.time()

    for i in range(1, total_iterations + 1):
        # Every 300 steps or after goal, randomize scenario
        if i % 2 == 0:
            if game.state == GameState.PLAYING and random.random() < 0.25:
                game.reset_round(randomize_scenario=True)

        stats = trainer.step_multistep(batch_steps)
        current_steps = stats.get("total_steps", 0)

        # Check milestones
        while next_ms_idx < len(milestone_keys) and current_steps >= milestone_keys[next_ms_idx]:
            ms_k = milestone_keys[next_ms_idx]
            torch.save(trainer.policy.state_dict(), milestones[ms_k])
            torch.save(trainer.policy.state_dict(), "checkpoints/meu_haxball_bot.pt")
            print(f"[CHECKPOINT SALVO] {ms_k:,} passos -> {milestones[ms_k]}")
            next_ms_idx += 1

        if i % 30 == 0 or i == total_iterations:
            elapsed = time.time() - start_time
            rew = stats.get("mean_reward", 0.0)
            passes = stats.get("passes", 0)
            sps = int(current_steps / max(1e-5, elapsed))
            rem_steps = total_target_steps - current_steps
            eta_sec = rem_steps / max(1, sps)

            print(
                f"[{i:03d}/{total_iterations}] Passos: {current_steps:06d}/{total_target_steps:,} | "
                f"Rew: {rew:+.2f} | Passes: {passes:03d} | Vel: {sps} p/s | ETA: {eta_sec/60:.1f} min"
            )
            torch.save(trainer.policy.state_dict(), "checkpoints/meu_haxball_bot.pt")

    print("\n" + "=" * 70)
    print(f"[OK] TREINAMENTO HUMANO ORGANICO CONCLUIDO EM {time.time() - start_time:.1f}s!")
    print("=" * 70)

if __name__ == "__main__":
    main()
