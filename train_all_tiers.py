"""
Multi-tier training pipeline for HaxBall RL.
Trains across all cognitive tiers from 0 to 1,000,000 steps and saves save states for each tier:
- Fase 1: Iniciante (25k)
- Fase 2: Amador (70k)
- Fase 3: Intermediario (200k)
- Fase 4: Veterano (500k)
- Fase 5: Pro Master (1M)
"""

import os
import sys
import time
import subprocess
import torch

# Limit PyTorch CPU threads for thermal safety and zero system lag
torch.set_num_threads(2)

from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer

def main():
    map_path = os.path.join("haxball", "maps", "futsal_2v2.hbs")
    stadium = Stadium.load_from_file(map_path)
    game = HaxBallGame(stadium=stadium, red_players_count=2, blue_players_count=2)

    trainer = SelfPlay2v2Trainer(game=game, rollout_steps=512)

    os.makedirs("checkpoints", exist_ok=True)

    milestones = {
        25000: "checkpoints/fase1_iniciante_25k.pt",
        70000: "checkpoints/fase2_amador_70k.pt",
        200000: "checkpoints/fase3_intermediario_200k.pt",
        500000: "checkpoints/fase4_veterano_500k.pt",
        1000000: "checkpoints/fase5_pro_master_1M.pt"
    }

    total_target_steps = 1000000
    batch_steps = 500
    total_iterations = total_target_steps // batch_steps

    print("=" * 65)
    print(" INICIANDO TREINAMENTO COMPLETO DE TODAS AS FASES COGNITIVAS")
    print(f" Total de passos: {total_target_steps:,} | Limite seguro: 2 threads CPU")
    print(" Checkpoints que serao gerados:")
    for ms, pth in milestones.items():
        print(f"  - {ms:,} passos -> {pth}")
    print("=" * 65)

    start_time = time.time()
    next_milestone_idx = 0
    milestone_keys = sorted(milestones.keys())

    for i in range(1, total_iterations + 1):
        stats = trainer.step_multistep(batch_steps)
        current_steps = stats.get("total_steps", 0)

        # Check milestones
        while next_milestone_idx < len(milestone_keys) and current_steps >= milestone_keys[next_milestone_idx]:
            ms_steps = milestone_keys[next_milestone_idx]
            ms_path = milestones[ms_steps]
            torch.save(trainer.policy.state_dict(), ms_path)
            # Also keep latest updated
            torch.save(trainer.policy.state_dict(), "checkpoints/meu_haxball_bot.pt")
            print(f"[CHECKPOINT SALVO] Meta {ms_steps:,} passos alcancada -> {ms_path}")
            next_milestone_idx += 1

        # Periodic logging
        if i % 40 == 0 or i == total_iterations:
            elapsed = time.time() - start_time
            rew = stats.get("mean_reward", 0.0)
            passes = stats.get("passes", 0)
            sps = int(current_steps / max(1e-5, elapsed))
            rem_steps = total_target_steps - current_steps
            eta_sec = rem_steps / max(1, sps)
            
            print(
                f"[{i:04d}/{total_iterations}] Passos: {current_steps:07d}/{total_target_steps:,} | "
                f"Rew: {rew:+.2f} | Passes: {passes:03d} | Vel: {sps} p/s | ETA: {eta_sec/60:.1f} min"
            )

    print("=" * 65)
    print(f"[OK] TREINAMENTO DE TODAS AS FASES CONCLUIDO EM {time.time() - start_time:.1f}s!")
    print("=" * 65)

if __name__ == "__main__":
    main()
