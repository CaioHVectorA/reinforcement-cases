"""
Safe, headless self-play training script for HaxBall RL.
Runs quietly in background with thread limits to prevent high CPU load or freezing.
"""

import os
import sys
import time
import torch

# Limit PyTorch to 2 CPU threads so it runs smoothly without freezing the PC
torch.set_num_threads(2)

from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer

def main():
    map_path = os.path.join("haxball", "maps", "futsal_2v2.hbs")
    stadium = Stadium.load_from_file(map_path)
    game = HaxBallGame(stadium=stadium, red_players_count=2, blue_players_count=2)

    trainer = SelfPlay2v2Trainer(game=game, rollout_steps=512)
    
    total_target_steps = 200000
    batch_steps = 500
    total_iterations = total_target_steps // batch_steps

    milestones = {
        25000: "checkpoints/fase1_iniciante_25k.pt",
        70000: "checkpoints/fase2_amador_70k.pt",
        200000: "checkpoints/fase3_intermediario_200k.pt",
    }
    milestone_keys = sorted(milestones.keys())
    next_milestone_idx = 0

    print("=" * 60)
    print(" INICIANDO TREINAMENTO SEGURO EM BACKGROUND (SELF-PLAY 2v2)")
    print(f" Meta: {total_target_steps:,} passos | Threads limitadas para estabilidade")
    print(" Checkpoints que serao gerados:")
    for ms, pth in milestones.items():
        print(f"  - {ms:,} passos -> {pth}")
    print("=" * 60)

    start_time = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "meu_haxball_bot.pt")

    for i in range(1, total_iterations + 1):
        stats = trainer.step_multistep(batch_steps)
        current_steps = stats.get("total_steps", 0)

        # Check milestones
        while next_milestone_idx < len(milestone_keys) and current_steps >= milestone_keys[next_milestone_idx]:
            ms_steps = milestone_keys[next_milestone_idx]
            ms_path = milestones[ms_steps]
            torch.save(trainer.policy.state_dict(), ms_path)
            torch.save(trainer.policy.state_dict(), save_path)
            print(f"[CHECKPOINT SALVO] Meta {ms_steps:,} passos alcancada -> {ms_path}")
            next_milestone_idx += 1

        if i % 20 == 0 or i == total_iterations:
            elapsed = time.time() - start_time
            rew = stats.get("mean_reward", 0.0)
            passes = stats.get("passes", 0)
            sps = int(current_steps / max(1e-5, elapsed))
            rem_steps = total_target_steps - current_steps
            eta_sec = rem_steps / max(1, sps)
            
            print(
                f"[{i:03d}/{total_iterations}] Passos: {current_steps:06d}/{total_target_steps:,} | "
                f"Reward: {rew:+.2f} | Passes: {passes:03d} | Vel: {sps} p/s | ETA: {eta_sec/60:.1f} min"
            )
            
            # Save latest checkpoint
            torch.save(trainer.policy.state_dict(), save_path)

    print("=" * 60)
    print(f"[OK] TREINO DE 200k CONCLUIDO COM SUCESSO EM {time.time() - start_time:.1f} SEGUNDOS!")
    print(f"[OK] Checkpoint final salvo em: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
