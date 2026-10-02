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
    
    total_target_steps = 100000
    batch_steps = 500
    total_iterations = total_target_steps // batch_steps

    print("=" * 60)
    print(" INICIANDO TREINAMENTO SEGURO EM BACKGROUND (SELF-PLAY 2v2)")
    print(f" Meta: {total_target_steps:,} passos | Threads limitadas para estabilidade")
    print("=" * 60)

    start_time = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "meu_haxball_bot.pt")

    for i in range(1, total_iterations + 1):
        stats = trainer.step_multistep(batch_steps)

        if i % 20 == 0 or i == total_iterations:
            elapsed = time.time() - start_time
            steps = stats.get("total_steps", 0)
            rew = stats.get("mean_reward", 0.0)
            passes = stats.get("passes", 0)
            sps = int(steps / max(1e-5, elapsed))
            
            print(
                f"[{i:03d}/{total_iterations}] Passos: {steps:06d}/{total_target_steps:,} | "
                f"Reward: {rew:+.2f} | Passes: {passes:03d} | Vel: {sps} passos/s | Tempo: {elapsed:.1f}s"
            )
            
            # Save checkpoint
            torch.save(trainer.policy.state_dict(), save_path)

    print("=" * 60)
    print(f"[OK] TREINO CONCLUIDO COM SUCESSO EM {time.time() - start_time:.1f} SEGUNDOS!")
    print(f"[OK] Checkpoint final salvo em: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
