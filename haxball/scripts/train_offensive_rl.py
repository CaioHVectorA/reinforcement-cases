import sys
import os
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add haxball/dodgeball directory to sys.path
haxball_dir = Path(__file__).resolve().parent.parent
dodgeball_dir = haxball_dir / "dodgeball"
sys.path.insert(0, str(dodgeball_dir))
sys.path.insert(0, str(haxball_dir.parent))

import torch
from core.stadium import Stadium
from core.dodgeball_game import DodgeballGame
from rl.trainer import DodgeballSelfPlayTrainer

def main():
    stadium_path = dodgeball_dir / "maps" / "dodgeball.hbs"
    stadium = Stadium.load_from_file(stadium_path)
    game = DodgeballGame(stadium, score_limit=15, red_players_count=1, blue_players_count=1)
    
    # High entropy bonus (0.02) to encourage active exploration of shooting and advancing
    trainer = DodgeballSelfPlayTrainer(
        game,
        lr=6e-4,
        ent_coef=0.02,
        rollout_steps=256,
        batch_size=64,
        update_epochs=4
    )

    total_target_steps = 120000
    print(f"[*] INICIANDO TREINAMENTO INTENSIVO PPO SELF-PLAY ({total_target_steps} PASSOS)...", flush=True)
    print("Mecanica: Recompensa negativa por distancia da bola, bonus de tiro ao gol, anti-camping!", flush=True)

    t0 = time.time()
    last_print = 0

    while trainer.total_env_steps < total_target_steps:
        stats = trainer.step_simulation()
        steps = trainer.total_env_steps
        if steps - last_print >= 10000 or steps >= total_target_steps:
            last_print = steps
            elapsed = time.time() - t0
            fps = steps / max(0.01, elapsed)
            m_rew = stats.get("mean_reward", 0.0)
            iters = stats.get("iteration", 0)
            elims = stats.get("wall_elims", 0)
            print(f"[{steps:06d}/{total_target_steps}] | Iter: {iters:03d} | Rew: {m_rew:+.2f} | Elims: {elims} | Speed: {fps:.0f} steps/s", flush=True)

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_path = str(ckpt_dir / "dodgeball_rl_best.pt")
    trainer.save_checkpoint(save_path)
    print(f"[OK] Treinamento de {total_target_steps} passos concluido em {time.time()-t0:.1f}s!", flush=True)
    print(f"Checkpoint salvo com sucesso em: {save_path}", flush=True)

if __name__ == "__main__":
    main()
