# CONTEXT.md — Project History, Context & Environment State

This document maintains the architectural history, past iterations, key decisions, and current runtime environment of the project.

---

## 1. Project Background & Architectural Pivot

### 1.1 The v0 Implementation (Archived in branch `v0_it_failed`)
- **Architecture**: Pure Python physics engine, custom PyTorch PPO and DQN trainers, and attempts at Behavioral Cloning (BC) from human `.hbr2` replays.
- **Why It Failed to Converge**:
  1. *Throughput Bottleneck*: The Python engine achieved only 2,000–8,000 ticks/sec across multiprocessing pools. Training for only 200k–1M steps is far below the threshold needed for contact-rich 2D physics.
  2. *Over-engineered Dense Rewards*: Too many conflicting reward terms (possession, distance, angular facing, camping penalties) caused severe reward-hacking. Agents circled the ball safely without shooting to exploit dense rewards, resulting in persistent draws.
  3. *Behavioral Cloning Failure*: Human replays suffered from high covariate shift ($O(T^2)$ error compounding) and erratic input sampling. Imitation learning failed to transfer to autonomous play and trapped policies in sub-optimal local minima.

### 1.2 The v1 Implementation (Current `main` Architecture)
- **Engine**: [HaxballGym](https://github.com/HaxballGym/HaxballGym) — a headless Rust simulation core with bit-exact ($10^{-9}$) analytical collision physics identical to the real Haxball engine.
- **Throughput**: 512 parallel environments simulated concurrently via Rayon multithreading, reaching 10,000 to 50,000 steps/sec on CPU.
- **Training Paradigm**: Clean self-play PPO with a frozen snapshot pool and goal-dominated reward shaping.

---

## 2. Infrastructure & System Specifications

- **OS / Platform**: Linux x86_64
- **Processor**: AMD Ryzen 7 5700U with Radeon Graphics (8 Cores, 16 Threads)
- **Accelerator**: None (CPU-only execution; PyTorch CPU wheels utilized to prevent disk bloat)
- **Package Manager**: Astral `uv` (`pyproject.toml` + `uv.lock`)
- **Key Dependencies**:
  - `haxball-core == 1.0.0`
  - `haxballgym == 1.0.1`
  - `torch == 2.14.1+cpu`
  - `numpy == 2.5.4`

---

## 3. Current Active Run

- **Command**:
  ```bash
  uv run src/train.py --stadium futsal-classic --red 1 --blue 1 --envs 512 --iters 4000
  ```
- **Configuration**:
  - Stadium: `futsal-classic` (solid boundary walls, futsal goal posts)
  - Matchup: 1v1
  - Envs: 512 parallel game instances
  - Total Target Iterations: 4,000 ($4,000 \times 64 \times 512 \approx 131,072,000$ steps)
  - Estimated Wall-clock Time: ~2h30m to 3h45m
- **Status**: Running. Policy rapidly achieved $+280$+ net score against `ChaseBot` in early iterations, with checkpoints saved continuously to `models/`.
