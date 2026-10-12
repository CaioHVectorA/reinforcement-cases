# TODO.md — Roadmap & Project Milestones

This file tracks the active tasks and upcoming development phases for the Haxball RL project.

---

## Active Phase: 1v1 Futsal Self-Play Convergence

- [x] Snapshot full v0 implementation into branch `v0_it_failed`.
- [x] Verify bit-exact collision handling and custom stadium support in Rust core.
- [x] Configure clean `uv` environment with CPU PyTorch wheels (preventing disk bloat).
- [x] Reset `main` branch to the clean HaxballGym architecture.
- [x] Author comprehensive documentation (`AGENTS.md`, `CONTEXT.md`, `PROTOCOL.md`, `TODO.md`).
- [/] **In Progress**: Run full 1v1 PPO self-play training on `futsal-classic` (4,000 iterations).
- [ ] Evaluate saved checkpoints via Round-Robin tournament (`uv run src/select.py`).
- [ ] Generate visual `.hbr2` match replays and verify play quality on `haxball.com/replay`.

---

## Phase 2: Multi-Agent Scaling (2v2 & 3v3 Futsal)

- [ ] Configure multi-agent environment with custom stadium (`maps/futsal_3v3.hbs`).
- [ ] Implement parameter sharing policy for 3v3 teamplay.
- [ ] Evaluate transfer learning (warm-starting 3v3 policy from converged 1v1 weights vs scratch).
- [ ] Train 3v3 self-play with snapshot pool.
- [ ] Verify emergent multi-agent coordination (passing, goal-line covering, wall bank passes).

---

## Phase 3: Deployment & Room Hosting

- [ ] Connect trained 3v3 / 1v1 model to the headless room bridge (`play.py` with Node token).
- [ ] Test live room gameplay against human players in real-time.
- [ ] Benchmark latency and inference response time during live matches.
