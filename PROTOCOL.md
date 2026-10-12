# PROTOCOL.md — Training, Reward & Evaluation Protocols

This document defines the mathematical formulation, hyperparameter choices, and verification procedures governing our reinforcement learning pipeline.

---

## 1. PPO Self-Play Training Protocol

### 1.1 Hyperparameter Specifications

| Parameter | Value | Rationale |
|---|---|---|
| Parallel Envs (`envs`) | 512 | Maximizes CPU core utilization across Rust simulation threads |
| Rollout Steps ($T$) | 64 | Captures multi-second ball trajectories and offensive sequences |
| Total Steps per Iteration | $32,768$ | $512 \text{ envs} \times 64 \text{ steps}$ |
| PPO Epochs | 4 | Balances sample reuse without policy drift |
| Minibatches ($N_{\text{MB}}$) | 4 | Batch size of $8,192$ per gradient step |
| Discount ($\gamma$) | $\approx 0.98$ | Calibrated to 5.0 seconds half-life: $\gamma = \exp(\ln(0.5) / ((60/\text{skip}) \times 5.0))$ |
| GAE Lambda ($\lambda$) | 0.95 | Standard Generalized Advantage Estimation smoothing |
| PPO Clip ($\epsilon$) | 0.20 | Standard policy ratio clipping threshold |
| Value Loss Coef ($c_1$) | 0.50 | Scales critic gradient with respect to policy loss |
| Entropy Coef ($c_2$) | 0.01 | Sustained exploration entropy to prevent policy collapse |
| Learning Rate ($\alpha$) | $3 \times 10^{-4} \to 1 \times 10^{-4}$ | Linearly annealed across total training iterations |
| Max Grad Norm | 0.50 | Gradient clipping for numerical stability |
| Tick Skip | 8 | Physics ticks per agent decision step (~7.5 decisions/sec) |

---

## 2. Reward Formulation Protocol

### 2.1 The Goal-Dominated Reward Function
The reward signal at time $t$ for player $i$ is defined as:

$$R_t^{(i)} = R_{\text{goal}}^{(i)}(s_t) + R_{\text{dense}}^{(i)}(s_t, a_t)$$

#### Sparse Goal Reward ($R_{\text{goal}}$)
$$R_{\text{goal}}^{(i)}(s_t) = \begin{cases} +10.0 & \text{if team of player } i \text{ scores} \\ -10.0 & \text{if opponent scores} \\ 0.0 & \text{otherwise} \end{cases}$$

#### Dense Shaping Reward ($R_{\text{dense}}$)
$$R_{\text{dense}}^{(i)}(s_t, a_t) = w_{\text{move}} \cdot \left(\vec{v}_{\text{player}} \cdot \hat{u}_{\text{to\_ball}}\right) + w_{\text{prog}} \cdot \left(\vec{v}_{\text{ball}} \cdot \hat{u}_{\text{to\_opp\_goal}}\right)$$

- $w_{\text{move}}, w_{\text{prog}} \sim 0.01$: Small kinematic compass guiding early exploration.
- Because $\left|R_{\text{goal}}\right| \gg \left|R_{\text{dense}}\right|$, no local micro-movement can compensate for conceding or failing to score.

---

## 3. Evaluation & Tournament Protocol

### 3.1 Live Sanity Evaluation (`eval_vs_chase`)
- Executed every 50 iterations during training.
- Plays 400 steps over 256 games against the heuristic `ChaseBot`.
- Computes net goal differential: $\Delta_{\text{goals}} = G_{\text{agent}} - G_{\text{chase}}$.
- Useful for detecting immediate regressions or early learning progress, but noisy for subtle late-stage checkpoint selection.

### 3.2 Head-to-Head Checkpoint Tournament (`select.py`)
- After training completes, all periodic checkpoints in `models/ckpts/` (e.g. `it1000.pt`, `it2000.pt`, ..., `it4000.pt`) compete in a full round-robin tournament.
- Each pair $(C_i, C_j)$ plays $N$ headless matches with alternating sides (Red vs Blue) to cancel initial-condition bias.
- The winrate matrix $W \in \mathbb{R}^{K \times K}$ is constructed:
  - If progress is transitive and monotonic, $W_{i, j} > 0.5$ for all $i > j$.
  - The checkpoint with the highest net score / TrueSkill rating is promoted to `models/model.pt`.

### 3.3 Visual Replay Verification
- Selected checkpoints can be rendered to `.hbr2` format via `src/replay.py`.
- Replays are loaded directly into the official web visualizer at [haxball.com/replay](https://www.haxball.com/replay) to inspect shot mechanics, rebounds, and defensive positioning.
