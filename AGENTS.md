# AGENTS.md — Agent Architecture & Multi-Agent Design

This document details the agent architectures, observation/action representations, and multi-agent scaling strategies used in the Haxball reinforcement learning environment.

---

## 1. Agent Classification

### 1.1 `NeuralBot` (Actor-Critic Policy)
The core reinforcement learning agent trained via PPO self-play.
- **Model Architecture**:
  - Multi-Layer Perceptron (MLP) with 2 hidden layers of 256 units and `Tanh` activations.
  - Orthogonal weight initialization (`gain=\sqrt{2}` for hidden layers, `gain=0.01` for policy heads, `gain=1.0` for value head).
- **Observation Space** (`obs_dim = 16` in 1v1):
  - Egocentric and goal-relative: the agent is always oriented so that the opponent's goal is forward (+X direction).
  - Encodes:
    - Own disc position $(x, y)$ and velocity $(v_x, v_y)$
    - Ball position $(x, y)$ and velocity $(v_x, v_y)$
    - Opponent disc position $(x, y)$ and velocity $(v_x, v_y)$
    - Relative displacement and distance vectors
- **Action Space** (`n_acts = [3, 3, 2]`):
  - Decoupled categorical actions:
    1. **X-Movement**: `{-1, 0, 1}` (Left, None, Right)
    2. **Y-Movement**: `{-1, 0, 1}` (Up, None, Down)
    3. **Kick**: `{0, 1}` (Release, Press)
  - Joint action sampled as independent categorical distributions:
    $$\pi(a_x, a_y, a_{\text{kick}} \mid s) = \pi_x(a_x \mid s) \cdot \pi_y(a_y \mid s) \cdot \pi_k(a_{\text{kick}} \mid s)$$

### 1.2 `ChaseBot` (Heuristic Scripted Baseline)
Used as an invariant baseline for evaluation:
- Directly tracks the ball coordinates from the simulator state.
- Predicts ball intercept trajectory and navigates toward the target point.
- Automatically triggers a kick when within kicking range ($\text{distance} \le r_{\text{player}} + r_{\text{ball}} + 4.0$).
- Fast and deterministic, providing a sanity score against random or untrained policies.

---

## 2. Multi-Agent Scaling (1v1 $\to$ 2v2 $\to$ 3v3)

### 2.1 Parameter Sharing (Homogeneous Teams)
For team games ($N \ge 2$), we employ **Decentralized Execution with Parameter Sharing**:
- All players on the same team share the exact same neural network policy $\theta$.
- Each player receives their own egocentric observation vector $o_i$.
- **Benefits**:
  - Sample efficiency scales linearly with team size (each game yields $N$ agent trajectories per team).
  - Eliminates the need to maintain distinct policies for each player.
  - Emergent role flexibility (any player can dynamically become the striker or defender depending on relative game state).

### 2.2 Cooperative Credit Assignment
- **Shared Sparse Objective**: When a goal is scored, every agent on the scoring team receives the full reward ($+10.0$), while every agent on the conceding team receives $(-10.0)$.
- **Emergent Behaviors**:
  - **Passing**: If passing the ball to an open teammate produces a higher expected state value $V(s')$, the agent learns to pass rather than attempt a solo dribble.
  - **Covering/Defending**: A teammate positioned further back learns that moving toward their own goal line lowers the risk of conceding a $-10.0$ penalty, producing natural goalkeeping/sweeper behavior.

---

## 3. Snapshot Pool (Anti-Cycling Adversary)

In self-play, playing exclusively against the latest checkpoint creates catastrophic policy cycles (rock-paper-scissors dynamics).

To guarantee monotonic skill progression:
1. Every 20 iterations, a copy of the current policy weights is appended to a snapshot pool (capped at 40 models, spanning ~800 iterations).
2. For each training episode:
   - With probability $p = 0.5$, the opponent is a frozen snapshot randomly sampled from the pool.
   - With probability $p = 0.5$, the opponent is a frozen clone of the current policy.
3. This forces the learner to beat both its current style and all historically viable strategies, ensuring transitive skill accumulation.
