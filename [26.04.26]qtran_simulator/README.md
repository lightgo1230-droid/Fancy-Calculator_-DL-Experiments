# QTRAN Simulator

A real-time reinforcement learning simulator built with **Rust + egui**.
Train up to 3 independent agents simultaneously on a configurable GridWorld maze,
compare five algorithms side-by-side, run hyperparameter search, and replay episodes —
all within a single native desktop application.

## Overview

| Item             | Details                                                              |
|------------------|----------------------------------------------------------------------|
| **Language**     | Rust (2021 edition)                                                  |
| **GUI**          | eframe / egui (immediate-mode)                                       |
| **Algorithms**   | Q-Learning · SARSA · Double-Q · DQN · PER-DQN                       |
| **Environments** | GridWorld 10×10 · MountainCar · Pendulum · CartPole · StockEnv      |
| **Multi-Agent**  | Up to 3 independent agents trained in parallel                       |
| **Curriculum**   | Auto difficulty scaling on goal-reach                                |
| **Reward**       | BFS potential-based shaping (maze-aware)                             |
| **Model I/O**    | Encrypted `.rls` files (XOR cipher + position-weighted checksum)     |
| **Hyper Search** | Grid search α × γ with heatmap visualisation                        |

## Quick Start

```bash
# Run the pre-built binary
qtran_simulator.exe       # Windows (double-click or terminal)

# Or build from source (requires Rust toolchain: https://rustup.rs)
cd rl_simulator
cargo build --release
.\target\release\rl_simulator.exe
```

## File Structure

```
qtran_simulator/
├── qtran_simulator.exe        # Pre-built Windows binary
├── addiction_results/         # Generated graphs and report images
├── QTRAN_Simulator_Report.docx
├── QMIX_vs_QTRAN_Comparison_Report.docx
```

## Algorithms

### Tabular Methods (Q-Learning / SARSA / Double-Q)

| Parameter | Value |
|-----------|-------|
| State space | 100 discrete cells (10×10) |
| Q-table | 100 × 4 float32 |
| α (learning rate) | 0.2 |
| γ (discount) | 0.99 |
| ε decay | 1.0 → 0.05  (rate = 0.993 per episode) |
| Init | Uniform ~ (−0.05, 0.05) |

**Q-Learning** — off-policy, greedy bootstrap target.
**SARSA** — on-policy, follows ε-greedy next action.
**Double-Q** — two separate tables reduce maximisation bias.

### Deep RL (DQN / PER-DQN)

| Component | Details |
|-----------|---------|
| State | 9-feature continuous vector |
| Network | 9 → Dense(64, ReLU) → Dense(32, ReLU) → Dense(4) |
| Optimiser | Adam (β₁=0.9, β₂=0.999) |
| Replay | 50 000 transitions (uniform or priority) |
| Target sync | Hard copy every 200 training steps |
| Gradient clip | L2 norm ≤ 10 |
| PER priority | ∝ \|TD error\|^0.6 + 0.01 |

**DQN** — experience replay + target network stabilises training.
**PER-DQN** — prioritised replay focuses on high-error transitions for faster convergence.

## Reward Shaping (BFS-Based)

Potential-based shaping adds a shaped bonus on each transition:

```
F(s, s') = 2.0 × ( d_BFS(s) − d_BFS(s') )
```

`d_BFS(s)` is the BFS shortest-path distance from cell `s` to the goal, computed once
per episode. This correctly rewards detours around walls and does **not** penalise
necessary lateral moves — a critical improvement over Manhattan distance.

## Curriculum Learning

Enable via the **Curriculum** toggle in the toolbar. Every N consecutive goal-reaches
(configurable 1–50) the difficulty increases by 5 % (max 0.90). Maps are validated
by BFS before each episode to guarantee solvability.

```
curriculum_difficulty += 0.05   (capped at 0.90)
```

## Environments

| Environment | State | Actions | Goal |
|-------------|-------|---------|------|
| GridWorld | 10×10 grid / 9-feat | 4 (UDLR) | Reach goal, avoid traps |
| MountainCar | pos + vel | 3 | Drive up the hill |
| Pendulum | cos θ, sin θ, ω | 3 | Balance upright |
| CartPole | x, ẋ, θ, θ̇ | 2 | Keep pole upright |
| StockEnv | price change, MA ratio | 3 | Maximise portfolio |


## License

MIT

---

Copyright © 2025 lightgo · lgithgo1230@gmail.com
