# RL Simulator

A real-time reinforcement learning simulator built with **Rust + egui**.
Train up to 3 independent agents simultaneously on a GridWorld maze environment,
compare algorithms, search hyperparameters, and replay episodes — all without
leaving the GUI.

## Features

| Feature | Details |
|---------|---------|
| Algorithms | Q-Learning, SARSA, Double-Q, DQN, PER-DQN |
| Environment | GridWorld 10×10 (walls, traps, goal) |
| Multi-agent | Up to 3 independent agents in parallel |
| Curriculum | Auto difficulty increase on goal reach |
| BFS Shaping | Maze-aware reward shaping (no detour penalty) |
| Visualisation | Q-value grid, V-value overlay, visit heatmap |
| Replay | Step-through last episode |
| Save / Load | Encrypted `.rls` model files (XOR + checksum) |
| Hyper Search | Grid search over α × γ with heatmap |

## Build & Run

```bash
# Prerequisites: Rust toolchain (https://rustup.rs)
cargo build --release
./target/release/rl_simulator          # Linux / macOS
target\release\rl_simulator.exe       # Windows
```

## File Structure

```
rl_simulator/
├── src/
│   ├── main.rs          # App UI, training loop, save/load
│   ├── agent.rs         # TabularAgent (Q/SARSA/DoubleQ), DqnAgent, PER-DQN
│   ├── env.rs           # GridWorld (BFS shaping), other environments
│   ├── main_app.rs      # Shared state, ViewOpts
│   ├── views.rs         # egui render helpers
│   └── crypto.rs        # XOR encryption + checksum for .rls files
├── Cargo.toml
├── README.md
├── addiction_results/   # Generated graphs and report assets
└── RL_Simulator_Report.docx
```

## Algorithms

### Tabular (Q-Learning / SARSA / Double-Q)
- State space: 100 discrete cells (10×10)
- Q-table: 100 × 4 float32 matrix
- α = 0.2 (fixed), γ = 0.99, ε decays 1.0 → 0.05 (dec=0.993)

### Deep (DQN / PER-DQN)
- 9-feature continuous state vector
  `[row/9, col/9, Δrow/9, Δcol/9, dist, block_up, block_dn, block_lt, block_rt]`
- 3-layer MLP: 9 → 64 → 32 → 4 (ReLU, He init)
- Adam optimiser, gradient clipping (norm ≤ 10), replay buffer 50 k, target-net sync every 200 steps
- PER-DQN: priority-proportional sampling (α=0.6)

## Reward Shaping

Potential-based shaping using **BFS shortest-path distance** from goal:

```
F(s, s') = 2.0 × (d_bfs(s) − d_bfs(s'))
```

This correctly rewards detours around walls, unlike Manhattan distance which
penalises necessary sidesteps and prevents convergence in maze tasks.

## Curriculum Learning

Enable **🎓 Curriculum** in the toolbar.
Every N successful goal-reaches (configurable 1–50), difficulty increases by 5 %:

```
curriculum_difficulty += 0.05   (max 0.90)
```

Random maps are generated via BFS-validated sampling to guarantee solvability.

## Save / Load (Encrypted)

Trained models are saved as binary `.rls` files encrypted with a 32-byte XOR
key and a position-weighted checksum.  Any modification of the file (even one
byte) is detected and rejected with an integrity error.

Click **💾 Save** / **📂 Load** in the toolbar, or type a custom path first.

## Results

See `addiction_results/` for training curves, success-rate charts, BFS vs
Manhattan comparison, and curriculum progression graphs.

## License

MIT

---

## Copyright

Copyright © 2026 lightgo · lgithgo1230@gmail.com
