# Blackjack RL — Learning Dashboard

A real-time desktop dashboard for training and visualizing a Q-Learning reinforcement learning agent in a Blackjack environment.

---

## Features

- **Q-Learning Agent** — Tabular Q-Table based reinforcement learning
- **Live Game Visualization** — Dealer/player cards, hand totals, and actions (Hit/Stand) displayed in real time
- **Training Phase Indicator** — Automatic transition: Exploring → Exploiting → Continuous
- **Policy Heatmap** — Agent policy visualization for Hard and Soft hands (with Basic Strategy diff overlay)
- **Statistics Dashboard** — Win rate, loss rate, push rate, expected value (EV), rolling win rate, and streaks
- **Charts** — Win rate trend / EV trend / Win rate by dealer up-card (bar chart)
- **Auto Save / Load** — Q-Table persisted as an XOR-encrypted `.bin` file, auto-loaded on restart
- **Episode Limit Presets** — 50K / 100K / 200K / 500K / 1M / 2M / Unlimited
- **Responsive UI** — Layout auto-adjusts to window size

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.03 |
| Discount factor (γ) | 1.0 |
| Initial epsilon (ε) | 1.0 |
| Minimum epsilon | 0.05 |
| Decay steps | 200,000 |
| Rolling window | 2,000 episodes |

---

## State Space

- **Player sum** (4 – 21) × **Dealer up-card** (A – 10) × **Usable Ace** (yes/no)
- 360 total states, 2 actions (Hit = 0, Stand = 1)

---

## Controls

| Button | Action |
|--------|--------|
| ⏸ Pause / ▶ Resume | Pause or resume training |
| ⏹ Stop | Stop training |
| ▶ Resume (keep table) | Restart while keeping the current Q-Table |
| ↺ Fresh Start | Reset Q-Table and start from scratch |
| Step delay slider | Control visualization speed (0 – 800 ms) |

---

## Build & Run

```bash
cargo build --release
./target/release/rl_blackjack.exe
```

### Requirements
- Rust 1.75 or later
- Windows 10 / 11

### Dependencies
| Crate | Purpose |
|-------|---------|
| eframe 0.28 | GUI framework |
| egui_plot 0.28 | Chart rendering |
| rand 0.8 | Random number generation |
| image 0.24 | Icon loading (ICO) |
| winres 0.1 | Executable icon embedding |

---

## Q-Table Save Path

Auto-saved every 100,000 episodes and automatically loaded on next launch.
