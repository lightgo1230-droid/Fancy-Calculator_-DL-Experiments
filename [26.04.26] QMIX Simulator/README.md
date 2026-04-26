# ⚙ QMIX Simulator 

**Generated:** 2026-04-25
**Contact:** lgithgo1230@gmail.com
**Copyright:** © 2026 lightgo. All rights reserved.

> *No rain, no flowers.*

---

## Overview

QMIX Simulator is a smart factory multi-agent simulation built with Rust + egui.
Agents autonomously transport raw materials → process at workstations → deliver finished goods.
Two control modes are supported: **BFS Heuristic** and **DQN Reinforcement Learning**.

---

## Simulation Results (200 Episodes)

| KPI | Value |
|-----|-------|
| Average Products Delivered | 0.0 |
| Max Products (single episode) | 0 |
| Min Products (single episode) | 0 |
| Average Episode Reward | -127.4 |
| Average Efficiency | 0.000 |
| Average Cycle Time (steps/product) | 300.0 |
| Average WIP | 0.00 |
| Average Total Distance | 13 |

---

## Graph Files (`graphs/` folder)

| File | Description |
|------|-------------|
| `01_products_per_episode.png` | Products delivered per episode + moving average |
| `02_episode_reward.png` | Cumulative reward per episode |
| `03_efficiency.png` | Efficiency (products per 100 distance units) |
| `04_cycle_time.png` | Cycle time — steps required per product |
| `05_wip.png` | Average Work In Progress per episode |
| `06_agent_deliveries.png` | Per-agent delivery counts |
| `07_kpi_dashboard.png` | 4-panel KPI overview dashboard |
| `08_kpi_distribution.png` | Box plot distribution for all KPIs |

---

## How to Use the Application

1. **Launch** `smart_factory.exe`
2. **Simulation tab** — Press `SPACE` to start/pause, `R` to reset
3. **Speed** — Use the slider (1×–16×) or `↑`/`↓` keys
4. **Mode** — Switch between *Heuristic* and *DQN Learn*
5. **Overlays** — Toggle *Path overlay* (BFS paths) and *Heatmap* (visit frequency)
6. **Map Design tab** — Paint cells, drag agents, adjust agent count (1–6)
7. **Training History tab** — View live DQN learning curves

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause |
| `R` | Reset episode |
| `P` | Toggle BFS path overlay |
| `↑` / `↓` | Increase / Decrease speed |
| `1`–`5` | Select cell type (Map Design) |
| `Ctrl+Z` | Undo (Map Design, up to 20 steps) |
| `ESC` | Cancel agent placement |

---

*© 2026 lightgo. All rights reserved.*
