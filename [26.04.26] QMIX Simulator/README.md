# ⚙ QMIX Simulator 

**Generated:** 2026-04-25
**Contact:** lightgo1230@gmail.com
**Copyright:** © 2026 lightgo. All rights reserved.

> *No rain, no flowers.*

---

## Overview

QMIX Simulator is a multi-agent simulation built with Rust + egui.
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
