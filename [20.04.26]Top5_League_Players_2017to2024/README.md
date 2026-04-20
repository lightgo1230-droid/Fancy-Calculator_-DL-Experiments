# ⚽ Football Analytics Dashboard

> **A standalone Windows desktop application for analyzing Top 5 European League player statistics (2017–2024)**
> Built with Rust · egui · Fully self-contained — no installation, no internet required

---

## Screenshot

| Tab | Description |
|-----|-------------|
| 👤 Player / Team | Filter & rank players by any metric |
| 📈 Time-Series Prediction | 3-season forecast with quarterly breakdown |
| 💶 Market Value | Estimated transfer value per player |
| 🗺 Tactical Trends | League-level metric trends across 7 seasons |

---

## Quick Start

1. Download `football_analyzer.exe`
2. Double-click to run — **no setup required**
3. All data (13 MB CSV) is embedded inside the executable

---

## Features

### Tab 1 — Player / Team Analysis
- Search by player name or team name
- Filter by **league**, **position**, **season**
- Top 20 bar chart ranked by selected metric
- Summary stats (total goals, avg xG, avg minutes)
- **Metrics:** Goals · Assists · xG · Shots · SCA · GCA · Touches · Tackles · Minutes

### Tab 2 — Time-Series Prediction
- **Cascade dropdown selection:** League → Team → Player → Metric
- Forecast model: **Holt Double Exponential Smoothing** (α=0.35, β=0.20)
- **3-season ahead forecast** (25/26, 26/27, 27/28)
- **Quarterly breakdown** per forecast season
  - Q1 Aug–Oct · Q2 Nov–Jan · Q3 Feb–Mar · Q4 Apr–Jun
- Annual trend chart with **95% confidence interval** band
- Historical season table + forecast table side by side

### Tab 3 — Market Value Estimation
- **Cascade dropdown selection:** League → Team → Player → Season
- Estimated transfer value in **€M** based on:
  - Age curve (peak at ~25 years)
  - Position base value (FW > MF > DF > GK)
  - Performance multiplier (Goals, Assists, xG, Minutes, GCA, SCA)
  - League premium coefficient
- **Player tier badge:** World Class / Top Player / Good Player / Squad Player / Rotation
- Team squad ranking bar chart (selected player highlighted in green)
- League average market value comparison chart

### Tab 4 — Tactical Trends
- Average metric per player-season across all 5 leagues
- Multi-line trend chart (2017/18 → 2023/24), color-coded by league
- Season-over-season **% change** chart

---

## Data

| Item | Detail |
|------|--------|
| Leagues | Premier League · La Liga · Bundesliga · Serie A · Ligue 1 |
| Seasons | 2017/18 – 2023/24 |
| Total records | 22,929 player-seasons |
| Filtered records | ~14,166 (≥ 90 minutes played) |
| File format | CSV, semicolon-separated, European decimal comma |
| Embedding | Compiled into the executable via `include_bytes!` |

---

## Build from Source

### Requirements

| Tool | Version |
|------|---------|
| Rust toolchain | 1.75+ (edition 2021) |
| Windows | 10 / 11 |
| Windows SDK | For icon embedding (`rc.exe`) |

### Steps

```bash
git clone <repository-url>
cd football_analyzer
cargo build --release
```

Output binary: `target/release/football_analyzer.exe` (~17 MB, fully self-contained)

---

## Project Structure

```
football_analyzer/
├── assets/
│   └── data.csv          # Source CSV (embedded at compile time)
├── src/
│   └── main.rs           # Full application (~1000 lines)
├── build.rs              # Generates icon.ico + embeds Windows resource
├── icon.ico              # Auto-generated L icon (16/32/48 px)
├── Cargo.toml
├── README.md
└── README_KR.md
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `eframe` | 0.27 | Native window + egui runtime |
| `egui_plot` | 0.27 | Line charts, bar charts |
| `winres` | 0.1 | Embed `.ico` icon into the `.exe` |

---

## Algorithms

### Time-Series Forecast — Holt Double Exponential Smoothing

```
Level:    L(t) = α · y(t) + (1 − α) · (L(t−1) + B(t−1))
Trend:    B(t) = β · (L(t) − L(t−1)) + (1 − β) · B(t−1)
Forecast: ŷ(T+h) = L(T) + h · B(T)

α = 0.35  (level smoothing)
β = 0.20  (trend smoothing)
```

Quarterly decomposition uses fixed seasonal weights `[0.30, 0.25, 0.20, 0.25]`
with a small golden-ratio perturbation per season to avoid identical quarters.

### Market Value Formula

```
Value (€M) = base × age_factor × (perf / 5.5) × league_premium

age_factor    = exp(−0.045 × (age − 25)²)

base          FW → 18   MF → 13   DF → 9   GK → 6

perf          = 1 + goals×1.0 + assists×0.6 + xG×0.4
                  + min(minutes / 3000, 1.2) × 1.8
                  + GCA×0.25 + SCA×0.04

league_prem   Premier League → 1.55   La Liga → 1.35
              Bundesliga → 1.30       Serie A → 1.20
              Ligue 1 → 1.10
```

---

## License

© lightgo · lightgo1230@gmail.com
