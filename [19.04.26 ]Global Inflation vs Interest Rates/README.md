# Global Inflation vs Interest Rates — 3-Year Forecast Dashboard

A desktop dashboard built with Rust + egui that visualizes global CPI inflation and policy rate data, with 3-year forecasts across all analysis tabs.

---

## Features

| Tab | Description |
|-----|-------------|
| **CPI Forecast** | Actual CPI YoY% with MA-3/6/12 overlays and 3-year recursive + OLS linear forecast |
| **Rate vs CPI** | Scatter plot coloured by rate decision (Hike/Hold/Cut) + MA-12 forecast trajectory in phase space |
| **Heatmap** | Pearson correlation matrix of CPI co-movement — toggle between historical-only and historical + 3Y forecast |
| **Comparison** | Multi-economy comparison for CPI, Policy Rate, and Real Rate with forecast extensions |
| **About** | Data metadata, forecast methodology, and Python ML pipeline overview |

---

## Forecast Methods

- **MA-3 / MA-6 / MA-12 Recursive** — Seeds a rolling buffer from the last N historical values, predicts each future month as the buffer mean, slides forward 36 months
- **OLS Linear Trend** — Fits slope/intercept on the last 24 months of valid data, extrapolates 36 months
- **Real Rate Forecast** — Computed as `CPI_ma12_forecast − Rate_ma12_forecast` per future month

---

## Economies Covered

Australia, Brazil, Canada, Eurozone, India, Japan, South Korea, United Kingdom, United States

---

## Project Structure

```
/
├── README.md
├── Cargo.toml          — Rust package manifest
├── build.rs            — Build-time icon generation (abstract "L" logo, ICO embedding)
└── src/
    ├── main.rs         — Full application source (~1200 lines)
    └── rates_vs_cpi_panel.csv  — Embedded data (compiled into binary)
```

---

## Build Instructions

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (stable, 1.70+)
- Windows with MSVC build tools (for icon embedding via `winres`)

### Build

```bash
cargo build --release
```

Output: `target/release/inflation_viewer.exe`

The CSV data is embedded at compile time via `include_bytes!` — no external data file required at runtime.

### Run

```bash
cargo run --release
# or double-click inflation_viewer.exe
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `eframe` | 0.28 | Native window + event loop |
| `egui` | 0.28 | Immediate-mode GUI |
| `egui_plot` | 0.28 | Plot / chart widgets |
| `csv` | 1.3 | CSV parsing |
| `winres` | 0.1 | Windows executable icon embedding (build-only) |

---

## Data Format (`rates_vs_cpi_panel.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | YYYY-MM-DD | Month |
| `economy` | string | Economy name |
| `policy_rate` | float | Central bank policy rate (%) |
| `cpi_yoy_pct` | float | CPI year-over-year change (%) |
| `real_rate_pct` | float | Real rate = policy_rate − cpi_yoy_pct |
| `rate_change_bps` | float | Rate change in basis points |
| `rate_action` | string | hike / hold / cut |
| `cycle_cumul_bps` | float | Cumulative bps change in current cycle |

---

## License

© lightgo — lightgo1230@gmail.com  
All rights reserved.
