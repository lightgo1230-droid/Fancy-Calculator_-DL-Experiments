# Lightgo SmartAnalytics Dashboard

> A professional smartphone market analytics dashboard built entirely in Rust.

---

## Overview

Lightgo SmartAnalytics is a native Windows desktop application that visualizes the **Smartprix Smartphones April 2026** dataset — 997 devices across 36 brands and 26 features — with zero external runtime dependencies.

All data is embedded directly in the binary at compile time. No internet connection, no installation, no CSV file required at runtime. Just run the `.exe`.

---

## Screenshots

| View | Description |
|------|-------------|
| Overview | KPI cards, stats strip, brand/category/processor charts, Top 5 table |
| Brand Stats | Score rankings, brand detail table, average price by category |
| Device Table | Sortable, filterable catalog of all 997 devices |
| About | Version and license information |

---

## Features

- **Single executable** — 4.7 MB, no installer, no runtime dependencies
- **Embedded dataset** — CSV data baked into the binary via `include_str!`
- **Embedded icon** — Custom `.ico` embedded in the PE resource section
- **4 views** — Overview, Brand Stats, Device Table, About
- **Interactive table** — Search by brand/model/processor/OS, filter by category, sort by any column
- **Live charts** — Bar charts with gradient coloring via egui_plot
- **Responsive layout** — All panels fit the window without scrolling (Dashboard & Brands)
- **Idle optimization** — Renders at ~10 fps when idle, full speed on input
- **Dark theme** — Custom deep-navy palette with role-based accent colors

---

## Tech Stack

| Component | Library / Version |
|-----------|-------------------|
| Language  | Rust 2021 edition |
| GUI framework | eframe 0.29 / egui 0.29 |
| Charts | egui_plot 0.29 |
| CSV parsing | csv 1 |
| Image loading | image 0.25 (ICO) |
| Icon embedding | winres 0.1 |
| Font | Segoe UI (Windows system font) |

---

## Dataset

| Field | Value |
|-------|-------|
| Source | [Smartprix](https://www.smartprix.com) |
| Period | April 2026 |
| Devices | 997 |
| Brands | 36 |
| Features | 26 columns |

Key columns used: `brand`, `model`, `category`, `price`, `spec_score`, `has_5g`, `has_nfc`, `processor`, `ram`, `storage`, `battery`, `screen_size`, `camera`, `os`

---

## Build

### Requirements

- Rust toolchain (MSVC, Windows 10/11)
- Windows SDK (for `rc.exe` — required by winres for icon embedding)
- The following files relative to the project root:

```
smartphone_dashboard/          ← project root
smartprix_smartphones_april_2026.csv   ← dataset (one level up)
icon.ico                               ← app icon  (one level up)
```

### Build commands

```powershell
# Development build
cargo build

# Optimized release build (LTO + stripped)
cargo build --release

# Output: target\release\smartphone_dashboard.exe
```

### Release profile

```toml
[profile.release]
opt-level     = 3
lto           = true
codegen-units = 1
strip         = true
```

---

## Project Structure

```
smartphone_dashboard/
├── src/
│   └── main.rs          # All application code (~1000 lines)
├── build.rs             # winres icon embedding
├── Cargo.toml
├── Cargo.lock
└── README.md
```

---

## Color System

| Token | Hex | Role |
|-------|-----|------|
| `BLUE` | `#6366F1` | Primary / Budget |
| `CYAN` | `#14B8A6` | Success / Mid-Range |
| `GOLD` | `#EAB308` | Warning / Premium |
| `RED`  | `#EF4444` | Danger / Flagship |
| `VIOLET` | `#A733D4` | Accent 5 |
| `SKY`  | `#0EA5E9` | Accent 6 |

---

## License

Copyright (c) 2026 **Lightgo**. All rights reserved.

Contact: [lightgo1230@gmail.com](mailto:lightgo1230@gmail.com)
