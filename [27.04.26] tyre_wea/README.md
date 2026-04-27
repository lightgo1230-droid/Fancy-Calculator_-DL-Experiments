# Tyre Wea — Tyre Wear Analysis Desktop Application

A high-performance desktop application for tyre wear simulation and brand analysis, built with **Rust + egui**.  
The full dataset (4,350 records · 16 brands) is embedded directly in the executable — no external files required.

---

## Quick Start

```
tyre_wea.exe
```

No installation. No dependencies. Single self-contained binary.

---

## Features

| Tab | Description |
|-----|-------------|
| **3D Simulation** | Interactive 3D surface — brand × distance × wear remaining. Drag to rotate, scroll to zoom. |
| **Brand Comparison** | Wear curves for all selected brands on a chosen road type. Replacement markers included. |
| **Road Analysis** | Compare wear across all 7 road types for a selected brand. Bar chart + data table. |
| **Cost Analysis** | Price vs durability scatter + annual cost bar chart. Supports advanced wear modifiers. |
| **Tyre Section** | Cross-section view of 5 tread zones (shoulder / center). Diagnosis panel included. |
| **Data Table** | Full brand statistics table. Sortable by any column. Live search filter. |

---

## Road Types

| Road | Wear Multiplier |
|------|----------------|
| Highway | ×0.75 |
| Mixed (baseline) | ×1.00 |
| Wet | ×1.10 |
| City | ×1.30 |
| Mountain | ×1.60 |
| Off-Road | ×2.20 |
| Racing | ×3.00 |

---

## Advanced Wear Model

When enabled, five additional modifiers are applied on top of the road multiplier:

- **Surface temperature** — above 30 °C: +0.6 % wear per degree
- **Load factor** — overloading accelerates wear non-linearly (power 1.6)
- **Speed factor** — above 80 km/h baseline (power 1.3)
- **Pressure factor** — both under- and over-inflation add wear
- **Compound type** — Touring (×0.85) → Sport (×1.40)

---

## Durability Formula

```
Durability (km) =
    20,000
  + max(0, load_index − 75) × 300
  + avg_rating × 4,000
  − max(0, width − 160)    × 30
  + max(0, aspect − 60)    × 100
  + price_normalised        × 15,000

  minimum: 22,000 km
```

Brand tier is assigned by price percentile:  
`> 60 % → Premium` · `25–60 % → Standard` · `< 25 % → Economy`

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start / Pause simulation |
| `R` | Reset distance to 0 km |
| `Enter` | Run wear snapshot (RUN) |
| `H` | Toggle help window |

---

## Dataset

- **Source**: Car Tyres Dataset (India market)  
- **Records**: 4,350 rows  
- **Brands**: 16 tyre brands  
- **Fields used**: Tyre Brand, Load Index, Size, Selling Price, Rating  
- **Embedded**: compiled into the binary via `include_bytes!`

---

## Analysis Charts (`addiction_results/`)

Pre-generated PNG charts exported from the dataset:

| File | Contents |
|------|----------|
| `01_brand_wear_comparison.png` | Wear curves for all 16 brands (Mixed road) |
| `02_road_type_analysis.png` | Road comparison curves + replacement distance bar chart |
| `03_cost_analysis.png` | Price vs durability scatter · Annual cost ranking |
| `04_brand_tier_distribution.png` | Tier share pie · Avg durability by tier |
| `05_top10_durability.png` | Top 10 brands ranked by estimated durability |
| `06_rating_vs_cost.png` | Rating vs cost-per-1000 km scatter |
| `07_replace_distance_heatmap.png` | Brand × Road type replacement distance heatmap |
| `08_tyre_section_wear.png` | Zone wear under 5 pressure/load/road scenarios |

---

## Technical Stack

| Component | Version |
|-----------|---------|
| Language | Rust 2021 edition |
| GUI framework | egui 0.28 / eframe 0.28 |
| Charts | egui_plot 0.28 |
| CSV parsing | csv 1.x |
| Build target | Windows x86-64 (native) |

---

## File Structure

```
20260427-1/
├── tyre_wea.exe            ← Main application (dataset embedded)
├── README.md               ← This file
└── addiction_results/      ← Pre-generated analysis charts (PNG)
    ├── 01_brand_wear_comparison.png
    ├── 02_road_type_analysis.png
    ├── 03_cost_analysis.png
    ├── 04_brand_tier_distribution.png
    ├── 05_top10_durability.png
    ├── 06_rating_vs_cost.png
    ├── 07_replace_distance_heatmap.png
    └── 08_tyre_section_wear.png
```
