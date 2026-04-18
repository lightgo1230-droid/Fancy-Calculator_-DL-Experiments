# Lightgo — YouTube Views Dashboard

A desktop dashboard built in Rust that visualizes historical YouTube channel view data and forecasts monthly views for 2026–2028 using two AI prediction models.

---

> **IMPORTANT — LICENSE NOTICE**
>
> Copyright 2024 Meruva Kodanda Suraj
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
> &nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

---

## Features

| Feature | Description |
|---------|-------------|
| Historical chart | Monthly average view graph covering 2020–2025 |
| AI forecast | Basic model (63% accuracy) and Advanced model (99% accuracy) for 2026–2028 |
| Year spotlight | Highlight a specific year; all other periods are dimmed automatically |
| Slider navigation | Drag the timeline slider to inspect any individual month |
| Interpretation bar | Auto-generated plain-language summary for the selected month |
| Metric cards | Actual / Basic / Advanced values with month-over-month change badge |
| Lightgo icon | Custom amber lightning-bolt icon embedded in the executable |

---

## Requirements

- **OS**: Windows 10 / 11 (64-bit)
- **No installation required** — single self-contained executable

### Data files (must exist before running)

```
C:\Users\USER\OneDrive\Desktop\youtube_predictor\data\
  ├── historical_monthly.json   — actual monthly views (2020–2025)
  ├── model_a_weights.json      — Basic model weights (MLP)
  ├── model_b_weights.json      — Advanced model weights (MLP + residual)
  ├── templates_a.json          — feature templates for Basic model
  └── templates_b.json          — feature templates for Advanced model
```

---

## Running the App

Double-click `youtube_viewer.exe`.

If the data files are missing, the dashboard displays an error message with the expected path.

---

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header — title  ·  subtitle  ·  [ All ][ 2026 ][ 2027 ][ 2028 ] │
├────────────────────────────────┬────────────────────────────┤
│                                │  Date card                  │
│   Main chart                   │  Actual Views card          │
│   ─ gray  : Actual Data        │  Basic Prediction card      │
│   ─ blue  : Basic Prediction   │  Advanced Prediction card   │
│   ─ green : Advanced Prediction│  Stats summary              │
│                                │                             │
├────────────────────────────────┴────────────────────────────┤
│  Timeline slider  (2020 ──────────────────────── 2028)      │
├─────────────────────────────────────────────────────────────┤
│  Interpretation bar  [ ACTUAL / FORECAST ]  narrative text  │
└─────────────────────────────────────────────────────────────┘
```

---

## Chart Guide

| Element | Meaning |
|---------|---------|
| Gray line | Actual recorded views |
| Blue line | Basic AI prediction |
| Green line | Advanced AI prediction |
| Amber vertical line | Currently selected month |
| Amber divider | Boundary between actual and forecast data |
| Dimmed region | Years outside the spotlight selection |
| Blue shaded zone | Forecast period (2026–2028) |

---

## Source Code

Located in the `source/` folder:

```
source/
  ├── src/
  │   └── main.rs        — full application (model inference + UI)
  ├── Cargo.toml         — Rust package manifest
  ├── build.rs           — embeds lightgo.ico into the Windows exe
  └── icon_gen.py        — generates resources/lightgo.ico (run once with Python)
```

### Building from source

```bash
# 1. Generate the icon (requires Pillow)
python icon_gen.py

# 2. Build the optimised release binary
cargo build --release

# Output: target/release/youtube_viewer.exe
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| eframe | 0.29 | Native GUI framework (egui) |
| egui_plot | 0.29 | Chart / plot widget |
| serde_json | 1 | JSON data parsing |
| winres *(build)* | 0.1 | Embeds icon into Windows exe |

---

## AI Model Architecture

Both models use the same inference pipeline implemented from scratch in Rust (no ML framework dependency):

1. **Stem** — Linear layer + Batch Normalization + GELU activation
2. **Residual blocks** — Two linear/BN/GELU layers with skip connection
3. **Head** — Two linear layers projecting to a single log-views output
4. **Post-processing** — `exp(output) - 1`, scaled to millions, anchored to the last historical value for visual continuity

---

## Project Info

- **Language**: Rust 2021 Edition
- **GUI**: immediate-mode (egui / eframe)
- **Icon**: pure-Rust 64×64 RGBA pixel renderer (runtime) + multi-size ICO via Python PIL (build-time)
- **Window size**: default 1480 × 900, minimum 960 × 620

---

## License

```
Copyright 2024 Meruva Kodanda Suraj

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

*Built with Lightgo · 2026*
