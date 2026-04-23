# AI4I 2020 Predictive Maintenance System

> A standalone desktop application for real-time machine failure prediction using a PyTorch-trained MLP model, built with Rust + egui.

---

## Overview

This project trains a binary classification model on the **AI4I 2020 Predictive Maintenance Dataset** to predict machine failures from industrial sensor readings. The trained model is exported and embedded into a single Windows executable — no Python, no runtime, no installation required.

---

## Features

- **Real-time prediction** — adjusting any slider instantly updates the failure probability gauge
- **Semicircular gauge** with color-coded risk levels (green → yellow → orange → red)
- **Adjustable decision threshold** — tune sensitivity to match operational cost requirements
- **Prediction history** — record and review up to 50 entries per session
- **Fully self-contained** — model weights, fonts, and icon are all compiled into the binary

---

## Dataset

| Item | Detail |
|---|---|
| Source | AI4I 2020 Predictive Maintenance Dataset |
| Samples | 10,000 |
| Normal | 9,661 (96.6 %) |
| Failure | 339 (3.4 %) |
| Class imbalance ratio | ~28.5 : 1 |
| Missing values | None |

**Input features used:**

| Feature | Type |
|---|---|
| Machine Type (L / M / H) | Categorical → One-Hot Encoded |
| Air Temperature \[K\] | Numerical → StandardScaler |
| Process Temperature \[K\] | Numerical → StandardScaler |
| Rotational Speed \[rpm\] | Numerical → StandardScaler |
| Torque \[Nm\] | Numerical → StandardScaler |
| Tool Wear \[min\] | Numerical → StandardScaler |

> Sub-type failure labels (TWF, HDF, PWF, OSF, RNF) are **excluded** to prevent data leakage.

---

## Model Architecture

```
Input (8) → Linear(128) → BN → ReLU → Dropout(0.3)
          → Linear(64)  → BN → ReLU → Dropout(0.3)
          → Linear(32)  → BN → ReLU
          → Linear(1)   → Sigmoid
```

| Setting | Value |
|---|---|
| Loss function | BCEWithLogitsLoss |
| pos_weight | 28.50 (= 9661 / 339) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Early Stopping | patience=20, metric=Val F1 |

---

## Model Performance

### Hold-Out Test Set (15 %)

| Metric | Value |
|---|---|
| AUC-ROC | **0.9727** |
| Recall | **0.9216** |
| F1-Score | 0.3837 |
| Precision | 0.2423 |

### 5-Fold Stratified Cross-Validation

| Metric | Mean ± Std |
|---|---|
| AUC-ROC | 0.9650 ± 0.0037 |
| Recall | 0.8879 ± 0.0068 |
| F1-Score | 0.4049 ± 0.0352 |

> In predictive maintenance, **Recall** (minimising missed failures) is the primary metric.

---

## Quick Start

1. Double-click `AI4I_Predictive_Maintenance.exe`
2. Select Machine Type: **L / M / H**
3. Drag the sensor sliders to match the equipment readings
4. Read the failure probability on the gauge
5. Click **▶ RECORD PREDICTION** to save the result to history

No installation, no Python, no internet connection required.

---

## UI Layout

```
┌─ Top Bar ──────────────────────────────────────────────────┐
│  ⚙ AI4I 2020 | Predictive Maintenance System               │
├─ Left Panel (380 px) ─────┬─ Central Panel ───────────────┤
│  Machine Type  [L][M][H]  │  Semicircular probability gauge│
│  Air Temp      ━━━●━━━    │  Status badge  ✓/⚠            │
│  Process Temp  ━━━●━━━    │  Metric cards (4)              │
│  Rot. Speed    ━━━●━━━    │  Prediction history table      │
│  Torque        ━━━●━━━    │                                │
│  Tool Wear     ━━━●━━━    │                                │
│  Threshold     ━━━●━━━    │                                │
│  [▶ RECORD]               │                                │
│  [✕ Clear]                │                                │
│  © 2026 lightgo           │                                │
└───────────────────────────┴────────────────────────────────┘
```

---

## Risk Level Reference

| Gauge Color | Probability | Risk Level | Recommended Action |
|---|---|---|---|
| Green | 0 – 30 % | LOW | Normal operation |
| Yellow | 30 – 55 % | MEDIUM | Increase monitoring frequency |
| Orange | 55 – 75 % | HIGH | Schedule maintenance soon |
| Red | 75 – 100 % | CRITICAL | Immediate inspection required |

---

## Threshold Guidance

| Use Case | Recommended Threshold |
|---|---|
| Critical equipment / high failure cost | 0.20 – 0.30 |
| General monitoring | 0.50 (default) |
| Low false-alarm tolerance | 0.65 – 0.80 |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Model training | Python · PyTorch · scikit-learn |
| Desktop UI | Rust · eframe · egui 0.29 |
| Font rendering | Segoe UI + Segoe UI Symbol (embedded) |
| Executable icon | winresource (embedded .ico) |
| Model weights | Embedded via `include_str!` (252 KB JSON) |
| C Runtime | Statically linked (`crt-static`) |

---

## Project Files

```
Desktop/
├── AI4I_Predictive_Maintenance.exe   ← Run this (standalone, 8 MB)
├── ai4i2020.csv                      ← Original dataset
├── ai4i_pytorch.py                   ← Model training script
├── AI4I_PyTorch_Report.docx          ← English model report
├── AI4I_PyTorch.docx                 ←  model report
├── AI4I_.docx                        ←  usage guide
├── addiction_results/                ← Saved graphs (10 PNG files)
└── maintenance_ui/                   ← Rust source project
    ├── src/main.rs
    ├── fonts/
    │   ├── segoeui.ttf
    │   └── seguisym.ttf
    └── Cargo.toml
```

---

## Contact

**lightgo** · lightgo1230@gmail.com

---

*© 2026 lightgo. All rights reserved.*
