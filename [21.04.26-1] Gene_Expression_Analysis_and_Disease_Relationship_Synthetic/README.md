# Gene Expression Analysis & Disease Relationship

> **Synthetic single-cell gene expression dataset — PyTorch Autoencoder + UMAP visualization**

---

## Overview

This project performs dimensionality reduction and cluster analysis on a synthetic single-cell gene expression dataset.
A **PyTorch Autoencoder** is trained to extract a compact 16-dimensional latent representation from 6 gene features,
followed by **UMAP** projection to 2D for visualization.
Results are saved as high-resolution PNG images and packaged into a professional **Rust desktop dashboard**.

---

## Dataset

| Item | Detail |
|---|---|
| Total Cells | 3,000 |
| Cell Types | T_Cell (1,000) · Cancer (1,000) · Fibroblast (1,000) |
| Disease Status | Tumor (2,595) · Healthy_Control (405) |
| Feature Columns | 6 (Gene_E_HK, Gene_A_Onco, Gene_B_Immune, Gene_C_Stromal, Gene_D_Therapy, Pathway_Inflam) |
| Normalization | Z-score via PyTorch (per-feature mean / std) |

---

## Analysis Pipeline

```
CSV Data
  └─ PyTorch Z-score Normalization
       └─ Autoencoder Training  (6 → 64 → 16 → 64 → 6, 100 epochs, Adam, MSELoss)
            ├─ Raw Feature UMAP     → 01, 02 PNG
            └─ Latent Space UMAP   → 03, 04 PNG
```

### Autoencoder Architecture

| Layer | Input | Output | Activation |
|---|---|---|---|
| Encoder FC-1 | 6 | 64 | BatchNorm + ReLU |
| Encoder FC-2 | 64 | 64 | BatchNorm + ReLU |
| Encoder FC-3 | 64 | 16 | — (Latent) |
| Decoder FC-1 | 16 | 64 | BatchNorm + ReLU |
| Decoder FC-2 | 64 | 64 | BatchNorm + ReLU |
| Decoder FC-3 | 64 | 6 | — (Reconstruction) |

**Training config:** Adam (lr=1e-3, weight_decay=1e-5) · StepLR (step=30, γ=0.5) · Batch 256 · 100 epochs  
**Final MSE loss:** ~0.006

---

## Output Files (`addiction_results/`)

| File | Description |
|---|---|
| `01_umap_raw_cell_type.png` | UMAP of raw features — Cell Type |
| `02_umap_raw_disease_status.png` | UMAP of raw features — Disease Status |
| `03_umap_latent_cell_type.png` | UMAP of AE latent space — Cell Type |
| `04_umap_latent_disease_status.png` | UMAP of AE latent space — Disease Status |
| `05_autoencoder_loss.png` | Training loss curve |
| `Gene_Expression_Dashboard.exe` | Rust desktop dashboard (standalone) |

---

## Key Findings

- **Cancer** cells show highest `Gene_A_Oncogene` (~14.0) and `Gene_D_Therapy` (~10.0)
- **T_Cell** has highest `Gene_B_Immune` (~12.0) and `Pathway_Score_Inflam` (~10.9)
- **Fibroblast** is dominated by `Gene_C_Stromal` (~13.5)
- All three cell types form **well-separated clusters** in both raw and latent UMAP space

---

## How to Run

### Python Analysis
```bash
# Install dependencies
pip install torch umap-learn matplotlib pandas scikit-learn reportlab python-docx Pillow

# Run UMAP analysis
python umap_analysis.py
```

### Dashboard (Windows)
```
Double-click  addiction_results/Gene_Expression_Dashboard.exe
```
> No installation required. Standalone single binary.

---

## Tech Stack

| Category | Library / Tool |
|---|---|
| Deep Learning | PyTorch 2.x |
| Dimensionality Reduction | umap-learn |
| Visualization | Matplotlib |
| Reporting | ReportLab · python-docx |
| Desktop UI | Rust · egui 0.29 · eframe 0.29 |

---

## License

© 2026 lightgo · lightgo1230@gmail.com
