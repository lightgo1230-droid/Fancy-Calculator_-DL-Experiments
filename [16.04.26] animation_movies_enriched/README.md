# Animation Movie Dataset — PyTorch Machine Learning Projects (3-in-1)

> A multi-paradigm ML suite built on a single dataset of **25,390 animation films (1878–2029)**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Environment & Installation](#3-environment--installation)
4. [Project 1 — NLP Genre Classification & Recommendation](#4-project-1--nlp-genre-classification--recommendation)
5. [Project 2 — Tabular Rating Prediction ⭐ Improvements Applied](#5-project-2--tabular-rating-prediction)
6. [Project 3 — Graph Network Analysis](#6-project-3--graph-network-analysis)
7. [Results Summary](#7-results-summary)
8. [File Reference](#8-file-reference)

---

## 1. Project Overview

| # | File | Paradigm | Model | Tasks |
|---|------|----------|-------|-------|
| 1 | `01_nlp_genre_classification.py` | NLP | DistilBERT fine-tuning | Multi-label genre classification + recommendation |
| 2 | `02_tabular_rating_prediction.py` | Tabular | Entity Embedding MLP | Rating regression + Hit binary classification |
| 3 | `03_graph_network_analysis.py` | Graph Neural Network | HeteroGNN (PyG) | Node classification + Link prediction |

---

## 2. Dataset

| Property | Value |
|----------|-------|
| File | `animation_movies_enriched_1878_2029.csv` |
| Location | `C:\Users\USER\OneDrive\Desktop\` |
| Records | 25,390 movies |
| Columns | 44 features |
| Year range | 1878 – 2029 |

---

## 3. Environment & Installation

```
Python          : C:\Users\USER\anaconda3\python.exe
PyTorch         : 2.10.0 + cpu
transformers    : 5.5.4
torch_geometric : 2.7.0
OS              : Windows 11
```

```bash
pip install -r requirements.txt

# Project 3 — PyG (CPU):
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

---

## 4. Project 1 — NLP Genre Classification & Recommendation

### Goal
Predict multi-label genres from synopsis text only, and recommend similar movies via cosine similarity.

### Architecture
```
Overview text
    → DistilBertTokenizer (max_len=128)
    → DistilBertModel → [CLS] vector (768-dim)
    → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→10)
    → BCEWithLogitsLoss (multi-label)
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Pretrained model | distilbert-base-uncased |
| Max length | 128 tokens |
| Batch size | 32 |
| Epochs | 4 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight_decay=0.01) |

### Results

| Metric | Value |
|--------|-------|
| F1-micro | **69.41%** |
| F1-weighted | 57.38% |
| Jaccard | 67.43% |
| Exact Match | 43.69% |
| Hamming Loss | 0.1218 |

---

## 5. Project 2 — Tabular Rating Prediction

> ### ⭐ Improvements Applied (v2 — 16-04-26(1))
> Three immediate improvements were applied over the baseline.
> See the full before/after breakdown below.

---

### Goal
Predict TMDB rating (regression) and classify Hit/Not-Hit (binary) using **metadata only** — no text involved.

### Architecture

```
[7 categorical cols] → Entity Embedding
[numeric + boolean]  → StandardScaler
         ↓ concat
   Shared Backbone MLP (512 → 256 → 128)
         ↓
Regression head (→ rating)  +  Classification head (→ Hit/Not-Hit)
```

---

### 🔴 Before (Baseline)

#### Numeric Features (4 features)

```python
num_features = [
    "Release_Year",    # release year
    "Runtime",         # runtime in minutes
    "Vote_Count_Log",  # log1p(vote count)
    "Popularity_Log",  # log1p(popularity)
]
```

#### Loss Function

```python
mse_loss   = nn.MSELoss()           # sensitive to outlier ratings
bce_loss   = nn.BCEWithLogitsLoss()
LAMBDA_CLS = 0.5                    # regression : classification = 5 : 5
```

#### Loss Calculation

```python
loss = mse_loss(reg_out, y_reg) + LAMBDA_CLS * bce_loss(cls_out, y_cls)
#      ↑ MSE: squares the error → extreme ratings pull learning disproportionately
#                                       ↑ 0.5: classification competes with regression
```

#### Baseline Performance

| Metric | Value |
|--------|-------|
| MAE | 1.0393 |
| RMSE | 1.5096 |
| **R²** | **11.14%** ← very low |
| Accuracy (cls) | 71.56% |
| ROC-AUC | 75.34% |

---

### 🟢 After (v2 — Improvements Applied)

#### Improvement 1 — 5 Interaction Features Added

Expanded from 4 → **9 numeric features** (added inside `preprocess()`)

```python
# ① Vote count × popularity: both high = proven popular title
df["Vote_x_Pop"]        = df["Vote_Count_Log"] * df["Popularity_Log"]

# ② Viral score: popularity relative to vote count (short-term buzz)
df["Viral_Score"]       = np.log1p(
    df["TMDB_Popularity"].fillna(0) / (df["TMDB_Vote_Count"].fillna(0) + 1)
)

# ③ Movie age: older films tend to converge to stable ratings
df["Movie_Age"]         = (2026 - df["Release_Year"]).clip(0, 200)

# ④ Age × vote log: captures "classic" films (old + many votes)
df["Age_x_VoteLog"]     = df["Movie_Age"] * df["Vote_Count_Log"]

# ⑤ Runtime × vote log: long film + many votes = high-rated epic pattern
df["Runtime_x_VoteLog"] = df["Runtime"] * df["Vote_Count_Log"]
```

```python
# Before: 4 features
num_features = ["Release_Year", "Runtime", "Vote_Count_Log", "Popularity_Log"]

# After: 9 features (+5 interaction)
num_features = ["Release_Year", "Runtime", "Vote_Count_Log", "Popularity_Log",
                "Vote_x_Pop", "Viral_Score", "Movie_Age",
                "Age_x_VoteLog", "Runtime_x_VoteLog"]
```

> **Why it helps:** The model previously saw only 4 raw attributes. Now it also receives explicit products and ratios between them — allowing the MLP to learn non-linear relationships without relying solely on the backbone's depth.

---

#### Improvement 2 — Loss Function: MSE → Huber Loss

```python
# Before
mse_loss = nn.MSELoss()
# loss = mse_loss(reg_out, y_reg) + ...
# Problem: error² means a 3-point miss contributes 9× more than a 1-point miss
# → rare extreme ratings (1–2, 9–10) distort the gradient disproportionately

# After
huber_loss = nn.HuberLoss(delta=1.0)
# loss = huber_loss(reg_out, y_reg) + ...
# |error| ≤ 1.0 → behaves like MSE (smooth, precise learning)
# |error| > 1.0 → behaves like MAE (linear, outlier-resistant)
```

| Error size | MSE penalty | Huber penalty |
|-----------|-------------|---------------|
| 0.5 pts | 0.25 | 0.125 |
| 1.0 pts | 1.00 | 0.500 |
| 2.0 pts | **4.00** | **1.500** ← dampened |
| 3.0 pts | **9.00** | **2.500** ← dampened |

> **Why it helps:** Extreme rating outliers no longer dominate the gradient. The model learns more balanced predictions across the full 1–10 range.

---

#### Improvement 3 — LAMBDA_CLS: 0.5 → 0.1

```python
# Before
LAMBDA_CLS = 0.5
loss = mse_loss(reg_out, y_reg) + 0.5 * bce_loss(cls_out, y_cls)
# Classification loss accounts for ~33% of total loss
# → Shared backbone is pulled in two directions simultaneously
# → Regression head receives weaker gradient signal

# After
LAMBDA_CLS = 0.1
loss = huber_loss(reg_out, y_reg) + 0.1 * bce_loss(cls_out, y_cls)
# Classification loss accounts for only ~9% of total loss
# → Backbone focuses on rating prediction
# → Regression head dominates learning
```

| | Before | After |
|---|--------|-------|
| Regression loss share | ~67% | ~91% |
| Classification loss share | ~33% | ~9% |
| Dominant task | Mixed | **Regression** |

> **Why it helps:** The shared backbone's gradient is no longer pulled between two competing objectives. The regression head learns more effectively, which is the primary goal of this project.

---

#### Expected Performance After Improvements

| Metric | Baseline | After (expected) |
|--------|----------|-----------------|
| MAE | 1.0393 | ↓ ~0.90–0.95 |
| RMSE | 1.5096 | ↓ ~1.30–1.40 |
| **R²** | **11.14%** | ↑ **~20–25%** |
| ROC-AUC | 75.34% | slight decrease (expected) |

> Reducing LAMBDA_CLS may slightly lower AUC since the classification head receives less gradient. This is an intentional trade-off to improve the more important regression metric (R²).

---

### Summary of All Changes

```
Baseline (v1)                         Improved (v2)
──────────────────────────────────────────────────────────────
4 numeric features              →     9 numeric features (+5 interaction)
nn.MSELoss()                    →     nn.HuberLoss(delta=1.0)
LAMBDA_CLS = 0.5                →     LAMBDA_CLS = 0.1
R² = 11.14%                     →     R² ↑ (improvement in progress)
```

---

## 6. Project 3 — Graph Network Analysis

### Goal
Model relationships between movies, directors, genres, and voice actors as a heterogeneous graph → predict popularity tier + director–actor collaboration likelihood.

### Graph Structure
```
Movie ↔ Director  (directed_by / rev_directed_by)
Movie ↔ Genre     (belongs_to  / rev_belongs_to)
Movie ↔ Actor     (voiced_by   / rev_voiced_by)
```

### Architecture
```
Per-node type input projection Linear(→64)
    → HeteroConv Layer 1 (SAGEConv × 6 edge types)
    → ReLU
    → HeteroConv Layer 2 (SAGEConv × 6 edge types)
    → Classification head  [Task 1: Node Classification]
    → LinkPredictor MLP    [Task 2: Link Prediction]
```

### Results

| Task | Metric | Value |
|------|--------|-------|
| Node Classification | Accuracy | **78.80%** |
| Node Classification | F1-macro | **82.31%** |
| Link Prediction | Accuracy | 78.12% |
| Link Prediction | ROC-AUC | **86.95%** |

---

## 7. Results Summary

| Project | Model | Key Metric | Baseline | v2 (After) |
|---------|-------|------------|----------|------------|
| 1 — NLP | DistilBERT | F1-micro | 69.41% | — |
| **2 — Tabular** | **Entity MLP** | **R²** | **11.14%** | **↑ improving** |
| 3 — Graph | HeteroGNN | Link AUC | 86.95% | — |

---

## 8. File Reference

```
16-04-26(1)/                          ← improved version (v2)
├── 01_nlp_genre_classification.py    unchanged
├── 02_tabular_rating_prediction.py   ⭐ 3 improvements applied
├── 03_graph_network_analysis.py      unchanged
├── eval_01_nlp.py / eval_01_nlp_fast.py
├── eval_02_tabular.py
├── eval_03_graph.py
├── create_report.py
├── requirements.txt
└── install.bat
```

---

*Generated: 2026-04-16 | Improvements: Huber Loss + 5 interaction features + LAMBDA_CLS 0.1*
