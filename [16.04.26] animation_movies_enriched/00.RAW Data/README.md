# Animation Movie Dataset — PyTorch Machine Learning Projects (3-in-1)

> A comprehensive multi-paradigm ML suite built on a single dataset of **25,390 animation films (1878–2029)**.  
> Each project approaches the same data from a completely different angle: language, tabular structure, and graph topology.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Environment & Installation](#3-environment--installation)
4. [Project 1 — NLP Genre Classification & Recommendation](#4-project-1--nlp-genre-classification--recommendation)
5. [Project 2 — Tabular Rating Prediction](#5-project-2--tabular-rating-prediction)
6. [Project 3 — Graph Network Analysis](#6-project-3--graph-network-analysis)
7. [Evaluation Files](#7-evaluation-files)
8. [Report Generation](#8-report-generation)
9. [Results Summary](#9-results-summary)
10. [How the Three Projects Connect](#10-how-the-three-projects-connect)
11. [Future Work](#11-future-work)
12. [File Reference](#12-file-reference)

---

## 1. Project Overview

This suite demonstrates how the **same dataset** can be exploited with three fundamentally different deep-learning paradigms:

| # | File | Paradigm | Model | Tasks |
|---|------|----------|-------|-------|
| 1 | `01_nlp_genre_classification.py` | Natural Language Processing | DistilBERT (fine-tuned) | Multi-label genre classification + cosine-similarity recommendation |
| 2 | `02_tabular_rating_prediction.py` | Tabular / Structured Data | Entity Embedding MLP | Rating regression + Hit/Not-Hit binary classification |
| 3 | `03_graph_network_analysis.py` | Graph Neural Network | Heterogeneous GraphSAGE (PyG) | Node classification (Popularity Tier) + Link prediction (director–voice actor collaboration) |

---

## 2. Dataset

| Property | Value |
|----------|-------|
| File | `animation_movies_enriched_1878_2029.csv` |
| Location | `C:\Users\USER\OneDrive\Desktop\` |
| Records | 25,390 movies |
| Columns | 44 features |
| Year range | 1878 – 2029 |

**Key columns used across projects:**

| Column | Type | Used In |
|--------|------|---------|
| `Overview` | Text (synopsis) | Project 1 |
| `Genre` | Categorical (multi-value) | All projects |
| `TMDB_Rating` | Float (0–10) | Projects 2 & 3 |
| `Release_Year`, `Runtime` | Numeric | Project 2 |
| `TMDB_Vote_Count`, `TMDB_Popularity` | Numeric | Projects 2 & 3 |
| `Animation_Style`, `MPAA_Rating`, `Target_Audience` | Categorical | Project 2 |
| `Director` | String | Projects 2 & 3 |
| `Voice_Cast` | String (comma-separated) | Project 3 |
| `Popularity_Tier` | Categorical (ordinal) | Project 3 (label) |

---

## 3. Environment & Installation

### Verified Environment

```
Python  : C:\Users\USER\anaconda3\python.exe
PyTorch : 2.10.0 + cpu
transformers : 5.5.4
torch_geometric : 2.7.0
OS      : Windows 11
```

### Step 1 — Install common dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Install PyTorch Geometric (Project 3 only)

**CPU:**
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**CUDA 12.x:**
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

> If `torch_geometric` is not installed, Project 3 automatically falls back to a pure-PyTorch bipartite embedding (Skip-Gram style). All other functionality remains intact.

### Quick install (Windows batch)

```bash
install.bat
```

### Execution order

```bash
python 01_nlp_genre_classification.py
python 02_tabular_rating_prediction.py
python 03_graph_network_analysis.py
```

---

## 4. Project 1 — NLP Genre Classification & Recommendation

### Goal

Given only a movie's synopsis (`Overview`), predict which genres it belongs to (multi-label), and retrieve semantically similar movies via embedding similarity.

### Architecture

```
Overview text
    │
    ▼
DistilBertTokenizer  (max_len=128, padding, truncation)
    │
    ▼
DistilBertModel  (distilbert-base-uncased, 66M params)
    │
    └── [CLS] token vector  (768-dim)
              │
              ▼
        Dropout(0.3)
        Linear(768 → 256)
        GELU
        Dropout(0.3)
        Linear(256 → 10)   ← number of top genres
              │
              ▼
        BCEWithLogitsLoss   (multi-label)
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Pre-trained model | `distilbert-base-uncased` |
| Max sequence length | 128 tokens |
| Batch size | 32 |
| Epochs | 4 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10% of total steps) |
| Loss | BCEWithLogitsLoss |
| Train / Val split | 85% / 15% |
| Top genres | 10 most frequent |

### Task 1 — Multi-label Genre Classification

Each movie may belong to multiple genres. The model outputs a 10-dimensional sigmoid score; predictions above 0.5 are treated as positive labels.

**Evaluation metrics:** F1-micro, F1-macro, Jaccard similarity, Exact Match ratio, Hamming Loss

### Task 2 — Synopsis-based Movie Recommendation

After fine-tuning, the `[CLS]` embeddings encode semantic content of synopses. Recommendation works by computing **cosine similarity** between the query embedding and all validation-set embeddings — no additional model required.

**Example queries used:**
- *"A young lion cub must reclaim his kingdom from a treacherous uncle."*
- *"Toys come to life when humans are not around and go on adventures."*
- *"A scientist accidentally creates chaos with a machine that makes food fall from the sky."*

### Output Files

| File | Description |
|------|-------------|
| `genre_classifier_best.pt` | Best model weights (highest Val F1-micro) |

### Key Results

| Metric | Value |
|--------|-------|
| F1-micro | **69.41%** |
| F1-weighted | 57.38% |
| Jaccard | 67.43% |
| Exact Match | 43.69% |
| Hamming Loss | 0.1218 |

---

## 5. Project 2 — Tabular Rating Prediction

### Goal

Predict a movie's TMDB rating (regression) and classify it as a "Hit" (rating ≥ 7.0) using structured metadata — no text involved.

### Feature Engineering

**Numeric features (StandardScaler normalized):**
- `Release_Year`, `Runtime` (Movie_Length_Minutes)
- `Vote_Count_Log` = log1p(TMDB_Vote_Count)
- `Popularity_Log` = log1p(TMDB_Popularity)

**Categorical features (Entity Embedding):**
- `Animation_Style`, `MPAA_Rating`, `Target_Audience`, `Era`, `Popularity_Tier`, `Original_Language`, `Primary_Genre`
- Embedding dimension per feature: min(50, (cardinality + 1) // 2)

**Boolean features (mapped to 0/1):**
- `Is_TV_Compilation`, `Hidden_Gem`, `Is_Adult_Content`, `Live_Action_Remake`

### Architecture

```
[Categorical features]              [Numeric + Boolean features]
       │                                        │
  Entity Embedding                        (already scaled)
  (per column)                                  │
       └──────────────── concat ────────────────┘
                              │
                         Backbone MLP
                    Linear → BN → GELU → Dropout
                    512 → 256 → 128
                         /         \
               Regression Head    Classification Head
               Linear(128 → 1)    Linear(128 → 1)
                    │                    │
                  MAE/RMSE           BCEWithLogits
                 (rating)             (Hit/Not-Hit)
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch size | 256 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss weights | MSE (reg) + 0.5 × BCE (cls) |
| Hit threshold | Rating ≥ 7.0 |
| Train / Val split | 80% / 20% |

### Key Results

**Task 1 — Regression**

| Metric | Value |
|--------|-------|
| MAE | **1.0393** |
| RMSE | 1.5096 |
| R² | 11.14% |

> Note: Low R² is expected — subjective ratings are inherently noisy and weakly correlated with metadata alone.

**Task 2 — Binary Classification (Hit detection)**

| Metric | Value |
|--------|-------|
| Accuracy | **71.56%** |
| ROC-AUC | **75.34%** |

### Output Files

| File | Description |
|------|-------------|
| `tabular_model_best.pt` | Best model weights (lowest Val MAE) |
| `tabular_training_curve.png` | Train Loss / Val MAE / Val AUC curves |
| `tabular_scatter.png` | Actual vs predicted rating scatter plot |

---

## 6. Project 3 — Graph Network Analysis

### Goal

Model the relational structure between movies, directors, genres, and voice actors as a **heterogeneous graph**, then perform node-level classification and edge-level link prediction.

### Graph Structure

```
        Movie node  ←──────────────────────┐
       /    |    \                          │
      /     |     \                         │ (rev edges for
 Director  Genre  Voice Actor               │  message passing)
  node     node    node                    │
      \     |     /                         │
       \    |    /                          │
        Movie node  ───────────────────────┘
```

**Node types:** movie (with 5 numeric features), director, genre, actor (latent 8-dim)

**Edge types (6 total, bidirectional):**
- movie → director (`directed_by`) / director → movie (`rev_directed_by`)
- movie → genre (`belongs_to`) / genre → movie (`rev_belongs_to`)
- movie → actor (`voiced_by`) / actor → movie (`rev_voiced_by`)

**Scale (after filtering top 5,000 by vote count):**
- Movies: up to 5,000
- Directors, genres, actors: derived from data

### Model — HeteroGNN (PyG version)

```
Input projections (per node type):
  movie    : Linear(5  → 64)
  director : Linear(8  → 64)
  genre    : Linear(8  → 64)
  actor    : Linear(8  → 64)

HeteroConv Layer 1  (SAGEConv, aggr=sum)
    → ReLU
HeteroConv Layer 2  (SAGEConv, aggr=sum)

Classification Head: Linear(64 → n_tiers)   [Task 1]
```

### Task 1 — Node Classification (Popularity Tier)

Predicts the popularity tier of each movie node using graph-propagated features.

**Split:** 60% train / 20% val / 20% test (random mask)

### Task 2 — Link Prediction (Director–Voice Actor Collaboration)

1. Extract director and actor embeddings from the trained GNN encoder
2. Build positive pairs (director, actor) that have collaborated on a movie
3. Build negative pairs via random sampling
4. Train a 2-layer MLP `LinkPredictor` on concatenated embeddings
5. Score unseen (director, actor) pairs by collaboration probability

### Fallback Mode (no PyG installed)

A pure-PyTorch `BipartiteEmbedding` model is used instead:
- Trains movie and director embeddings using BPR loss (Skip-Gram style)
- Provides cosine-similarity-based movie recommendation

### Key Results

| Task | Metric | Value |
|------|--------|-------|
| Node Classification | Accuracy | **78.80%** |
| Node Classification | F1-macro | **82.31%** |
| Link Prediction | Accuracy | 78.12% |
| Link Prediction | ROC-AUC | **86.95%** |

### Output Files

| File | Description |
|------|-------------|
| `gnn_nc_best.pt` | Best GNN weights (highest Val Accuracy) |
| `graph_network_viz.png` | Director–Movie–Genre network visualization (top 30 films) |

---

## 7. Evaluation Files

Separate, faster evaluation scripts allow you to measure accuracy without re-running full training:

| File | Purpose | Notes |
|------|---------|-------|
| `eval_01_nlp_fast.py` | NLP accuracy measurement | 8,000-sample fast version |
| `eval_01_nlp.py` | NLP accuracy measurement | Full dataset version |
| `eval_02_tabular.py` | Tabular model accuracy | Loads `tabular_model_best.pt` |
| `eval_03_graph.py` | Graph model accuracy | Loads `gnn_nc_best.pt` |

---

## 8. Report Generation

`create_report.py` generates a formatted Word document (`.docx`) summarizing all results, metrics, and visualizations.

```bash
pip install python-docx
python create_report.py
```

---

## 9. Results Summary

| Project | Model | Best Metric | Value |
|---------|-------|-------------|-------|
| 1 — NLP | DistilBERT | F1-micro | **69.41%** |
| 2 — Tabular | Entity Embedding MLP | ROC-AUC | **75.34%** |
| 3 — Graph | HeteroGNN | Link Pred AUC | **86.95%** |

---

## 10. How the Three Projects Connect

All three share the same data pipeline origin but diverge at the feature extraction step:

```
Raw CSV
  │
  ├─ Overview text   ──►  [Project 1] DistilBERT   ──►  768-dim [CLS] embedding
  │
  ├─ Numeric/Categorical ─►  [Project 2] Entity MLP  ──►  128-dim backbone feature
  │
  └─ Entity relationships ─►  [Project 3] HeteroGNN  ──►  64-dim node embedding
```

**Shared design pattern:** All three projects reuse their internal embeddings for a second task (recommendation or link prediction) without training a separate model — a cost-effective transfer learning strategy within each project.

**Performance hierarchy:**
- Graph (structure) > NLP (content) > Tabular (metadata) for classification tasks
- This reflects how much information each modality provides beyond raw attributes

---

## 11. Future Work

Ideas documented in `context.md`:

| Idea | Expected Benefit |
|------|-----------------|
| **Cross-modal ensemble** — concat [CLS] + Entity + GNN embeddings | Richer representation, higher recommendation quality |
| **GPU retraining** on full data | Faster convergence, potentially higher NLP accuracy |
| **Class imbalance correction** — `pos_weight`, Focal Loss | Improve recall for rare genres / minority tiers |
| **Temporal GNN** — incorporate release year as graph signal | Model industry trend evolution over time |
| **REST API movie recommendation server** | Deploy models as a live inference endpoint |

---

## 12. File Reference

```
16-04-26/
├── 01_nlp_genre_classification.py   # DistilBERT fine-tuning + recommendation
├── 02_tabular_rating_prediction.py  # Entity Embedding MLP (regression + classification)
├── 03_graph_network_analysis.py     # HeteroGNN node classification + link prediction
├── eval_01_nlp.py                   # NLP evaluation (full)
├── eval_01_nlp_fast.py              # NLP evaluation (8k sample, fast)
├── eval_02_tabular.py               # Tabular model evaluation
├── eval_03_graph.py                 # Graph model evaluation
├── create_report.py                 # Word report generator
├── requirements.txt                 # Python dependencies
├── install.bat                      # Windows quick-install script
├── README.txt                       # Brief project overview (original)
└── context.md                       # Session context & measured results
```

---

*Generated: 2026-04-16 | Dataset: animation_movies_enriched_1878_2029.csv | Framework: PyTorch 2.10.0*
