# AnimationTabular Dashboard 

A standalone Windows desktop application that predicts animation movie ratings using a trained deep learning model.  
No Python, no internet connection, no external files required — everything is embedded in a single `.exe`.

---

## Quick Start

1. Double-click **`AnimationTabular_Dashboard.exe`** on the Desktop.
2. Answer 9 questions by clicking the icon-button options.
3. Read the predicted rating and box-office probability on the result screen.
4. Click **🔄 Predict Again** to start over.

---

## What the App Does

| Step | Description |
|------|-------------|
| 1 | Select the **release era** of the movie (Classic → 2020s) |
| 2 | Select the **runtime** (Under 60 min → Over 180 min) |
| 3 | Select the expected **vote count** (Very Low → Very High) |
| 4 | Select the **popularity score** (Very Low → Viral) |
| 5 | Select the **animation style** (2D Traditional, 3D CGI, Stop Motion, etc.) |
| 6 | Select the **MPAA rating** (G, PG, PG-13, R, Not Rated) |
| 7 | Select the **target audience** (Children, Family, Teens, Adults, All Ages) |
| 8 | Select the **original language** (English, Japanese, Korean, French, Other) |
| 9 | Select the **primary genre** (Adventure, Drama, Comedy, Action, Fantasy, etc.) |

After all 9 answers are submitted, the model runs instantly and shows:

- **Predicted Rating** — a score from 1.0 to 10.0
- **Box-office Probability** — percentage likelihood of being a hit
- **Rank label** — Top 10% Masterpiece / Top 25% Strong / Above Average / Average / Below Average
- **Visual rating bar** — gradient bar with hit threshold marker at 7.0

---

## Technical Details

### Model Architecture

An **Entity Embedding MLP** trained on a tabular animation movie dataset:

- **7 categorical columns** → learned embeddings (Animation_Style, MPAA_Rating, Target_Audience, Era, Popularity_Tier, Original_Language, Primary_Genre)
- **9 numeric features** → StandardScaler normalized (release_year, runtime, vote_log, pop_log, Vote×Pop, Viral_Score, Movie_Age, Age×VoteLog, Runtime×VoteLog)
- **Backbone**: 3 hidden layers (512 → 256 → 128) with BatchNorm + GELU + Dropout
- **Dual heads**: regression head (predicted rating) + classification head (hit probability)

### Inference Engine (Pure Rust)

The entire PyTorch forward pass is reimplemented in Rust:

- `linear_fwd` — matrix-vector multiply + bias
- `batchnorm_fwd` — eval-mode batch normalization
- `gelu` — Gauss error function approximation
- `sigmoid` — for hit probability output
- StandardScaler and LabelEncoder logic replicated exactly

Model weights (`model_weights.json`, ~4.9 MB) and preprocessor config (`preprocessor.json`) are embedded at compile time using `include_str!`.

### Technology Stack

| Component | Detail |
|-----------|--------|
| Language  | Rust 2021 edition |
| GUI       | eframe 0.29 + egui 0.29 (immediate-mode) |
| Fonts     | Segoe UI → Malgun Gothic → Segoe UI Emoji |
| ML Model  | Entity Embedding MLP (PyTorch, exported to JSON) |
| Serialization | serde + serde_json |
| Binary size | ~9.3 MB (self-contained) |

---

## ML Project Structure

The model was trained in Python at `C:/Users/USER/OneDrive/Desktop/animation_ml_projects/`.

### Training Pipeline

| File | Purpose |
|------|---------|
| `01_data_collection_cleaning.py` | Collects and cleans animation movie data |
| `02_tabular_rating_prediction.py` | Trains the Entity Embedding MLP model |
| `export_model_for_rust.py` | Exports trained weights → JSON files for Rust |
| `predict_tabular.py` | Standalone Python prediction script (reference) |
| `tabular_model_best.pt` | Saved PyTorch model weights |
| `tabular_preprocessors.pkl` | Saved scaler + label encoders |

### Retraining

To retrain and update the app:

```bash
# 1. Train the model
python 02_tabular_rating_prediction.py

# 2. Export weights to JSON
python export_model_for_rust.py

# 3. Rebuild the exe
cd C:\Users\USER\OneDrive\Desktop\animation_tabular_app
cargo build --release

# 4. Copy to Desktop
copy target\release\AnimationTabular_Dashboard.exe C:\Users\USER\OneDrive\Desktop\
```

---

## Hit Threshold

A movie is classified as a **HIT** when `predicted rating ≥ 7.0`.  
The threshold is stored in `preprocessor.json` and used automatically at inference time.

---

## Copyright

© lightgo · lightgo1230@gmail.com
