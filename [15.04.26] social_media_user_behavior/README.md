# SNS Addiction Level Prediction System

**Developer:** lightgo  
**Email:** lightgo1230@gmail.com

---

## Overview

A deep learning system trained on social media usage behavior data (25,000 users) that predicts a user's SNS addiction level on a scale of 1 to 10.  
Built as a native Windows GUI app using Rust + egui, it presents a chat-style interface where users answer questions and the AI analyzes their addiction level in real time.

---

## File Structure

```
SNS_Addiction_Prediction/
│
├── SNS_Addiction_Prediction.exe     ← Main executable (double-click to run)
│
├── predict_chat.py                  ← Chat-based prediction Python backend
├── social_media_dataset.py          ← PyTorch Custom Dataset class
├── social_media_model.py            ← TabResNet neural network definition
├── plot_addiction.py                ← Training runner & result graph generator
│
├── social_media_user_behavior.csv   ← Raw dataset (25,000 rows × 45 columns)
├── chat_model.pt                    ← Trained model weights
├── chat_meta.pkl                    ← Encoder / scaler metadata
├── addiction_results.png            ← 6-panel analysis result graph
│
└── sns_app/                         ← Rust source code
    ├── src/main.rs                  ← Full GUI source (926 lines)
    ├── Cargo.toml                   ← Dependency configuration
    ├── build.rs                     ← Icon embedding build script
    └── icon.ico                     ← App icon (L logo)
```

---

## Requirements

- Python 3.9 or higher
- Required packages:

```bash
pip install torch numpy pandas scikit-learn Pillow
```

---

## How to Run

Double-click `SNS_Addiction_Prediction.exe`.

---

## How to Use

### Chat Prediction Tab
1. Launch the app and click the **Start** button
2. Answer 12 questions one by one
3. After all questions are answered, the AI automatically analyzes your addiction level
4. Result displays: level (1–10), risk label, assessment, and recommended action

| Question | Input |
|----------|-------|
| Age | Enter a number |
| Gender | Select an option number |
| Primary platform | Choose from 12 options (Instagram, TikTok, etc.) |
| Daily screen time | Enter minutes |
| Weekly sessions | Enter a number |
| Followers count | Enter a number |
| Likes per day | Enter a number |
| Sleep hours | Enter hours |
| Posts per week | Enter a number |
| Content creator | Yes / No |
| Daily notifications | Enter a number |
| Check phone in morning | Yes / No |

### Training & Eval Tab
1. Click **Run Training & Predict** to start model training
2. Watch real-time training logs and progress bar
3. After completion, performance metrics are displayed automatically
4. **Graph** button — opens the result chart (PNG)
5. **Report** button — opens the analysis report (DOCX)

---

## Addiction Level Reference

| Level | Label | Description |
|-------|-------|-------------|
| 1 – 2 | Very Low | Very healthy social media usage |
| 3 – 4 | Low | Healthy usage — maintain current pattern |
| 5 – 6 | Moderate | Caution advised — monitor screen time |
| 7 – 8 | High | Addiction warning — digital detox recommended |
| 9 – 10 | Very High | Severe addiction — professional counseling recommended |

---

## Model Architecture

### TabResNet
- **Embedding Layer** — maps categorical variables to dense vectors
- **Pre-Activation Residual Block** — BN → ReLU → Linear → BN → ReLU → Dropout → Linear + skip connection
- **Total parameters** — 559,832

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Loss function | MSELoss |
| Scheduler | ReduceLROnPlateau |
| Batch size | 256 |
| Max epochs | 20 |
| Early stopping | patience=6 |

### Performance (test set)

| Metric | Value |
|--------|-------|
| RMSE | ~1.11 |
| R² | ~0.57 |
| ±1 Accuracy | ~83.5% |

---

## Dataset

**File:** `social_media_user_behavior.csv`  
**Source:** Kaggle — Social Media User Behavior Dataset  
**Size:** 25,000 users × 45 columns

### Feature Types

| Type | Count | Examples |
|------|-------|---------|
| Categorical | 16 | gender, primary_platform, country |
| Numerical | 17 | daily_screen_time_minutes, followers_count |
| Boolean | 9 | is_content_creator, checks_phone_first_morning |
| Target | 1 | addiction_level_1_to_10 |

---

## Tech Stack

| Area | Technology |
|------|------------|
| GUI | Rust + egui 0.29 + eframe 0.29 |
| Deep Learning | Python + PyTorch |
| Data Processing | pandas, numpy, scikit-learn |
| Visualization | matplotlib |
| Report | python-docx |
| Icon embedding | winres 0.1 |

---

## Rebuild (for developers)

After modifying Rust source, rebuild with:

```bash
cd sns_app
cargo build --release
copy target\release\SNS_Addiction_Prediction.exe ..\
```

Full Python dependency list:
```bash
pip install torch pandas numpy scikit-learn matplotlib python-docx Pillow
```

---

## License

Copyright (c) 2026 lightgo  
lightgo1230@gmail.com
