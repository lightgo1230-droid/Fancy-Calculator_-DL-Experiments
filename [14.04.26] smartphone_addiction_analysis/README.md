# Smartphone Addiction Prediction — GAN Data Augmentation + Binary Classification

> Conditional GAN 10x Data Augmentation | Binary Label Reformulation | Accuracy 47% -> 90%

---

## Project Overview

A deep learning model that predicts smartphone addiction using usage data (7,500 samples).
To overcome the low accuracy (47%) of the original 4-class model (None/Mild/Moderate/Severe),
we applied **Conditional GAN to augment data 10x** and **reformulated the task as binary classification**,
achieving a final **Test Accuracy of 90.49% and ROC-AUC of 0.9722**.

---

## Accuracy Improvement Steps

| Step | Method | Test Accuracy |
|------|--------|--------------|
| Step 1 | Original 4-class MLP | 47.20% |
| Step 2 | Binary label reformulation | Improvement begins |
| Step 3 | **GAN 10x Augmentation + Binary Classification** | **90.49%** |

---

## Why Was Accuracy Low?

### Problem 1. Insufficient Effective Features
- Only **2 out of 12 features** meaningfully contributed (daily_screen_time, social_media_hours)
- The remaining 10 features were essentially noise with no predictive power

### Problem 2. Ambiguous Label Boundaries
- The Moderate class sits between None and Severe with no clear boundary
- Labels were artificially generated (synthetic dataset)
- Moderate Recall = **2%** (nearly impossible to predict)

### Solution
```
4-class (None / Mild / Moderate / Severe)
           |
           v
2-class (Not Addicted / Addicted)
  - None + Mild       -> 0 (Not Addicted)
  - Moderate + Severe -> 1 (Addicted)
```

---

## Project Structure

```
Desktop/
├── gan_binary_classification.py       # Main analysis script
├── smartphone_addiction_analysis.py   # Original 4-class script
├── README.md                          # This document
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
└── addiction_results_binary/
    ├── best_binary_model.pth          # Best model weights
    ├── 00_gan_loss.png                # GAN training loss curve
    ├── 01_augmentation.png            # Class distribution before/after augmentation
    ├── 02_learning_curve.png          # Train/Val loss and accuracy curve
    ├── 03_confusion_matrix.png        # Confusion matrix
    ├── 04_roc_curve.png               # ROC curve
    └── 05_feature_importance.png      # Feature importance chart
```

---

## Dataset Overview

| Item | Details |
|------|---------|
| Total samples | 7,500 |
| Input features | 12 |
| Original target | addiction_level (None / Mild / Moderate / Severe) |
| Converted target | label (0: Not Addicted / 1: Addicted) |
| Data split | Train 70% / Val 15% / Test 15% (stratified) |

### Binary Label Distribution

| Class | Count | Ratio |
|-------|-------|-------|
| 0 (Not Addicted) | 2,192 | 29.2% |
| 1 (Addicted) | 5,308 | 70.8% |

---

## Key Technique 1 — Conditional GAN (Data Augmentation)

### What is GAN?

A **GAN (Generative Adversarial Network)** consists of a Generator and a Discriminator
that compete against each other during training.

```
Real data ---> Discriminator ---> Real / Fake judgment
                    ^
Noise     ---> Generator    ---> Fake data generation
```

### What is Conditional GAN?

By feeding a class label alongside the noise, the model generates
**data for a specific target class**.

```python
# Class condition concatenated with noise
z     = torch.randn(batch, NOISE_DIM)    # random noise
label = torch.tensor([[0.], [1.], ...])  # class condition

fake_data = Generator(torch.cat([z, label], dim=1))
```

### Augmentation Results

| Item | Value |
|------|-------|
| Original train samples | 5,249 |
| GAN-generated samples | 47,241 |
| Total augmented samples | **52,490 (10x)** |

### GAN Architecture

**Generator**
```
Noise(64) + Label(1)
    |
Linear(65 -> 256) -> BatchNorm -> LeakyReLU(0.2)
    |
Linear(256 -> 256) -> BatchNorm -> LeakyReLU(0.2)
    |
Linear(256 -> 128) -> BatchNorm -> LeakyReLU(0.2)
    |
Linear(128 -> 12) -> Tanh
    |
Fake data [12 features]
```

**Discriminator**
```
Data(12) + Label(1)
    |
Linear(13 -> 256) -> LeakyReLU -> Dropout(0.3)
    |
Linear(256 -> 128) -> LeakyReLU -> Dropout(0.3)
    |
Linear(128 -> 1)
    |
Real / Fake logit
```

### GAN Training Configuration

| Item | Value |
|------|-------|
| Noise dimension | 64 |
| GAN epochs | 300 |
| Batch size | 256 |
| Generator LR | 2e-4 |
| Discriminator LR | 2e-4 |
| Optimizer | Adam (beta1=0.5, beta2=0.999) |
| Loss function | BCEWithLogitsLoss |

---

## Key Technique 2 — Binary Classification

### Loss Function: nn.BCEWithLogitsLoss

Since this is a binary classification task, `BCEWithLogitsLoss` is used.

```python
# Binary classification -> BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

# Multi-class classification -> CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()
```

| Scenario | Loss Function |
|----------|--------------|
| Binary classification (this task) | nn.BCEWithLogitsLoss |
| Multi-class classification | nn.CrossEntropyLoss |
| Regression (continuous output) | nn.MSELoss |

`BCEWithLogitsLoss` internally applies Sigmoid + Binary Cross Entropy.

```python
prob = torch.sigmoid(logit)   # convert logit to 0~1 probability
pred = (prob >= 0.5).float()  # threshold 0.5 -> addicted (1)
```

### Classifier Architecture

```
Input [12 features]
    |
Linear(12 -> 256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
    |
Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
    |
Linear(128 -> 64) -> BatchNorm1d -> ReLU -> Dropout(0.2)
    |
Linear(64 -> 1)
    |
Output [single logit -> Sigmoid -> addiction probability]
```

| Item | Value |
|------|-------|
| Total parameters | 45,441 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Epochs | 60 |
| Batch size | 128 |

---

## Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **91.39%** |
| **Test Accuracy** | **90.49%** |
| **ROC-AUC** | **0.9722** |
| Macro F1-Score | **0.89** |

### Per-class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Addicted (0) | 0.78 | 0.93 | 0.85 | 329 |
| Addicted (1) | 0.97 | 0.89 | 0.93 | 796 |

> **ROC-AUC 0.97**: Model clearly distinguishes addicted vs non-addicted — ready for practical use.

---

## Feature Importance (Permutation Importance)

### Results

| Rank | Feature | Accuracy Drop | Interpretation |
|------|---------|---------------|----------------|
| 1st | daily_screen_time_hours | +0.1964 | Most important factor |
| 2nd | social_media_hours | +0.1760 | Second most important |
| 3rd | gender | +0.0124 | Minor contribution |
| 4th | notifications_per_day | +0.0089 | Minor contribution |
| 5th | app_opens_per_day | +0.0089 | Minor contribution |

> Compared to 4-class model: lower-ranked features (gender, notifications) gained slight importance.
> Binary simplification allowed the model to pick up on subtler patterns.

---

## How to Run

### Install dependencies

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

### Run

```bash
python gan_binary_classification.py
```

### Output files

```
addiction_results_binary/
    best_binary_model.pth     # Best model checkpoint (highest Val Acc)
    00_gan_loss.png           # GAN Generator / Discriminator loss curves
    01_augmentation.png       # Class distribution before vs after augmentation
    02_learning_curve.png     # Train/Val loss and accuracy over epochs
    03_confusion_matrix.png   # Actual vs predicted heatmap
    04_roc_curve.png          # ROC curve with AUC score
    05_feature_importance.png # Feature importance bar chart
```

---

## Key Code Explained

### Conditional GAN Training Loop

```python
for epoch in range(1, GAN_EPOCHS + 1):
    # Train Discriminator
    fake_x     = Generator(noise, label).detach()
    real_logit = Discriminator(real_x, label)
    fake_logit = Discriminator(fake_x, label)
    d_loss = (BCE(real_logit, ones) + BCE(fake_logit, zeros)) / 2

    # Train Generator
    fake_x     = Generator(noise, label)
    fake_logit = Discriminator(fake_x, label)
    g_loss = BCE(fake_logit, ones)  # fool the discriminator
```

### Synthetic Data Generation

```python
G.eval()
with torch.no_grad():
    for cls in [0, 1]:
        label = torch.full((n_cls, 1), float(cls))
        z     = torch.randn(n_cls, NOISE_DIM)
        fake  = Generator(z, label)   # generate per-class synthetic data
```

### Binary Prediction

```python
logit = model(x)                      # single logit output
prob  = torch.sigmoid(logit)          # convert to 0~1 probability
pred  = (prob >= 0.5).float()         # threshold 0.5 -> class decision
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| PyTorch | GAN and classifier implementation |
| scikit-learn | Preprocessing, evaluation metrics |
| pandas / numpy | Data handling and array operations |
| matplotlib / seaborn | Visualization (ROC curve, confusion matrix, etc.) |
