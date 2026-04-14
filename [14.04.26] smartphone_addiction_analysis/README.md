# Smartphone Usage & Addiction Prediction Analysis

> PyTorch Multi-class Classification | Feature Importance | Class Imbalance Handling

---

## Project Structure

```
Desktop/
├── smartphone_addiction_analysis.py   # Main analysis script
├── README.md                          # This document
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
└── addiction_results/
    ├── best_model.pth                 # Best trained model weights
    ├── 01_learning_curve.png          # Training curve (loss / accuracy)
    ├── 02_confusion_matrix.png        # Confusion matrix
    ├── 03_feature_importance.png      # Feature importance bar chart
    └── 04_class_balance.png           # Class balance before/after comparison
```

---

## Dataset Overview

| Item | Details |
|------|---------|
| Total samples | 7,500 |
| Input features | 12 |
| Target variable | addiction_level (None / Mild / Moderate / Severe) |
| Missing values | 819 NaN in addiction_level filled as "None" (not addicted) |

### Feature List

| Feature | Description | Type |
|---------|-------------|------|
| age | Age of user | Numeric |
| gender | Gender (Male / Female / Other) | Categorical |
| daily_screen_time_hours | Total daily screen time (hours) | Numeric |
| social_media_hours | Daily social media usage (hours) | Numeric |
| gaming_hours | Daily gaming time (hours) | Numeric |
| work_study_hours | Work or study screen time (hours) | Numeric |
| sleep_hours | Hours of sleep per night | Numeric |
| notifications_per_day | Number of notifications received daily | Numeric |
| app_opens_per_day | Number of app launches per day | Numeric |
| weekend_screen_time | Screen time during weekends (hours) | Numeric |
| stress_level | Stress level (Low / Medium / High) | Categorical |
| academic_work_impact | Whether phone affects work/study (Yes / No) | Categorical |

### Target Class Distribution

| Class | Count | Ratio |
|-------|-------|-------|
| None (not addicted) | 819 | 10.9% |
| Mild | 1,373 | 18.3% |
| Moderate | 2,874 | 38.3% |
| Severe | 2,434 | 32.5% |

> **Note**: "None" class is only 10.9% of data — handled with WeightedRandomSampler.

---

## Preprocessing Pipeline

```
Raw CSV
   |
   +-- Fill NaN: addiction_level NaN -> "None"
   +-- Categorical encoding: LabelEncoder (gender, stress_level, academic_work_impact)
   +-- Target encoding: None=0, Mild=1, Moderate=2, Severe=3
   +-- Normalization: StandardScaler (mean=0, std=1)
   +-- Split: Train 70% / Val 15% / Test 15% (stratified)
```

---

## Model Architecture

**4-layer MLP (Multi-Layer Perceptron)**

```
Input  [12 features]
   |
Linear(12 -> 256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
   |
Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
   |
Linear(128 -> 64) -> BatchNorm1d -> ReLU -> Dropout(0.2)
   |
Linear(64 -> 4)
   |
Output [4 class logits]
```

| Item | Value |
|------|-------|
| Total parameters | 45,636 |
| BatchNorm1d | Stabilizes training, reduces overfitting |
| Dropout | Regularization (0.2 ~ 0.3) |

---

## Loss Function Selection

This is a **4-class multi-class classification** task, so `nn.CrossEntropyLoss` is used.

```python
# Multi-class classification -> CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=ce_class_weights)

# If predicting a continuous addiction score -> MSELoss
# criterion = nn.MSELoss()
```

| Scenario | Loss Function | Reason |
|----------|---------------|--------|
| Multi-class (this task) | nn.CrossEntropyLoss | Applies LogSoftmax + NLLLoss internally |
| Binary classification | nn.BCEWithLogitsLoss | Sigmoid + binary cross entropy |
| Regression (continuous output) | nn.MSELoss | Minimizes mean squared error |

Applied class weights: None (minority) -> **2.290** / Moderate (majority) -> **0.653**

---

## Class Imbalance Handling

### Problem

- `None` class: only **573 samples** in train set (smallest)
- `Moderate` class: **2,011 samples** in train set (largest)
- ~**3.5x difference** -> model biased toward majority classes

### Solution: WeightedRandomSampler

```python
# Sample weight = 1 / class count
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in y_train]

sampler = WeightedRandomSampler(
    weights=torch.FloatTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True   # oversample minority classes
)
train_loader = DataLoader(dataset, sampler=sampler, drop_last=True)
```

| Class | Train Count | Sample Weight |
|-------|-------------|---------------|
| None | 573 | 0.00175 (high) |
| Mild | 961 | 0.00104 |
| Moderate | 2,011 | 0.00050 (low) |
| Severe | 1,704 | 0.00059 |

**Result**: None class Recall achieved **80%**

---

## Training Configuration

```python
optimizer  = torch.optim.Adam(lr=1e-3, weight_decay=1e-4)
scheduler  = ReduceLROnPlateau(mode="max", patience=5, factor=0.5)
EPOCHS     = 60
BATCH_SIZE = 128
```

| Item | Value | Description |
|------|-------|-------------|
| Optimizer | Adam | Adaptive learning rate, fast convergence |
| weight_decay | 1e-4 | L2 regularization to prevent overfitting |
| LR Scheduler | ReduceLROnPlateau | Halve LR when Val Acc stops improving |
| drop_last | True | Prevents BatchNorm error on single-sample last batch |

---

## Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **49.02%** |
| Test Accuracy | **47.20%** |
| Macro Avg F1-Score | 0.43 |

### Per-class Performance (Classification Report)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| None | 0.46 | **0.80** | 0.59 | 123 |
| Mild | 0.48 | 0.46 | 0.47 | 206 |
| Moderate | 0.28 | 0.02 | 0.04 | 431 |
| Severe | 0.48 | **0.90** | 0.63 | 365 |

> **Moderate Recall = 2%**
> Moderate sits between None and Severe, making its boundary ambiguous.
> Fix: increase epochs or apply Focal Loss.

---

## Feature Importance (Permutation Importance)

### Method

```
1. Measure baseline accuracy on the test set (Baseline = 47.20%)
2. Shuffle one feature at a time (all others unchanged)
3. Re-measure accuracy after shuffle
4. Accuracy drop = importance of that feature
   (larger drop -> more important for predicting addiction)
```

```python
baseline_acc = predict_accuracy(X_test, y_test)

for i, feat in enumerate(FEATURE_COLS):
    X_permuted = X_test.copy()
    np.random.shuffle(X_permuted[:, i])       # shuffle only this feature
    perm_acc = predict_accuracy(X_permuted, y_test)
    importance = baseline_acc - perm_acc      # drop = importance score
```

### Ranking

| Rank | Feature | Accuracy Drop | Interpretation |
|------|---------|---------------|----------------|
| 1st | daily_screen_time_hours | +0.1396 | Most important factor |
| 2nd | social_media_hours | +0.1191 | Second most important |
| 3rd | app_opens_per_day | +0.0053 | Minor contribution |
| 4th | work_study_hours | +0.0018 | Negligible |
| 5th | sleep_hours | +0.0009 | Negligible |
| - | notifications_per_day | -0.0009 | Noise level |
| - | age | -0.0018 | Noise level |
| - | gender | -0.0044 | Noise level |
| - | academic_work_impact | -0.0089 | Noise level |

### Key Insight

> **Addiction level is determined almost entirely by how long and how much social media a user consumes.**
>
> - Shuffling `daily_screen_time_hours` alone drops accuracy by 14%
> - Notification count, age, and gender contribute almost nothing
> - Academic/work impact acts as noise rather than a predictive signal

---

## How to Run

### Install dependencies

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

### Run

```bash
python smartphone_addiction_analysis.py
```

### Output files

```
addiction_results/
    best_model.pth             # Best model checkpoint (highest Val Acc)
    01_learning_curve.png      # Train/Val loss and accuracy over epochs
    02_confusion_matrix.png    # Actual vs predicted class heatmap
    03_feature_importance.png  # Feature importance horizontal bar chart
    04_class_balance.png       # Class distribution before vs after sampler
```

---

## Potential Improvements

| Method | Expected Effect |
|--------|----------------|
| More epochs (60 -> 100+) | Improve Moderate class Recall |
| Focal Loss | Focus on hard boundary samples |
| Feature engineering | Add derived features (e.g., social_media / total_screen ratio) |
| SMOTE | Synthetic oversampling; compare against WeightedRandomSampler |
| Ensemble | Combine Random Forest + PyTorch for cross-validated feature importance |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning model (MLP, DataLoader, WeightedRandomSampler) |
| scikit-learn | Preprocessing (StandardScaler), evaluation (classification_report) |
| pandas | Data loading and manipulation |
| numpy | Array operations and feature shuffling |
| matplotlib / seaborn | Learning curves, confusion matrix, feature importance plots |
