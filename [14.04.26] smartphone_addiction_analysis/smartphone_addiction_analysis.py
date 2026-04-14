"""
Smartphone Addiction Analysis — PyTorch Multi-class Classification
Target: addiction_level (None / Mild / Moderate / Severe)
Features: age, gender, daily usage, social media, gaming, notifications, stress, etc.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

plt.rcParams["axes.unicode_minus"] = False

SAVE_DIR = "C:/Users/USER/OneDrive/Desktop/addiction_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. Load & Preprocess Data
# ─────────────────────────────────────────
print("=" * 60)
print("1. Load & Preprocess Data")
print("=" * 60)

df = pd.read_csv("C:/Users/USER/OneDrive/Desktop/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")

# NaN in addiction_level = "None" (not addicted)
df["addiction_level"] = df["addiction_level"].fillna("None")

print(f"Dataset shape: {df.shape}")
print("\nTarget distribution (addiction_level):")
label_counts = df["addiction_level"].value_counts().sort_index()
for k, v in label_counts.items():
    print(f"  {k:10s}: {v:5d}  ({v/len(df)*100:.1f}%)")

# Categorical encoding
cat_cols = ["gender", "stress_level", "academic_work_impact"]
df_enc = df.copy()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Target encoding (ordered: None < Mild < Moderate < Severe)
level_order = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
df_enc["target"] = df_enc["addiction_level"].map(level_order)
CLASS_NAMES = ["None", "Mild", "Moderate", "Severe"]
NUM_CLASSES = 4

# Feature selection (drop id columns)
FEATURE_COLS = [
    "age", "gender",
    "daily_screen_time_hours", "social_media_hours",
    "gaming_hours", "work_study_hours", "sleep_hours",
    "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time", "stress_level", "academic_work_impact",
]

X = df_enc[FEATURE_COLS].values.astype(np.float32)
y = df_enc["target"].values.astype(np.int64)

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Validation / Test split (70 / 15 / 15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)
print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ─────────────────────────────────────────
# 2. Class Imbalance — WeightedRandomSampler
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Class Imbalance Handling (WeightedRandomSampler)")
print("=" * 60)

class_counts = Counter(y_train)
print("Train class distribution:", dict(class_counts))

# Weight per class = 1 / count
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = np.array([class_weights[label] for label in y_train])
sampler = WeightedRandomSampler(
    weights=torch.FloatTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)
print("WeightedRandomSampler applied (minority class oversampling)")

# ─────────────────────────────────────────
# 3. Dataset & DataLoader
# ─────────────────────────────────────────
class AddictionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 128

train_dataset = AddictionDataset(X_train, y_train)
val_dataset   = AddictionDataset(X_val,   y_val)
test_dataset  = AddictionDataset(X_test,  y_test)

# Train: WeightedRandomSampler (shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# 4. Model Definition
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PyTorch Model Definition")
print("=" * 60)

class AddictionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AddictionClassifier(input_dim=len(FEATURE_COLS), num_classes=NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(model)

# ─────────────────────────────────────────
# 5. Loss Function — CrossEntropyLoss (multi-class)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Loss Function: nn.CrossEntropyLoss (multi-class)")
print("=" * 60)

# Apply class weights to CrossEntropyLoss
ce_class_weights = torch.FloatTensor([
    len(y_train) / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)
]).to(device)
print("CrossEntropyLoss class weights:", [f"{w:.3f}" for w in ce_class_weights.cpu().tolist()])

criterion = nn.CrossEntropyLoss(weight=ce_class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

# ─────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Model Training")
print("=" * 60)

EPOCHS = 60
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc = 0
best_model_path = os.path.join(SAVE_DIR, "best_model.pth")

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_b)
        correct += (out.argmax(1) == y_b).sum().item()
        total += len(y_b)

    train_loss = running_loss / total
    train_acc  = correct / total

    # --- Validation ---
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            loss = criterion(out, y_b)
            v_loss += loss.item() * len(y_b)
            v_correct += (out.argmax(1) == y_b).sum().item()
            v_total += len(y_b)

    val_loss = v_loss / v_total
    val_acc  = v_correct / v_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

# ─────────────────────────────────────────
# 7. Test Evaluation
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Test Set Evaluation")
print("=" * 60)

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b = X_b.to(device)
        out = model(X_b)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_acc:.4f}\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────
# 8. Feature Importance — Permutation Importance
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Feature Importance Analysis (Permutation Importance)")
print("=" * 60)

def predict_accuracy(X_data, y_data):
    """Return accuracy for given data"""
    dataset = AddictionDataset(X_data, y_data)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            out = model(X_b)
            correct += (out.argmax(1).cpu() == y_b).sum().item()
            total += len(y_b)
    return correct / total

# Baseline accuracy (test set)
baseline_acc = predict_accuracy(X_test, y_test)
print(f"Baseline accuracy: {baseline_acc:.4f}")

# Shuffle each feature and measure accuracy drop
importance_scores = []
for i, feat in enumerate(FEATURE_COLS):
    X_permuted = X_test.copy()
    np.random.seed(42)
    np.random.shuffle(X_permuted[:, i])
    perm_acc = predict_accuracy(X_permuted, y_test)
    drop = baseline_acc - perm_acc
    importance_scores.append((feat, drop))
    print(f"  {feat:35s}: accuracy drop = {drop:+.4f}")

# Sort by importance descending
importance_scores.sort(key=lambda x: x[1], reverse=True)

feat_names = [x[0] for x in importance_scores]
feat_drops = [x[1] for x in importance_scores]

print("\n=== Feature Importance Ranking ===")
for rank, (feat, drop) in enumerate(importance_scores, 1):
    bar = "#" * max(0, int(drop * 1000))
    print(f"  {rank:2d}. {feat:35s}: {drop:+.4f}  {bar}")

# ─────────────────────────────────────────
# 9. Save Visualizations
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Save Result Visualizations")
print("=" * 60)

# (A) Learning Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss", color="#e74c3c")
axes[0].plot(val_losses,   label="Val Loss",   color="#3498db")
axes[0].set_title("Loss (CrossEntropyLoss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(train_accs, label="Train Acc", color="#e74c3c")
axes[1].plot(val_accs,   label="Val Acc",   color="#3498db")
axes[1].axhline(y=test_acc, color="#2ecc71", linestyle="--",
                label=f"Test Acc ({test_acc:.3f})")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
path_a = os.path.join(SAVE_DIR, "01_learning_curve.png")
plt.savefig(path_a, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path_a}")

# (B) Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
path_b = os.path.join(SAVE_DIR, "02_confusion_matrix.png")
plt.savefig(path_b, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path_b}")

# (C) Feature Importance Bar Chart
colors = ["#e74c3c" if d > 0 else "#95a5a6" for d in feat_drops]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(feat_names, feat_drops, color=colors)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title("Feature Importance (Permutation Importance)\nHigher value = more important for addiction prediction")
ax.set_xlabel("Accuracy Drop (vs baseline)")
for bar, val in zip(bars, feat_drops):
    ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", fontsize=9)
plt.tight_layout()
path_c = os.path.join(SAVE_DIR, "03_feature_importance.png")
plt.savefig(path_c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path_c}")

# (D) Class Imbalance & WeightedRandomSampler Effect
original_dist = [label_counts.get(c, 0) for c in CLASS_NAMES]
sampled_y = []
sampler_iter = iter(sampler)
for _ in range(len(train_dataset)):
    idx = next(sampler_iter)
    sampled_y.append(y_train[idx])
sampled_counts = Counter(sampled_y)
sampled_dist = [sampled_counts.get(i, 0) for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(CLASS_NAMES, original_dist, color=["#3498db","#2ecc71","#e67e22","#e74c3c"])
axes[0].set_title("Original Class Distribution (Imbalanced)")
axes[0].set_ylabel("Count")
for i, v in enumerate(original_dist):
    axes[0].text(i, v + 20, str(v), ha="center", fontsize=10)

axes[1].bar(CLASS_NAMES, sampled_dist, color=["#3498db","#2ecc71","#e67e22","#e74c3c"])
axes[1].set_title("After WeightedRandomSampler")
axes[1].set_ylabel("Count")
for i, v in enumerate(sampled_dist):
    axes[1].text(i, v + 20, str(v), ha="center", fontsize=10)

plt.tight_layout()
path_d = os.path.join(SAVE_DIR, "04_class_balance.png")
plt.savefig(path_d, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path_d}")

# ─────────────────────────────────────────
# 10. Final Summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Final Summary")
print("=" * 60)
print(f"  Model        : 4-layer MLP (256 -> 128 -> 64 -> 4)")
print(f"  Loss Function: nn.CrossEntropyLoss (with class weights)")
print(f"  Imbalance    : WeightedRandomSampler")
print(f"  Best Val Acc : {best_val_acc:.4f}")
print(f"  Test Acc     : {test_acc:.4f}")
print()
print("  [Feature Importance Top 5]")
for rank, (feat, drop) in enumerate(importance_scores[:5], 1):
    print(f"    {rank}. {feat}: {drop:+.4f}")
print()
print(f"  Results saved to: {SAVE_DIR}")
print("=" * 60)
print("Analysis complete!")
