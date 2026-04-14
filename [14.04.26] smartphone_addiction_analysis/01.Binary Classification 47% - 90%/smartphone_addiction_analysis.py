"""
Smartphone Addiction — GAN Data Augmentation (10x) + Binary Classification
  - Label: None + Mild = 0 (not addicted) / Moderate + Severe = 1 (addicted)
  - Conditional GAN for tabular data augmentation
  - Binary classifier with BCEWithLogitsLoss
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)

plt.rcParams["axes.unicode_minus"] = False

SAVE_DIR = "C:/Users/USER/OneDrive/Desktop/addiction_results_binary"
os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. Load & Preprocess
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("1. Load & Preprocess  (Binary Label)")
print("=" * 60)

df = pd.read_csv("C:/Users/USER/OneDrive/Desktop/"
                 "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
df["addiction_level"] = df["addiction_level"].fillna("None")

# ── Binary label ──
# None + Mild  → 0  (not addicted)
# Moderate + Severe → 1  (addicted)
df["label"] = df["addiction_level"].map(
    {"None": 0, "Mild": 0, "Moderate": 1, "Severe": 1}
)

print("Binary label distribution:")
vc = df["label"].value_counts().sort_index()
for k, v in vc.items():
    tag = "Not Addicted" if k == 0 else "Addicted"
    print(f"  {k} ({tag}): {v:5d}  ({v/len(df)*100:.1f}%)")

cat_cols = ["gender", "stress_level", "academic_work_impact"]
df_enc = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col].astype(str))

FEATURE_COLS = [
    "age", "gender",
    "daily_screen_time_hours", "social_media_hours",
    "gaming_hours", "work_study_hours", "sleep_hours",
    "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time", "stress_level", "academic_work_impact",
]
INPUT_DIM = len(FEATURE_COLS)

X = df_enc[FEATURE_COLS].values.astype(np.float32)
y = df_enc["label"].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

print(f"\nTrain: {len(X_train)},  Val: {len(X_val)},  Test: {len(X_test)}")
print(f"Train label dist: {dict(Counter(y_train.astype(int)))}")

# ═══════════════════════════════════════════════════════════
# 2. Conditional GAN — tabular data augmentation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Conditional GAN Training  (10x augmentation)")
print("=" * 60)

NOISE_DIM   = 64
GAN_EPOCHS  = 300
GAN_BATCH   = 256
LR_G        = 2e-4
LR_D        = 2e-4
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + 1, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, INPUT_DIM),
            nn.Tanh(),
        )
    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM + 1, 256),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
bce   = nn.BCEWithLogitsLoss()

X_tr = torch.FloatTensor(X_train).to(device)
y_tr = torch.FloatTensor(y_train).to(device)

g_losses, d_losses = [], []

for epoch in range(1, GAN_EPOCHS + 1):
    # mini-batch
    idx    = torch.randperm(len(X_tr))[:GAN_BATCH]
    real_x = X_tr[idx]
    real_c = y_tr[idx].unsqueeze(1)

    # ── Train Discriminator ──
    z      = torch.randn(GAN_BATCH, NOISE_DIM).to(device)
    fake_x = G(z, real_c).detach()

    real_logit = D(real_x, real_c)
    fake_logit = D(fake_x, real_c)

    d_loss = (bce(real_logit, torch.ones_like(real_logit)) +
              bce(fake_logit, torch.zeros_like(fake_logit))) / 2

    opt_D.zero_grad(); d_loss.backward(); opt_D.step()

    # ── Train Generator ──
    z      = torch.randn(GAN_BATCH, NOISE_DIM).to(device)
    fake_x = G(z, real_c)
    fake_logit = D(fake_x, real_c)
    g_loss = bce(fake_logit, torch.ones_like(fake_logit))

    opt_G.zero_grad(); g_loss.backward(); opt_G.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    if epoch % 50 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{GAN_EPOCHS} | "
              f"G Loss: {g_loss.item():.4f}  D Loss: {d_loss.item():.4f}")

print("GAN training complete.")

# ── Generate synthetic data (10x) ──
print("\nGenerating synthetic data (10x original train size)...")
G.eval()
n_orig    = len(X_train)
n_gen     = n_orig * 9          # original 1x + synthetic 9x = 10x total

class_cnt = Counter(y_train.astype(int))
syn_list_x, syn_list_y = [], []

with torch.no_grad():
    for cls in [0, 1]:
        n_cls  = int(n_gen * (class_cnt[cls] / n_orig))
        label  = torch.full((n_cls, 1), float(cls)).to(device)
        z      = torch.randn(n_cls, NOISE_DIM).to(device)
        fake   = G(z, label).cpu().numpy()
        syn_list_x.append(fake)
        syn_list_y.append(np.full(n_cls, float(cls)))

X_syn = np.vstack(syn_list_x).astype(np.float32)
y_syn = np.concatenate(syn_list_y).astype(np.float32)

# combine original + synthetic
X_aug = np.vstack([X_train, X_syn])
y_aug = np.concatenate([y_train, y_syn])

print(f"Original train : {n_orig:,}")
print(f"Synthetic added: {len(X_syn):,}")
print(f"Total augmented: {len(X_aug):,}  ({len(X_aug)/n_orig:.1f}x)")
print(f"Augmented label dist: {dict(Counter(y_aug.astype(int)))}")

# ═══════════════════════════════════════════════════════════
# 3. Binary Classifier
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Binary Classifier  (BCEWithLogitsLoss)")
print("=" * 60)


class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),   # single logit
        )
    def forward(self, x): return self.net(x).squeeze(1)


BATCH_SIZE = 128

aug_counts = Counter(y_aug.astype(int))
aug_weights = {cls: 1.0 / cnt for cls, cnt in aug_counts.items()}
samp_w = torch.FloatTensor([aug_weights[int(l)] for l in y_aug])
aug_sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

train_aug_ds = BinaryDataset(X_aug,   y_aug)
val_ds       = BinaryDataset(X_val,   y_val)
test_ds      = BinaryDataset(X_test,  y_test)

train_loader = DataLoader(train_aug_ds, batch_size=BATCH_SIZE,
                          sampler=aug_sampler, drop_last=True)
val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model     = BinaryClassifier(INPUT_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print("Loss function: nn.BCEWithLogitsLoss  (binary classification)")

# ── Training ──
EPOCHS = 60
tr_losses, vl_losses, tr_accs, vl_accs = [], [], [], []
best_val_acc = 0
best_path    = os.path.join(SAVE_DIR, "best_binary_model.pth")

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    rl = correct = total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logit = model(xb)
        loss  = criterion(logit, yb)
        loss.backward(); optimizer.step()
        rl      += loss.item() * len(yb)
        pred     = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == yb).sum().item()
        total   += len(yb)
    tr_losses.append(rl / total); tr_accs.append(correct / total)

    model.eval()
    vl = vc = vt = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logit  = model(xb)
            loss   = criterion(logit, yb)
            vl    += loss.item() * len(yb)
            pred   = (torch.sigmoid(logit) >= 0.5).float()
            vc    += (pred == yb).sum().item()
            vt    += len(yb)
    vl_losses.append(vl / vt); vl_accs.append(vc / vt)
    scheduler.step(vl_accs[-1])

    if vl_accs[-1] > best_val_acc:
        best_val_acc = vl_accs[-1]
        torch.save(model.state_dict(), best_path)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {tr_losses[-1]:.4f}, Acc: {tr_accs[-1]:.4f} | "
              f"Val   Loss: {vl_losses[-1]:.4f}, Acc: {vl_accs[-1]:.4f}")

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

# ═══════════════════════════════════════════════════════════
# 4. Test Evaluation
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Test Set Evaluation")
print("=" * 60)

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logit = model(xb.to(device))
        prob  = torch.sigmoid(logit).cpu().numpy()
        pred  = (prob >= 0.5).astype(int)
        all_preds.extend(pred)
        all_probs.extend(prob)
        all_labels.extend(yb.numpy().astype(int))

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds)
auc      = roc_auc_score(all_labels, all_probs)

print(f"Test Accuracy : {test_acc:.4f}")
print(f"ROC-AUC Score : {auc:.4f}")
print()
print("Classification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=["Not Addicted (0)", "Addicted (1)"]))

# ═══════════════════════════════════════════════════════════
# 5. Feature Importance (Permutation)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Feature Importance (Permutation)")
print("=" * 60)

def eval_acc(X_data, y_data):
    ds  = BinaryDataset(X_data, y_data.astype(np.float32))
    ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in ldr:
            logit = model(xb.to(device))
            pred  = (torch.sigmoid(logit).cpu() >= 0.5).float()
            correct += (pred == yb).sum().item()
            total   += len(yb)
    return correct / total

baseline = eval_acc(X_test, y_test)
print(f"Baseline accuracy: {baseline:.4f}")

imp_scores = []
for i, feat in enumerate(FEATURE_COLS):
    Xp = X_test.copy()
    np.random.seed(42)
    np.random.shuffle(Xp[:, i])
    drop = baseline - eval_acc(Xp, y_test)
    imp_scores.append((feat, drop))
    print(f"  {feat:35s}: drop = {drop:+.4f}")

imp_scores.sort(key=lambda x: x[1], reverse=True)
feat_names = [s[0] for s in imp_scores]
feat_drops = [s[1] for s in imp_scores]

print("\n=== Feature Importance Ranking ===")
for rank, (f, d) in enumerate(imp_scores, 1):
    bar = "#" * max(0, int(d * 500))
    print(f"  {rank:2d}. {f:35s}: {d:+.4f}  {bar}")

# ═══════════════════════════════════════════════════════════
# 6. Visualizations
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Save Visualizations")
print("=" * 60)

CLASS_NAMES = ["Not Addicted", "Addicted"]

# (A) GAN Training Loss
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(g_losses, label="Generator Loss",     color="#e74c3c", alpha=0.8)
ax.plot(d_losses, label="Discriminator Loss", color="#3498db", alpha=0.8)
ax.set_title("Conditional GAN Training Loss", fontsize=13)
ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "00_gan_loss.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 00_gan_loss.png")

# (B) Data Augmentation — before vs after
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
orig_cnt = [int((y_train == 0).sum()), int((y_train == 1).sum())]
aug_cnt  = [int((y_aug  == 0).sum()), int((y_aug  == 1).sum())]
colors2  = ["#3498db", "#e74c3c"]

axes[0].bar(CLASS_NAMES, orig_cnt, color=colors2)
axes[0].set_title("Original Train Distribution", fontsize=12)
axes[0].set_ylabel("Sample Count")
for i, v in enumerate(orig_cnt):
    axes[0].text(i, v + 30, str(v), ha="center", fontsize=11)

axes[1].bar(CLASS_NAMES, aug_cnt, color=colors2)
axes[1].set_title(f"After GAN Augmentation (10x = {len(X_aug):,} samples)", fontsize=12)
axes[1].set_ylabel("Sample Count")
for i, v in enumerate(aug_cnt):
    axes[1].text(i, v + 100, str(v), ha="center", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "01_augmentation.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 01_augmentation.png")

# (C) Learning Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(tr_losses, label="Train Loss", color="#e74c3c")
axes[0].plot(vl_losses, label="Val Loss",   color="#3498db")
axes[0].set_title("Loss (BCEWithLogitsLoss)", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(tr_accs, label="Train Acc", color="#e74c3c")
axes[1].plot(vl_accs, label="Val Acc",   color="#3498db")
axes[1].axhline(y=test_acc, color="#2ecc71", linestyle="--",
                label=f"Test Acc ({test_acc:.3f})")
axes[1].set_title("Accuracy", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "02_learning_curve.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 02_learning_curve.png")

# (D) Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix", fontsize=13)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 03_confusion_matrix.png")

# (E) ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
ax.set_title("ROC Curve", fontsize=13)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "04_roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 04_roc_curve.png")

# (F) Feature Importance
colors_f = ["#e74c3c" if d > 0 else "#95a5a6" for d in feat_drops]
fig, ax  = plt.subplots(figsize=(11, 7))
bars = ax.barh(feat_names, feat_drops, color=colors_f)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title("Feature Importance (Permutation Importance)\n"
             "Higher value = stronger impact on addiction prediction", fontsize=12)
ax.set_xlabel("Accuracy Drop vs Baseline")
for bar, val in zip(bars, feat_drops):
    ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "05_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 05_feature_importance.png")

# ═══════════════════════════════════════════════════════════
# 7. Final Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. Final Summary")
print("=" * 60)
print(f"  Task          : Binary Classification (Not Addicted / Addicted)")
print(f"  Augmentation  : Conditional GAN  x10  ({n_orig:,} -> {len(X_aug):,})")
print(f"  Loss Function : nn.BCEWithLogitsLoss")
print(f"  Best Val Acc  : {best_val_acc:.4f}")
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  ROC-AUC       : {auc:.4f}")
print()
print("  [Feature Importance Top 5]")
for rank, (f, d) in enumerate(imp_scores[:5], 1):
    print(f"    {rank}. {f}: {d:+.4f}")
print()
print(f"  Results saved to: {SAVE_DIR}")
print("=" * 60)
print("Done!")
