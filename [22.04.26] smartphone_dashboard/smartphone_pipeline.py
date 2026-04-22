"""
Smartprix Smartphone Dataset - PyTorch MLP Pipeline
====================================================
Task 1 : Price Regression   (log-transformed MSE)
Task 2 : Price Category Classification  (4-class)
"""

# ─────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # headless rendering

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, r2_score,
                              classification_report, confusion_matrix)

import os, re

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH   = r"C:\Users\USER\OneDrive\Desktop\smartprix_smartphones_april_2026.csv"
OUTPUT_DIR  = r"C:\Users\USER\OneDrive\Desktop\smartphone_pytorch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# ─────────────────────────────────────────────────
# 1. Data Loading & Cleaning
# ─────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Extract OS version number only (e.g., "Android v16" → 16.0)
    def extract_os_ver(s):
        if pd.isna(s):
            return np.nan
        m = re.search(r"(\d+\.?\d*)", str(s))
        return float(m.group(1)) if m else np.nan

    df["os_version"] = df["os"].apply(extract_os_ver)

    # ── boolean → int
    for col in ["has_5G", "has_NFC", "has_IR"]:
        df[col] = df[col].astype(int)

    # ── Fill numeric missing values with median
    num_cols = [
        "num_core", "processor_speed", "fast_charging(W)",
        "charging_ratio", "refresh_rate", "rear_camera",
        "front_camera", "os_version"
    ]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # ── Fill categorical missing values with mode
    cat_cols = ["processor_brand"]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ─────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "spec_score", "vfm_score", "num_core", "processor_speed",
    "ram", "memory", "battery_capacity(mAh)", "fast_charging(W)",
    "charging_ratio", "screen_size", "refresh_rate",
    "rear_camera", "front_camera", "rear_camera_count",
    "has_5G", "has_NFC", "has_IR", "os_version"
]

CATEGORICAL_FEATURES = [
    "brand_name", "processor_brand", "charging_speed_type"
]

TARGET_REG  = "price"
TARGET_CLF  = "price_category"

PRICE_CATEGORY_ORDER = ["Budget", "Mid-Range", "Premium", "Flagship"]


def encode_categoricals(df, cat_encoders=None, fit=True):
    """Apply LabelEncoder and return encoder dictionary."""
    if cat_encoders is None:
        cat_encoders = {}
    encoded = {}
    for col in CATEGORICAL_FEATURES:
        le = cat_encoders.get(col, LabelEncoder())
        if fit:
            le.fit(df[col].astype(str))
            cat_encoders[col] = le
        encoded[col] = le.transform(df[col].astype(str))
    return encoded, cat_encoders


def prepare_data(df):
    """Full preprocessing → return (X_num, X_cat, y_reg, y_clf)."""
    X_num = df[NUMERIC_FEATURES].values.astype(np.float32)

    encoded, cat_encoders = encode_categoricals(df, fit=True)
    X_cat = np.stack([encoded[c] for c in CATEGORICAL_FEATURES], axis=1)

    y_reg = np.log1p(df[TARGET_REG].values).astype(np.float32)   # log1p transform

    le_clf = LabelEncoder()
    le_clf.fit(PRICE_CATEGORY_ORDER)
    y_clf = le_clf.transform(df[TARGET_CLF]).astype(np.int64)

    # Compute embedding dim per category: min(50, (n_unique//2)+1)
    cat_vocab_sizes = [len(cat_encoders[c].classes_) for c in CATEGORICAL_FEATURES]
    cat_embed_dims  = [min(50, (v // 2) + 1) for v in cat_vocab_sizes]

    scaler = StandardScaler()
    X_num  = scaler.fit_transform(X_num)

    return (X_num, X_cat, y_reg, y_clf,
            cat_vocab_sizes, cat_embed_dims, scaler, cat_encoders, le_clf)


# ─────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────
class SmartphoneDataset(Dataset):
    def __init__(self, X_num, X_cat, y_reg, y_clf):
        self.X_num  = torch.tensor(X_num,  dtype=torch.float32)
        self.X_cat  = torch.tensor(X_cat,  dtype=torch.long)
        self.y_reg  = torch.tensor(y_reg,  dtype=torch.float32).unsqueeze(1)
        self.y_clf  = torch.tensor(y_clf,  dtype=torch.long)

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_cat[idx],
                self.y_reg[idx],  self.y_clf[idx])


# ─────────────────────────────────────────────────
# 4. Model Architecture
# ─────────────────────────────────────────────────
class SmartphoneMLP(nn.Module):
    """
    MLP combining Embedding layers (categorical) + Dense layers (numeric).
    Outputs both a Regression head and a Classification head simultaneously.
    """
    def __init__(self, num_input_dim, cat_vocab_sizes, cat_embed_dims,
                 hidden_dims=(256, 128, 64), dropout=0.3, n_classes=4):
        super().__init__()

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, dim)
            for vocab, dim in zip(cat_vocab_sizes, cat_embed_dims)
        ])
        embed_total_dim = sum(cat_embed_dims)

        input_dim = num_input_dim + embed_total_dim

        # Shared Backbone
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Regression Head: predict log1p(price)
        self.reg_head = nn.Linear(in_dim, 1)

        # Classification Head: predict price_category
        self.clf_head = nn.Linear(in_dim, n_classes)

    def forward(self, x_num, x_cat):
        # Concatenate embeddings
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embeds, dim=1)

        # Learn shared representation
        x = self.backbone(x)

        return self.reg_head(x), self.clf_head(x)


# ─────────────────────────────────────────────────
# 5. Multi-Task Loss
# ─────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Learnable uncertainty weighting (Kendall et al. 2018).
    Automatically balances two task losses via log_var.
    """
    def __init__(self):
        super().__init__()
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_clf = nn.Parameter(torch.zeros(1))
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, pred_reg, y_reg, pred_clf, y_clf):
        loss_reg = self.mse(pred_reg, y_reg)
        loss_clf = self.ce(pred_clf, y_clf)

        # Uncertainty-weighted sum
        precision_reg = torch.exp(-self.log_var_reg)
        precision_clf = torch.exp(-self.log_var_clf)

        total = (precision_reg * loss_reg + self.log_var_reg +
                 precision_clf * loss_clf + self.log_var_clf)
        return total, loss_reg, loss_clf


# ─────────────────────────────────────────────────
# 6. Training & Evaluation
# ─────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss = reg_loss_sum = clf_loss_sum = 0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_num, x_cat, y_reg, y_clf in loader:
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            y_reg = y_reg.to(DEVICE)
            y_clf = y_clf.to(DEVICE)

            pred_reg, pred_clf = model(x_num, x_cat)
            loss, l_reg, l_clf = criterion(pred_reg, y_reg, pred_clf, y_clf)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = x_num.size(0)
            total_loss    += loss.item() * bs
            reg_loss_sum  += l_reg.item() * bs
            clf_loss_sum  += l_clf.item() * bs
            n += bs

    return total_loss / n, reg_loss_sum / n, clf_loss_sum / n


def evaluate(model, loader, scaler_y=None):
    """Collect predictions and ground truths, return as numpy arrays."""
    model.eval()
    all_reg_pred, all_reg_true = [], []
    all_clf_pred, all_clf_true = [], []

    with torch.no_grad():
        for x_num, x_cat, y_reg, y_clf in loader:
            pred_r, pred_c = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            all_reg_pred.append(pred_r.cpu().numpy())
            all_reg_true.append(y_reg.numpy())
            all_clf_pred.append(pred_c.argmax(dim=1).cpu().numpy())
            all_clf_true.append(y_clf.numpy())

    reg_pred = np.concatenate(all_reg_pred).flatten()
    reg_true = np.concatenate(all_reg_true).flatten()
    clf_pred = np.concatenate(all_clf_pred)
    clf_true = np.concatenate(all_clf_true)

    # Inverse log1p transform
    reg_pred_orig = np.expm1(reg_pred)
    reg_true_orig = np.expm1(reg_true)

    mae = mean_absolute_error(reg_true_orig, reg_pred_orig)
    r2  = r2_score(reg_true_orig, reg_pred_orig)
    acc = (clf_pred == clf_true).mean()

    return mae, r2, acc, reg_pred_orig, reg_true_orig, clf_pred, clf_true


# ─────────────────────────────────────────────────
# 7. Plotting
# ─────────────────────────────────────────────────
def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = range(1, len(history["train_total"]) + 1)

    for ax, key, title in zip(
        axes,
        [("train_total", "val_total"),
         ("train_reg",   "val_reg"),
         ("train_clf",   "val_clf")],
        ["Total Loss", "Regression Loss (MSE)", "Classification Loss (CE)"]
    ):
        ax.plot(epochs, history[key[0]], label="Train", linewidth=2)
        ax.plot(epochs, history[key[1]], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_regression_scatter(reg_true, reg_pred, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: predicted vs actual
    ax = axes[0]
    ax.scatter(reg_true, reg_pred, alpha=0.4, s=20, color="steelblue")
    lim = [min(reg_true.min(), reg_pred.min()),
           max(reg_true.max(), reg_pred.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect Fit")
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Regression: Actual vs Predicted")
    ax.legend()
    ax.grid(alpha=0.3)

    # Residual Distribution
    ax2 = axes[1]
    residuals = reg_pred - reg_true
    ax2.hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Residual (Predicted − Actual, ₹)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_confusion_matrix(clf_true, clf_pred, class_names, save_path):
    cm = confusion_matrix(clf_true, clf_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Row %")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_pct[i, j] > 55 else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Price Category)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")


# ─────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  Smartphone PyTorch Multi-Task Learning Pipeline")
    print("="*55)

    # ── Data preparation
    df = load_and_clean(DATA_PATH)
    (X_num, X_cat, y_reg, y_clf,
     cat_vocab_sizes, cat_embed_dims,
     scaler, cat_encoders, le_clf) = prepare_data(df)

    print(f"\n[DATA] Samples       : {len(df)}")
    print(f"[DATA] Numeric feats : {X_num.shape[1]}")
    print(f"[DATA] Categorical   : {CATEGORICAL_FEATURES}")
    print(f"[DATA] Vocab sizes   : {cat_vocab_sizes}")
    print(f"[DATA] Embed dims    : {cat_embed_dims}")

    # ── Train / Val / Test  (70 / 15 / 15)
    idx = np.arange(len(df))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=SEED)
    va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=SEED)

    def make_ds(i):
        return SmartphoneDataset(X_num[i], X_cat[i], y_reg[i], y_clf[i])

    train_ds = make_ds(tr_idx)
    val_ds   = make_ds(va_idx)
    test_ds  = make_ds(te_idx)

    BATCH = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    print(f"\n[SPLIT] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    # ── Model initialization
    model = SmartphoneMLP(
        num_input_dim   = X_num.shape[1],
        cat_vocab_sizes = cat_vocab_sizes,
        cat_embed_dims  = cat_embed_dims,
        hidden_dims     = (256, 128, 64),
        dropout         = 0.3,
        n_classes       = len(PRICE_CATEGORY_ORDER)
    ).to(DEVICE)

    criterion = MultiTaskLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=10, min_lr=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {total_params:,}")

    # ── Training loop
    N_EPOCHS   = 150
    PATIENCE   = 20
    best_val   = float("inf")
    patience_cnt = 0
    best_state = None

    history = {k: [] for k in
               ["train_total","train_reg","train_clf",
                "val_total",  "val_reg",  "val_clf"]}

    print(f"\n[TRAIN] Epochs={N_EPOCHS} | Batch={BATCH} | EarlyStop patience={PATIENCE}\n")

    for epoch in range(1, N_EPOCHS + 1):
        tr_tot, tr_reg, tr_clf = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va_tot, va_reg, va_clf = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step(va_tot)

        history["train_total"].append(tr_tot)
        history["train_reg"].append(tr_reg)
        history["train_clf"].append(tr_clf)
        history["val_total"].append(va_tot)
        history["val_reg"].append(va_reg)
        history["val_clf"].append(va_clf)

        # Early Stopping
        if va_tot < best_val:
            best_val   = va_tot
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train [{tr_tot:.4f} = Reg {tr_reg:.4f} + Clf {tr_clf:.4f}] | "
                  f"Val [{va_tot:.4f} = Reg {va_reg:.4f} + Clf {va_clf:.4f}] | "
                  f"LR {lr_now:.6f}")

        if patience_cnt >= PATIENCE:
            print(f"\n  [EarlyStop] Stopped at epoch {epoch}.")
            break

    # ── Restore best weights and evaluate on test set
    model.load_state_dict(best_state)

    mae, r2, acc, reg_pred, reg_true, clf_pred, clf_true = evaluate(model, test_loader)

    print("\n" + "="*55)
    print("  Test Set Results")
    print("="*55)
    print(f"  [Regression]     MAE : ₹{mae:,.0f}")
    print(f"  [Regression]     R²  : {r2:.4f}")
    print(f"  [Classification] Acc : {acc*100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(clf_true, clf_pred,
                                 target_names=PRICE_CATEGORY_ORDER, digits=3))

    # ── Save model
    model_path = os.path.join(OUTPUT_DIR, "smartphone_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "cat_vocab_sizes": cat_vocab_sizes,
        "cat_embed_dims":  cat_embed_dims,
        "num_input_dim":   X_num.shape[1],
    }, model_path)
    print(f"\n[SAVE] Model → {model_path}")

    # ── Save visualizations
    print("\n[PLOT] Generating charts ...")
    plot_training_curves(
        history,
        os.path.join(OUTPUT_DIR, "01_training_curves.png")
    )
    plot_regression_scatter(
        reg_true, reg_pred,
        os.path.join(OUTPUT_DIR, "02_regression_scatter.png")
    )
    plot_confusion_matrix(
        clf_true, clf_pred,
        PRICE_CATEGORY_ORDER,
        os.path.join(OUTPUT_DIR, "03_confusion_matrix.png")
    )

    print("\n[DONE] Pipeline complete.")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
