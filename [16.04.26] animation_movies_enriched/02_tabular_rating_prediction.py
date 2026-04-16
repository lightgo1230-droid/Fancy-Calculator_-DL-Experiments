"""
[프로젝트 2] 정형 데이터: 평점(Rating) 예측 & Hit 분류
==========================================================
모델: PyTorch MLP + Entity Embedding (범주형 변수)
입력: Release_Year, Runtime, Genre, Animation_Style, TMDB_Vote_Count, MPAA_Rating 등
출력:
  - Task 1: TMDB_Rating Regression (평점 예측)
  - Task 2: Binary Classification – Rating ≥ 7.0 → "Hit"

설치:
  pip install torch scikit-learn pandas numpy matplotlib tqdm

실행:
  python 02_tabular_rating_prediction.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_PATH = r"C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv"
BATCH_SIZE  = 256
EPOCHS      = 50
LR          = 1e-3
HIT_THRESH  = 7.0      # 이 점수 이상을 "Hit" 으로 정의
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Device] {DEVICE}")


# ──────────────────────────────────────────────
# 1. 데이터 로드 & 피처 엔지니어링
# ──────────────────────────────────────────────
def preprocess(path: str):
    df = pd.read_csv(path)

    # 타겟이 있는 행만
    df = df.dropna(subset=["TMDB_Rating"]).copy()

    # ── 수치형 피처 ──────────────────────────────
    df["Release_Year"]    = df["Release_Year"].fillna(df["Release_Year"].median())
    df["Runtime"]         = df["Movie_Length_Minutes"].fillna(df["Movie_Length_Minutes"].median())
    df["Vote_Count_Log"]  = np.log1p(df["TMDB_Vote_Count"].fillna(0))
    df["Popularity_Log"]  = np.log1p(df["TMDB_Popularity"].fillna(0))

    # ── 상호작용 피처 5개 (Interaction Features) ──────────
    # ① 투표수 × 인기도 (함께 높을수록 검증된 인기작)
    df["Vote_x_Pop"]         = df["Vote_Count_Log"] * df["Popularity_Log"]
    # ② 바이럴 지수: 인기도를 투표수로 나눈 비율 (화제성)
    df["Viral_Score"]        = np.log1p(
        df["TMDB_Popularity"].fillna(0) / (df["TMDB_Vote_Count"].fillna(0) + 1)
    )
    # ③ 영화 나이: 오래된 영화일수록 평점이 안정적으로 수렴
    df["Movie_Age"]          = (2026 - df["Release_Year"]).clip(0, 200)
    # ④ 영화 나이 × 투표 로그: 오래됐지만 투표 많은 클래식 포착
    df["Age_x_VoteLog"]      = df["Movie_Age"] * df["Vote_Count_Log"]
    # ⑤ 러닝타임 × 투표 로그: 긴 영화 + 많은 투표 = 서사형 고평점 패턴
    df["Runtime_x_VoteLog"]  = df["Runtime"] * df["Vote_Count_Log"]

    # ── 범주형 피처 ──────────────────────────────
    cat_cols = ["Animation_Style", "MPAA_Rating", "Target_Audience",
                "Era", "Popularity_Tier", "Original_Language"]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # 주 장르 (첫 번째 항목만)
    df["Primary_Genre"] = df["Genre"].apply(
        lambda x: x.split(",")[0].split("|")[0].strip() if pd.notna(x) else "Unknown"
    )
    cat_cols.append("Primary_Genre")

    # LabelEncoding → Embedding 에 사용
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 불리언 피처
    bool_cols = ["Is_TV_Compilation", "Hidden_Gem", "Is_Adult_Content",
                 "Live_Action_Remake", "Belongs_To_Collection"]
    for col in bool_cols:
        if col in df.columns:
            df[col + "_int"] = df[col].map({"Yes": 1, "No": 0, True: 1, False: 0}).fillna(0).astype(int)

    return df, cat_cols, encoders


df, cat_cols, encoders = preprocess(DATA_PATH)
print(f"[Data] 학습 가능 샘플: {len(df):,}개")

num_features = ["Release_Year", "Runtime", "Vote_Count_Log", "Popularity_Log",
                "Vote_x_Pop", "Viral_Score", "Movie_Age", "Age_x_VoteLog", "Runtime_x_VoteLog"]
bool_features = [c + "_int" for c in ["Is_TV_Compilation", "Hidden_Gem",
                                       "Is_Adult_Content", "Live_Action_Remake"]
                 if c + "_int" in df.columns]
cat_enc_cols = [c + "_enc" for c in cat_cols]

# 범주형 임베딩 차원 설정: min(50, (카디널리티+1)//2)
cat_dims = []
for col in cat_cols:
    n = df[col + "_enc"].nunique()
    emb_dim = min(50, (n + 1) // 2)
    cat_dims.append((n, emb_dim))

# 수치형 스케일링
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# 타겟
y_reg = df["TMDB_Rating"].values.astype(np.float32)
y_cls = (y_reg >= HIT_THRESH).astype(np.float32)   # 0 or 1

X_cat = df[cat_enc_cols].values.astype(np.int64)
X_num = df[num_features + bool_features].values.astype(np.float32)

(X_cat_tr, X_cat_val,
 X_num_tr,  X_num_val,
 y_reg_tr,  y_reg_val,
 y_cls_tr,  y_cls_val) = train_test_split(
    X_cat, X_num, y_reg, y_cls, test_size=0.2, random_state=42
)

print(f"[Split] Train {len(X_cat_tr):,}  /  Val {len(X_cat_val):,}")
print(f"[Hit Ratio] Train {y_cls_tr.mean():.2%}  /  Val {y_cls_val.mean():.2%}")


# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X_cat, X_num, y_reg, y_cls):
        self.X_cat = torch.LongTensor(X_cat)
        self.X_num = torch.FloatTensor(X_num)
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.FloatTensor(y_cls)

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y_reg[idx], self.y_cls[idx]


train_ds = TabularDataset(X_cat_tr, X_num_tr, y_reg_tr, y_cls_tr)
val_ds   = TabularDataset(X_cat_val, X_num_val, y_reg_val, y_cls_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ──────────────────────────────────────────────
# 3. 모델: Entity Embedding MLP
# ──────────────────────────────────────────────
class EntityEmbeddingMLP(nn.Module):
    """
    범주형 변수 → Entity Embedding → 수치형과 concat → MLP
    두 헤드: Regression 헤드(rating) + Classification 헤드(hit)
    """
    def __init__(self, cat_dims, num_dim, hidden_dims=(512, 256, 128), dropout=0.3):
        super().__init__()
        # 범주형 임베딩 레이어
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, emb_dim) for n_cats, emb_dim in cat_dims
        ])
        total_emb = sum(emb_dim for _, emb_dim in cat_dims)
        in_dim = total_emb + num_dim

        # 공유 Backbone
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Regression 헤드
        self.reg_head = nn.Linear(prev, 1)

        # Classification 헤드
        self.cls_head = nn.Linear(prev, 1)

    def forward(self, X_cat, X_num):
        embs = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [X_num], dim=1)
        feat = self.backbone(x)
        return self.reg_head(feat).squeeze(-1), self.cls_head(feat).squeeze(-1)


num_dim = X_num_tr.shape[1]
model   = EntityEmbeddingMLP(cat_dims, num_dim).to(DEVICE)
print(f"\n[모델] 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

optimizer    = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
huber_loss   = nn.HuberLoss(delta=1.0)   # MSE → Huber (이상치 영향 억제)
bce_loss     = nn.BCEWithLogitsLoss()
LAMBDA_CLS   = 0.1   # 0.5 → 0.1: 회귀 헤드에 더 많은 학습 집중


# ──────────────────────────────────────────────
# 4. 학습
# ──────────────────────────────────────────────
history = {"train_loss": [], "val_mae": [], "val_auc": []}

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_reg_pred, all_reg_true = [], []
    all_cls_pred, all_cls_true = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X_cat, X_num, y_reg, y_cls in loader:
            X_cat, X_num = X_cat.to(DEVICE), X_num.to(DEVICE)
            y_reg, y_cls = y_reg.to(DEVICE),  y_cls.to(DEVICE)

            reg_out, cls_out = model(X_cat, X_num)
            loss = huber_loss(reg_out, y_reg) + LAMBDA_CLS * bce_loss(cls_out, y_cls)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_reg_pred.extend(reg_out.cpu().detach().numpy())
            all_reg_true.extend(y_reg.cpu().numpy())
            all_cls_pred.extend(torch.sigmoid(cls_out).cpu().detach().numpy())
            all_cls_true.extend(y_cls.cpu().numpy())

    mae = mean_absolute_error(all_reg_true, all_reg_pred)
    rmse = mean_squared_error(all_reg_true, all_reg_pred) ** 0.5
    try:
        auc = roc_auc_score(all_cls_true, all_cls_pred)
    except Exception:
        auc = 0.0
    return total_loss / len(loader), mae, rmse, auc


print("\n" + "="*60)
print("  [학습] Entity Embedding MLP (Regression + Classification)")
print("="*60)

best_val_mae = float("inf")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_mae, tr_rmse, tr_auc = run_epoch(model, train_loader, optimizer)
    va_loss, va_mae, va_rmse, va_auc = run_epoch(model, val_loader)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_mae"].append(va_mae)
    history["val_auc"].append(va_auc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val MAE: {va_mae:.4f}  RMSE: {va_rmse:.4f}  AUC: {va_auc:.4f}")

    if va_mae < best_val_mae:
        best_val_mae = va_mae
        torch.save(model.state_dict(), "tabular_model_best.pt")

print(f"\n  최고 Val MAE: {best_val_mae:.4f}  →  tabular_model_best.pt 저장")


# ──────────────────────────────────────────────
# 5. 최종 평가 & 결과 시각화
# ──────────────────────────────────────────────
model.load_state_dict(torch.load("tabular_model_best.pt", map_location=DEVICE))
model.eval()

all_reg_pred, all_reg_true = [], []
all_cls_pred_prob, all_cls_pred_bin, all_cls_true = [], [], []

with torch.no_grad():
    for X_cat, X_num, y_reg, y_cls in val_loader:
        reg_out, cls_out = model(X_cat.to(DEVICE), X_num.to(DEVICE))
        all_reg_pred.extend(reg_out.cpu().numpy())
        all_reg_true.extend(y_reg.numpy())
        prob = torch.sigmoid(cls_out).cpu().numpy()
        all_cls_pred_prob.extend(prob)
        all_cls_pred_bin.extend((prob > 0.5).astype(int))
        all_cls_true.extend(y_cls.numpy())

print("\n[Task 1 - Regression 결과]")
mae  = mean_absolute_error(all_reg_true, all_reg_pred)
rmse = mean_squared_error(all_reg_true, all_reg_pred) ** 0.5
r2   = r2_score(all_reg_true, all_reg_pred)
print(f"  MAE : {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²  : {r2:.4f}  (기존 베이스라인: 0.1114)")

print(f"\n[Task 2 - Binary Classification 결과] (임계값: Rating ≥ {HIT_THRESH})")
print(classification_report(all_cls_true, all_cls_pred_bin,
                             target_names=["Not Hit", "Hit"], zero_division=0))
print(f"  ROC-AUC: {roc_auc_score(all_cls_true, all_cls_pred_prob):.4f}")

# 학습 곡선 저장
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["train_loss"]); axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")
axes[1].plot(history["val_mae"]);    axes[1].set_title("Val MAE");    axes[1].set_xlabel("Epoch")
axes[2].plot(history["val_auc"]);    axes[2].set_title("Val AUC");    axes[2].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("tabular_training_curve.png", dpi=120)
print("\n  학습 곡선 → tabular_training_curve.png 저장")

# 예측 산점도
plt.figure(figsize=(6, 6))
plt.scatter(all_reg_true, all_reg_pred, alpha=0.3, s=10)
plt.plot([0, 10], [0, 10], "r--")
plt.xlabel("실제 Rating"); plt.ylabel("예측 Rating")
plt.title(f"Rating Prediction  (MAE={mae:.3f})")
plt.tight_layout()
plt.savefig("tabular_scatter.png", dpi=120)
print("  산점도      → tabular_scatter.png 저장")

print("\n[완료] 02_tabular_rating_prediction.py 실행 종료")
