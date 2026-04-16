"""
[정확도 측정] 프로젝트 2 - 정형 데이터: Entity Embedding MLP 평점 예측
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)
from tqdm import tqdm

# ── 설정 ────────────────────────────────────────────
DATA_PATH  = r"C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv"
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 1e-3
HIT_THRESH = 7.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}\n")

# ── 데이터 ──────────────────────────────────────────
df = pd.read_csv(DATA_PATH).dropna(subset=["TMDB_Rating"]).copy()
df["Release_Year"]   = df["Release_Year"].fillna(df["Release_Year"].median())
df["Runtime"]        = df["Movie_Length_Minutes"].fillna(df["Movie_Length_Minutes"].median())
df["Vote_Log"]       = np.log1p(df["TMDB_Vote_Count"].fillna(0))
df["Pop_Log"]        = np.log1p(df["TMDB_Popularity"].fillna(0))
df["Budget_Log"]     = np.log1p(df["Budget_Million_USD"].fillna(0))

# 범주형
cat_cols = ["Animation_Style","MPAA_Rating","Target_Audience","Era","Popularity_Tier","Original_Language"]
for c in cat_cols:
    df[c] = df[c].fillna("Unknown")
df["Primary_Genre"] = df["Genre"].apply(lambda x: x.split(",")[0].split("|")[0].strip() if pd.notna(x) else "Unknown")
cat_cols.append("Primary_Genre")

encoders, cat_dims = {}, []
for c in cat_cols:
    le = LabelEncoder()
    df[c+"_enc"] = le.fit_transform(df[c].astype(str))
    encoders[c]  = le
    n = df[c+"_enc"].nunique()
    cat_dims.append((n, min(50,(n+1)//2)))

# 불리언
bool_cols = ["Is_TV_Compilation","Hidden_Gem","Is_Adult_Content","Live_Action_Remake"]
for c in bool_cols:
    df[c+"_int"] = df[c].map({True:1,False:0,"Yes":1,"No":0}).fillna(0).astype(int)

num_cols  = ["Release_Year","Runtime","Vote_Log","Pop_Log","Budget_Log"]
bool_feats = [c+"_int" for c in bool_cols]
cat_encs  = [c+"_enc" for c in cat_cols]

scaler    = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X_cat = df[cat_encs].values.astype(np.int64)
X_num = df[num_cols + bool_feats].values.astype(np.float32)
y_reg = df["TMDB_Rating"].values.astype(np.float32)
y_cls = (y_reg >= HIT_THRESH).astype(np.float32)

(X_cat_tr, X_cat_va,
 X_num_tr,  X_num_va,
 y_reg_tr,  y_reg_va,
 y_cls_tr,  y_cls_va) = train_test_split(X_cat, X_num, y_reg, y_cls,
                                          test_size=0.2, random_state=42)
print(f"  Train {len(y_reg_tr):,}  /  Val {len(y_reg_va):,}")
print(f"  Hit 비율  Train {y_cls_tr.mean():.2%}  /  Val {y_cls_va.mean():.2%}\n")

# ── Dataset ─────────────────────────────────────────
class TabDS(Dataset):
    def __init__(self, Xc, Xn, yr, yc):
        self.Xc=torch.LongTensor(Xc); self.Xn=torch.FloatTensor(Xn)
        self.yr=torch.FloatTensor(yr); self.yc=torch.FloatTensor(yc)
    def __len__(self): return len(self.yr)
    def __getitem__(self,i): return self.Xc[i],self.Xn[i],self.yr[i],self.yc[i]

tr_ld = DataLoader(TabDS(X_cat_tr,X_num_tr,y_reg_tr,y_cls_tr), batch_size=BATCH_SIZE, shuffle=True)
va_ld = DataLoader(TabDS(X_cat_va,X_num_va,y_reg_va,y_cls_va), batch_size=BATCH_SIZE, shuffle=False)

# ── 모델 ────────────────────────────────────────────
class EmbMLP(nn.Module):
    def __init__(self, cat_dims, num_dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n,d) for n,d in cat_dims])
        in_dim    = sum(d for _,d in cat_dims) + num_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512,256),    nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256,128),    nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
        )
        self.reg = nn.Linear(128,1)
        self.cls = nn.Linear(128,1)
    def forward(self, Xc, Xn):
        x = torch.cat([e(Xc[:,i]) for i,e in enumerate(self.embs)] + [Xn], dim=1)
        f = self.body(x)
        return self.reg(f).squeeze(-1), self.cls(f).squeeze(-1)

model   = EmbMLP(cat_dims, X_num_tr.shape[1]).to(DEVICE)
opt     = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
mse_fn  = nn.MSELoss()
bce_fn  = nn.BCEWithLogitsLoss()
print(f"  파라미터 수 : {sum(p.numel() for p in model.parameters()):,}\n")

# ── 학습 ────────────────────────────────────────────
print("=" * 62)
print("  [학습]  Entity Embedding MLP (Regression + Classification)")
print("=" * 62)

def evaluate(loader):
    model.eval()
    rp, rt, cp, ct = [], [], [], []
    with torch.no_grad():
        for Xc,Xn,yr,yc in loader:
            ro, co = model(Xc.to(DEVICE), Xn.to(DEVICE))
            rp.extend(ro.cpu().numpy()); rt.extend(yr.numpy())
            cp.extend(torch.sigmoid(co).cpu().numpy()); ct.extend(yc.numpy())
    rp,rt = np.array(rp), np.array(rt)
    cp,ct = np.array(cp), np.array(ct)
    mae   = mean_absolute_error(rt, rp)
    rmse  = mean_squared_error(rt, rp)**0.5
    r2    = r2_score(rt, rp)
    auc   = roc_auc_score(ct, cp) if len(np.unique(ct))>1 else 0.0
    cb    = (cp > 0.5).astype(int)
    acc   = accuracy_score(ct, cb)
    return mae, rmse, r2, auc, acc, rp, rt, cp, ct

best_mae = float("inf")
for ep in tqdm(range(1, EPOCHS+1), desc="  Training"):
    model.train()
    for Xc,Xn,yr,yc in tr_ld:
        ro,co = model(Xc.to(DEVICE), Xn.to(DEVICE))
        loss  = mse_fn(ro,yr.to(DEVICE)) + 0.5*bce_fn(co,yc.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if ep % 10 == 0:
        mae,rmse,r2,auc,acc,*_ = evaluate(va_ld)
        tqdm.write(f"  Epoch {ep:>3} | MAE {mae:.4f}  RMSE {rmse:.4f}  "
                   f"R² {r2:.4f}  AUC {auc:.4f}  Acc {acc:.4f}")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "eval_tab_best.pt")

# ── 최종 결과 ────────────────────────────────────────
model.load_state_dict(torch.load("eval_tab_best.pt", map_location=DEVICE))
mae, rmse, r2, auc, acc, rp, rt, cp, ct = evaluate(va_ld)
cb = (cp > 0.5).astype(int)

print("\n" + "=" * 62)
print("  [최종 정확도 측정 결과] - 정형 데이터 MLP (Validation Set)")
print("=" * 62)
print()
print("  ┌─ Task 1: Rating 회귀 (연속값 예측) ───────────────────┐")
print(f"  │  MAE  (평균 절대 오차)  : {mae:.4f}  점")
print(f"  │  RMSE (평균 제곱근 오차): {rmse:.4f}  점")
print(f"  │  R²   (결정계수)        : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  │  평균 예측값            : {rp.mean():.3f}  (실제 {rt.mean():.3f})")
print(f"  │  예측 범위              : [{rp.min():.2f} ~ {rp.max():.2f}]")
print("  └───────────────────────────────────────────────────────┘\n")
print(f"  ┌─ Task 2: Hit 분류 (Rating ≥ {HIT_THRESH}) ──────────────────┐")
print(f"  │  Accuracy  (정확도)     : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  │  ROC-AUC               : {auc:.4f}  ({auc*100:.2f}%)")
print(f"  │  Hit 실제 비율         : {ct.mean():.3f}")
print(f"  │  Hit 예측 비율         : {cb.mean():.3f}")
print("  └───────────────────────────────────────────────────────┘\n")

print("  [Hit 분류 상세 리포트]")
print(classification_report(ct, cb, target_names=["Not Hit","Hit"], zero_division=0))

print("  [혼동 행렬]")
cm = confusion_matrix(ct, cb)
print(f"               예측 Not-Hit  예측 Hit")
print(f"  실제 Not-Hit      {cm[0,0]:>6}    {cm[0,1]:>6}")
print(f"  실제 Hit          {cm[1,0]:>6}    {cm[1,1]:>6}")

# 평점 구간별 MAE
print("\n  [평점 구간별 예측 오차]")
print(f"  {'구간':<12} {'샘플수':>6} {'MAE':>8} {'RMSE':>8}")
print("  " + "-"*36)
bins = [(0,5,"0~5점"),(5,6,"5~6점"),(6,7,"6~7점"),(7,8,"7~8점"),(8,10,"8~10점")]
for lo,hi,label in bins:
    mask = (rt >= lo) & (rt < hi)
    if mask.sum() > 0:
        m = mean_absolute_error(rt[mask], rp[mask])
        r = mean_squared_error(rt[mask], rp[mask])**0.5
        print(f"  {label:<12} {mask.sum():>6} {m:>8.4f} {r:>8.4f}")
print()
