"""
[정확도 측정] 프로젝트 1 - NLP: DistilBERT 시놉시스 기반 장르 분류 (빠른 버전)
전략: 5,000개 샘플 + 2 에폭 → CPU 약 8분
"""
import warnings, re, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (f1_score, hamming_loss, jaccard_score,
                              accuracy_score, classification_report)
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# ── 설정 ────────────────────────────────────────────
DATA_PATH   = r"C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv"
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 64          # ↑ 배치 크기 증가
EPOCHS      = 3
LR          = 3e-5
TOP_GENRES  = 8           # 상위 8개 장르
THRESHOLD   = 0.5
MAX_SAMPLES = 8000        # 샘플 수 제한
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}")
t0 = time.time()

# ── 데이터 ──────────────────────────────────────────
def clean_genre(g):
    return [x.strip() for x in re.split(r"[,|]", g) if x.strip()]

df = pd.read_csv(DATA_PATH).dropna(subset=["Overview", "Genre"]).copy()
df["genre_list"] = df["Genre"].apply(clean_genre)

from collections import Counter
all_genres  = [g for gs in df["genre_list"] for g in gs]
top_genres  = {g for g, _ in Counter(all_genres).most_common(TOP_GENRES)}
df["genre_list"] = df["genre_list"].apply(lambda gs: [g for g in gs if g in top_genres])
df = df[df["genre_list"].map(len) > 0].reset_index(drop=True)

# 샘플 제한 (Overview 길이 기준 필터링: 너무 짧은 건 제거)
df["ov_len"] = df["Overview"].str.len()
df = df[df["ov_len"] >= 30].reset_index(drop=True)
if len(df) > MAX_SAMPLES:
    df = df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)

mlb    = MultiLabelBinarizer(classes=sorted(top_genres))
labels = mlb.fit_transform(df["genre_list"])
texts  = df["Overview"].tolist()

(tr_txt, va_txt,
 tr_lbl, va_lbl) = train_test_split(texts, labels, test_size=0.2, random_state=42)

print(f"  Train  : {len(tr_txt):,}  /  Val : {len(va_txt):,}")
print(f"  장르   : {list(mlb.classes_)}\n")

# ── Dataset ─────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

class MovieDS(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), self.labels[i]

tr_ds = MovieDS(tr_txt, tr_lbl)
va_ds = MovieDS(va_txt, va_lbl)
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 모델 ────────────────────────────────────────────
class GenreModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        # 하위 4 레이어 동결 (속도 향상)
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < 4:
                for p in layer.parameters():
                    p.requires_grad = False
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(h, 256),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n)
        )
    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.head(out.last_hidden_state[:, 0, :])

n_labels = len(top_genres)
model    = GenreModel(n_labels).to(DEVICE)
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  파라미터 수: 전체 {total_params:,}  /  학습 {train_params:,}\n")

opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
steps = len(tr_loader) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, steps//10, steps)
crit  = nn.BCEWithLogitsLoss()

# ── 평가 함수 ────────────────────────────────────────
def run_eval():
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ids, mask, lbl in va_loader:
            logits = model(ids.to(DEVICE), mask.to(DEVICE))
            prob   = torch.sigmoid(logits).cpu().numpy()
            preds.append((prob > THRESHOLD).astype(int))
            trues.append(lbl.numpy())
    P = np.vstack(preds); T = np.vstack(trues)
    f1_mi = f1_score(T, P, average="micro",    zero_division=0)
    f1_ma = f1_score(T, P, average="macro",    zero_division=0)
    f1_w  = f1_score(T, P, average="weighted", zero_division=0)
    hl    = hamming_loss(T, P)
    jac   = jaccard_score(T, P, average="samples", zero_division=0)
    ex    = accuracy_score(T, P)
    return f1_mi, f1_ma, f1_w, hl, jac, ex, P, T

# ── 학습 ────────────────────────────────────────────
print("=" * 65)
print("  [학습]  DistilBERT Multi-label 장르 분류 (하위 레이어 Frozen)")
print("=" * 65)

best_f1, best_P, best_T = 0, None, None
for ep in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0
    for ids, mask, lbl in tqdm(tr_loader, desc=f"  Epoch {ep}", leave=False):
        logits = model(ids.to(DEVICE), mask.to(DEVICE))
        loss   = crit(logits, lbl.to(DEVICE))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        ep_loss += loss.item()

    f1_mi, f1_ma, f1_w, hl, jac, ex, P, T = run_eval()
    elapsed = (time.time() - t0) / 60
    print(f"  Epoch {ep}/{EPOCHS} | Loss {ep_loss/len(tr_loader):.4f} | "
          f"F1-micro {f1_mi:.4f} | F1-macro {f1_ma:.4f} | "
          f"Hamming {hl:.4f} | Jaccard {jac:.4f}  [{elapsed:.1f}분]")
    if f1_mi > best_f1:
        best_f1, best_P, best_T = f1_mi, P.copy(), T.copy()

# ── 최종 결과 ────────────────────────────────────────
f1_mi, f1_ma, f1_w, hl, jac, ex, _, _ = run_eval()

print("\n" + "=" * 65)
print("  [최종 정확도 측정 결과] - NLP 장르 분류 (Validation Set)")
print("=" * 65)
print()
print("  ┌─ Multi-label 분류 지표 ─────────────────────────────────┐")
print(f"  │  F1-micro   (전체 레이블 균등)  : {best_f1:.4f}  ({best_f1*100:.2f}%)")
print(f"  │  F1-macro   (클래스별 단순평균) : {f1_ma:.4f}  ({f1_ma*100:.2f}%)")
print(f"  │  F1-weighted (샘플수 가중평균) : {f1_w:.4f}  ({f1_w*100:.2f}%)")
print(f"  │  Hamming Loss (낮을수록 좋음)  : {hl:.4f}")
print(f"  │  Jaccard Similarity (샘플평균): {jac:.4f}  ({jac*100:.2f}%)")
print(f"  │  Exact Match  (완전일치)       : {ex:.4f}  ({ex*100:.2f}%)")
print("  └───────────────────────────────────────────────────────────┘\n")

print("  [장르별 상세 분류 리포트]")
print(classification_report(best_T, best_P,
                             target_names=list(mlb.classes_), zero_division=0))

# 장르별 예측 분포 비교
print("  [장르별 예측 vs 실제 비율]")
print(f"  {'장르':<20} {'실제':>8} {'예측':>8} {'차이':>8}")
print("  " + "-"*46)
for i, g in enumerate(mlb.classes_):
    r = best_T[:, i].mean()
    p = best_P[:, i].mean()
    print(f"  {g:<20} {r:>8.3f} {p:>8.3f} {abs(r-p):>8.3f}")

elapsed = (time.time() - t0) / 60
print(f"\n  총 소요 시간: {elapsed:.1f}분")
print("\n[완료] eval_01_nlp_fast.py 실행 종료")
