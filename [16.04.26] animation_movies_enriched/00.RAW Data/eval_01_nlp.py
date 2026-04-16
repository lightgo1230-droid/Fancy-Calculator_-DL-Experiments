"""
[정확도 측정] 프로젝트 1 - NLP: DistilBERT 시놉시스 기반 장르 분류
"""
import warnings, re
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
BATCH_SIZE  = 32
EPOCHS      = 4
LR          = 2e-5
TOP_GENRES  = 10
THRESHOLD   = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}")

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

mlb    = MultiLabelBinarizer(classes=sorted(top_genres))
labels = mlb.fit_transform(df["genre_list"])
texts  = df["Overview"].tolist()

(tr_txt, va_txt,
 tr_lbl, va_lbl) = train_test_split(texts, labels, test_size=0.15, random_state=42)

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
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# ── 모델 ────────────────────────────────────────────
class GenreModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 256),
                                   nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n))
    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.head(out.last_hidden_state[:, 0, :])

n_labels = len(top_genres)
model    = GenreModel(n_labels).to(DEVICE)
opt      = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps    = len(tr_loader) * EPOCHS
sched    = get_linear_schedule_with_warmup(opt, steps//10, steps)
crit     = nn.BCEWithLogitsLoss()

# ── 학습 ────────────────────────────────────────────
print("=" * 62)
print("  [학습]  DistilBERT Multi-label 장르 분류")
print("=" * 62)

def run_val():
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ids, mask, lbl in va_loader:
            logits = model(ids.to(DEVICE), mask.to(DEVICE))
            prob   = torch.sigmoid(logits).cpu().numpy()
            preds.append((prob > THRESHOLD).astype(int))
            trues.append(lbl.numpy())
    P = np.vstack(preds); T = np.vstack(trues)
    return (f1_score(T,P,average="micro",zero_division=0),
            f1_score(T,P,average="macro",zero_division=0),
            hamming_loss(T,P),
            jaccard_score(T,P,average="samples",zero_division=0),
            P, T)

best, best_P, best_T = 0, None, None
for ep in range(1, EPOCHS+1):
    model.train()
    ep_loss = 0
    for ids, mask, lbl in tqdm(tr_loader, desc=f"  Epoch {ep}", leave=False):
        logits = model(ids.to(DEVICE), mask.to(DEVICE))
        loss   = crit(logits, lbl.to(DEVICE))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        ep_loss += loss.item()
    f1_mi, f1_ma, hl, jac, P, T = run_val()
    print(f"  Epoch {ep}/{EPOCHS} | Loss {ep_loss/len(tr_loader):.4f} | "
          f"F1-micro {f1_mi:.4f} | F1-macro {f1_ma:.4f} | "
          f"Hamming {hl:.4f} | Jaccard {jac:.4f}")
    if f1_mi > best:
        best, best_P, best_T = f1_mi, P, T

# ── 최종 결과 ────────────────────────────────────────
print("\n" + "=" * 62)
print("  [최종 정확도 측정 결과] - NLP 장르 분류 (Validation Set)")
print("=" * 62)

f1_mi  = f1_score(best_T, best_P, average="micro",   zero_division=0)
f1_ma  = f1_score(best_T, best_P, average="macro",   zero_division=0)
f1_w   = f1_score(best_T, best_P, average="weighted",zero_division=0)
hl     = hamming_loss(best_T, best_P)
jac    = jaccard_score(best_T, best_P, average="samples", zero_division=0)
# Exact Match (모든 레이블 일치)
exact  = accuracy_score(best_T, best_P)

print(f"  F1-micro  (모든 레이블 균등 평균) : {f1_mi:.4f}  ({f1_mi*100:.2f}%)")
print(f"  F1-macro  (클래스별 단순 평균)    : {f1_ma:.4f}  ({f1_ma*100:.2f}%)")
print(f"  F1-weighted (샘플 수 가중 평균)   : {f1_w:.4f}  ({f1_w*100:.2f}%)")
print(f"  Hamming Loss (낮을수록 좋음)      : {hl:.4f}")
print(f"  Jaccard Similarity (샘플 평균)   : {jac:.4f}  ({jac*100:.2f}%)")
print(f"  Exact Match (완전일치 정확도)     : {exact:.4f}  ({exact*100:.2f}%)")

print("\n  [장르별 상세 분류 리포트]")
print(classification_report(best_T, best_P,
                             target_names=list(mlb.classes_), zero_division=0))

# 각 장르 예측 분포
print("  [예측 분포 vs 실제 분포]")
print(f"  {'장르':<30} {'실제 비율':>10} {'예측 비율':>10}")
print("  " + "-"*52)
for i, g in enumerate(mlb.classes_):
    real = best_T[:, i].mean()
    pred = best_P[:, i].mean()
    print(f"  {g:<30} {real:>10.3f} {pred:>10.3f}")
print()
