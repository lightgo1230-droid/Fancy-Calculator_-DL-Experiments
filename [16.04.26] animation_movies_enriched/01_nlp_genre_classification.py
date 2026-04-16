"""
[프로젝트 1] NLP 접근: 시놉시스 기반 장르 분류 & 영화 추천
=============================================================
모델: DistilBERT (HuggingFace Transformers + PyTorch)
입력: Overview (시놉시스 텍스트)
출력:
  - Task 1: Multi-label Genre Classification (장르 예측)
  - Task 2: Sentence Similarity 기반 영화 추천

설치:
  pip install torch transformers scikit-learn pandas numpy tqdm

실행:
  python 01_nlp_genre_classification.py
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_PATH = r"C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_N_GENRES = 10          # 상위 N개 장르만 사용
MIN_VOTES_RECOMMEND = 10   # 추천 시 최소 투표수

print(f"[Device] {DEVICE}")


# ──────────────────────────────────────────────
# 1. 데이터 로드 & 전처리
# ──────────────────────────────────────────────
def clean_genre(g: str) -> list[str]:
    """'Animation, Comedy | Family' 같은 혼합 구분자를 정리"""
    return [x.strip() for x in re.split(r"[,|]", g) if x.strip()]


def load_data(path: str):
    df = pd.read_csv(path)
    # Overview와 Genre가 모두 있는 행만 사용
    df = df.dropna(subset=["Overview", "Genre"]).copy()
    df["genre_list"] = df["Genre"].apply(clean_genre)

    # 상위 N개 장르 선정
    from collections import Counter
    all_genres = [g for genres in df["genre_list"] for g in genres]
    top_genres = {g for g, _ in Counter(all_genres).most_common(TOP_N_GENRES)}
    df["genre_list"] = df["genre_list"].apply(
        lambda gs: [g for g in gs if g in top_genres]
    )
    # 최소 1개 장르가 남은 행만 사용
    df = df[df["genre_list"].map(len) > 0].reset_index(drop=True)
    return df, sorted(top_genres)


df, top_genres = load_data(DATA_PATH)
print(f"[Data] 학습 가능 샘플: {len(df):,}개  /  사용 장르: {top_genres}")

# Multi-label 이진화
mlb = MultiLabelBinarizer(classes=top_genres)
labels = mlb.fit_transform(df["genre_list"])          # (N, num_genres)

texts = df["Overview"].tolist()
train_texts, val_texts, train_labels, val_labels, train_idx, val_idx = \
    train_test_split(texts, labels, df.index.tolist(),
                     test_size=0.15, random_state=42)

print(f"[Split] Train {len(train_texts):,}  /  Val {len(val_texts):,}")


# ──────────────────────────────────────────────
# 2. Dataset & DataLoader
# ──────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


class MovieDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts = texts
        self.labels = torch.FloatTensor(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


train_ds = MovieDataset(train_texts, train_labels)
val_ds   = MovieDataset(val_texts,   val_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ──────────────────────────────────────────────
# 3. 모델 정의 (DistilBERT + Classification Head)
# ──────────────────────────────────────────────
class GenreClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        hidden = self.bert.config.hidden_size        # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] 토큰 벡터 사용
        cls_emb = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)


num_labels = len(top_genres)
model = GenreClassifier(num_labels).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)
criterion = nn.BCEWithLogitsLoss()


# ──────────────────────────────────────────────
# 4. 학습 루프
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        input_ids  = batch["input_ids"].to(DEVICE)
        attn_mask  = batch["attention_mask"].to(DEVICE)
        targets    = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)
        loss   = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            targets   = batch["labels"].to(DEVICE)

            logits = model(input_ids, attn_mask)
            loss   = criterion(logits, targets)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    f1_micro = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1_micro, f1_macro


print("\n" + "="*60)
print("  [Task 1] Multi-label 장르 분류 Fine-tuning")
print("="*60)
best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_loss, f1_micro, f1_macro = evaluate(model, val_loader, criterion)
    print(f"  Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f}")
    if f1_micro > best_f1:
        best_f1 = f1_micro
        torch.save(model.state_dict(), "genre_classifier_best.pt")

print(f"\n  최고 Val F1-micro: {best_f1:.4f}  →  genre_classifier_best.pt 저장")

# 상세 분류 리포트 (마지막 에폭 기준)
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(batch["labels"].numpy())
all_preds   = np.vstack(all_preds)
all_targets = np.vstack(all_targets)
print("\n[분류 리포트]")
print(classification_report(all_targets, all_preds,
                             target_names=mlb.classes_, zero_division=0))


# ──────────────────────────────────────────────
# 5. Task 2: Sentence Similarity 기반 영화 추천
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  [Task 2] Sentence Similarity 영화 추천")
print("="*60)

# 투표수가 충분한 영화만 추천 풀에 포함
rec_df = df[df.index.isin(val_idx)].copy()

def get_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """[CLS] 임베딩을 배치 단위로 추출"""
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model.bert(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            )
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
    return np.vstack(embeddings)


def recommend(query_text: str, df_pool: pd.DataFrame, emb_pool: np.ndarray,
              top_k: int = 5) -> pd.DataFrame:
    """쿼리 텍스트와 가장 유사한 top_k 영화 반환"""
    query_emb = get_embeddings([query_text])                # (1, 768)
    # 코사인 유사도
    pool_norm  = emb_pool  / (np.linalg.norm(emb_pool,  axis=1, keepdims=True) + 1e-8)
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    sims = (pool_norm @ query_norm.T).squeeze()             # (N,)
    top_idx = np.argsort(sims)[::-1][:top_k]
    result = df_pool.iloc[top_idx][["Movie_Name", "Genre", "TMDB_Rating", "Overview"]].copy()
    result["Similarity"] = sims[top_idx].round(4)
    return result


print("  Validation 영화 임베딩 추출 중...")
val_overview_list = df.loc[val_idx, "Overview"].tolist()
val_embeddings    = get_embeddings(val_overview_list)
rec_df = rec_df.reset_index(drop=True)

# 예시 추천 쿼리
queries = [
    "A young lion cub must reclaim his kingdom from a treacherous uncle.",
    "Toys come to life when humans are not around and go on adventures.",
    "A scientist accidentally creates chaos with a machine that makes food fall from the sky.",
]

for q in queries:
    print(f"\n  [쿼리] {q[:70]}...")
    result = recommend(q, rec_df, val_embeddings, top_k=5)
    print(result[["Movie_Name", "Genre", "TMDB_Rating", "Similarity"]].to_string(index=False))

print("\n[완료] 01_nlp_genre_classification.py 실행 종료")
