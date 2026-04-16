"""
[정확도 측정] 프로젝트 3 - 그래프: Heterogeneous GNN (PyTorch Geometric)
"""
import warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score)

try:
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, HeteroConv
    PYGEO = True
    print(f"  PyG {torch_geometric.__version__} 로드 성공")
except ImportError:
    PYGEO = False
    print("  [경고] PyG 미설치 → 순수 PyTorch 대체 실행")

from tqdm import tqdm

# ── 설정 ────────────────────────────────────────────
DATA_PATH  = r"C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv"
MAX_MOVIES = 5000
MAX_ACTORS = 3
HIDDEN     = 64
EP_NC      = 100   # Node Classification 에폭
EP_LP      = 80    # Link Prediction 에폭
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}\n")

# ── 데이터 ──────────────────────────────────────────
def clean_genre(g):
    return [x.strip() for x in re.split(r"[,|]", g) if x.strip()]

def parse_actors(v):
    if pd.isna(v): return []
    return [x.strip() for x in str(v).split(",") if x.strip()][:MAX_ACTORS]

df = pd.read_csv(DATA_PATH).dropna(subset=["TMDB_Rating"]).copy()
df = df.sort_values("TMDB_Vote_Count", ascending=False).head(MAX_MOVIES).reset_index(drop=True)

# 타겟: Popularity_Tier
tier_le = LabelEncoder()
df["tier_label"] = tier_le.fit_transform(df["Popularity_Tier"].astype(str))
n_tiers = df["tier_label"].nunique()
tier_names = list(tier_le.classes_)

# 노드 수치 피처
num_cols = ["TMDB_Rating","Movie_Length_Minutes","TMDB_Vote_Count","TMDB_Popularity","Release_Year"]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
scaler = StandardScaler()
movie_feats = scaler.fit_transform(df[num_cols].values).astype(np.float32)

# 엔티티
df["Director"] = df["Director"].fillna("Unknown_Director")
df["Primary_Genre"] = df["Genre"].apply(lambda x: clean_genre(str(x))[0] if pd.notna(x) else "Unknown")
df["actor_list"] = df["Voice_Cast"].apply(parse_actors)

directors = sorted(df["Director"].unique())
genres    = sorted(df["Primary_Genre"].unique())
actors    = sorted({a for lst in df["actor_list"] for a in lst})

dir2id   = {d:i for i,d in enumerate(directors)}
genre2id = {g:i for i,g in enumerate(genres)}
actor2id = {a:i for i,a in enumerate(actors)}

n_movies = len(df);  n_dirs = len(dir2id)
n_genres = len(genre2id); n_actors = len(actor2id)

print(f"  영화:{n_movies}  감독:{n_dirs}  장르:{n_genres}  성우:{n_actors}  클래스:{n_tiers}")
print(f"  클래스 목록: {tier_names}\n")

# 엣지 구성
md_src,md_dst,mg_src,mg_dst,ma_src,ma_dst = [],[],[],[],[],[]
for i, row in df.iterrows():
    if row["Director"] in dir2id:
        md_src.append(i); md_dst.append(dir2id[row["Director"]])
    if row["Primary_Genre"] in genre2id:
        mg_src.append(i); mg_dst.append(genre2id[row["Primary_Genre"]])
    for a in row["actor_list"]:
        if a in actor2id:
            ma_src.append(i); ma_dst.append(actor2id[a])

ei_md = torch.tensor([md_src,md_dst], dtype=torch.long)
ei_mg = torch.tensor([mg_src,mg_dst], dtype=torch.long)
ei_ma = torch.tensor([ma_src,ma_dst], dtype=torch.long)

print(f"  엣지 — 영화-감독:{ei_md.shape[1]}  영화-장르:{ei_mg.shape[1]}  영화-성우:{ei_ma.shape[1]}\n")

# ── 마스크 (60/20/20) ───────────────────────────────
n = n_movies
idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
tr_m = torch.zeros(n, dtype=torch.bool); tr_m[idx[:int(n*0.6)]]              = True
va_m = torch.zeros(n, dtype=torch.bool); va_m[idx[int(n*0.6):int(n*0.8)]]   = True
te_m = torch.zeros(n, dtype=torch.bool); te_m[idx[int(n*0.8):]]              = True

# ════════════════════════════════════════════════════
#  ── PyG 버전 ─────────────────────────────────────
# ════════════════════════════════════════════════════
if PYGEO:
    data = HeteroData()
    data["movie"].x     = torch.FloatTensor(movie_feats)
    data["movie"].y     = torch.LongTensor(df["tier_label"].values)
    data["director"].x  = torch.zeros(n_dirs,  8)
    data["genre"].x     = torch.zeros(n_genres,8)
    data["actor"].x     = torch.zeros(n_actors,8)
    data["movie","directed_by","director"].edge_index = ei_md
    data["movie","belongs_to","genre"].edge_index     = ei_mg
    data["movie","voiced_by","actor"].edge_index      = ei_ma
    data["director","rev_directed_by","movie"].edge_index = ei_md.flip(0)
    data["genre","rev_belongs_to","movie"].edge_index     = ei_mg.flip(0)
    data["actor","rev_voiced_by","movie"].edge_index      = ei_ma.flip(0)

    data["movie"].train_mask = tr_m
    data["movie"].val_mask   = va_m
    data["movie"].test_mask  = te_m
    data = data.to(DEVICE)

    class HeteroGNN(nn.Module):
        def __init__(self, hidden, n_cls):
            super().__init__()
            self.proj = nn.ModuleDict({
                "movie":    nn.Linear(movie_feats.shape[1], hidden),
                "director": nn.Linear(8, hidden),
                "genre":    nn.Linear(8, hidden),
                "actor":    nn.Linear(8, hidden),
            })
            conv_cfg = {
                ("movie","directed_by","director"):    SAGEConv(hidden,hidden),
                ("movie","belongs_to","genre"):        SAGEConv(hidden,hidden),
                ("movie","voiced_by","actor"):         SAGEConv(hidden,hidden),
                ("director","rev_directed_by","movie"):SAGEConv(hidden,hidden),
                ("genre","rev_belongs_to","movie"):    SAGEConv(hidden,hidden),
                ("actor","rev_voiced_by","movie"):     SAGEConv(hidden,hidden),
            }
            self.conv1 = HeteroConv(conv_cfg, aggr="sum")
            self.conv2 = HeteroConv({k: SAGEConv(hidden,hidden) for k in conv_cfg}, aggr="sum")
            self.head  = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_cls))

        def encode(self, data):
            xd = {k: F.relu(self.proj[k](data[k].x)) for k in self.proj}
            xd = self.conv1(xd, data.edge_index_dict)
            xd = {k: F.relu(v) for k,v in xd.items()}
            xd = self.conv2(xd, data.edge_index_dict)
            return xd

        def forward(self, data):
            return self.head(self.encode(data)["movie"])

    gnn = HeteroGNN(HIDDEN, n_tiers).to(DEVICE)
    opt = Adam(gnn.parameters(), lr=LR, weight_decay=5e-4)
    print("=" * 62)
    print("  [학습]  Heterogeneous GraphSAGE - Node Classification")
    print("=" * 62)

    @torch.no_grad()
    def nc_metrics(mask):
        gnn.eval()
        out  = gnn(data)
        prob = F.softmax(out, dim=-1).cpu().numpy()
        pred = out.argmax(-1).cpu().numpy()
        true = data["movie"].y.cpu().numpy()
        pm, tm = pred[mask.cpu()], true[mask.cpu()]
        acc = accuracy_score(tm, pm)
        f1w = f1_score(tm, pm, average="weighted", zero_division=0)
        return acc, f1w, pm, tm

    best_va, best_pred, best_true = 0, None, None
    for ep in tqdm(range(1, EP_NC+1), desc="  GNN NC Training"):
        gnn.train(); opt.zero_grad()
        out  = gnn(data)
        loss = F.cross_entropy(out[data["movie"].train_mask],
                               data["movie"].y[data["movie"].train_mask])
        loss.backward(); opt.step()
        if ep % 20 == 0:
            va_acc, va_f1, *_ = nc_metrics(va_m)
            tqdm.write(f"  Epoch {ep:>3} | Loss {loss.item():.4f} | "
                       f"Val Acc {va_acc:.4f} | Val F1-w {va_f1:.4f}")
            if va_acc > best_va:
                best_va = va_acc
                torch.save(gnn.state_dict(), "eval_gnn_nc.pt")

    gnn.load_state_dict(torch.load("eval_gnn_nc.pt", map_location=DEVICE))
    te_acc, te_f1, te_pred, te_true = nc_metrics(te_m)

    print("\n" + "=" * 62)
    print("  [정확도] Task 1: Node Classification (Popularity Tier)")
    print("=" * 62)
    print(f"  Test Accuracy  : {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(f"  Test F1-weighted: {te_f1:.4f}  ({te_f1*100:.2f}%)")
    print(f"  Test F1-macro  : {f1_score(te_true,te_pred,average='macro',zero_division=0):.4f}")
    print()
    print("  [클래스별 분류 리포트]")
    print(classification_report(te_true, te_pred, target_names=tier_names, zero_division=0))

    print("  [혼동 행렬]")
    cm = confusion_matrix(te_true, te_pred)
    header = "       " + "".join(f"{t[:8]:>10}" for t in tier_names)
    print("  " + header)
    for i, row in enumerate(cm):
        print(f"  {tier_names[i][:8]:<7}" + "".join(f"{v:>10}" for v in row))

    # ── Task 2: Link Prediction (감독-성우 협업) ──────
    print("\n" + "=" * 62)
    print("  [학습]  Link Prediction - 감독 ↔ 성우 협업 가능성")
    print("=" * 62)

    # 감독-성우 양성 쌍 생성
    m_dir_map, m_act_map = {}, {}
    for m, d in zip(md_src, md_dst):
        m_dir_map.setdefault(m,[]).append(d)
    for m, a in zip(ma_src, ma_dst):
        m_act_map.setdefault(m,[]).append(a)
    pos_set = set()
    for m in m_dir_map:
        if m in m_act_map:
            for d in m_dir_map[m]:
                for a in m_act_map[m]:
                    pos_set.add((d,a))
    pos_pairs = list(pos_set)

    if len(pos_pairs) < 50:
        print(f"  [건너뜀] 양성 쌍 부족 ({len(pos_pairs)}개)")
    else:
        np.random.seed(42)
        neg_d = np.random.randint(0, n_dirs,   len(pos_pairs)*2)
        neg_a = np.random.randint(0, n_actors, len(pos_pairs)*2)
        neg_pairs = [(d,a) for d,a in zip(neg_d,neg_a) if (d,a) not in pos_set][:len(pos_pairs)]

        all_pairs  = pos_pairs + neg_pairs
        all_labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)
        print(f"  양성 {len(pos_pairs):,}  /  음성 {len(neg_pairs):,}  총 {len(all_pairs):,}쌍\n")

        # GNN 임베딩 추출
        gnn.eval()
        with torch.no_grad():
            xd = gnn.encode(data)
            dir_emb   = xd["director"].cpu()
            actor_emb = xd["actor"].cpu()

        pair_d = torch.LongTensor([p[0] for p in all_pairs])
        pair_a = torch.LongTensor([p[1] for p in all_pairs])
        pair_y = torch.FloatTensor(all_labels)

        idx2 = torch.randperm(len(all_pairs), generator=torch.Generator().manual_seed(42))
        sp   = int(len(idx2)*0.7); sp2 = int(len(idx2)*0.85)
        tr_i, va_i, te_i = idx2[:sp], idx2[sp:sp2], idx2[sp2:]

        class LP(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(h*2, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1))
            def forward(self, de, ae):
                return self.net(torch.cat([de,ae],dim=-1)).squeeze(-1)

        lp  = LP(HIDDEN).to(DEVICE)
        olp = Adam(lp.parameters(), lr=LR)

        best_lp_auc = 0
        for ep in tqdm(range(1, EP_LP+1), desc="  Link Pred Training"):
            lp.train(); olp.zero_grad()
            de = dir_emb[pair_d[tr_i]].to(DEVICE)
            ae = actor_emb[pair_a[tr_i]].to(DEVICE)
            logits = lp(de,ae)
            loss   = F.binary_cross_entropy_with_logits(logits, pair_y[tr_i].to(DEVICE))
            loss.backward(); olp.step()
            if ep % 20 == 0:
                lp.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(lp(dir_emb[pair_d[va_i]].to(DEVICE),
                                          actor_emb[pair_a[va_i]].to(DEVICE))).cpu().numpy()
                yv = pair_y[va_i].numpy()
                try:
                    auc_v = roc_auc_score(yv, pv)
                    tqdm.write(f"  Epoch {ep:>3} | Loss {loss.item():.4f} | Val AUC {auc_v:.4f}")
                    if auc_v > best_lp_auc:
                        best_lp_auc = auc_v
                        torch.save(lp.state_dict(),"eval_lp_best.pt")
                except: pass

        lp.load_state_dict(torch.load("eval_lp_best.pt", map_location=DEVICE))
        lp.eval()
        with torch.no_grad():
            pt = torch.sigmoid(lp(dir_emb[pair_d[te_i]].to(DEVICE),
                                   actor_emb[pair_a[te_i]].to(DEVICE))).cpu().numpy()
        yt = pair_y[te_i].numpy()
        pb = (pt > 0.5).astype(int)

        te_auc = roc_auc_score(yt, pt)
        te_acc_lp = accuracy_score(yt, pb)
        te_f1_lp  = f1_score(yt, pb, average="binary", zero_division=0)

        print("\n" + "=" * 62)
        print("  [정확도] Task 2: Link Prediction (감독-성우 협업)")
        print("=" * 62)
        print(f"  Test Accuracy : {te_acc_lp:.4f}  ({te_acc_lp*100:.2f}%)")
        print(f"  Test ROC-AUC  : {te_auc:.4f}  ({te_auc*100:.2f}%)")
        print(f"  Test F1-binary: {te_f1_lp:.4f}  ({te_f1_lp*100:.2f}%)")
        print()
        print("  [Link Prediction 분류 리포트]")
        print(classification_report(yt, pb, target_names=["No Link","Link"], zero_division=0))

        cm2 = confusion_matrix(yt, pb)
        print("  [혼동 행렬]")
        print(f"               예측 No-Link  예측 Link")
        print(f"  실제 No-Link      {cm2[0,0]:>7}    {cm2[0,1]:>7}")
        print(f"  실제 Link         {cm2[1,0]:>7}    {cm2[1,1]:>7}")

# ════════════════════════════════════════════════════
#  ── PyG 미설치 대체 버전 ──────────────────────────
# ════════════════════════════════════════════════════
else:
    print("  PyG 미설치: 이분 그래프 임베딩 (Skip-Gram 방식)")
    class BiEmb(nn.Module):
        def __init__(self, nm, ne, d=32):
            super().__init__()
            self.m = nn.Embedding(nm,d); self.e = nn.Embedding(ne,d)
            nn.init.xavier_uniform_(self.m.weight); nn.init.xavier_uniform_(self.e.weight)
        def forward(self,mi,ei): return (self.m(mi)*self.e(ei)).sum(-1)

    pos_m = ei_md[0]; pos_d = ei_md[1]
    em  = BiEmb(n_movies,n_dirs,32).to(DEVICE)
    oem = Adam(em.parameters(),lr=5e-3)
    for ep in tqdm(range(40), desc="  Embedding"):
        em.train(); oem.zero_grad()
        ps = em(pos_m.to(DEVICE), pos_d.to(DEVICE))
        nd = torch.randint(0,n_dirs,(len(pos_m),))
        ns = em(pos_m.to(DEVICE), nd.to(DEVICE))
        loss = -F.logsigmoid(ps-ns).mean()
        loss.backward(); oem.step()
    print(f"  Final Loss: {loss.item():.4f}")
    em.eval()
    with torch.no_grad():
        embs = em.m.weight.cpu().numpy()
    norm = embs/(np.linalg.norm(embs,axis=1,keepdims=True)+1e-8)
    avg_sim = (norm @ norm.T).mean()
    print(f"  평균 영화 임베딩 코사인 유사도: {avg_sim:.4f}")
    print("  (PyG 설치 후 재실행 시 GNN 기반 정확도 측정 가능)")
print()
