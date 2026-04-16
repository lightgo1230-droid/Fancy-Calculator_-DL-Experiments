"""
[프로젝트 3] 그래프 데이터 분석: 인물/장르 간 관계 네트워크 (PyTorch Geometric)
=================================================================================
이분 그래프(Bipartite Graph) 구조:
  영화 노드 ↔ 감독 노드
  영화 노드 ↔ 장르 노드
  영화 노드 ↔ 성우 노드

Task 1: Node Classification (영화 노드의 Popularity_Tier 예측)
Task 2: Link Prediction    (특정 감독 ↔ 성우 협업 가능성 예측)

설치:
  pip install torch torch_geometric pandas numpy scikit-learn tqdm matplotlib
  # PyG 추가 라이브러리 (CPU 기준):
  pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

실행:
  python 03_graph_network_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

# PyG 임포트 (설치 확인 포함)
try:
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
    from torch_geometric.transforms import ToUndirected, RandomLinkSplit
    PYGEO_AVAILABLE = True
    print(f"[PyG] torch_geometric {torch_geometric.__version__} 로드 성공")
except ImportError:
    PYGEO_AVAILABLE = False
    print("[경고] torch_geometric 이 설치되지 않았습니다.")
    print("       pip install torch_geometric 후 재실행하세요.")
    print("       현재는 순수 PyTorch로 단순화된 버전을 실행합니다.\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_PATH   = r"C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MOVIES  = 5000    # 그래프 크기 제한 (메모리)
MAX_ACTORS  = 3       # 성우 파싱 최대 인원 수
HIDDEN_DIM  = 64
EPOCHS_NC   = 80      # Node Classification 에폭
EPOCHS_LP   = 60      # Link Prediction 에폭
LR          = 1e-3

print(f"[Device] {DEVICE}")


# ──────────────────────────────────────────────
# 1. 데이터 로드 & 그래프 구성
# ──────────────────────────────────────────────
def clean_genre(g: str) -> list:
    return [x.strip() for x in re.split(r"[,|]", g) if x.strip()]


def parse_actors(val) -> list:
    if pd.isna(val):
        return []
    parts = [x.strip() for x in str(val).split(",")]
    return [p for p in parts if p][:MAX_ACTORS]


def load_graph_data(path: str):
    df = pd.read_csv(path)

    # Rating이 있는 영화 중 상위 MAX_MOVIES
    df = df.dropna(subset=["TMDB_Rating"]).copy()
    df = df.sort_values("TMDB_Vote_Count", ascending=False).head(MAX_MOVIES).reset_index(drop=True)

    # 타겟: Popularity_Tier (3분류로 단순화)
    tier_map = {t: i for i, t in enumerate(df["Popularity_Tier"].unique())}
    df["tier_label"] = df["Popularity_Tier"].map(tier_map)
    df["tier_label"] = df["tier_label"].fillna(0).astype(int)

    # 노드 피처 (수치형)
    num_cols = ["TMDB_Rating", "Movie_Length_Minutes", "TMDB_Vote_Count", "TMDB_Popularity",
                "Release_Year"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    scaler = StandardScaler()
    movie_feats = scaler.fit_transform(df[num_cols].values).astype(np.float32)

    # ── 엔티티 추출 ──────────────────────────────
    # 감독
    df["Director"] = df["Director"].fillna("Unknown_Director")
    directors = sorted(df["Director"].unique())
    dir2id = {d: i for i, d in enumerate(directors)}

    # 장르 (주 장르)
    df["Primary_Genre"] = df["Genre"].apply(
        lambda x: clean_genre(str(x))[0] if pd.notna(x) else "Unknown"
    )
    genres = sorted(df["Primary_Genre"].unique())
    genre2id = {g: i for i, g in enumerate(genres)}

    # 성우
    df["actor_list"] = df["Voice_Cast"].apply(parse_actors)
    all_actors = sorted({a for lst in df["actor_list"] for a in lst})
    actor2id = {a: i for i, a in enumerate(all_actors)}

    return df, movie_feats, dir2id, genre2id, actor2id, tier_map


df, movie_feats, dir2id, genre2id, actor2id, tier_map = load_graph_data(DATA_PATH)
n_movies  = len(df)
n_dirs    = len(dir2id)
n_genres  = len(genre2id)
n_actors  = len(actor2id)
n_tiers   = len(tier_map)

print(f"[Graph] 영화:{n_movies}  감독:{n_dirs}  장르:{n_genres}  성우:{n_actors}  분류:{n_tiers}")

# ── 엣지 인덱스 구성 ────────────────────────────
def build_edges(df, dir2id, genre2id, actor2id):
    movie_dir_src, movie_dir_dst = [], []
    movie_genre_src, movie_genre_dst = [], []
    movie_actor_src, movie_actor_dst = [], []

    for i, row in df.iterrows():
        d = row["Director"]
        if d in dir2id:
            movie_dir_src.append(i); movie_dir_dst.append(dir2id[d])
        g = row["Primary_Genre"]
        if g in genre2id:
            movie_genre_src.append(i); movie_genre_dst.append(genre2id[g])
        for a in row["actor_list"]:
            if a in actor2id:
                movie_actor_src.append(i); movie_actor_dst.append(actor2id[a])

    ei_md = torch.tensor([movie_dir_src,   movie_dir_dst],   dtype=torch.long)
    ei_mg = torch.tensor([movie_genre_src, movie_genre_dst], dtype=torch.long)
    ei_ma = torch.tensor([movie_actor_src, movie_actor_dst], dtype=torch.long)
    return ei_md, ei_mg, ei_ma


ei_movie_dir, ei_movie_genre, ei_movie_actor = build_edges(df, dir2id, genre2id, actor2id)
print(f"[Edges] 영화-감독:{ei_movie_dir.shape[1]}  "
      f"영화-장르:{ei_movie_genre.shape[1]}  "
      f"영화-성우:{ei_movie_actor.shape[1]}")


# ──────────────────────────────────────────────────────────────────────────────
# ── A. PyTorch Geometric 버전 ────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
if PYGEO_AVAILABLE:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, HeteroConv
    from torch_geometric.utils import negative_sampling

    # ── HeteroData 구성 ──────────────────────────────
    data = HeteroData()

    data["movie"].x     = torch.FloatTensor(movie_feats)
    data["movie"].y     = torch.LongTensor(df["tier_label"].values)
    data["director"].x  = torch.zeros(n_dirs,   8)   # 감독은 ID 임베딩으로 표현
    data["genre"].x     = torch.zeros(n_genres, 8)
    data["actor"].x     = torch.zeros(n_actors, 8)

    data["movie", "directed_by", "director"].edge_index = ei_movie_dir
    data["movie", "belongs_to",  "genre"].edge_index    = ei_movie_genre
    data["movie", "voiced_by",   "actor"].edge_index    = ei_movie_actor

    # 역방향 엣지도 추가 (undirected)
    data["director", "rev_directed_by", "movie"].edge_index = ei_movie_dir.flip(0)
    data["genre",    "rev_belongs_to",  "movie"].edge_index = ei_movie_genre.flip(0)
    data["actor",    "rev_voiced_by",   "movie"].edge_index = ei_movie_actor.flip(0)

    # ── Task 1: Node Classification with HeteroGNN ──
    n = n_movies
    idx = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(n*0.6)]] = True
    val_mask  [idx[int(n*0.6):int(n*0.8)]] = True
    test_mask [idx[int(n*0.8):]] = True
    data["movie"].train_mask = train_mask
    data["movie"].val_mask   = val_mask
    data["movie"].test_mask  = test_mask

    class HeteroGNN(nn.Module):
        """Heterogeneous GraphSAGE"""
        def __init__(self, hidden, n_classes):
            super().__init__()
            # 입력 투영 (각 노드 타입별)
            self.proj_movie    = nn.Linear(movie_feats.shape[1], hidden)
            self.proj_director = nn.Linear(8, hidden)
            self.proj_genre    = nn.Linear(8, hidden)
            self.proj_actor    = nn.Linear(8, hidden)

            # Conv 레이어 (메시지 패싱)
            self.conv1 = HeteroConv({
                ("movie",    "directed_by",     "director"): SAGEConv(hidden, hidden),
                ("movie",    "belongs_to",       "genre"):   SAGEConv(hidden, hidden),
                ("movie",    "voiced_by",        "actor"):   SAGEConv(hidden, hidden),
                ("director", "rev_directed_by",  "movie"):   SAGEConv(hidden, hidden),
                ("genre",    "rev_belongs_to",   "movie"):   SAGEConv(hidden, hidden),
                ("actor",    "rev_voiced_by",    "movie"):   SAGEConv(hidden, hidden),
            }, aggr="sum")
            self.conv2 = HeteroConv({
                ("movie",    "directed_by",     "director"): SAGEConv(hidden, hidden),
                ("movie",    "belongs_to",       "genre"):   SAGEConv(hidden, hidden),
                ("movie",    "voiced_by",        "actor"):   SAGEConv(hidden, hidden),
                ("director", "rev_directed_by",  "movie"):   SAGEConv(hidden, hidden),
                ("genre",    "rev_belongs_to",   "movie"):   SAGEConv(hidden, hidden),
                ("actor",    "rev_voiced_by",    "movie"):   SAGEConv(hidden, hidden),
            }, aggr="sum")

            self.cls_head = nn.Linear(hidden, n_classes)

        def encode(self, data):
            x_dict = {
                "movie":    F.relu(self.proj_movie(data["movie"].x)),
                "director": F.relu(self.proj_director(data["director"].x)),
                "genre":    F.relu(self.proj_genre(data["genre"].x)),
                "actor":    F.relu(self.proj_actor(data["actor"].x)),
            }
            x_dict = self.conv1(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = self.conv2(x_dict, data.edge_index_dict)
            return x_dict

        def forward(self, data):
            x_dict = self.encode(data)
            return self.cls_head(x_dict["movie"])

    data = data.to(DEVICE)
    gnn_nc = HeteroGNN(HIDDEN_DIM, n_tiers).to(DEVICE)
    opt_nc = Adam(gnn_nc.parameters(), lr=LR, weight_decay=5e-4)

    print("\n" + "="*60)
    print("  [Task 1] Node Classification (Popularity Tier 예측)")
    print("="*60)

    def nc_train():
        gnn_nc.train()
        opt_nc.zero_grad()
        out  = gnn_nc(data)
        loss = F.cross_entropy(out[data["movie"].train_mask],
                               data["movie"].y[data["movie"].train_mask])
        loss.backward()
        opt_nc.step()
        return loss.item()

    @torch.no_grad()
    def nc_eval(mask):
        gnn_nc.eval()
        out  = gnn_nc(data)
        pred = out.argmax(dim=-1)
        y    = data["movie"].y
        acc  = (pred[mask] == y[mask]).float().mean().item()
        return acc

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS_NC + 1):
        loss = nc_train()
        if epoch % 10 == 0:
            val_acc  = nc_eval(data["movie"].val_mask)
            test_acc = nc_eval(data["movie"].test_mask)
            print(f"  Epoch {epoch:>3} | Loss: {loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(gnn_nc.state_dict(), "gnn_nc_best.pt")

    # 최종 리포트
    gnn_nc.load_state_dict(torch.load("gnn_nc_best.pt", map_location=DEVICE))
    gnn_nc.eval()
    with torch.no_grad():
        out  = gnn_nc(data)
        pred = out.argmax(dim=-1)[data["movie"].test_mask].cpu().numpy()
        true = data["movie"].y[data["movie"].test_mask].cpu().numpy()
    tier_names = list(tier_map.keys())
    print("\n[분류 리포트 - Test Set]")
    print(classification_report(true, pred,
                                 target_names=[str(t) for t in tier_names],
                                 zero_division=0))
    print(f"  최고 Val Acc: {best_val_acc:.4f}  →  gnn_nc_best.pt 저장")


    # ── Task 2: Link Prediction (감독 ↔ 성우 협업 예측) ──
    print("\n" + "="*60)
    print("  [Task 2] Link Prediction (감독 ↔ 성우 협업 가능성)")
    print("="*60)

    # 감독-성우 간접 연결 (영화를 통한 공유 협업)
    # 영화-감독 엣지와 영화-성우 엣지를 결합해 (감독, 성우) 쌍 생성
    # 양성: 실제 협업한 쌍 / 음성: 협업하지 않은 쌍 (negative sampling)

    dir_nodes  = ei_movie_dir[1]    # 각 영화의 감독 인덱스
    actor_nodes = ei_movie_actor[1]  # 각 영화의 성우 인덱스
    movie_for_dir   = ei_movie_dir[0]
    movie_for_actor = ei_movie_actor[0]

    # 영화별 감독-성우 매핑
    pos_pairs = set()
    movie_dir_map  = {}
    movie_actor_map = {}
    for m, d in zip(movie_for_dir.tolist(),   dir_nodes.tolist()):
        movie_dir_map.setdefault(m, []).append(d)
    for m, a in zip(movie_for_actor.tolist(), actor_nodes.tolist()):
        movie_actor_map.setdefault(m, []).append(a)
    for m in movie_dir_map:
        if m in movie_actor_map:
            for d in movie_dir_map[m]:
                for a in movie_actor_map[m]:
                    pos_pairs.add((d, a))
    pos_pairs = list(pos_pairs)

    if len(pos_pairs) < 50:
        print("  [경고] 양성 (감독,성우) 쌍이 너무 적습니다. Task 2를 건너뜁니다.")
    else:
        print(f"  양성 쌍: {len(pos_pairs):,}개")
        np.random.seed(42)
        neg_dirs   = np.random.randint(0, n_dirs,   len(pos_pairs))
        neg_actors = np.random.randint(0, n_actors, len(pos_pairs))
        neg_pairs  = [(d, a) for d, a in zip(neg_dirs, neg_actors)
                      if (d, a) not in pos_pairs][:len(pos_pairs)]

        all_pairs  = pos_pairs + neg_pairs
        all_labels = [1]*len(pos_pairs) + [0]*len(neg_pairs)

        # 임베딩 사용 (GNN 인코더 재활용)
        class LinkPredictor(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
            def forward(self, dir_emb, actor_emb):
                return self.fc(torch.cat([dir_emb, actor_emb], dim=-1)).squeeze(-1)

        # GNN 인코더로 임베딩 추출
        with torch.no_grad():
            gnn_nc.eval()
            x_dict = gnn_nc.encode(data)
            dir_embs   = x_dict["director"].cpu()
            actor_embs = x_dict["actor"].cpu()

        pair_d = torch.LongTensor([p[0] for p in all_pairs])
        pair_a = torch.LongTensor([p[1] for p in all_pairs])
        pair_y = torch.FloatTensor(all_labels)

        # Train/Val split
        idx = torch.randperm(len(all_pairs))
        sp  = int(len(idx) * 0.8)
        tr_idx, va_idx = idx[:sp], idx[sp:]

        lp_model = LinkPredictor(HIDDEN_DIM).to(DEVICE)
        opt_lp   = Adam(lp_model.parameters(), lr=LR)

        for epoch in range(1, EPOCHS_LP + 1):
            lp_model.train()
            opt_lp.zero_grad()
            d_emb = dir_embs[pair_d[tr_idx]].to(DEVICE)
            a_emb = actor_embs[pair_a[tr_idx]].to(DEVICE)
            y_tr  = pair_y[tr_idx].to(DEVICE)
            logits = lp_model(d_emb, a_emb)
            loss   = F.binary_cross_entropy_with_logits(logits, y_tr)
            loss.backward()
            opt_lp.step()

            if epoch % 10 == 0:
                lp_model.eval()
                with torch.no_grad():
                    d_emb_v = dir_embs[pair_d[va_idx]].to(DEVICE)
                    a_emb_v = actor_embs[pair_a[va_idx]].to(DEVICE)
                    prob_v  = torch.sigmoid(lp_model(d_emb_v, a_emb_v)).cpu().numpy()
                    y_v     = pair_y[va_idx].numpy()
                try:
                    auc_v = roc_auc_score(y_v, prob_v)
                except Exception:
                    auc_v = 0.0
                print(f"  Epoch {epoch:>3} | Loss: {loss.item():.4f} | Val AUC: {auc_v:.4f}")

        # 상위 5쌍 추천 예시
        print("\n  [협업 가능성 상위 5쌍]")
        id2dir   = {v: k for k, v in dir2id.items()}
        id2actor = {v: k for k, v in actor2id.items()}

        lp_model.eval()
        with torch.no_grad():
            # 투표수 상위 10명 감독 기준
            top_dirs_idx = list(range(min(10, n_dirs)))
            preds_all = []
            for d_i in top_dirs_idx:
                d_emb_q = dir_embs[d_i].unsqueeze(0).expand(n_actors, -1).to(DEVICE)
                a_emb_q = actor_embs.to(DEVICE)
                probs   = torch.sigmoid(lp_model(d_emb_q, a_emb_q)).cpu().numpy()
                for a_i, p in enumerate(probs):
                    preds_all.append((p, id2dir[d_i], id2actor[a_i]))

        preds_all.sort(reverse=True)
        print(f"  {'확률':>6}  {'감독':<25}  {'성우'}")
        for prob, d, a in preds_all[:5]:
            print(f"  {prob:.4f}  {d:<25}  {a}")


# ──────────────────────────────────────────────────────────────────────────────
# ── B. PyG 미설치 시: 순수 PyTorch Graph Embedding (DeepWalk-style) ──────────
# ──────────────────────────────────────────────────────────────────────────────
else:
    print("\n[대체] PyG 미설치 → 순수 PyTorch 노드 임베딩 (Skip-Gram 방식)")
    print("       영화-감독 이분 그래프를 Node2Vec 스타일로 학습합니다.\n")

    class BipartiteEmbedding(nn.Module):
        """간단한 이분 그래프 임베딩: (영화, 감독) 쌍으로 점수 학습"""
        def __init__(self, n_movies, n_entities, emb_dim=32):
            super().__init__()
            self.movie_emb  = nn.Embedding(n_movies,   emb_dim)
            self.entity_emb = nn.Embedding(n_entities, emb_dim)
            nn.init.xavier_uniform_(self.movie_emb.weight)
            nn.init.xavier_uniform_(self.entity_emb.weight)

        def forward(self, movie_ids, entity_ids):
            m = self.movie_emb(movie_ids)
            e = self.entity_emb(entity_ids)
            return (m * e).sum(dim=-1)

    # 영화-감독 임베딩 학습
    pos_m, pos_d = ei_movie_dir[0], ei_movie_dir[1]
    emb_model = BipartiteEmbedding(n_movies, n_dirs, emb_dim=32).to(DEVICE)
    opt_emb   = Adam(emb_model.parameters(), lr=5e-3)

    print("  영화-감독 임베딩 학습 중...")
    for epoch in range(1, 31):
        emb_model.train()
        # 양성 샘플
        pos_score = emb_model(pos_m.to(DEVICE), pos_d.to(DEVICE))
        # 음성 샘플
        neg_d_rand = torch.randint(0, n_dirs, (len(pos_m),))
        neg_score  = emb_model(pos_m.to(DEVICE), neg_d_rand.to(DEVICE))
        # BPR 손실
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        opt_emb.zero_grad()
        loss.backward()
        opt_emb.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>2}/30 | Loss: {loss.item():.4f}")

    # 영화 임베딩으로 유사 영화 추천
    print("\n  [유사 영화 추천 (임베딩 코사인 유사도)]")
    emb_model.eval()
    with torch.no_grad():
        all_movie_emb = emb_model.movie_emb.weight.cpu().numpy()  # (N, 32)

    norm = all_movie_emb / (np.linalg.norm(all_movie_emb, axis=1, keepdims=True) + 1e-8)

    for query_idx in [0, 10, 100]:
        if query_idx >= n_movies:
            continue
        sims = norm @ norm[query_idx]
        top5 = np.argsort(sims)[::-1][1:6]
        q_name = df.iloc[query_idx]["Movie_Name"]
        print(f"\n  '{q_name}' 와 유사한 영화:")
        for i in top5:
            print(f"    {df.iloc[i]['Movie_Name']:<40} (sim={sims[i]:.4f})")


# ──────────────────────────────────────────────
# 3. 네트워크 시각화 (감독-영화-장르 소규모)
# ──────────────────────────────────────────────
print("\n[시각화] 감독-영화-장르 관계 그래프 (상위 30개 영화) 저장 중...")

G = nx.Graph()
top30 = df.head(30)

for _, row in top30.iterrows():
    m_node  = f"M:{row['Movie_Name'][:20]}"
    d_node  = f"D:{row['Director']}"
    g_node  = f"G:{row['Primary_Genre']}"
    G.add_node(m_node,  node_type="movie")
    G.add_node(d_node,  node_type="director")
    G.add_node(g_node,  node_type="genre")
    G.add_edge(m_node, d_node)
    G.add_edge(m_node, g_node)

color_map = {"movie": "#4C72B0", "director": "#DD8452", "genre": "#55A868"}
colors = [color_map.get(G.nodes[n].get("node_type", "movie"), "#4C72B0") for n in G.nodes]

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=2)
nx.draw(G, pos, node_color=colors, with_labels=True,
        font_size=6, node_size=300, edge_color="gray", alpha=0.8)
legend_elements = [
    plt.scatter([], [], c="#4C72B0", s=100, label="Movie"),
    plt.scatter([], [], c="#DD8452", s=100, label="Director"),
    plt.scatter([], [], c="#55A868", s=100, label="Genre"),
]
plt.legend(handles=legend_elements, loc="upper left")
plt.title("Animation Movie – Director – Genre Network (Top 30)")
plt.tight_layout()
plt.savefig("graph_network_viz.png", dpi=120)
print("  그래프 시각화 → graph_network_viz.png 저장")

print("\n[완료] 03_graph_network_analysis.py 실행 종료")
