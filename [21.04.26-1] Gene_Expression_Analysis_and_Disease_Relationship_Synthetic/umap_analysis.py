"""
Gene Expression UMAP Analysis using PyTorch + UMAP
- PyTorch: data normalization, Autoencoder latent representation
- UMAP: high-dimensional -> 2D visualization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
import warnings
warnings.filterwarnings('ignore')

# ── Font & reproducibility ────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42)
np.random.seed(42)

# ── 1. Load data ─────────────────────────────────────────────
DATA_PATH = r"C:\Users\USER\OneDrive\Desktop\Gene_Expression_Analysis_and_Disease_Relationship_Synthetic\Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"
SAVE_DIR  = r"C:\Users\USER\OneDrive\Desktop\Gene_Expression_Analysis_and_Disease_Relationship_Synthetic\addiction_results"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Cell_Type:      {df['Cell_Type'].value_counts().to_dict()}")
print(f"Disease_Status: {df['Disease_Status'].value_counts().to_dict()}")

GENE_COLS = ['Gene_E_Housekeeping', 'Gene_A_Oncogene', 'Gene_B_Immune',
             'Gene_C_Stromal', 'Gene_D_Therapy', 'Pathway_Score_Inflam']

X_raw = df[GENE_COLS].values.astype(np.float32)

# ── 2. PyTorch normalization ──────────────────────────────────
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

X_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
mean     = X_tensor.mean(dim=0)
std      = X_tensor.std(dim=0) + 1e-8
X_norm   = (X_tensor - mean) / std

# ── 3. Autoencoder definition ────────────────────────────────
INPUT_DIM  = len(GENE_COLS)   # 6
LATENT_DIM = 16
HIDDEN_DIM = 64

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# ── 4. Train Autoencoder ──────────────────────────────────────
model     = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

dataset    = TensorDataset(X_norm)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

EPOCHS = 100
losses = []
print("\n[Autoencoder Training]")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for (batch,) in dataloader:
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    epoch_loss /= len(X_norm)
    losses.append(epoch_loss)
    scheduler.step()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  Loss: {epoch_loss:.6f}")

# ── 5. Extract latent representations ────────────────────────
model.eval()
with torch.no_grad():
    _, Z = model(X_norm)
Z_np = Z.cpu().numpy()
print(f"\nLatent shape: {Z_np.shape}")

# ── 6. UMAP (raw features & latent space) ────────────────────
print("\n[Running UMAP]")
X_cpu      = X_norm.cpu().numpy()
emb_raw    = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(X_cpu)
emb_latent = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(Z_np)
print("UMAP done")

# ── 7. Color palettes ─────────────────────────────────────────
cell_palette    = {'T_Cell': '#4C72B0', 'Cancer': '#DD8452', 'Fibroblast': '#55A868'}
disease_palette = {'Tumor': '#C44E52', 'Healthy_Control': '#4C72B0'}

cell_colors    = [cell_palette[c]    for c in df['Cell_Type'].values]
disease_colors = [disease_palette[d] for d in df['Disease_Status'].values]

def save_umap(emb, colors, palette, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=8, alpha=0.6, linewidths=0)
    handles = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
    ax.legend(handles=handles, loc='best', fontsize=10, markerscale=2)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP-1', fontsize=11)
    ax.set_ylabel('UMAP-2', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ── 8. Save 4 individual UMAP plots ──────────────────────────
save_umap(emb_raw,    cell_colors,    cell_palette,
          'UMAP (Raw Features) — Cell Type',
          '01_umap_raw_cell_type.png')

save_umap(emb_raw,    disease_colors, disease_palette,
          'UMAP (Raw Features) — Disease Status',
          '02_umap_raw_disease_status.png')

save_umap(emb_latent, cell_colors,    cell_palette,
          'UMAP (AE Latent Space) — Cell Type',
          '03_umap_latent_cell_type.png')

save_umap(emb_latent, disease_colors, disease_palette,
          'UMAP (AE Latent Space) — Disease Status',
          '04_umap_latent_disease_status.png')

# ── 9. Autoencoder training loss curve ───────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, EPOCHS + 1), losses, color='steelblue', linewidth=2)
ax.set_title('Autoencoder Training Loss (MSE)', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss (MSE)', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(SAVE_DIR, '05_autoencoder_loss.png')
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {loss_path}")

# ── 10. Cluster statistics ────────────────────────────────────
print("\n" + "="*50)
print("Mean Gene Expression by Cell Type")
print("="*50)
print(df.groupby('Cell_Type')[GENE_COLS].mean().round(3).to_string())

print("\n" + "="*50)
print("Mean Gene Expression by Disease Status")
print("="*50)
print(df.groupby('Disease_Status')[GENE_COLS].mean().round(3).to_string())

print("\nDone! All plots saved to:", SAVE_DIR)
