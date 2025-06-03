import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans

# ===================== CUSTOM DATASET =====================
class MetaDataset(Dataset):
    def __init__(self, features, filenames):
        self.features = features          # torch.Tensor (N, 128)
        self.filenames = filenames        # list of length N

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.filenames[idx]

# ===================== VQ-VAE DEFINITION =====================
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=128, code_dim=32, beta=1.0):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        self.beta = beta

    def forward(self, z):
        dist = torch.cdist(z, self.codebook)
        indices = torch.argmin(dist, dim=1)
        z_q = self.codebook[indices]
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.quantizer = VectorQuantizer(num_codes=128, code_dim=32, beta=1.0)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 128)
        )

    def init_codebook(self, data_sample):
        with torch.no_grad():
            z_latent = self.encoder(data_sample)
        kmeans = MiniBatchKMeans(n_clusters=128, batch_size=256, n_init=10)
        kmeans.fit(z_latent.cpu().numpy())
        with torch.no_grad():
            self.quantizer.codebook.copy_(
                torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            )

    def forward(self, x):
        z = self.encoder(x)
        if self.training:
            z = z + 0.05 * torch.randn_like(z)
        z_q, vq_loss, indices = self.quantizer(z)
        x_hat = self.decoder(z_q)
        rec_loss = F.mse_loss(x_hat, x)
        # Code usage regularization
        unique_codes = torch.unique(indices)
        usage_ratio = len(unique_codes) / self.quantizer.codebook.shape[0]
        usage_reg = 0.1 * (1.0 - usage_ratio)
        total_loss = rec_loss + vq_loss + usage_reg
        return x_hat, total_loss, indices

# ===================== LOAD PCA FEATURES WITH FILENAMES =====================
def load_pca_data(base_dir):
    feats_list, fnames_list = [], []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == 'z_pca.pickle':
                data = pickle.load(open(os.path.join(root, file), 'rb'))
                feats_list.append(data['features'])       # numpy (n_i,128)
                fnames_list.extend(data['filenames'])     # list length n_i
    all_feats = np.concatenate(feats_list, axis=0)
    return torch.tensor(all_feats, dtype=torch.float32), fnames_list

# Paths
PCA_BASE = '/kaggle/working/dcase2025t2/dev_data/pca_128'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
posix = os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Load
z_pca_tensor, filenames = load_pca_data(PCA_BASE)

# 2. Dataset & DataLoader
dataset = MetaDataset(z_pca_tensor, filenames)
loader  = DataLoader(dataset, batch_size=512, shuffle=True)

# 3. Model & Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VQVAE().to(device)
# Initialize codebook on first 10000 samples
model.init_codebook(z_pca_tensor[:10000].to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. Training Loop
mapping = []  # to store (filename, code_idx) across all epochs
for epoch in range(200):
    model.train()
    total_loss = 0.0
    codes_used = set()
    for feats, fnames in loader:
        feats = feats.to(device)
        optimizer.zero_grad()
        _, loss, indices = model(feats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        for nm, idx in zip(fnames, indices.cpu().tolist()):
            mapping.append((nm, idx))
            codes_used.add(idx)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.4f} | Codes Used: {len(codes_used)}/128")

# 5. Save model, codebook, and mapping
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# Model
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'vqvae_model.pth'))
# Codebook
with open(os.path.join(CHECKPOINT_DIR, 'codebook.npy'), 'wb') as f:
    np.save(f, model.quantizer.codebook.detach().cpu().numpy())
# Mapping
with open(os.path.join(CHECKPOINT_DIR, 'usage_mapping.pkl'), 'wb') as f:
    pickle.dump(mapping, f)

print("✅ Training complete. Model, codebook, and mapping saved.")
