import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# --- VectorQuantizer definition ---
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
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices, z_q

# --- VQVAE definition ---
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

    def forward(self, x):
        z_latent = self.encoder(x)
        z_q, vq_loss, indices, z_q_vec = self.quantizer(z_latent)
        z_hat = self.decoder(z_q)
        rec_loss = F.mse_loss(z_hat, x)
        total_loss = rec_loss + vq_loss
        return z_hat, z_q, total_loss, indices

# --- Load trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE().to(device)
model.load_state_dict(torch.load("/kaggle/working/checkpoints/vqvae_model.pth", map_location=device))
model.eval()

# --- Loop over PCA dirs and process ---
input_base = "/kaggle/working/dcase2025t2/dev_data/pca_128"
output_base = "/kaggle/working/processed"

for root, _, files in os.walk(input_base):
    for file in files:
        if file == "z_pca.pickle":
            z_path = os.path.join(root, file)
            with open(z_path, "rb") as f:
                data = pickle.load(f)
            z_pca = torch.tensor(data['features'], dtype=torch.float32).to(device)
            filenames = data['filenames']

            with torch.no_grad():
                z_hat, z_q, _, indices = model(z_pca)

            # Construct output directory path (preserve structure)
            rel_dir = os.path.relpath(root, input_base)
            out_dir = os.path.join(output_base, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            # Save outputs in the working directory
            np.save(os.path.join(out_dir, "z_hat.npy"), z_hat.cpu().numpy())
            np.save(os.path.join(out_dir, "z_q.npy"), z_q.cpu().numpy())
            np.save(os.path.join(out_dir, "vq_codes.npy"), indices.cpu().numpy())
            with open(os.path.join(out_dir, "filenames.pkl"), "wb") as f:
                pickle.dump(filenames, f)
            print(f"[DONE] {z_path} → {out_dir}")

print("✅ All files processed and saved to /kaggle/working/processed/")
