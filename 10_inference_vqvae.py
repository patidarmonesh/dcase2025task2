import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# ===================== MODEL DEFINITION (Matches Training) =====================
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
        z = self.encoder(x)
        z_q, vq_loss, indices, z_q_vec = self.quantizer(z)
        z_hat = self.decoder(z_q)
        rec_loss = F.mse_loss(z_hat, x)
        return z_hat, z_q, rec_loss + vq_loss, indices

# ===================== INFERENCE PIPELINE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference():
    # Load trained model
    model = VQVAE().to(device)
    model.load_state_dict(torch.load("/kaggle/working/checkpoints/vqvae_model.pth", map_location=device))
    model.eval()

    # Process all PCA files
    input_base = "/kaggle/working/data/dcase2025t2/dev_data/pca_128"
    output_base = "/kaggle/working/processed"
    
    for root, _, files in os.walk(input_base):
        for file in files:
            if file == "z_pca.pickle":
                # Load PCA features and filenames
                z_path = os.path.join(root, file)
                with open(z_path, "rb") as f:
                    data = pickle.load(f)
                z_pca = torch.tensor(data['features'], dtype=torch.float32).to(device)
                filenames = data['filenames']

                # Inference
                with torch.no_grad():
                    z_hat, z_q, _, indices = model(z_pca)

                # Create output directory (preserve input structure)
                rel_path = os.path.relpath(root, input_base)
                out_dir = os.path.join(output_base, rel_path)
                os.makedirs(out_dir, exist_ok=True)

                # Save with traceability
                np.save(os.path.join(out_dir, "z_hat.npy"), z_hat.cpu().numpy())
                np.save(os.path.join(out_dir, "z_q.npy"), z_q.cpu().numpy())
                np.save(os.path.join(out_dir, "vq_codes.npy"), indices.cpu().numpy())
                with open(os.path.join(out_dir, "filenames.pkl"), "wb") as f:
                    pickle.dump(filenames, f)
                print(f"Processed {len(filenames)} segments → {out_dir}")

if __name__ == "__main__":
    run_inference()
    print("✅ Inference complete. All outputs saved with full traceability.")
