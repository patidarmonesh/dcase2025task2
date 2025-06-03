import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

# Use processed directory for GMM training and scoring
input_base = "/kaggle/working/processed"
checkpoint_path = "/kaggle/working/checkpoints/branch_a_gmm.pkl"
machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

# STEP 1: Collect z_q and filenames from train splits for all machines
zq_train_all = []
fnames_train_all = []
for machine in machine_types:
    train_dir = os.path.join(input_base, machine, "train")
    zq_path = os.path.join(train_dir, "z_q.npy")
    fnames_path = os.path.join(train_dir, "filenames.pkl")
    if os.path.exists(zq_path) and os.path.exists(fnames_path):
        z_q = np.load(zq_path)
        with open(fnames_path, "rb") as f:
            fnames = pickle.load(f)
        zq_train_all.append(z_q)
        fnames_train_all.extend(fnames)
        print(f"[LOAD] {machine}/train: {z_q.shape}")
    else:
        print(f"[SKIP] {machine}/train: missing z_q.npy or filenames.pkl")
if len(zq_train_all) == 0:
    raise RuntimeError("No train z_q.npy files found! Please run VQ-VAE inference first.")
zq_all = np.concatenate(zq_train_all, axis=0)  # shape: (N_train, 32)

# STEP 2: Fit GMM
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
gmm.fit(zq_all)
print("✅ GMM trained on z_q, shape:", zq_all.shape)

# Save GMM
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
with open(checkpoint_path, "wb") as f:
    pickle.dump(gmm, f)

# STEP 3: Apply to all splits and save s_GMM with filenames
for machine in machine_types:
    for split in splits:
        dir_path = os.path.join(input_base, machine, split)
        zq_path = os.path.join(dir_path, "z_q.npy")
        fnames_path = os.path.join(dir_path, "filenames.pkl")
        if os.path.exists(zq_path) and os.path.exists(fnames_path):
            z_q = np.load(zq_path)
            with open(fnames_path, "rb") as f:
                filenames = pickle.load(f)
            if len(filenames) != z_q.shape[0]:
                print(f"[ERROR] Mismatch in {dir_path}: {len(filenames)} filenames vs {z_q.shape[0]} z_qs")
                continue
            s_gmm = -gmm.score_samples(z_q)  # shape: (n_clips,)
            np.save(os.path.join(dir_path, "s_GMM.npy"), s_gmm)
            with open(os.path.join(dir_path, "s_GMM.pkl"), "wb") as f:
                pickle.dump({'scores': s_gmm, 'filenames': filenames}, f)
            print(f"[SAVED] {dir_path} → s_GMM.npy & s_GMM.pkl shape: {s_gmm.shape}")
        else:
            print(f"[SKIP] {dir_path}: missing z_q.npy or filenames.pkl")

print("✅ All splits processed. GMM scores and filenames saved traceably.")
