import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

input_base = "/kaggle/working/processed"
checkpoint_base = "/kaggle/working/checkpoints/gmm_per_machine"
machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

os.makedirs(checkpoint_base, exist_ok=True)

for machine in machine_types:
    print(f"\n=== Processing {machine} ===")
    # STEP 1: Collect z_q and filenames from train split
    train_dir = os.path.join(input_base, machine, "train")
    zq_path = os.path.join(train_dir, "z_q.npy")
    fnames_path = os.path.join(train_dir, "filenames.pkl")
    if not (os.path.exists(zq_path) and os.path.exists(fnames_path)):
        print(f"[SKIP] {machine}/train: missing z_q.npy or filenames.pkl")
        continue
    z_q = np.load(zq_path)
    with open(fnames_path, "rb") as f:
        fnames = pickle.load(f)
    if len(fnames) != z_q.shape[0]:
        print(f"[ERROR] {machine}/train: {len(fnames)} filenames vs {z_q.shape[0]} z_q vectors")
        continue

    # STEP 2: Fit GMM for this machine
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(z_q)
    gmm_ckpt_path = os.path.join(checkpoint_base, f"gmm_{machine}.pkl")
    with open(gmm_ckpt_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"✅ GMM trained & saved for {machine} (train shape: {z_q.shape})")

    # STEP 3: Score all splits for this machine
    for split in splits:
        dir_path = os.path.join(input_base, machine, split)
        zq_path = os.path.join(dir_path, "z_q.npy")
        fnames_path = os.path.join(dir_path, "filenames.pkl")
        if not (os.path.exists(zq_path) and os.path.exists(fnames_path)):
            print(f"[SKIP] {machine}/{split}: missing z_q.npy or filenames.pkl")
            continue
        z_q_split = np.load(zq_path)
        with open(fnames_path, "rb") as f:
            filenames_split = pickle.load(f)
        if len(filenames_split) != z_q_split.shape[0]:
            print(f"[ERROR] {machine}/{split}: {len(filenames_split)} filenames vs {z_q_split.shape[0]} z_q vectors")
            continue

        # Score and normalize
        s_gmm = -gmm.score_samples(z_q_split)
        # Normalization (fit only on train split)
        if split == "train":
            mean_s = np.mean(s_gmm)
            std_s = np.std(s_gmm) if np.std(s_gmm) > 0 else 1.0
        s_gmm_norm = (s_gmm - mean_s) / std_s

        # Save both raw and normalized scores
        np.save(os.path.join(dir_path, "s_GMM.npy"), s_gmm)
        np.save(os.path.join(dir_path, "s_GMM_norm.npy"), s_gmm_norm)
        with open(os.path.join(dir_path, "s_GMM.pkl"), "wb") as f:
            pickle.dump({'scores': s_gmm, 'scores_norm': s_gmm_norm, 'filenames': filenames_split}, f)
        print(f"[SAVED] {machine}/{split}: {s_gmm.shape} scores (raw+norm)")

print("\n✅ All machines and splits processed with full traceability and normalization!")
