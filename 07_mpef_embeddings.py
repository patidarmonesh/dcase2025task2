import os
import pickle
import numpy as np

# ─── USE THE SAME ROOT AS ALL OTHER SCRIPTS ──────────────────────────────
EMB_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

machine_types = [
    "ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"
]
splits = ["train", "test", "supplemental"]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

for machine in machine_types:
    for split in splits:
        base_dir = os.path.join(EMB_ROOT, machine, split)

        paths = {
            "panns":    os.path.join(base_dir, "panns_embeddings.pickle"),
            "wav2vec":  os.path.join(base_dir, "wav2vec2_embeddings.pickle"),
            "beats":    os.path.join(base_dir, "beats_embeddings.pickle"),
            "clap":     os.path.join(base_dir, "clap_embeddings.pickle"),
        }

        if not all(os.path.isfile(p) for p in paths.values()):
            print(f"[SKIP] Missing embeddings for {machine}/{split}")
            continue

        arrs = []
        for key, p in paths.items():
            emb = load_pickle(p)
            arrs.append(emb)

        n_clips = arrs[0].shape[0]
        if any(a.shape[0] != n_clips for a in arrs[1:]):
            print(f"[ERROR] Mismatch in clip count for {machine}/{split}")
            continue

        mpef_emb = np.concatenate(arrs, axis=1)

        out_path = os.path.join(OUT_ROOT, machine, split, "mpef_embeddings.pickle")
        save_pickle(mpef_emb, out_path)
        print(f"[SAVED] {machine}/{split} → mpef_embeddings: {mpef_emb.shape}")
