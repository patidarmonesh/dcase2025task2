import os
import pickle
import numpy as np

EMB_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
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

        features_list = []
        filenames = None

        for name, path in paths.items():
            data = load_pickle(path)

            # Expect: dict with 'features' and 'filenames'
            feats = data["features"]
            names = data["filenames"]

            if filenames is None:
                filenames = names
            elif filenames != names:
                print(f"[ERROR] Filename mismatch in {machine}/{split} → {name}")
                break

            features_list.append(feats)

        else:
            # All matched, safe to concatenate
            merged_features = np.concatenate(features_list, axis=1)
            mpef_dict = {
                "features": merged_features,
                "filenames": filenames
            }

            out_path = os.path.join(OUT_ROOT, machine, split, "mpef_embeddings.pickle")
            save_pickle(mpef_dict, out_path)
            print(f"[SAVED] {machine}/{split} → mpef_embeddings: {merged_features.shape}")
