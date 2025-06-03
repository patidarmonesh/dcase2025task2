import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define root paths for each embedding type
embedding_roots = {
    "beats": "/kaggle/input/beats-embedding/dcase2025t2/dev_data/processed",
    "clap": "/kaggle/input/clap-embeddings/dcase2025t2/dev_data/processed",
    "htsat": "/kaggle/input/hts-embeddings",
    "wav2vec": "/kaggle/input/wav2vec2/dcase2025t2/dev_data/processed"
}

OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {str(e)}")
        return None

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def process_embeddings(features):
    std_devs = np.std(features, axis=0)
    non_constant_mask = std_devs > 1e-6
    features_clean = features[:, non_constant_mask]
    if features_clean.shape[1] == 0:
        return None
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features_clean)
    features_clipped = np.clip(features_norm, -5, 5)
    return features_clipped

for machine in machine_types:
    for split in splits:
        print(f"\n=== Processing {machine}/{split} ===")

        # Build embedding file paths for this machine/split
        paths = {
            "beats": os.path.join(embedding_roots["beats"], machine, split, "beats_embeddings.pickle"),
            "clap": os.path.join(embedding_roots["clap"], machine, split, "clap_embeddings.pickle"),
            "htsat": os.path.join(embedding_roots["htsat"], machine, split, "htsat_embeddings.pickle"),
            "wav2vec": os.path.join(embedding_roots["wav2vec"], machine, split, "wav2vec2_embeddings.pickle"),
        }

        # Check for missing files
        missing = [k for k, v in paths.items() if not os.path.isfile(v)]
        if missing:
            print(f"[SKIP] Missing embeddings: {', '.join(missing)}")
            continue

        processed_embeddings = []
        filenames = None
        feature_dims = {}

        for emb_name, path in paths.items():
            if not os.path.isfile(path):
                print(f"[ERROR] File missing: {path}")
                processed_embeddings = None
                break
            data = load_pickle(path)
            if data is None:
                print(f"[ERROR] Failed to process {emb_name}, skipping this split")
                processed_embeddings = None
                break

            # Handle both dict and array formats
            if isinstance(data, dict) and "features" in data:
                curr_features = data["features"]
                curr_filenames = data["filenames"]
            elif isinstance(data, np.ndarray):
                curr_features = data
                curr_filenames = [f"file_{i}" for i in range(len(curr_features))]
            else:
                print(f"[ERROR] Unexpected data format in {emb_name} for {path}")
                processed_embeddings = None
                break

            if filenames is None:
                filenames = curr_filenames
            elif filenames != curr_filenames:
                print(f"[ERROR] Filename mismatch in {emb_name} embeddings!")
                print(f"First 5 expected: {filenames[:5]}")
                print(f"First 5 actual: {curr_filenames[:5]}")
                processed_embeddings = None
                break

            cleaned_features = process_embeddings(curr_features)
            if cleaned_features is None:
                print(f"[ERROR] All features removed for {emb_name}, check data!")
                processed_embeddings = None
                break

            processed_embeddings.append(cleaned_features)
            feature_dims[emb_name] = cleaned_features.shape[1]
            print(f"{emb_name}: {cleaned_features.shape} features after cleaning")

        if processed_embeddings is None:
            continue

        merged_features = np.concatenate(processed_embeddings, axis=1)
        print(f"Total merged features: {merged_features.shape}")

        mpef_dict = {
            "features": merged_features,
            "filenames": filenames,
            "feature_dims": feature_dims
        }

        out_path = os.path.join(OUT_ROOT, machine, split, "mpef_embeddings.pickle")
        save_pickle(mpef_dict, out_path)
        print(f"[SAVED] Merged embeddings to {out_path}")

print("\n=== Merging Complete ===")
