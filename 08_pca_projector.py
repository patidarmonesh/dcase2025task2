import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

# 1. Set paths
base_dir = "/kaggle/working/data/dcase2025t2/dev_data/processed"       # Directory containing machine/split/mpef_embeddings.pickle
pca_params_path = "/kaggle/working/checkpoints/pca_params.pkl"    # Where to save PCA mean & components
output_base = "/kaggle/working/data/dcase2025t2/dev_data/pca_128"      # Where to save 128-D projected vectors

# 2. Collect all 2575-D embeddings and filenames from mpef_embeddings.pickle files
embeddings_list = []
filenames_list = []
file_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "mpef_embeddings.pickle":
            path = os.path.join(root, file)
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Expect data to be a dict with 'features' and 'filenames'
            embeddings = data['features']      # shape (n_i, 2575)
            filenames = data['filenames']      # list of length n_i
            embeddings_list.append(embeddings)
            filenames_list.append(filenames)
            file_paths.append(path)

# 3. Concatenate all embeddings for PCA fitting
all_embeddings = np.concatenate(embeddings_list, axis=0)  # shape (N_total, 2575)

# 4. Compute global mean and center data
mean_vector = np.mean(all_embeddings, axis=0)      # shape (2575,)
centered = all_embeddings - mean_vector            # shape (N_total, 2575)

# 5. Fit PCA (2575 → 128)
pca = PCA(n_components=128, svd_solver="randomized", whiten=False)
pca.fit(centered)
components = pca.components_  # shape (128, 2575)

# 6. Save PCA parameters
os.makedirs(os.path.dirname(pca_params_path), exist_ok=True)
with open(pca_params_path, "wb") as f:
    pickle.dump({"mean": mean_vector, "components": components}, f)

# 7. Project each file's embeddings and save 128-D vectors + filenames
for path, embeddings, filenames in zip(file_paths, embeddings_list, filenames_list):
    emb_array = np.array(embeddings)              # shape (n_i, 2575)
    # --- Shape check for safety ---
    if emb_array.shape[1] != components.shape[1]:
        raise ValueError(f"Embedding dim {emb_array.shape[1]} does not match PCA dim {components.shape[1]} for {path}")
    emb_centered = emb_array - mean_vector         # shape (n_i, 2575)
    z_pca = np.dot(emb_centered, components.T)     # shape (n_i, 128)

    # Determine relative output path under output_base, preserve machine/split folder structure
    rel_path = os.path.relpath(path, base_dir)
    save_dir = os.path.join(output_base, os.path.dirname(rel_path))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "z_pca.pickle")
    # Save both z_pca and filenames for traceability
    with open(save_path, "wb") as f:
        pickle.dump({'features': z_pca, 'filenames': filenames}, f)

print("PCA projection complete. Parameters and 128-D vectors with filenames saved.")
