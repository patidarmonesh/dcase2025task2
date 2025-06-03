import os
import numpy as np
import pickle

input_base_pca = "/kaggle/working/dcase2025t2/dev_data/pca_128"
input_base_processed = "/kaggle/working/processed"

for root, _, files in os.walk(input_base_pca):
    if "z_pca.pickle" in files:  # 👈 Correct file extension
        rel_dir = os.path.relpath(root, input_base_pca)
        processed_dir = os.path.join(input_base_processed, rel_dir)
        
        # 1. Load original PCA data (features + filenames)
        with open(os.path.join(root, "z_pca.pickle"), "rb") as f:
            pca_data = pickle.load(f)
        z_pca = pca_data['features']
        orig_fnames = pca_data['filenames']  # 👈 Original filenames
        
        # 2. Load processed data
        try:
            z_hat = np.load(os.path.join(processed_dir, "z_hat.npy"))
            with open(os.path.join(processed_dir, "filenames.pkl"), "rb") as f:
                processed_fnames = pickle.load(f)
        except FileNotFoundError:
            print(f"[SKIP] Missing files in {processed_dir}")
            continue
        
        # 3. Verify filename alignment
        if orig_fnames != processed_fnames:
            print(f"[ERROR] Filename mismatch in {processed_dir}")
            continue  # 👈 Critical check
        
        # 4. Compute & save scores with filenames
        s_AE = np.sum((z_pca - z_hat) ** 2, axis=1)
        np.save(os.path.join(processed_dir, "s_AE.npy"), s_AE)
        
        # 5. Optional: Save scores with filenames for full traceability
        with open(os.path.join(processed_dir, "s_AE.pkl"), "wb") as f:
            pickle.dump({'scores': s_AE, 'filenames': processed_fnames}, f)
        
        print(f"[DONE] {processed_dir} → s_AE.npy (shape: {s_AE.shape})")
