import os
import numpy as np
import pickle
from tqdm import tqdm

# ===================================================================
# CONFIGURATION
# ===================================================================
input_base = "/kaggle/working/processed"  # Directory where scores and filenames are stored
output_base = "/kaggle/working/dcase2025t2/dev_data/final_scores"
machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

# ===================================================================
# STEP 1: COMPUTE NORMALIZATION PARAMETERS (from normal/train data)
# ===================================================================
print("Computing normalization parameters from training data...")

train_s_AE, train_s_GMM = [], []

for machine in machine_types:
    train_dir = os.path.join(input_base, machine, "train")
    s_ae_path = os.path.join(train_dir, "s_AE.npy")
    s_gmm_path = os.path.join(train_dir, "s_GMM.npy")
    
    if os.path.exists(s_ae_path):
        train_s_AE.append(np.load(s_ae_path))
    else:
        print(f"[WARNING] Missing {s_ae_path}")
    
    if os.path.exists(s_gmm_path):
        train_s_GMM.append(np.load(s_gmm_path))
    else:
        print(f"[WARNING] Missing {s_gmm_path}")

# Check if training data exists
if not train_s_AE or not train_s_GMM:
    raise RuntimeError("No training data found! Did you run GMM scoring first?")

# Compute global min/max
global_min_AE = np.min(np.concatenate(train_s_AE))
global_max_AE = np.max(np.concatenate(train_s_AE))
global_min_GMM = np.min(np.concatenate(train_s_GMM))
global_max_GMM = np.max(np.concatenate(train_s_GMM))

print(f"Normalization parameters computed:")
print(f"AE:  min={global_min_AE:.4f}, max={global_max_AE:.4f}")
print(f"GMM: min={global_min_GMM:.4f}, max={global_max_GMM:.4f}")

# ===================================================================
# STEP 2: NORMALIZATION & AVERAGING FUNCTION
# ===================================================================
def normalize_and_average(s_AE, s_GMM):
    norm_AE = (s_AE - global_min_AE) / (global_max_AE - global_min_AE + 1e-8)
    norm_GMM = (s_GMM - global_min_GMM) / (global_max_GMM - global_min_GMM + 1e-8)
    norm_AE = np.clip(norm_AE, 0, 1)
    norm_GMM = np.clip(norm_GMM, 0, 1)
    s_tilde_A = 0.5 * (norm_AE + norm_GMM)
    return norm_AE, norm_GMM, s_tilde_A

# ===================================================================
# STEP 3: PROCESS ALL SPLITS AND MACHINES (WITH TRACEABILITY)
# ===================================================================
print("\nProcessing all splits...")

for machine in tqdm(machine_types, desc="Machines"):
    for split in splits:
        current_dir = os.path.join(input_base, machine, split)
        output_dir = os.path.join(output_base, machine, split)
        os.makedirs(output_dir, exist_ok=True)
        
        # Skip if files missing
        required_files = ["s_AE.npy", "s_GMM.npy", "filenames.pkl"]
        if not all(os.path.exists(os.path.join(current_dir, f)) for f in required_files):
            print(f"[SKIP] Missing files in {current_dir}")
            continue
            
        # Load data
        s_AE = np.load(os.path.join(current_dir, "s_AE.npy"))
        s_GMM = np.load(os.path.join(current_dir, "s_GMM.npy"))
        with open(os.path.join(current_dir, "filenames.pkl"), "rb") as f:
            filenames = pickle.load(f)
        
        # Verify alignment
        if len(s_AE) != len(filenames) or len(s_GMM) != len(filenames):
            print(f"[ERROR] Length mismatch in {current_dir}")
            continue
        
        # Normalize and average
        norm_AE, norm_GMM, s_tilde_A = normalize_and_average(s_AE, s_GMM)
        
        # Save normalized scores WITH FILENAMES
        np.save(os.path.join(output_dir, "norm_AE.npy"), norm_AE)
        np.save(os.path.join(output_dir, "norm_GMM.npy"), norm_GMM)
        np.save(os.path.join(output_dir, "s_tilde_A.npy"), s_tilde_A)
        with open(os.path.join(output_dir, "filenames.pkl"), "wb") as f:
            pickle.dump(filenames, f)

print("✅ Normalization and averaging complete! Traceability preserved.")
