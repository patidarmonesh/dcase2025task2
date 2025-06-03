import os
import numpy as np
import pickle
from collections import defaultdict

# ===================================================================
# CONFIGURATION
# ===================================================================
input_base = "/kaggle/input/branch-a-materials/branchA_data/dev_data/final_scores"  # Normalized segment-level scores
output_base = "/kaggle/working/dcase2025t2/dev_data/file_level_scores"
machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

# ===================================================================
# CORE FUNCTION: MAX-POOLING WITH TRACEABILITY
# ===================================================================
def process_max_pooling(machine, split):
    current_dir = os.path.join(input_base, machine, split)
    output_dir = os.path.join(output_base, machine, split)
    os.makedirs(output_dir, exist_ok=True)

    # Load segment scores and filenames
    try:
        s_tilde_A = np.load(os.path.join(current_dir, "s_tilde_A.npy"))
        with open(os.path.join(current_dir, "filenames.pkl"), "rb") as f:
            segment_fnames = pickle.load(f)
    except FileNotFoundError:
        print(f"[SKIP] Missing files in {current_dir}")
        return

    # Verify alignment
    if len(s_tilde_A) != len(segment_fnames):
        print(f"[ERROR] Length mismatch in {current_dir}")
        return

    # Group scores by original file
    file_scores = defaultdict(list)
    for seg_name, score in zip(segment_fnames, s_tilde_A):
        # Universal filename parsing (handles both "_seg" and "_segment")
        if "_seg" in seg_name:
            original_file = seg_name.split("_seg")[0] + ".wav"
        elif "_segment" in seg_name:
            original_file = seg_name.split("_segment")[0] + ".wav"
        else:  # Fallback for non-standard names
            original_file = seg_name.rsplit("_", 1)[0] + ".wav"
        file_scores[original_file].append(score)

    # Max-pooling and save
    max_scores = {fname: np.max(scores) for fname, scores in file_scores.items()}
    np.save(os.path.join(output_dir, "branch_a_scores.npy"), max_scores)
    with open(os.path.join(output_dir, "original_filenames.pkl"), "wb") as f:
        pickle.dump(list(max_scores.keys()), f)

# ===================================================================
# MAIN PROCESSING LOOP
# ===================================================================
if __name__ == "__main__":
    print("Starting file-level max-pooling...")
    for machine in machine_types:
        for split in splits:
            process_max_pooling(machine, split)
    print("✅ All file-level scores saved. Ready for AUC calculation.")
