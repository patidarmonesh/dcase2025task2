# ==================== SETUP BEATs EMBEDDING EXTRACTION ====================

# 1. Clone the UNILM repository (only if missing)
import os, sys

if not os.path.isdir("/kaggle/working/unilm"):
    !git clone https://github.com/microsoft/unilm.git /kaggle/working/unilm
else:
    print("⏩ /kaggle/working/unilm already exists – skipping clone.")

# 2. Add the 'beats' subfolder to Python’s import path so we can import BEATs classes
sys.path.append("/kaggle/working/unilm/beats")

# 3. Create the checkpoints directory and copy your pretrained checkpoint there.
!mkdir -p /kaggle/working/unilm/beats/checkpoints
!cp /kaggle/input/checkpoint/BEATs_iter3_plus_AS2M.pt \
      /kaggle/working/unilm/beats/checkpoints/BEATs_iter3_plus_AS2M.pt

# 4. (Optional) Upgrade torch/torchaudio if necessary
!pip install --upgrade torch torchaudio

# ==================== BEGIN BEATs EMBEDDING EXTRACTION SCRIPT ====================

import os
import numpy as np
import soundfile as sf
import pickle
from tqdm import tqdm

import torch
import torchaudio

# Import BEATsConfig and BEATs from BEATs.py (located under unilm/beats/)
from BEATs import BEATs, BEATsConfig

# 1) Load BEATs model from checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "/kaggle/working/unilm/beats/checkpoints/BEATs_iter3_plus_AS2M.pt"

ckpt = torch.load(checkpoint_path, map_location=device)
cfg = BEATsConfig(ckpt["cfg"])
model = BEATs(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# 2) Define a helper to extract a 768-D BEATs embedding from a raw waveform
def extract_beats_embedding(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Input:
      wav: 1D NumPy array of length 16000 (1 second at 16 kHz)
      sr: sampling rate of wav
    Output:
      768-D NumPy array from BEATs model
    """
    # Resample to 16 kHz if needed
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)       # shape: (1, L)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
        wav = wav_tensor.squeeze(0).cpu().numpy()

    # Convert to float32 and send to device
    wav_tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)  # shape: (1, 16000)

    with torch.no_grad():
        # BEATs.extract_features returns a tuple; first element is hidden states (1, T', 768)
        hidden_states, _ = model.extract_features(wav_tensor, None)
        # Mean-pool over time dimension -> shape (1, 768)
        emb = hidden_states.mean(dim=1).cpu().numpy().squeeze(0)  # shape: (768,)
    return emb

# 3) Batch-process each <machine>/<split>/raw_segments/*.wav and save embeddings WITH FILENAMES
def process_machine_split(machine: str, split: str, in_root: str, out_root: str):
    """
    Reads all 1-second WAVs under in_root/<machine>/<split>/raw_segments/,
    extracts a 768-D BEATs embedding for each, and saves the stack as:
      out_root/<machine>/<split>/beats_embeddings.pickle
    Now also stores corresponding filenames for traceability.
    """
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing folder: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "beats_embeddings.pickle")

    embeddings = []
    filenames = []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        try:
            wav, sr = sf.read(wav_path)
            # Skip any clip shorter than 1 second (optional)
            if len(wav) < 16000:
                print(f"[SKIP] Too short: {fname}")
                continue
            emb = extract_beats_embedding(wav, sr)  # shape: (768,)
            embeddings.append(emb)
            filenames.append(fname)  # <-- this is the key fix
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        arr = np.stack(embeddings, axis=0)
        out_data = {
            'features': arr,
            'filenames': filenames  # <-- now traceability is preserved
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_data, f)
        print(f"[SAVED] {save_path}: {arr.shape} with {len(filenames)} filenames")
    else:
        print(f"[EMPTY] No embeddings generated for {machine}/{split}")

# 4) Main loop: iterate over all machines and splits
if __name__ == "__main__":
    IN_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]

    for machine in machine_types:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT)
