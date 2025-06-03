# ===================== INSTALLATION & SETUP =====================

# 1. Remove preinstalled torch, torchvision, torchaudio to avoid version conflicts
!pip uninstall -y torch torchvision torchaudio

# 2. Install compatible torch/torchaudio/torchvision (CUDA 11.8 wheels)
!pip install --quiet torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Install CLAP and audio dependencies
!pip install --quiet timm transformers librosa soundfile h5py
!pip install --quiet torchlibrosa

# 4. Clone the official CLAP repo into /kaggle/working
!rm -rf /kaggle/working/CLAP
!git clone https://github.com/LAION-AI/CLAP.git /kaggle/working/CLAP

# 5. Install CLAP in editable mode
%cd /kaggle/working/CLAP
!pip install -e .
%cd -


# ===================== IMPORTS & DEVICE SETUP =====================
import os
import numpy as np
import soundfile as sf
import pickle
import torch
import torchaudio
from tqdm import tqdm

# PyTorch 2.6+ safe globals fix for CLAP checkpoints
from numpy.core.multiarray import scalar, _reconstruct
from numpy import dtype
from numpy.dtypes import Float64DType, Float32DType
from torch.serialization import add_safe_globals
add_safe_globals([scalar, dtype, _reconstruct, Float64DType, Float32DType])

import laion_clap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== MODEL LOADING =====================
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()  # Should now work without UnpicklingError

# ===================== HELPER FUNCTIONS =====================
def resample_to_48k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Convert waveform to 48 kHz mono float32"""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 48000:
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 48000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    return wav.astype(np.float32)

def extract_clap_embedding(wav: np.ndarray, sr: int) -> np.ndarray:
    """Process audio and extract CLAP embedding"""
    wav_48k = resample_to_48k(wav, sr)
    tmp_path = "/kaggle/working/_clap_temp.wav"
    sf.write(tmp_path, wav_48k, 48000, subtype="PCM_16")
    embed = model.get_audio_embedding_from_filelist([tmp_path], use_tensor=False)
    try: os.remove(tmp_path)
    except: pass
    return embed[0]

def process_machine_split(machine: str, split: str, in_root: str, out_root: str):
    """Process all audio files for a machine/split and save with traceability"""
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing directory: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "clap_embeddings.pickle")

    embeddings = []
    filenames = []
    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(seg_dir, fname)
        try:
            wav, sr = sf.read(wav_path)
            if len(wav) < sr:
                print(f"[SKIP] Too short: {fname}")
                continue
            embeddings.append(extract_clap_embedding(wav, sr))
            filenames.append(fname)  # <-- Store the filename for traceability!
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        arr = np.stack(embeddings)
        out_dict = {
            "features": arr,
            "filenames": filenames
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_dict, f)
        print(f"[SAVED] {save_path}: {arr.shape}, {len(filenames)} filenames")
    else:
        print(f"[EMPTY] No embeddings for {machine}/{split}")

# ===================== EXECUTION =====================
IN_ROOT = "/kaggle/input/dcase2025/data/dcase2025t2/dev_data/processed"
OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

for machine in machine_types:
    for split in splits:
        process_machine_split(machine, split, IN_ROOT, OUT_ROOT)

print("=== CLAP embedding extraction complete with traceability ===")
