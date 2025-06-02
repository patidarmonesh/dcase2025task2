# ===================== INSTALL & IMPORT =====================

# 1) Uninstall any existing conflicting versions (optional but recommended)
!pip uninstall -y torch torchvision torchaudio timm laion-clap

# 2) Install matching torch/torchvision/torchaudio versions along with timm and laion-clap
#    - torch==2.7.0, torchvision==0.15.3, torchaudio==2.7.0
#    - timm==0.9.10 is required by laion-clap
#    - soundfile for WAV I/O
!pip install --quiet \
    torch==2.7.0+cpu torchvision==0.15.3+cpu torchaudio==2.7.0+cpu \
    timm==0.9.10 laion-clap soundfile --extra-index-url https://download.pytorch.org/whl/cpu

# 3) Standard imports
import os
import numpy as np
import soundfile as sf
import pickle
import torch
import torchaudio
from tqdm import tqdm

# 4) Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5) Import CLAP after torch/torchvision/timm are initialized
from laion_clap import CLAP_Module

# ===================== LOAD CLAP MODEL =====================
model = CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()  # Downloads & loads the default pretrained checkpoint

# ===================== HELPERS =====================
def resample_to_48k(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Ensure the waveform is 48 kHz mono float32.
    """
    # If stereo/multi-channel, convert to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # Resample if not already 48 kHz
    if sr != 48000:
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)
        wav_tensor = torchaudio.functional.resample(
            wav_tensor, orig_freq=sr, new_freq=48000
        )
        wav = wav_tensor.squeeze(0).cpu().numpy()
    # Clamp to float32 in [-1, +1]
    wav = wav.astype(np.float32)
    return wav

def extract_clap_embedding(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Inputs:
      - wav: 1D NumPy array
      - sr: its sample rate
    Output:
      -  Embed with shape (D,), where D is CLAP’s audio embedding dimension.
    """
    # 1) Resample to 48 kHz mono float32
    wav_48k = resample_to_48k(wav, sr)  # (N,)
    # 2) Write to a temporary file under the working directory
    tmp_path = "temp_clap_input.wav"
    sf.write(tmp_path, wav_48k, 48000, subtype="PCM_16")
    # 3) Call CLAP on that file (returns NumPy array of shape (1, D) if use_tensor=False)
    embed = model.get_audio_embedding_from_filelist(x=[tmp_path], use_tensor=False)
    # 4) Delete temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return embed[0]  # shape (D,)

# ===================== TRACEABLE CLAP EMBEDDING EXTRACTION =====================
def process_machine_split(machine: str, split: str,
                          in_root: str, out_root: str):
    """
    For each 1 s WAV under:
      in_root/<machine>/<split>/raw_segments/*.wav
    extracts a CLAP embedding and saves a pickle:
      {
        'features': np.ndarray of shape (num_clips, D),
        'filenames': list of segment filenames
      }
    to:
      out_root/<machine>/<split>/clap_embeddings.pickle
    """
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
            if len(wav) < 16000:
                print(f"[SKIP] Too short: {fname}")
                continue
            emb = extract_clap_embedding(wav, sr)  # (D,)
            embeddings.append(emb)
            filenames.append(fname)  # 👈 Store segment name for traceability
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        out_dict = {
            "features": np.stack(embeddings, axis=0),  # shape: (n_clips, D)
            "filenames": filenames                     # shape: (n_clips,)
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_dict, f)
        print(f"[SAVED] {save_path}: {out_dict['features'].shape} features, {len(filenames)} names")
    else:
        print(f"[EMPTY] No embeddings generated for {machine}/{split}")

# ===================== MAIN =====================
if __name__ == "__main__":
    # Adjust these root paths as needed:
    IN_ROOT = "/kaggle/input/dcase2025/data/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]

    for machine in machine_types:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT)
