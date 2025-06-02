# -------------------- SETUP: install dependencies --------------------
!pip install --upgrade torch torchaudio
!pip install panns-inference

# -------------------- BEGIN EXTRACTION SCRIPT ------------------------
import os
import numpy as np
import soundfile as sf
import pickle
from tqdm import tqdm
import torch
from panns_inference import AudioTagging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
at = AudioTagging(checkpoint_path=None, device=device)  # Uses default Cnn14 checkpoint (expects 32kHz audio)

def extract_panns_embedding(wav: np.ndarray, sr: int = 32000) -> np.ndarray:
    # Resample if needed (handles 1s/16kHz -> 1s/32kHz)
    if sr != 32000:
        import torchaudio
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 32000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    # Model expects (batch, time)
    audio = wav[None, :]
    _, embedding = at.inference(audio)
    return embedding.squeeze(0)  # (2048,)

def process_machine_split(machine, split, in_root, out_root):
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing folder: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "panns_embeddings.pickle")

    embeddings = []
    filenames = []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        try:
            wav, sr = sf.read(wav_path)
            emb = extract_panns_embedding(wav, sr)  # shape: (2048,)
            embeddings.append(emb)
            filenames.append(fname)  # 👈 store file identity
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    if embeddings:
        out_dict = {
            'features': np.stack(embeddings, axis=0),  # shape: (N, 2048)
            'filenames': filenames                     # shape: (N,)
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_dict, f)
        print(f"[SAVED] {save_path}: {out_dict['features'].shape} features, {len(filenames)} names")
    else:
        print(f"[EMPTY] No embeddings generated for {machine}/{split}")

if __name__ == "__main__":
    IN_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

    machine_types = [
        "ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"
    ]
    splits = ["train", "test", "supplemental"]

    for machine in machine_types:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT)
