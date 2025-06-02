!pip uninstall -y torch torchaudio transformers numpy
!pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.41.1 numpy==1.26.4

import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable TorchDynamo for stability

import numpy as np
import soundfile as sf
import pickle
from tqdm.auto import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

def extract_wav2vec2_embedding(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), orig_freq=sr, new_freq=16000).squeeze(0).numpy()
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def process_machine_split(machine, split, in_root, out_root):
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.exists(seg_dir):
        print(f"[SKIP] Missing directory: {seg_dir}")
        return

    save_path = os.path.join(out_root, machine, split, "wav2vec2_embeddings.pickle")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    embeddings = []
    filenames = []  # 👈 Track segment filenames

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.endswith(".wav"):
            continue
        try:
            wav, sr = sf.read(os.path.join(seg_dir, fname))
            if len(wav) < 16000:
                print(f"[SKIP] Too short: {fname}")
                continue
            embeddings.append(extract_wav2vec2_embedding(wav, sr))
            filenames.append(fname)  # 👈 Store filename
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")

    if embeddings:
        # 👇 Save both features and filenames in dictionary
        with open(save_path, "wb") as f:
            pickle.dump({
                'features': np.stack(embeddings),
                'filenames': filenames
            }, f)
        print(f"[SAVED] {save_path}: {len(embeddings)} embeddings, {len(filenames)} filenames")

if __name__ == "__main__":
    IN_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    machines = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]
    for machine in machines:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT)
