# =================== 1. Install dependencies and clone repo ===================
!pip install torchlibrosa
!pip install museval

!pip install -U torch==2.0.1 torchaudio==2.0.2
!pip install librosa==0.10.0 sox tqdm soundfile
!apt-get install -y sox ffmpeg
!git clone https://github.com/RetroCirce/HTS-Audio-Transformer.git /kaggle/working/HTS-Audio-Transformer

# =================== 2. Embedding extraction script ===================
import sys
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import pickle
from tqdm import tqdm

sys.path.append('/kaggle/working/HTS-Audio-Transformer')
from model.htsat import HTSAT_Swin_Transformer
import config as htsat_config

class HTSATExtractor:
    def __init__(self, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = HTSAT_Swin_Transformer(
            spec_size=256,
            patch_size=4,
            in_chans=1,
            num_classes=527,
            window_size=8,
            config=htsat_config,  # Use imported config as required by the repo
            depths=[2, 2, 6, 2],
            embed_dim=96,
            patch_stride=(4, 4),
            num_heads=[4, 8, 16, 32]
        )
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.sample_rate = 32000

    def _load_audio(self, path):
        wav, sr = sf.read(path)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        return torch.from_numpy(wav).float()

    def extract_embedding(self, audio_path):
        waveform = self._load_audio(audio_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output_dict = self.model(waveform, None, True)
            framewise_emb = output_dict['framewise_output'].cpu().numpy()[0]  # (T, 768)
            emb = framewise_emb.mean(axis=0)  # (768,)
        return emb

def process_machine_split(machine, split, in_root, out_root, extractor):
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing folder: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "htsat_embeddings.pickle")

    embeddings, filenames = [], []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        try:
            emb = extractor.extract_embedding(wav_path)
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    if embeddings:
        out_dict = {
            'features': np.stack(embeddings, axis=0),
            'filenames': filenames
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_dict, f)
        print(f"[SAVED] {save_path}: {out_dict['features'].shape} features, {len(filenames)} names")
    else:
        print(f"[EMPTY] No embeddings generated for {machine}/{split}")

# =================== 3. Set paths and run extraction ===================
CKPT_PATH = "/kaggle/input/hts-audio-transformer/HTSAT_AudioSet_Saved_1.ckpt"
IN_ROOT = "/kaggle/input/dcase2025/data/dcase2025t2/dev_data/processed"
OUT_ROOT = "/kaggle/working/hts_embeddings"
machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits = ["train", "test", "supplemental"]

extractor = HTSATExtractor(CKPT_PATH)

for machine in machine_types:
    for split in splits:
        print(f"\nProcessing {machine}/{split}")
        process_machine_split(machine, split, IN_ROOT, OUT_ROOT, extractor)

print("\nAll done. Embeddings saved in:", OUT_ROOT)
