import os
import pickle
import numpy as np
import torch
import torchaudio

def compute_log_mel_gpu(
    seg_root: str,
    out_root: str,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: str = 'power',  # torchaudio uses 'power' or 'magnitude'
    device: str = 'cuda'
):
    """
    GPU-accelerated log-Mel spectrogram computation.
    Walks seg_root/<machine>/<split>/raw_segments/*.wav,
    computes via torchaudio.transforms on GPU, then pickles to
    out_root/<machine>/<split>/mels_standard.pickle.
    Stores both features and corresponding filenames for traceability.
    """
    # configure transforms on GPU
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    ).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB(
        stype=power,
        top_db=80.0
    ).to(device)

    for machine in os.listdir(seg_root):
        mdir = os.path.join(seg_root, machine)
        if not os.path.isdir(mdir):
            continue
        for split in os.listdir(mdir):
            seg_dir = os.path.join(mdir, split, 'raw_segments')
            if not os.path.isdir(seg_dir):
                continue

            save_dir = os.path.join(out_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)

            specs = []
            meta = []

            for fname in sorted(os.listdir(seg_dir)):
                if not fname.endswith('.wav'):
                    continue
                wav_path = os.path.join(seg_dir, fname)
                waveform, sr_loaded = torchaudio.load(wav_path)
                if sr_loaded != sr:
                    waveform = torchaudio.functional.resample(waveform, sr_loaded, sr)
                waveform = waveform.to(device)

                # compute mel & log-scale on GPU
                mels = mel_spec(waveform)
                log_mels = db_transform(mels)

                arr = log_mels.squeeze(0).transpose(0, 1).cpu().numpy()
                specs.append(arr)
                meta.append(fname)  # add segment file name for traceability

            if not specs:
                continue

            out_path = os.path.join(save_dir, 'mels_standard.pickle')
            with open(out_path, 'wb') as f:
                pickle.dump({'features': np.array(specs), 'filenames': meta}, f)

            print(f"[GPU] Saved {len(specs)} segments to {out_path}")

if __name__ == "__main__":
    SEG_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    OUT_ROOT = SEG_ROOT
    compute_log_mel_gpu(SEG_ROOT, OUT_ROOT)
