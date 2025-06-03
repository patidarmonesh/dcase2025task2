import os
import pickle
import torch
import torchaudio
from tqdm import tqdm

def compute_multi_res_mel_gpu(
    seg_root: str,
    out_root: str,
    sr: int = 16000,
    device: str = 'cuda'
):
    """
    GPU-accelerated multi-resolution Mel spectrogram computation.
    Generates three spectrograms per audio clip with different window sizes.
    """
    # Configuration for three resolutions
    configs = [
        {'name': '64ms', 'n_fft': 1024, 'hop': 512},
        {'name': '256ms', 'n_fft': 4096, 'hop': 2048},
        {'name': '1000ms', 'n_fft': 16000, 'hop': 8000}
    ]
    
    # Initialize transforms for each resolution
    transforms = {}
    for cfg in configs:
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg['n_fft'],
            hop_length=cfg['hop'],
            n_mels=128,
            power=2.0
        ).to(device)
        db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        ).to(device)
        transforms[cfg['name']] = (mel_spec, db_transform)

    # Process each machine type
    for machine in tqdm(os.listdir(seg_root), desc="Machines"):
        mdir = os.path.join(seg_root, machine)
        if not os.path.isdir(mdir):
            continue
            
        # Process each split
        for split in ['train', 'test', 'supplemental']:
            seg_dir = os.path.join(mdir, split, 'raw_segments')
            if not os.path.isdir(seg_dir):
                continue

            # Prepare output directory
            save_dir = os.path.join(out_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)
            output_data = {}

            # Process each audio file
            for fname in tqdm(os.listdir(seg_dir), desc=f"{machine}/{split}", leave=False):
                if not fname.endswith('.wav'):
                    continue
                
                # Load and resample audio
                wav_path = os.path.join(seg_dir, fname)
                waveform, orig_sr = torchaudio.load(wav_path)
                if orig_sr != sr:
                    waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
                waveform = waveform.to(device)

                # Compute all three spectrograms
                clip_mels = {}
                for res_name, (mel_spec, db_transform) in transforms.items():
                    try:
                        # Compute Mel spectrogram
                        mels = mel_spec(waveform)
                        log_mels = db_transform(mels)
                        
                        # Convert to numpy array (Time x Mel)
                        arr = log_mels.squeeze(0).transpose(0, 1).cpu().numpy()
                        clip_mels[res_name] = arr
                    except Exception as e:
                        print(f"Error processing {fname} ({res_name}): {str(e)}")
                        continue
                
                # Store results
                clip_id = os.path.splitext(fname)[0]
                output_data[clip_id] = clip_mels

            # Save all spectrograms for this split
            if output_data:
                out_path = os.path.join(save_dir, 'mels_multires.pickle')
                with open(out_path, 'wb') as f:
                    pickle.dump(output_data, f)
                print(f"Saved {len(output_data)} clips to {out_path}")

if __name__ == "__main__":
    # Input: Raw audio files organized as:
    # /kaggle/input/d/moneshpatidar/dcase2025/data/dcase2025t2/dev_data/processed/<machine>/<split>/raw_segments/*.wav
    SEG_ROOT = "/kaggle/input/d/moneshpatidar/dcase2025/data/dcase2025t2/dev_data/processed"
    
    # Output: Processed Mel spectrograms
    OUT_ROOT = "/kaggle/working/dcase2025t2/dev_data/processed"
    
    compute_multi_res_mel_gpu(SEG_ROOT, OUT_ROOT)
