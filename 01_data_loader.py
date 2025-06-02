import os
import librosa
import soundfile as sf


def segment_and_resample(
    raw_root: str,
    processed_root: str,
    sr: int = 16000,
    segment_duration: float = 1.0
):
    """
    Walk the `raw_root` directory containing multiple machine-type subfolders.
    For each machine and for each split folder (e.g., train, test, supplemental):
      1. Load each WAV at original sample rate, resample to `sr`.
      2. Split into non-overlapping segments of `segment_duration` seconds.
      3. Save segments under:
         processed_root/<machine>/<split>/raw_segments/

    Args:
        raw_root: path to raw audio root
        processed_root: base path to save processed segments
        sr: target sampling rate (default 16 kHz)
        segment_duration: segment length in seconds
    """
    seg_len = int(sr * segment_duration)
    # Discover machine types
    machines = [m for m in os.listdir(raw_root)
                if os.path.isdir(os.path.join(raw_root, m))]
    for machine in machines:
        machine_raw = os.path.join(raw_root, machine)
        for split in os.listdir(machine_raw):
            split_in = os.path.join(machine_raw, split)
            if not os.path.isdir(split_in):
                continue
            # Prepare output directory
            split_out = os.path.join(processed_root, machine, split, 'raw_segments')
            os.makedirs(split_out, exist_ok=True)

            # Process WAV files
            for fname in os.listdir(split_in):
                if not fname.lower().endswith('.wav'):
                    continue
                path = os.path.join(split_in, fname)
                audio, orig_sr = librosa.load(path, sr=None)
                # Resample if needed
                if orig_sr != sr:
                    audio = librosa.resample(audio, orig_sr, sr)
                # Split into fixed-length segments
                total = len(audio)
                n_segs = total // seg_len
                for idx in range(n_segs):
                    start = idx * seg_len
                    end = start + seg_len
                    segment = audio[start:end]
                    out_fname = f"{os.path.splitext(fname)[0]}_seg{idx:02d}.wav"
                    out_path = os.path.join(split_out, out_fname)
                    sf.write(out_path, segment, sr)


if __name__ == "__main__":
    RAW_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/raw"
    PROC_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    segment_and_resample(RAW_ROOT, PROC_ROOT)
