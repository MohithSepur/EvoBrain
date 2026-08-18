import os
import sys
import argparse
import h5py
from tqdm import tqdm
from dataloader_detection import computeSliceMatrix

FILE_MARKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_markers_detection")


def main(
        resample_dir,
        raw_data_dir,
        output_dir,
        clip_len,
        time_step_size,
        is_fft=True):
    # Read all marker files if they exist
    all_tuples = []
    for split_name in ["trainSet", "devSet", "testSet"]:
        for label_type in ["sz", "nosz"]:
            marker_file = os.path.join(FILE_MARKER_DIR, f"{split_name}_seq2seq_{clip_len}s_{label_type}.txt")
            if os.path.exists(marker_file):
                with open(marker_file, "r") as f:
                    for line in f:
                        if line.strip():
                            all_tuples.append(line.strip().split(','))

    print(f"Loaded {len(all_tuples)} clips to preprocess across splits.")

    edf_files = []
    for path, subdirs, files in os.walk(raw_data_dir):
        for name in files:
            if name.endswith(".edf"):
                edf_files.append(os.path.join(path, name))

    # Build O(1) lookup dictionary for EDF paths
    edf_map = {os.path.basename(f): f for f in edf_files}
    print(f"Indexed {len(edf_map)} raw EDF files.")

    output_dir = os.path.join(
        output_dir,
        'clipLen' +
        str(clip_len) +
        '_timeStepSize' +
        str(time_step_size))
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(all_tuples))):
        h5_fn, _ = all_tuples[idx]
        target_edf = h5_fn.split('.edf')[0] + '.edf'
        if target_edf not in edf_map:
            continue
        edf_fn_full = edf_map[target_edf]
        clip_idx = int(h5_fn.split('_')[-1].split('.h5')[0])
        base_h5 = h5_fn.split('.edf')[0] + '.h5'

        # Check flat and nested resampled paths
        resampled_path = os.path.join(resample_dir, base_h5)
        if not os.path.exists(resampled_path):
            for s in ["train", "dev", "eval", "test"]:
                nested = os.path.join(resample_dir, s, base_h5)
                if os.path.exists(nested):
                    resampled_path = nested
                    break

        if not os.path.exists(resampled_path):
            continue

        eeg_clip, _ = computeSliceMatrix(
            h5_fn=resampled_path,
            edf_fn=edf_fn_full,
            clip_idx=clip_idx,
            time_step_size=time_step_size,
            clip_len=clip_len,
            is_fft=is_fft)

        out_fn = os.path.join(output_dir, f"{target_edf}_{clip_idx}.h5")
        with h5py.File(out_fn, 'w') as hf:
            hf.create_dataset('clip', data=eeg_clip)

    print("Preprocessing DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resampled_dir",
        type=str,
        default=None,
        help="Directory to resampled signals.")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Directory to raw edf files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory.")
    parser.add_argument(
        "--clip_len",
        type=int,
        default=60,
        help="EEG clip length in seconds.")
    parser.add_argument(
        "--time_step_size",
        type=int,
        default=1,
        help="Time step size in seconds.")
    parser.add_argument(
        "--is_fft",
        action="store_true",
        default=False,
        help="Whether to perform FFT.")

    args = parser.parse_args()
    main(
        args.resampled_dir,
        args.raw_data_dir,
        args.output_dir,
        args.clip_len,
        args.time_step_size,
        args.is_fft)
