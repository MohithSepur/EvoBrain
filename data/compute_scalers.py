import os
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.dataloader_chb import computeSliceMatrix as computeSliceMatrix_CHB
    from data.dataloader_detection import computeSliceMatrix as computeSliceMatrix_TUSZ
except ImportError:
    from dataloader_chb import computeSliceMatrix as computeSliceMatrix_CHB
    from dataloader_detection import computeSliceMatrix as computeSliceMatrix_TUSZ


def compute_dataset_scaler(dataset, resampled_dir, raw_data_dir, marker_dir, clip_len=12, time_step_size=1, max_samples=2000):
    """
    Computes global mean and standard deviation of Fourier transformed EEG signals across the training split.
    """
    is_chb = dataset in ["CHBMIT", "CHB-MIT"]
    compute_fn = computeSliceMatrix_CHB if is_chb else computeSliceMatrix_TUSZ
    
    train_sz = os.path.join(marker_dir, f"trainSet_seq2seq_{clip_len}s_sz.txt")
    train_nosz = os.path.join(marker_dir, f"trainSet_seq2seq_{clip_len}s_nosz.txt")
    
    if not os.path.exists(train_sz) or not os.path.exists(train_nosz):
        raise FileNotFoundError(f"Missing train marker files in {marker_dir} for clip_len={clip_len}s")
        
    lines = []
    with open(train_sz, 'r') as f:
        lines.extend(f.readlines())
    with open(train_nosz, 'r') as f:
        lines.extend(f.readlines())
        
    np.random.seed(123)
    np.random.shuffle(lines)
    
    if max_samples and len(lines) > max_samples:
        print(f"Subsampling {max_samples}/{len(lines)} train clips to estimate mean/std...")
        lines = lines[:max_samples]
    else:
        print(f"Using all {len(lines)} train clips to compute mean/std...")
        
    # Index all raw EDFs for fast lookup
    edf_map = {}
    for root, _, files in os.walk(raw_data_dir):
        for f in files:
            if f.endswith('.edf'):
                edf_map[f] = os.path.join(root, f)
                
    # Index all resampled H5 files for fast lookup across all subdirectories (train, dev, eval, subject dirs)
    h5_map = {}
    for root, _, files in os.walk(resampled_dir):
        for f in files:
            if f.endswith('.h5'):
                h5_map[f] = os.path.join(root, f)
                
    print(f"Indexed {len(edf_map)} raw .edf files from {raw_data_dir}")
    print(f"Indexed {len(h5_map)} resampled .h5 files from {resampled_dir}")
    if len(h5_map) == 0:
        raise FileNotFoundError(f"Found 0 .h5 files in {resampled_dir}! Please ensure your --resampled_dir path is correct.")
    if len(edf_map) == 0:
        raise FileNotFoundError(f"Found 0 .edf files in {raw_data_dir}! Please ensure your --raw_data_dir path is correct.")
        
    # Running accumulation
    sum_vals = 0.0
    sum_sq_vals = 0.0
    total_count = 0
    
    first_error_logged = False
    successful_clips = 0
    for line in tqdm(lines, desc=f"Computing scaler ({dataset} {clip_len}s)"):
        line = line.strip()
        if not line:
            continue
        h5_fn, _ = line.split(',')
        clip_idx = int(h5_fn.split('_')[-1].split('.h5')[0])
        edf_name = h5_fn.split('.edf')[0] + '.edf'
        
        if edf_name not in edf_map:
            continue
        edf_full_path = edf_map[edf_name]
        
        base_h5 = h5_fn.split('.edf')[0] + '.h5'
        h5_path = h5_map.get(base_h5, None)
        if h5_path is None or not os.path.exists(h5_path):
            continue
            
        try:
            eeg_clip, _ = compute_fn(
                h5_fn=h5_path,
                edf_fn=edf_full_path,
                clip_idx=clip_idx,
                time_step_size=time_step_size,
                clip_len=clip_len,
                is_fft=True
            )
            
            sum_vals += np.sum(eeg_clip)
            sum_sq_vals += np.sum(np.square(eeg_clip))
            total_count += eeg_clip.size
            successful_clips += 1
        except Exception as e:
            if not first_error_logged:
                print(f"\n[Debug] Error reading clip {base_h5} (idx {clip_idx}): {e}")
                first_error_logged = True
            continue
            
    if total_count == 0 or successful_clips == 0:
        raise RuntimeError(
            f"Failed to read any valid resampled clips to compute scaler! "
            f"Verified {len(h5_map)} .h5 files and {len(edf_map)} .edf files. "
            f"Please check that the filenames in your marker files match the .h5 files."
        )
        
    mean = sum_vals / total_count
    var = (sum_sq_vals / total_count) - (mean ** 2)
    std = np.sqrt(max(var, 1e-8))
    
    mean_val = np.float64(mean)
    std_val = np.float64(std)
    
    out_mean_file = os.path.join(marker_dir, f"means_seq2seq_fft_{clip_len}s_szdetect_single.pkl")
    out_std_file = os.path.join(marker_dir, f"stds_seq2seq_fft_{clip_len}s_szdetect_single.pkl")
    
    with open(out_mean_file, 'wb') as f:
        pickle.dump(mean_val, f)
    with open(out_std_file, 'wb') as f:
        pickle.dump(std_val, f)
        
    print(f"\nSuccessfully computed scaler over {successful_clips} clips:")
    print(f"  Mean: {mean_val:.6f} -> Saved to {out_mean_file}")
    print(f"  Std:  {std_val:.6f} -> Saved to {out_std_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute StandardScaler pkl files for EEG training")
    parser.add_argument("--dataset", type=str, default="CHB-MIT", choices=["TUSZ", "CHBMIT", "CHB-MIT"])
    parser.add_argument("--resampled_dir", type=str, required=True, help="Path to resampled h5 directory")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to raw edf directory")
    parser.add_argument("--marker_dir", type=str, default=None, help="Path to marker file directory")
    parser.add_argument("--clip_lens", nargs="+", type=int, default=[10, 12, 60], help="Clip lengths in seconds")
    parser.add_argument("--max_samples", type=int, default=3000, help="Number of training clips to sample for statistics")
    
    args = parser.parse_args()
    
    is_chb = args.dataset in ["CHBMIT", "CHB-MIT"]
    if args.marker_dir is None:
        args.marker_dir = "./data/file_markers_chb" if is_chb else "./data/file_markers_detection"
        
    for cl in args.clip_lens:
        compute_dataset_scaler(
            dataset=args.dataset,
            resampled_dir=args.resampled_dir,
            raw_data_dir=args.raw_data_dir,
            marker_dir=args.marker_dir,
            clip_len=cl,
            max_samples=args.max_samples
        )
