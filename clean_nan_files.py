import os
import sys
import argparse
import pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import shutil

def check_file(file_path):
    """
    Checks if a .pkl file is valid or contains NaN/Inf/all-zeros/corruption.
    Returns: (file_path, is_valid, reason, label)
    """
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        return (file_path, False, f"Corrupt/Unpicklable: {type(e).__name__}", None)

    if isinstance(data_dict, dict):
        raw_data = data_dict.get('X', data_dict.get('data', None))
        label = data_dict.get('y', data_dict.get('label', 0))
    elif isinstance(data_dict, (tuple, list)):
        raw_data, label = data_dict[0], data_dict[1]
    else:
        raw_data, label = data_dict, 0

    if raw_data is None:
        return (file_path, False, "Missing 'X' or 'data' key", None)

    raw_arr = np.array(raw_data)

    if np.isnan(raw_arr).any():
        return (file_path, False, "Contains NaN", label)
    if np.isinf(raw_arr).any():
        return (file_path, False, "Contains Inf", label)
    if np.all(raw_arr == 0):
        return (file_path, False, "All-Zeros / Disconnected flatline", label)

    # Check if signal has zero standard deviation across all channels
    if raw_arr.ndim >= 2 and np.all(np.std(raw_arr, axis=-1) == 0):
        return (file_path, False, "Zero variance across all channels", label)

    return (file_path, True, "OK", label)


def restore_quarantine(data_dir):
    """
    Restores all quarantined files back to their original train/val/test directories.
    """
    quarantine_dir = os.path.join(data_dir, "_quarantined_nan_files")
    if not os.path.exists(quarantine_dir):
        print(f"No quarantine folder found at: {quarantine_dir}")
        return

    quarantined_files = []
    for root, _, files in os.walk(quarantine_dir):
        for f in files:
            if f.endswith('.pkl'):
                quarantined_files.append(os.path.join(root, f))

    if not quarantined_files:
        print("No files found in quarantine.")
        return

    print(f"Restoring {len(quarantined_files):,} files from quarantine back to original locations...")
    for fpath in tqdm(quarantined_files, desc="Restoring"):
        rel_path = os.path.relpath(fpath, quarantine_dir)
        target_path = os.path.join(data_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(fpath, target_path)

    shutil.rmtree(quarantine_dir, ignore_errors=True)
    print(f"Successfully restored {len(quarantined_files):,} files back to {data_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Fast multi-threaded scanner to remove, quarantine, or restore NaN/Inf/corrupted .pkl files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the processed_seg directory (containing train/, val/, test/ folders)."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["quarantine", "restore", "delete", "dry-run"],
        default="quarantine",
        help="Action: 'quarantine' (move bad files aside), 'restore' (undo quarantine), 'delete' (permanently remove), 'dry-run' (scan only)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel CPU worker processes (default: {mp.cpu_count()})."
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        sys.exit(1)

    if args.action == "restore":
        restore_quarantine(data_dir)
        return

    print(f"=== CHB-MIT NaN/Corrupt File Cleaner ===")
    print(f"Scanning directory : {data_dir}")
    print(f"Action             : {args.action.upper()}")
    print(f"Worker processes   : {args.num_workers}")
    print("=" * 40)

    # Collect all .pkl files across train, val, dev, test or subdirectories
    all_files = []
    splits = ['train', 'val', 'dev', 'test']
    has_splits = any(os.path.exists(os.path.join(data_dir, s)) for s in splits)

    if has_splits:
        for s in splits:
            s_dir = os.path.join(data_dir, s)
            if os.path.exists(s_dir):
                for f in os.listdir(s_dir):
                    if f.endswith('.pkl') and not f.startswith('.'):
                        all_files.append(os.path.join(s_dir, f))
    else:
        for root, _, files in os.walk(data_dir):
            if '_quarantined' in root:
                continue
            for f in files:
                if f.endswith('.pkl') and not f.startswith('.'):
                    all_files.append(os.path.join(root, f))

    total_files = len(all_files)
    print(f"Found {total_files:,} .pkl files to scan.")

    if total_files == 0:
        print("No .pkl files found. Exiting.")
        return

    # Run multi-threaded check with tqdm
    invalid_files = []
    valid_count = 0
    reason_counts = {}
    label_counts = {0: 0, 1: 0, 'unknown': 0}

    chunksize = max(100, total_files // (args.num_workers * 20))

    with mp.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(check_file, all_files, chunksize=chunksize),
                total=total_files,
                desc="Scanning files",
                unit="file"
            )
        )

    for file_path, is_valid, reason, label in results:
        if is_valid:
            valid_count += 1
        else:
            invalid_files.append((file_path, reason, label))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if label == 1:
                label_counts[1] += 1
            elif label == 0:
                label_counts[0] += 1
            else:
                label_counts['unknown'] += 1

    num_invalid = len(invalid_files)
    print("\n" + "=" * 40)
    print("=== SCAN RESULTS ===")
    print(f"Total files scanned : {total_files:,}")
    print(f"Valid clean files   : {valid_count:,} ({valid_count/total_files*100:.2f}%)")
    print(f"Invalid / NaN files : {num_invalid:,} ({num_invalid/total_files*100:.2f}%)")
    print("=" * 40)

    if num_invalid > 0:
        print("\nBreakdown by issue:")
        for r, c in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {r}: {c:,} files")

        print(f"\nLabel breakdown of invalid files: Non-Seizure (0): {label_counts[0]:,}, Seizure (1): {label_counts[1]:,}")

        # Execute action
        if args.action == "quarantine":
            quarantine_dir = os.path.join(data_dir, "_quarantined_nan_files")
            os.makedirs(quarantine_dir, exist_ok=True)
            print(f"\nMoving {num_invalid:,} invalid files to: {quarantine_dir}")

            for file_path, reason, _ in tqdm(invalid_files, desc="Quarantining"):
                rel_path = os.path.relpath(file_path, data_dir)
                target_path = os.path.join(quarantine_dir, rel_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.move(file_path, target_path)

            print(f"Successfully quarantined {num_invalid:,} files. They will not be loaded by the DataLoader.")

        elif args.action == "delete":
            confirm = input(f"\nAre you sure you want to PERMANENTLY DELETE {num_invalid:,} files? [y/N]: ")
            if confirm.lower() == 'y':
                for file_path, _, _ in tqdm(invalid_files, desc="Deleting"):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                print(f"Successfully deleted {num_invalid:,} files.")
            else:
                print("Deletion aborted. No files were deleted.")

        elif args.action == "dry-run":
            print("\nDry-run complete. No files were moved or deleted.")
    else:
        print("\n🎉 All files are 100% clean! No NaNs, Infs, or corruptions found.")

if __name__ == "__main__":
    main()
