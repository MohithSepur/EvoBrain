import os
import pyedflib
from collections import Counter, defaultdict

# Exact channels from constants.py
INCLUDED_CHANNELS = [
    'EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3',
    'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
    'EEG T5', 'EEG T6', 'EEG FZ', 'EEG CZ', 'EEG PZ'
]

LOG_FILE = "./data/resampled/list of all the failed files.txt"

def parse_failed_files(log_path):
    failed = []
    # If the exact path doesn't exist, check the root directory as a fallback
    if not os.path.exists(log_path):
        log_path = "./list_of_all_the_failed_files.txt"
        
    if not os.path.exists(log_path):
        print(f"Could not find log file at {log_path}. Please check the path.")
        return []

    with open(log_path, 'r') as f:
        for line in f:
            if "Failed to process" in line:
                path = line.split("Failed to process ")[-1].strip()
                if path.endswith(".edf"):
                    failed.append(path)
    return failed[:50] # Just check the first 50 to save time

def diagnose_file(edf_path):
    parts = edf_path.split('/')
    try:
        split = parts[parts.index('edf') + 1]
        montage = next(p for p in parts if 'tcp' in p)
    except:
        split, montage = "unknown", "unknown"
        
    try:
        f = pyedflib.EdfReader(edf_path)
        raw_labels = f.getSignalLabels()
        f._close()
        
        stripped_labels = [l.split("-")[0] for l in raw_labels]
        missing = [ch for ch in INCLUDED_CHANNELS if ch not in stripped_labels]
        return montage, split, tuple(missing)
    except Exception as e:
        return montage, split, ("FILE_READ_ERROR",)

def main():
    failed_paths = parse_failed_files(LOG_FILE)
    if not failed_paths:
        return
        
    summary = defaultdict(lambda: defaultdict(list))
    for f in failed_paths:
        montage, split, missing = diagnose_file(f)
        if missing:
            summary[montage][split].append(missing)
            
    print("\n================ AGGREGATE SUMMARY ================")
    for montage, splits in summary.items():
        for split, missing_lists in splits.items():
            counter = Counter(missing_lists)
            print(f"Montage: {montage} | Split: {split}")
            for missing_tuple, count in counter.most_common():
                print(f"  -> {count} files missing these channels: {list(missing_tuple)}")

if __name__ == "__main__":
    main()
