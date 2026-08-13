import os
import pyedflib
from collections import Counter, defaultdict

# Exact channels from constants.py
INCLUDED_CHANNELS = [
    'EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3',
    'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
    'EEG T5', 'EEG T6', 'EEG FZ', 'EEG CZ', 'EEG PZ'
]

# Hardcoded a few sample passing files just for comparison
PASSING_FILES = [
    "/home/stud1/Desktop/Swathi/TUSZ/edf/dev/aaaaaabc/s001_2011/01_tcp_ar/aaaaaabc_s001_t000.edf"
]

LOG_FILE = "./data/resampled/list of all the failed files.txt"

def parse_failed_files(log_path):
    failed = []
    if not os.path.exists(log_path):
        print(f"Could not find {log_path}, using a hardcoded sample instead.")
        return [
            "/home/stud1/Desktop/Swathi/TUSZ/edf/eval/aaaaaaaq/s007_2014/01_tcp_ar/aaaaaaaq_s007_t000.edf",
            "/home/stud1/Desktop/Swathi/TUSZ/edf/dev/aaaaadkb/s010_2016/01_tcp_ar/aaaaadkb_s010_t001.edf",
            "/home/stud1/Desktop/Swathi/TUSZ/edf/train/aaaaates/s007_2015/01_tcp_ar/aaaaates_s007_t004.edf",
            "/home/stud1/Desktop/Swathi/TUSZ/edf/dev/aaaaamva/s004_2016/01_tcp_ar/aaaaamva_s004_t018.edf"
        ]
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Failed to process" in line:
                path = line.split("Failed to process ")[-1].strip()
                if path.endswith(".edf"):
                    failed.append(path)
    return failed[:50] 

def diagnose_file(edf_path, is_passing=False):
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
        
        # Apply the exact TUSZ logic from data_utils.py
        stripped_labels = [l.split("-")[0] for l in raw_labels]
        
        missing = [ch for ch in INCLUDED_CHANNELS if ch not in stripped_labels]
        
        print(f"\n[{'PASSING' if is_passing else 'FAILING'}] Split: {split} | Montage: {montage}")
        print(f"File: {edf_path}")
        print(f"Raw Labels:      {raw_labels}")
        print(f"Stripped Labels: {stripped_labels}")
        print(f"Missing from INCLUDED_CHANNELS: {missing}")
        
        return montage, split, tuple(missing)
    except Exception as e:
        print(f"Could not read {edf_path}: {e}")
        return montage, split, ("FILE_READ_ERROR",)

def main():
    print("--- BASELINE: PASSING FILES ---")
    for f in PASSING_FILES:
        # We don't care if the mock passing files fail to read in this diagnostic
        if os.path.exists(f): 
            diagnose_file(f, is_passing=True)
        
    print("\n--- DIAGNOSING FAILING FILES ---")
    failed_paths = parse_failed_files(LOG_FILE)
    
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
