import os
import sys
import argparse
import pyedflib
from collections import Counter, defaultdict

try:
    from constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS
    from data_utils import MODERN_CHANNEL_ALIASES
except ImportError:
    from data.constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS
    from data.data_utils import MODERN_CHANNEL_ALIASES


def clean_channel_name(ch):
    """Normalize channel name by removing 'EEG', 'EEG ', trailing dots, and whitespace."""
    return ch.upper().replace('EEG ', '').replace('EEG', '').strip().rstrip('.').replace(' ', '')


def parse_failed_files(log_path):
    possible_paths = [
        log_path,
        "./list_of_all_the_failed_files.txt",
        "./data/resampled/list_of_all_the_failed_files.txt",
        "./data/resampled/list of all the failed files.txt",
        "./data/resampled/CHB-MIT/list_of_all_the_failed_files.txt",
        "./data/resampled/TUSZ/list_of_all_the_failed_files.txt",
    ]
    actual_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            actual_path = p
            break
            
    if not actual_path:
        print(f"Could not find log file in any of: {possible_paths}")
        return []

    print(f"Reading failure log from: {actual_path}")
    failed = []
    with open(actual_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "Failed to process" in line:
                path = line.split("Failed to process ")[-1].strip()
                if path.endswith(".edf") and path not in failed:
                    failed.append(path)
    return failed


def diagnose_file(edf_path, dataset="TUSZ", raw_data_dir=None):
    parts = edf_path.split('/')
    split = "unknown"
    montage = "unknown"
    
    if "edf" in parts:
        try:
            split = parts[parts.index('edf') + 1]
        except Exception:
            pass
    for p in parts:
        if 'tcp' in p or p.startswith('chb'):
            montage = p
            break
            
    actual_path = edf_path
    if not os.path.exists(actual_path) and raw_data_dir and os.path.exists(raw_data_dir):
        # Search for the edf file basename inside raw_data_dir
        base_name = os.path.basename(edf_path)
        for root, _, files in os.walk(raw_data_dir):
            if base_name in files:
                actual_path = os.path.join(root, base_name)
                break
                
    if not os.path.exists(actual_path):
        return montage, split, ("FILE_NOT_FOUND_ON_LOCAL_DISK",), ("FILE_NOT_FOUND_ON_LOCAL_DISK",)
        
    target_channels = CHBMIT_INCLUDED_CHANNELS if dataset in ["CHBMIT", "CHB-MIT"] else INCLUDED_CHANNELS
    
    try:
        f = pyedflib.EdfReader(actual_path)
        raw_labels = f.getSignalLabels()
        f._close()
        
        if dataset in ["CHBMIT", "CHB-MIT"]:
            labels_raw_upper = [l.strip().upper() for l in raw_labels]
            labels_cleaned = [clean_channel_name(l) for l in raw_labels]
        else:
            labels_raw_upper = [l.split("-")[0].strip().upper() for l in raw_labels]
            labels_cleaned = [clean_channel_name(l.split("-")[0]) for l in raw_labels]
        
        missing_exact = []
        missing_after_alias = []
        
        for ch in target_channels:
            ch_clean = clean_channel_name(ch)
            
            # Exact match check
            if ch.upper() not in labels_raw_upper and ch_clean not in labels_cleaned:
                missing_exact.append(ch)
                
                # Alias check (using MODERN_CHANNEL_ALIASES)
                alias = MODERN_CHANNEL_ALIASES.get(ch, "") or MODERN_CHANNEL_ALIASES.get(ch_clean, "")
                alias_clean = clean_channel_name(alias) if alias else ""
                
                if not (alias and (alias.upper() in labels_raw_upper or alias_clean in labels_cleaned)):
                    missing_after_alias.append(ch)
                    
        return montage, split, tuple(missing_exact), tuple(missing_after_alias)
    except Exception as e:
        return montage, split, (f"ERROR: {e}",), (f"ERROR: {e}",)


def main():
    parser = argparse.ArgumentParser(description="Diagnose EDF Channel Mismatches")
    parser.add_argument("--log_file", type=str, default="./list_of_all_the_failed_files.txt", help="Path to failure log")
    parser.add_argument("--dataset", type=str, default="TUSZ", choices=["TUSZ", "CHBMIT", "CHB-MIT"], help="Dataset name")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw", help="Path to local raw data directory for resolving paths")
    args = parser.parse_args()

    failed_paths = parse_failed_files(args.log_file)
    if not failed_paths:
        print("No failed files found to diagnose.")
        return
        
    print(f"Found {len(failed_paths)} unique failed files in log.")
    
    summary = defaultdict(lambda: defaultdict(list))
    resolvable_by_alias = 0
    truly_missing = 0
    not_found = 0
    
    for f in failed_paths:
        montage, split, missing_exact, missing_alias = diagnose_file(f, dataset=args.dataset, raw_data_dir=args.raw_data_dir)
        summary[montage][split].append((missing_exact, missing_alias))
        
        if "FILE_NOT_FOUND_ON_LOCAL_DISK" in missing_exact:
            not_found += 1
        elif len(missing_exact) > 0 and len(missing_alias) == 0:
            resolvable_by_alias += 1
        elif len(missing_alias) > 0:
            truly_missing += 1
            
    print("\n================ AGGREGATE SUMMARY ================")
    print(f"Total failed files evaluated: {len(failed_paths)}")
    print(f"  -> Successfully resolved by modern alias mapping: {resolvable_by_alias}")
    print(f"  -> File not found on local disk: {not_found}")
    print(f"  -> Truly missing channels after alias mapping: {truly_missing}")
    
    for montage, splits in summary.items():
        for split, entries in splits.items():
            print(f"\nMontage: {montage} | Split: {split} (Total logged: {len(entries)})")
            after_alias_lists = [e[1] for e in entries if e[1]]
            if after_alias_lists:
                counter_alias = Counter(after_alias_lists)
                for missing_tuple, count in counter_alias.most_common():
                    print(f"  -> {count} files still missing: {list(missing_tuple)}")
            else:
                print("  -> All channel mismatches in this group are 100% resolved by modern alias mapping!")


if __name__ == "__main__":
    main()
