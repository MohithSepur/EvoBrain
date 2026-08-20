import os
import glob
import re
import argparse
import random
from collections import defaultdict
import pyedflib

def parse_chbmit_summary(summary_file):
    """
    Parse a single chbXX-summary.txt file to extract file names and seizure start/end times.
    Returns: dict mapping edf_basename -> list of (start_sec, end_sec)
    """
    if not os.path.exists(summary_file):
        return {}
    
    file_seizures = defaultdict(list)
    current_file = None
    
    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    current_start = None
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_file = line.split("File Name:")[-1].strip()
            # In some summaries, file name is given as chb01_01.edf
            current_file = os.path.basename(current_file)
            current_start = None
            if current_file not in file_seizures:
                file_seizures[current_file] = []
        elif current_file is not None:
            start_match = re.search(r'Seizure\s*(?:\d+\s*)?Start\s*Time:\s*(\d+)', line, re.IGNORECASE)
            end_match = re.search(r'Seizure\s*(?:\d+\s*)?End\s*Time:\s*(\d+)', line, re.IGNORECASE)
            
            if start_match:
                current_start = float(start_match.group(1))
            elif end_match and current_start is not None:
                current_end = float(end_match.group(1))
                file_seizures[current_file].append((current_start, current_end))
                current_start = None
                
    return file_seizures

def get_edf_duration(edf_path):
    """
    Read EDF duration in seconds using pyedflib or return default 3600.
    """
    try:
        f = pyedflib.EdfReader(edf_path)
        duration = f.file_duration
        f._close()
        return duration
    except Exception:
        # Fallback to standard 1 hour (3600s)
        return 3600.0

def generate_markers_chbmit(raw_dir, out_dir, clip_len, split_mode="patient_15", seed=123, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15):
    """
    Generate marker files for CHB-MIT dataset.
    Track A: patient_15 (15% test data per patient as in Kotoge et al. NeurIPS 2025 paper)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all subject folders: chb01, chb02, ...
    subject_dirs = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and d.lower().startswith('chb')
    ])
    
    if not subject_dirs:
        # Check if raw_dir itself is a single subject folder
        if any(f.endswith('.edf') for f in os.listdir(raw_dir)):
            subject_dirs = [os.path.basename(raw_dir.rstrip('/'))]
            raw_dir = os.path.dirname(raw_dir.rstrip('/'))
            
    print(f"Found {len(subject_dirs)} subjects: {subject_dirs}")
    
    train_sz, train_nosz = [], []
    dev_sz, dev_nosz = [], []
    test_sz, test_nosz = [], []
    
    total_sz_clips = 0
    total_nosz_clips = 0
    
    for subj_idx, subj in enumerate(subject_dirs):
        subj_dir = os.path.join(raw_dir, subj)
        summary_file = os.path.join(subj_dir, f"{subj}-summary.txt")
        if not os.path.exists(summary_file):
            summary_file = os.path.join(subj_dir, f"{subj.lower()}-summary.txt")
            
        file_seizures_dict = parse_chbmit_summary(summary_file)
        
        # Scan for all EDF files in subject dir
        edf_files = sorted([
            f for f in os.listdir(subj_dir)
            if f.endswith('.edf') and not f.startswith('.') and ':Zone.Identifier' not in f
        ])
        
        subj_sz_lines = []
        subj_nosz_lines = []
        
        for edf_name in edf_files:
            edf_path = os.path.join(subj_dir, edf_name)
            duration = get_edf_duration(edf_path)
            seizures = file_seizures_dict.get(edf_name, [])
            
            num_clips = int(duration // clip_len)
            basename = edf_name.replace('.edf', '')
            
            for c_idx in range(num_clips):
                c_start = c_idx * clip_len
                c_end = (c_idx + 1) * clip_len
                
                is_sz = 0
                for sz_s, sz_e in seizures:
                    if max(c_start, sz_s) < min(c_end, sz_e):
                        is_sz = 1
                        break
                        
                line_str = f"{basename}.edf_{c_idx}.h5,{is_sz}\n"
                if is_sz == 1:
                    subj_sz_lines.append(line_str)
                else:
                    subj_nosz_lines.append(line_str)
                    
        total_sz_clips += len(subj_sz_lines)
        total_nosz_clips += len(subj_nosz_lines)
        
        if split_mode == "patient_15":
            # Track A: Random 15% test split per patient
            rng = random.Random(seed + subj_idx * 1000)
            rng.shuffle(subj_sz_lines)
            rng.shuffle(subj_nosz_lines)
            
            # Split seizure clips
            n_sz = len(subj_sz_lines)
            n_sz_train = int(n_sz * train_ratio)
            n_sz_dev = int(n_sz * dev_ratio)
            # remaining goes to test
            train_sz.extend(subj_sz_lines[:n_sz_train])
            dev_sz.extend(subj_sz_lines[n_sz_train:n_sz_train + n_sz_dev])
            test_sz.extend(subj_sz_lines[n_sz_train + n_sz_dev:])
            
            # Split non-seizure clips
            n_nosz = len(subj_nosz_lines)
            n_nosz_train = int(n_nosz * train_ratio)
            n_nosz_dev = int(n_nosz * dev_ratio)
            train_nosz.extend(subj_nosz_lines[:n_nosz_train])
            dev_nosz.extend(subj_nosz_lines[n_nosz_train:n_nosz_train + n_nosz_dev])
            test_nosz.extend(subj_nosz_lines[n_nosz_train + n_nosz_dev:])
            
            print(f"  [{subj}] Seizures: {n_sz} (Tr:{n_sz_train}, Dev:{n_sz_dev}, Te:{n_sz - n_sz_train - n_sz_dev}) | Non-Sz: {n_nosz} (Tr:{n_nosz_train}, Dev:{n_nosz_dev}, Te:{n_nosz - n_nosz_train - n_nosz_dev})")
            
        elif split_mode == "subject_holdout":
            # Track B: Subject holdout split
            # e.g., chb01-16+21: train, chb17-19: dev, chb20,22-24: test
            if subj in ['chb17', 'chb18', 'chb19']:
                dev_sz.extend(subj_sz_lines)
                dev_nosz.extend(subj_nosz_lines)
            elif subj in ['chb20', 'chb22', 'chb23', 'chb24']:
                test_sz.extend(subj_sz_lines)
                test_nosz.extend(subj_nosz_lines)
            else:
                train_sz.extend(subj_sz_lines)
                train_nosz.extend(subj_nosz_lines)
                
    # Write output files
    splits = {
        'train': (train_sz, train_nosz),
        'dev': (dev_sz, dev_nosz),
        'test': (test_sz, test_nosz)
    }
    
    for sname, (sz_list, nosz_list) in splits.items():
        with open(os.path.join(out_dir, f"{sname}Set_seq2seq_{clip_len}s_sz.txt"), 'w') as f:
            f.writelines(sz_list)
        with open(os.path.join(out_dir, f"{sname}Set_seq2seq_{clip_len}s_nosz.txt"), 'w') as f:
            f.writelines(nosz_list)
        print(f"  -> {sname.upper()}: {len(sz_list)} seizure clips, {len(nosz_list)} non-seizure clips written.")
        
    print(f"\nCHB-MIT {clip_len}s Total: {total_sz_clips} seizure clips, {total_nosz_clips} non-seizure clips across {len(subject_dirs)} subjects.")

def generate_markers_tusz(raw_dir, out_dir, clip_len):
    splits = []
    for s in ['train', 'dev', 'eval', 'test']:
        if os.path.exists(os.path.join(raw_dir, s)):
            splits.append(s)
            
    if not splits:
        splits = ['all']
        
    os.makedirs(out_dir, exist_ok=True)
    generated_splits = {}
    
    for split in splits:
        sz_lines = []
        nosz_lines = []
        
        split_dir = raw_dir if split == 'all' else os.path.join(raw_dir, split)
        print(f"Scanning {split_dir} for {clip_len}s clips...")
        
        edf_files = []
        for root, _, files in os.walk(split_dir):
            for name in files:
                if name.endswith('.edf') and not name.startswith('.'):
                    edf_files.append(os.path.join(root, name))
                    
        print(f"  -> Found {len(edf_files)} .edf files in {split_dir}")
        
        missing_anno_count = 0
        for edf in edf_files:
            basename = os.path.basename(edf).replace('.edf', '')
            
            tse_bi = edf.replace('.edf', '.tse_bi')
            csv_bi = edf.replace('.edf', '.csv_bi')
            tse = edf.replace('.edf', '.tse')
            csv = edf.replace('.edf', '.csv')
            
            anno_file = None
            if os.path.exists(tse_bi):
                anno_file = tse_bi
            elif os.path.exists(csv_bi):
                anno_file = csv_bi
            elif os.path.exists(tse):
                anno_file = tse
            elif os.path.exists(csv):
                anno_file = csv
            else:
                missing_anno_count += 1
                continue
            
            seizures = []
            max_time = 0
            try:
                with open(anno_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'version' in line or line.startswith("#") or not line.strip() or 'start_time' in line: 
                            continue
                        parts = line.strip().replace(',', ' ').split()
                        if len(parts) >= 4 and parts[0].upper() == "TERM":
                            start, end, label = float(parts[1]), float(parts[2]), parts[3]
                            max_time = max(max_time, end)
                            if label.lower() != 'bckg' or 'seiz' in label.lower():
                                seizures.append((start, end))
                        elif len(parts) >= 3:
                            start, end, label = float(parts[0]), float(parts[1]), parts[2]
                            max_time = max(max_time, end)
                            if label.lower() != 'bckg' or 'seiz' in line.lower():
                                seizures.append((start, end))
                        elif len(parts) >= 2 and any(k in line.lower() for k in ["seiz", "fnsz", "gnsz", "cpsz", "spsz", "tcsz"]):
                            seizures.append((float(parts[0]), float(parts[1])))
            except Exception as e:
                print(f"Error reading {anno_file}: {e}")
                continue
                            
            num_clips = int(max_time // clip_len)
            for i in range(num_clips):
                clip_start = i * clip_len
                clip_end = (i + 1) * clip_len
                
                is_sz = 0
                for sz_s, sz_e in seizures:
                    if max(clip_start, sz_s) < min(clip_end, sz_e):
                        is_sz = 1
                        break
                        
                line_str = f"{basename}.edf_{i}.h5,{is_sz}\n"
                if is_sz == 1:
                    sz_lines.append(line_str)
                else:
                    nosz_lines.append(line_str)
                    
        if split == 'all':
            for sname in ["train", "dev", "test"]:
                generated_splits[sname] = (sz_lines, nosz_lines)
                with open(os.path.join(out_dir, f"{sname}Set_seq2seq_{clip_len}s_sz.txt"), 'w') as f:
                    f.writelines(sz_lines)
                with open(os.path.join(out_dir, f"{sname}Set_seq2seq_{clip_len}s_nosz.txt"), 'w') as f:
                    f.writelines(nosz_lines)
            print(f"  -> Generated {len(sz_lines)} seizure clips and {len(nosz_lines)} non-seizure clips (applied to train/dev/test).")
        else:
            split_name = "test" if split in ["eval", "test"] else split
            generated_splits[split_name] = (sz_lines, nosz_lines)
            with open(os.path.join(out_dir, f"{split_name}Set_seq2seq_{clip_len}s_sz.txt"), 'w') as f:
                f.writelines(sz_lines)
            with open(os.path.join(out_dir, f"{split_name}Set_seq2seq_{clip_len}s_nosz.txt"), 'w') as f:
                f.writelines(nosz_lines)
            print(f"  -> {split_name}: Generated {len(sz_lines)} seizure clips and {len(nosz_lines)} non-seizure clips.")
            
        if missing_anno_count > 0:
            print(f"  -> WARNING: {missing_anno_count} EDF files had NO matching annotation file (.tse_bi, .tse, or .csv)!")

    if "test" not in generated_splits and "dev" in generated_splits:
        print("  -> Notice: No 'eval' or 'test' split folder found. Mirroring 'dev' markers to 'test' marker files.")
        dev_sz, dev_nosz = generated_splits["dev"]
        with open(os.path.join(out_dir, f"testSet_seq2seq_{clip_len}s_sz.txt"), 'w') as f:
            f.writelines(dev_sz)
        with open(os.path.join(out_dir, f"testSet_seq2seq_{clip_len}s_nosz.txt"), 'w') as f:
            f.writelines(dev_nosz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Marker Files for TUSZ and CHB-MIT")
    parser.add_argument("--dataset", type=str, default="CHB-MIT", choices=["TUSZ", "CHBMIT", "CHB-MIT"], help="Dataset name")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to raw edf directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for marker files")
    parser.add_argument("--clip_lens", nargs="+", type=int, default=[10, 12, 60], help="Clip lengths in seconds (default: 10 12 60)")
    parser.add_argument("--split_mode", type=str, default="patient_15", choices=["patient_15", "subject_holdout"], help="Split mode for CHB-MIT: patient_15 (Track A) or subject_holdout (Track B)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible patient split")
    
    args = parser.parse_args()
    
    is_chb = args.dataset in ["CHBMIT", "CHB-MIT"]
    if args.out_dir is None:
        args.out_dir = "./data/file_markers_chb" if is_chb else "./data/file_markers_detection"
        
    for cl in args.clip_lens:
        print(f"\n==========================================")
        print(f"Generating {cl}s markers for {args.dataset} (Mode: {args.split_mode if is_chb else 'split_folders'})...")
        print(f"==========================================")
        if is_chb:
            generate_markers_chbmit(args.raw_data_dir, args.out_dir, cl, split_mode=args.split_mode, seed=args.seed)
        else:
            generate_markers_tusz(args.raw_data_dir, args.out_dir, cl)
            
    print("\nDONE! All marker files successfully generated.")
