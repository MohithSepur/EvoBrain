import os
import glob
import argparse

def generate_markers(raw_dir, out_dir, clip_len):
    splits = ['train', 'dev', 'eval']
    os.makedirs(out_dir, exist_ok=True)
    
    for split in splits:
        sz_lines = []
        nosz_lines = []
        
        # In TUSZ, the raw dir might have split folders directly
        split_dir = os.path.join(raw_dir, split)
        if not os.path.exists(split_dir): 
            print(f"Warning: Could not find {split_dir}")
            continue
            
        print(f"Scanning {split_dir} for {clip_len}s clips...")
        tse_files = glob.glob(os.path.join(split_dir, '**', '*.tse'), recursive=True)
        
        for tse in tse_files:
            basename = os.path.basename(tse).replace('.tse', '')
            
            seizures = []
            max_time = 0
            try:
                with open(tse, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'version' in line or not line.strip(): 
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start = float(parts[0])
                            end = float(parts[1])
                            label = parts[2]
                            max_time = max(max_time, end)
                            
                            # 'bckg' is background (non-seizure). Anything else is a seizure type.
                            if label != 'bckg':
                                seizures.append((start, end))
            except Exception as e:
                print(f"Error reading {tse}: {e}")
                continue
                            
            num_clips = int(max_time // clip_len)
            for i in range(num_clips):
                clip_start = i * clip_len
                clip_end = (i + 1) * clip_len
                
                is_sz = 0
                for sz_s, sz_e in seizures:
                    # Check for overlap between the clip window and the seizure window
                    if max(clip_start, sz_s) < min(clip_end, sz_e):
                        is_sz = 1
                        break
                        
                line_str = f"{basename}.edf_{i}.h5,{is_sz}\n"
                if is_sz == 1:
                    sz_lines.append(line_str)
                else:
                    nosz_lines.append(line_str)
                    
        # Write the marker files
        # TUSZ uses 'eval', but EvoBrain names the split 'test'
        split_name = "test" if split == "eval" else split
        
        sz_file = os.path.join(out_dir, f"{split_name}Set_seq2seq_{clip_len}s_sz.txt")
        nosz_file = os.path.join(out_dir, f"{split_name}Set_seq2seq_{clip_len}s_nosz.txt")
        
        with open(sz_file, 'w') as f:
            f.writelines(sz_lines)
            
        with open(nosz_file, 'w') as f:
            f.writelines(nosz_lines)
            
        print(f" -> {split_name}: Generated {len(sz_lines)} seizure clips and {len(nosz_lines)} non-seizure clips.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Marker Files for TUSZ v1.5.2")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to raw edf files (e.g. /home/.../TUSZ/edf)")
    parser.add_argument("--out_dir", type=str, default="./data/file_markers_detection", help="Output directory for marker files")
    
    args = parser.parse_args()
    
    # Generate for both 60s and 12s just to be safe
    print("Generating 60s markers...")
    generate_markers(args.raw_data_dir, args.out_dir, 60)
    
    print("\nGenerating 12s markers...")
    generate_markers(args.raw_data_dir, args.out_dir, 12)
    
    print("\nDONE! You can now run main.py!")
