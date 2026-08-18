import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS, FREQUENCY
    from data_utils import resampleData, getEDFsignals, getOrderedChannels
except ImportError:
    from data.constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS, FREQUENCY
    from data.data_utils import resampleData, getEDFsignals, getOrderedChannels
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import h5py
import scipy
import time


def read_with_retry(file_name, max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            f = pyedflib.EdfReader(file_name)
            return f
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay) 
    raise Exception(f"All {max_retries} retries failed.")

def is_valid_h5(filepath, expected_channels=None):
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        return False
    try:
        with h5py.File(filepath, "r") as hf:
            if "resampled_signal" in hf and "resample_freq" in hf:
                if expected_channels is not None:
                    return hf["resampled_signal"].shape[0] == expected_channels
                return True
            return False
    except Exception:
        return False

def process_single_file(edf_fn, save_dir, dataset="TUSZ"):
    # Determine the split (train, dev, eval) from the EDF path
    split_folder = ""
    for s in ["train", "dev", "eval"]:
        if f"/{s}/" in edf_fn or edf_fn.endswith(f"/{s}"):
            split_folder = s
            break
            
    # Avoid duplicating the folder name if save_dir already ends with it
    out_dir = save_dir
    if split_folder and not save_dir.rstrip('/').endswith(split_folder):
        out_dir = os.path.join(save_dir, split_folder)
        
    os.makedirs(out_dir, exist_ok=True)
    save_fn = os.path.join(out_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
    
    channels = CHBMIT_INCLUDED_CHANNELS if dataset in ["CHBMIT", "CHB-MIT"] else INCLUDED_CHANNELS
    if is_valid_h5(save_fn, expected_channels=len(channels)):
        return None  # Skip if already exists and valid with correct channels

    tmp_save_fn = save_fn + f".tmp.{os.getpid()}"
    f = None
    try:
        f = read_with_retry(edf_fn)

        channels = CHBMIT_INCLUDED_CHANNELS if dataset in ["CHBMIT", "CHB-MIT"] else INCLUDED_CHANNELS
        raw_labels = f.getSignalLabels()
        sample_freq = f.getSampleFrequency(0)

        try:
            orderedChannels = getOrderedChannels(
                edf_fn, False, raw_labels, channels, dataset=dataset
            )
            signals = getEDFsignals(f)
            signal_array = np.array(signals[orderedChannels, :], dtype=np.float32)
        except Exception as err:
            if dataset in ["CHBMIT", "CHB-MIT"]:
                # Synthesize 18 bipolar channels from reference/monopolar electrodes (e.g. chb12)
                label_map = {}
                for idx, l in enumerate(raw_labels):
                    clean = l.upper().replace("EEG", "").replace("-CS2", "").replace(" ", "").rstrip(".")
                    if clean == "01":
                        clean = "O1"
                    if clean:
                        label_map[clean] = idx

                bipolar_list = []
                for ch_pair in channels:
                    ch1, ch2 = ch_pair.split("-")
                    if ch1 in label_map and ch2 in label_map:
                        s1 = f.readSignal(label_map[ch1])
                        s2 = f.readSignal(label_map[ch2])
                        bipolar_list.append(s1 - s2)
                    else:
                        raise Exception(f"Channel pair {ch_pair} could not be derived: {err}")
                signal_array = np.array(bipolar_list, dtype=np.float32)
            else:
                raise err
        f._close()
        f = None
        if sample_freq != FREQUENCY:
            signal_array = resampleData(
                signal_array,
                to_freq=FREQUENCY,
                window_size=int(signal_array.shape[1] / sample_freq),
                orig_freq=sample_freq,
            )

        with h5py.File(tmp_save_fn, "w") as hf:
            hf.create_dataset("resampled_signal", data=signal_array)
            hf.create_dataset("resample_freq", data=FREQUENCY)
        os.replace(tmp_save_fn, save_fn)
        return None
    except Exception as e:
        if os.path.exists(tmp_save_fn):
            try:
                os.remove(tmp_save_fn)
            except Exception:
                pass
        print(f"An error occurred: {e}")  
        print(f"Failed to process {edf_fn}")
        return (edf_fn, str(e))
    finally:
        if f is not None:
            try:
                f._close()
            except Exception:
                pass

def resample_all(raw_edf_dir, save_dir, dataset="TUSZ", num_cores=None):
    os.makedirs(save_dir, exist_ok=True)
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if name.endswith(".edf") and not name.startswith(".") and ":Zone.Identifier" not in name:
                edf_files.append(os.path.join(path, name))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    import json
    
    failed_files = []
    # Safe default: 4 workers prevents RAM exhaustion during heavy scipy FFT resampling on 1-hour files
    if num_cores is None:
        num_cores = min(4, max(1, multiprocessing.cpu_count() // 4))
    print(f"Starting multiprocessing safely across {num_cores} worker processes for {len(edf_files)} files...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, fn, save_dir, dataset): fn for fn in edf_files}
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(futures), total=len(edf_files)):
            result = future.result()
            if result is not None:
                failed_files.append(result)

    print(f"DONE. {len(failed_files)} files failed.")
    if failed_files:
        failed_json = os.path.join(save_dir, "failed_files.json")
        failed_txt = os.path.join(save_dir, "list_of_all_the_failed_files.txt")
        with open(failed_json, "w") as f:
            json.dump([{"file": fn, "error": err} for fn, err in failed_files], f, indent=2)
        with open(failed_txt, "w") as f:
            for fn, err in failed_files:
                f.write(f"Failed to process {fn}\nAn error occurred: {err}\n")
        print(f"Failed files log saved to {failed_json} and {failed_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample.")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        default=None,
        help="Full path to raw edf files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Full path to dir to save resampled signals.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="TUSZ",
        help="Dataset name: TUSZ or CHBMIT",
    )
    parser.add_argument(
        "--num_cores",
        type=int,
        default=None,
        help="Number of cores for multiprocessing",
    )
    args = parser.parse_args()

    resample_all(args.raw_edf_dir, args.save_dir, args.dataset, args.num_cores)
