import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS, FREQUENCY
from data_utils import resampleData, getEDFsignals, getOrderedChannels
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

def process_single_file(edf_fn, save_dir, dataset="TUSZ"):
    save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
    if os.path.exists(save_fn):
        return None  # Skip if already exists
    try:
        f = read_with_retry(edf_fn)

        channels = CHBMIT_INCLUDED_CHANNELS if dataset == "CHBMIT" else INCLUDED_CHANNELS
        orderedChannels = getOrderedChannels(
            edf_fn, False, f.getSignalLabels(), channels, dataset=dataset
        )
        signals = getEDFsignals(f)
        signal_array = np.array(signals[orderedChannels, :])
        sample_freq = f.getSampleFrequency(0)
        if sample_freq != FREQUENCY:
            signal_array = resampleData(
                signal_array,
                to_freq=FREQUENCY,
                window_size=int(signal_array.shape[1] / sample_freq),
            )

        with h5py.File(save_fn, "w") as hf:
            hf.create_dataset("resampled_signal", data=signal_array)
            hf.create_dataset("resample_freq", data=FREQUENCY)
        return None
    except Exception as e:
        print(f"An error occurred: {e}")  
        print(f"Failed to process {edf_fn}")
        return edf_fn

def resample_all(raw_edf_dir, save_dir, dataset="TUSZ"):
    os.makedirs(save_dir, exist_ok=True)
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if name.endswith(".edf"):
                edf_files.append(os.path.join(path, name))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    failed_files = []
    num_cores = multiprocessing.cpu_count()
    print(f"Starting multiprocessing across {num_cores} CPU cores...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_file, fn, save_dir, dataset): fn for fn in edf_files}
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(futures), total=len(edf_files)):
            result = future.result()
            if result is not None:
                failed_files.append(result)

    print("DONE. {} files failed.".format(len(failed_files)))


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
    args = parser.parse_args()

    resample_all(args.raw_edf_dir, args.save_dir, args.dataset)
