import numpy as np
import random
import os
import sys

sys.path.append("../")
import pyedflib
from data.constants import INCLUDED_CHANNELS, FREQUENCY, ALL_LABEL_DICT
from scipy.fftpack import fft
from scipy.signal import resample, correlate


def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P


def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Args:
        channels: list of channel names
    Returns:
        list of tuples, each a pair of channel indices being swapped
    """
    swap_pairs = []
    pairs = [
        ("EEG FP1", "EEG FP2"),
        ("EEG Fp1", "EEG Fp2"),
        ("EEG F3", "EEG F4"),
        ("EEG F7", "EEG F8"),
        ("EEG C3", "EEG C4"),
        ("EEG T3", "EEG T4"),
        ("EEG T5", "EEG T6"),
        ("EEG O1", "EEG O2"),
        ("FP1-F7", "FP2-F8"),
        ("F7-T7", "F8-T8"),
        ("T7-P7", "T8-P8"),
        ("P7-O1", "P8-O2"),
        ("FP1-F3", "FP2-F4"),
        ("F3-C3", "F4-C4"),
        ("C3-P3", "C4-P4"),
        ("P3-O1", "P4-O2")
    ]
    
    for left, right in pairs:
        if left in channels and right in channels:
            swap_pairs.append((channels.index(left), channels.index(right)))

    return swap_pairs


MODERN_CHANNEL_ALIASES = {
    'EEG T3': 'EEG T7',
    'EEG T4': 'EEG T8',
    'EEG T5': 'EEG P7',
    'EEG T6': 'EEG P8',
    'EEG T7': 'EEG T3',
    'EEG T8': 'EEG T4',
    'EEG P7': 'EEG T5',
    'EEG P8': 'EEG T6',
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6',
    'F7-T3': 'F7-T7',
    'F7-T7': 'F7-T3',
    'T3-T5': 'T7-P7',
    'T7-P7': 'T3-T5',
    'T5-O1': 'P7-O1',
    'P7-O1': 'T5-O1',
    'F8-T4': 'F8-T8',
    'F8-T8': 'F8-T4',
    'T4-T6': 'T8-P8',
    'T8-P8': 'T4-T6',
    'T6-O2': 'P8-O2',
    'P8-O2': 'T6-O2',
}

def getOrderedChannels(file_name, verbose, labels_object, channel_names, dataset="TUSZ"):
    raw_labels = list(labels_object)
    
    if dataset == "TUSZ":
        labels = [l.split("-")[0].strip().upper() for l in raw_labels]
    elif dataset in ["CHBMIT", "CHB"]:
        labels = []
        for l in raw_labels:
            clean_l = l.upper().replace('EEG ', '').replace('EEG', '').strip().rstrip('.').replace(' ', '')
            if clean_l.endswith('-0') or clean_l.endswith('-1'):
                clean_l = clean_l[:-2]
            labels.append(clean_l)
    else:
        labels = [l.upper().strip() for l in raw_labels]

    ordered_channels = []
    for ch in channel_names:
        ch_clean = ch.upper().replace('EEG ', '').replace('EEG', '').strip().rstrip('.').replace(' ', '')
        if ch_clean in labels:
            ordered_channels.append(labels.index(ch_clean))
        elif ch.upper() in labels:
            ordered_channels.append(labels.index(ch.upper()))
        else:
            # Try standard 10-20 modern alias
            alias = MODERN_CHANNEL_ALIASES.get(ch, "") or MODERN_CHANNEL_ALIASES.get(ch_clean, "")
            alias_clean = alias.upper().replace('EEG ', '').replace('EEG', '').strip().rstrip('.').replace(' ', '')
            if alias_clean and alias_clean in labels:
                ordered_channels.append(labels.index(alias_clean))
            else:
                if verbose:
                    print(file_name + " failed to get channel " + ch)
                raise Exception("channel not match")
    return ordered_channels


def getSeizureTimes(file_name):
    """
    Args:
        file_name: edf file name
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_bi = file_name.split(".edf")[0] + ".tse_bi"
    csv_bi = file_name.split(".edf")[0] + ".csv_bi"
    tse = file_name.split(".edf")[0] + ".tse"
    csv = file_name.split(".edf")[0] + ".csv"

    anno_file = None
    if os.path.exists(tse_bi):
        anno_file = tse_bi
    elif os.path.exists(csv_bi):
        anno_file = csv_bi
    elif os.path.exists(tse):
        anno_file = tse
    elif os.path.exists(csv):
        anno_file = csv

    seizure_times = []
    if anno_file is None:
        return seizure_times

    with open(anno_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'version' in line or line.startswith("#") or not line.strip() or 'start_time' in line:
                continue
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 4 and parts[0].upper() == "TERM":
                try:
                    start, end, label = float(parts[1]), float(parts[2]), parts[3]
                    if label.lower() != 'bckg' or 'seiz' in label.lower():
                        seizure_times.append([start, end])
                except ValueError:
                    pass
            elif len(parts) >= 3:
                try:
                    start, end, label = float(parts[0]), float(parts[1]), parts[2]
                    if label.lower() != 'bckg' or 'seiz' in line.lower():
                        seizure_times.append([start, end])
                except ValueError:
                    pass
            elif len(parts) >= 2 and any(k in line.lower() for k in ["seiz", "fnsz", "gnsz", "cpsz", "spsz", "tcsz"]):
                try:
                    seizure_times.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
    return seizure_times


def getSeizureTimes_CHBMIT(file_name):
    import re
    # file_name example: /path/to/raw/.../chb09/chb09_06.edf
    edf_basename = os.path.basename(file_name)
    subject = edf_basename.split('_')[0]
    
    summary_file = os.path.join(os.path.dirname(file_name), f"{subject}-summary.txt")
    if not os.path.exists(summary_file):
        summary_file = os.path.join(os.path.dirname(file_name), f"{subject.lower()}-summary.txt")
        
    seizure_times = []
    if not os.path.exists(summary_file):
        return seizure_times
        
    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    in_target_file = False
    current_start = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            if edf_basename in line:
                in_target_file = True
            else:
                in_target_file = False
                current_start = None
                
        if in_target_file:
            start_match = re.search(r'Seizure\s*(?:\d+\s*)?Start\s*Time:\s*(\d+)', line, re.IGNORECASE)
            end_match = re.search(r'Seizure\s*(?:\d+\s*)?End\s*Time:\s*(\d+)', line, re.IGNORECASE)
            
            if start_match:
                current_start = float(start_match.group(1))
            elif end_match and current_start is not None:
                current_end = float(end_match.group(1))
                seizure_times.append([current_start, current_end])
                current_start = None
                
    return seizure_times


def getSeizureClass(file_name, target_labels_dict=None, file_type="edf"):
    """
    Args:
        file_name: file name of .edf file etc.
        target_labels_dict: dict, key is seizure class str, value is seizure class number,
                        e.g. {'fnsz': 0, 'gnsz': 1}
        file_type: "edf" or "tse"
    Returns:
        seizure_class: list of seizure class in the .edf file
    """
    label_dict = (
        target_labels_dict if target_labels_dict is not None else ALL_LABEL_DICT
    )
    target_labels = list(label_dict.keys())

    tse_file = ""
    if file_type == "edf":
        tse_file = file_name[:-4] + ".tse"
    elif file_type == "tse":
        tse_file = file_name
    else:
        raise ValueError("Unrecognized file type.")

    seizure_class = []
    with open(tse_file) as f:
        for line in f.readlines():
            if any(
                s in line for s in target_labels
            ):  # if this is one of the seizure types of interest
                seizure_str = [s for s in target_labels if s in line]
                seizure_class.append(label_dict[seizure_str[0]])
    return seizure_class


def getEDFsignals(edf):
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples), dtype=np.float32)
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except Exception:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=None, orig_freq=None):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds (optional)
        orig_freq: original frequency in Hz (optional)
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    from math import gcd
    from scipy.signal import resample_poly, resample

    num_samples = signals.shape[1]
    if orig_freq is None:
        if window_size is not None and window_size > 0:
            orig_freq = int(round(num_samples / window_size))
        else:
            orig_freq = 256  # default fallback

    if int(orig_freq) == int(to_freq):
        return signals

    try:
        g = gcd(int(to_freq), int(orig_freq))
        up = int(to_freq) // g
        down = int(orig_freq) // g
        resampled = resample_poly(signals, up, down, axis=-1)

        if window_size is not None:
            target_num = int(to_freq * window_size)
            if resampled.shape[-1] > target_num:
                resampled = resampled[..., :target_num]
            elif resampled.shape[-1] < target_num:
                pad_width = [(0, 0)] * (resampled.ndim - 1) + [(0, target_num - resampled.shape[-1])]
                resampled = np.pad(resampled, pad_width, mode='edge')
        return resampled
    except Exception:
        # Fallback to Fourier-based resample
        if window_size is not None:
            num = int(to_freq * window_size)
        else:
            num = int(num_samples * (to_freq / orig_freq))
        return resample(signals, num=num, axis=-1)


######## Graph related data utils ########
def keep_topk(adj_mat, top_k=3, directed=True):
    """ "
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors
    for each node.
    Args:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        top_k: int
        directed: whether or not a directed graph
    Returns:
        adj_mat: sparse adjacency matrix, directed graph
    """
    # Set values that are not of top-k neighbors to 0:
    adj_mat_noSelfEdge = adj_mat.copy()
    for i in range(adj_mat_noSelfEdge.shape[0]):
        adj_mat_noSelfEdge[i, i] = 0

    top_k_idx = (-adj_mat_noSelfEdge).argsort(axis=-1)[:, :top_k]

    mask = np.eye(adj_mat.shape[0], dtype=bool)
    for i in range(0, top_k_idx.shape[0]):
        for j in range(0, top_k_idx.shape[1]):
            mask[i, top_k_idx[i, j]] = 1
            if not directed:
                mask[top_k_idx[i, j], i] = 1  # symmetric

    adj_mat = mask * adj_mat
    return adj_mat


def comp_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 1D array
        y: 1D array
        mode: 'valid', 'full' or 'same',
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        normalize: If True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """
    xcorr = correlate(x, y, mode=mode)
    # the below normalization code refers to matlab xcorr function
    cxx0 = np.sum(np.absolute(x) ** 2)
    cyy0 = np.sum(np.absolute(y) ** 2)
    if normalize and (cxx0 != 0) and (cyy0 != 0):
        scale = (cxx0 * cyy0) ** 0.5
        xcorr /= scale
    if hasattr(xcorr, 'size') and xcorr.size == 1:
        return xcorr.item()
    return xcorr


######## Graph related data utils ########