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


def getOrderedChannels(file_name, verbose, labels_object, channel_names, dataset="TUSZ"):
    labels = list(labels_object)
    if dataset == "TUSZ":
        for i in range(len(labels)):
            labels[i] = labels[i].split("-")[0]
            
    # Convert all labels to uppercase for case-insensitive matching
    labels = [label.upper() for label in labels]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch.upper()))
        except Exception:
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
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times


def getSeizureTimes_CHBMIT(file_name):
    import re
    # file_name example: /path/to/raw/.../chb09/chb09_06.edf
    edf_basename = os.path.basename(file_name)
    subject = edf_basename.split('_')[0]
    summary_file = os.path.join(os.path.dirname(file_name), f"{subject}-summary.txt")
    
    seizure_times = []
    if not os.path.exists(summary_file):
        return seizure_times
        
    with open(summary_file, 'r') as f:
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
                
        if in_target_file:
            # Match "Seizure Start Time: XXX seconds" or "Seizure 1 Start Time: XXX seconds"
            start_match = re.search(r'Seizure\s+(?:\d+\s+)?Start Time:\s+(\d+)', line)
            end_match = re.search(r'Seizure\s+(?:\d+\s+)?End Time:\s+(\d+)', line)
            
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
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except Exception:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled


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