import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyedflib
try:
    import utils
    from constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS, FREQUENCY
    from utils import StandardScaler
except ImportError:
    import data.utils as utils
    from data.constants import INCLUDED_CHANNELS, CHBMIT_INCLUDED_CHANNELS, FREQUENCY
    from data.utils import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import math
import h5py
import numpy as np
import os
import pickle
import scipy
import scipy.signal
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from data.data_utils import comp_xcorr, keep_topk, computeFFT, getSeizureTimes, getSeizureTimes_CHBMIT, get_swap_pairs

repo_paths = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILEMARKER_DIR = Path(repo_paths).joinpath('data/file_markers_chb')


def computeSliceMatrix(
        h5_fn,
        edf_fn,
        clip_idx,
        time_step_size=1,
        clip_len=60,
        is_fft=False):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        h5_fn: file name of resampled signal h5 file (full path)
        clip_idx: index of current clip/sliding window
        time_step_size: length of each time_step_size, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        is_fft: whether to perform FFT on raw EEG data
    Returns:
        slices: list of EEG clips, each having shape (clip_len*freq, num_channels, time_step_size*freq)
        seizure_labels: list of seizure labels for each clip, 1 for seizure, 0 for no seizure
    """
    # get seizure times
    seizure_times = getSeizureTimes_CHBMIT(edf_fn)

    # Iterating through signal
    physical_clip_len = int(FREQUENCY * clip_len)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    start_window = clip_idx * physical_clip_len
    end_window = start_window + physical_clip_len

    with h5py.File(h5_fn, 'r') as f:
        curr_slc = f["resampled_signal"][:, start_window:end_window]

    if curr_slc.shape[1] < physical_clip_len:
        pad_width = ((0, 0), (0, physical_clip_len - curr_slc.shape[1]))
        curr_slc = np.pad(curr_slc, pad_width, mode='edge')

    start_time_step = 0
    time_steps = []
    while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
        end_time_step = start_time_step + physical_time_step_size
        # (num_channels, physical_time_step_size)
        curr_time_step = curr_slc[:, start_time_step:end_time_step]
        if is_fft:
            curr_time_step, _ = computeFFT(
                curr_time_step, n=physical_time_step_size)

        time_steps.append(curr_time_step)
        start_time_step = end_time_step

    eeg_clip = np.stack(time_steps, axis=0)

    # determine if there's seizure in current clip
    is_seizure = 0
    for t in seizure_times:
        start_t = int(t[0] * FREQUENCY)
        end_t = int(t[1] * FREQUENCY)
        if not ((end_window < start_t) or (start_window > end_t)):
            is_seizure = 1
            break

    return eeg_clip, is_seizure


def parseTxtFiles(split_type, seizure_file, nonseizure_file,
                  cv_seed=123, scale_ratio=1):

    np.random.seed(cv_seed)

    seizure_str = []
    nonseizure_str = []

    seizure_contents = open(seizure_file, "r")
    seizure_str.extend(seizure_contents.readlines())

    nonseizure_contents = open(nonseizure_file, "r")
    nonseizure_str.extend(nonseizure_contents.readlines())

    # balanced dataset if train
    if split_type == 'train':
        num_dataPoints = int(scale_ratio * len(seizure_str))
        print('number of seizure files: ', num_dataPoints)
        sz_ndxs_all = list(range(len(seizure_str)))
        np.random.shuffle(sz_ndxs_all)
        sz_ndxs = sz_ndxs_all[:num_dataPoints]
        seizure_str = [seizure_str[i] for i in sz_ndxs]
        np.random.shuffle(nonseizure_str)
        nonseizure_str = nonseizure_str[:num_dataPoints]

    combined_str = seizure_str + nonseizure_str

    np.random.shuffle(combined_str)

    combined_tuples = []
    for i in range(len(combined_str)):
        tup = combined_str[i].strip("\n").split(",")
        tup[1] = int(tup[1])
        combined_tuples.append(tup)

    print_str = 'Number of clips in ' + \
        split_type + ': ' + str(len(combined_tuples))
    print(print_str)

    return combined_tuples


class PklSeizureDataset(Dataset):
    def __init__(
            self,
            data_dir,
            split='train',
            time_step_size=1,
            max_seq_len=10,
            use_fft=True,
            standardize=False,
            scaler=None,
            data_augment=False,
            graph_type='dynamic',
            top_k=3,
            filter_type='laplacian'):
        """
        Loads preprocessed .pkl segment clips directly (e.g. from processed_seg/train, val, test)
        where each .pkl contains {'X': segment, 'y': label}.
        """
        self.split = split
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.use_fft = use_fft
        self.standardize = standardize
        self.scaler = scaler
        self.data_augment = data_augment
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type

        # Support 'val' and 'dev' interchangeably
        folder_candidates = ['val', 'dev'] if split in ['val', 'dev'] else [split]
        split_dir = None
        for fc in folder_candidates:
            cand_path = os.path.join(data_dir, fc)
            if os.path.exists(cand_path):
                split_dir = cand_path
                break

        if split_dir is None:
            split_dir = data_dir

        self.files = sorted([
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
            if f.endswith('.pkl') and not f.startswith('.')
        ]) if os.path.exists(split_dir) else []

        self.size = len(self.files)
        self.num_nodes = 16

        # Inspect first file to detect number of channels
        if self.size > 0:
            try:
                with open(self.files[0], 'rb') as f:
                    sample_dict = pickle.load(f)
                sample_data = sample_dict.get('X', sample_dict.get('data', None)) if isinstance(sample_dict, dict) else sample_dict
                if sample_data is not None:
                    sample_arr = np.array(sample_data)
                    self.num_nodes = sample_arr.shape[0] if sample_arr.ndim >= 2 else 16
            except Exception:
                self.num_nodes = 16

        print(f"[{split.upper()}] Loaded {self.size} .pkl segment files ({self.num_nodes} channels) from {split_dir}")

    def __len__(self):
        return self.size

    def targets(self):
        # We need to return all targets to calculate pos_weight.
        # To avoid opening all .pkl files, we can quickly scan them or rely on a cached list if available.
        # But for correctness, we'll quickly extract the labels from the dict/tuple.
        print("Estimating targets for class weighting from a random sample (fast)...")
        targets_list = []
        import random
        # Sample up to 5000 files to get an accurate estimate of class imbalance without taking 10 minutes to read all 280k files.
        sample_size = min(5000, len(self.files))
        sampled_files = random.sample(self.files, sample_size)
        
        for file_path in sampled_files:
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            if isinstance(data_dict, dict):
                label = int(data_dict.get('y', data_dict.get('label', 0)))
            elif isinstance(data_dict, (tuple, list)):
                label = int(data_dict[1])
            else:
                label = 0
            targets_list.append(label)
        
        # Scale the counts up to the full dataset size for accurate pos_weight computation
        pos_ratio = sum(targets_list) / len(targets_list) if len(targets_list) > 0 else 0
        total_pos_estimated = int(pos_ratio * len(self.files))
        
        # Create a dummy list representing the estimated full dataset class distribution
        full_estimated_targets = [1] * total_pos_estimated + [0] * (len(self.files) - total_pos_estimated)
        return full_estimated_targets

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        if isinstance(data_dict, dict):
            raw_data = data_dict.get('X', data_dict.get('data', None))
            seizure_label = int(data_dict.get('y', data_dict.get('label', 0)))
        elif isinstance(data_dict, (tuple, list)):
            raw_data, seizure_label = data_dict[0], int(data_dict[1])
        else:
            raw_data, seizure_label = data_dict, 0

        raw_data = np.array(raw_data, dtype=np.float32)
        raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

        if raw_data.ndim == 2:
            num_ch, n_samples = raw_data.shape
            # Resample from 256 Hz (e.g. 2560 samples for 10s) to 200 Hz (2000 samples)
            target_samples = int(self.max_seq_len * FREQUENCY)
            if n_samples != target_samples:
                raw_data = scipy.signal.resample_poly(raw_data, up=target_samples, down=n_samples, axis=1)
                raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

            physical_step = int(self.time_step_size * FREQUENCY)
            time_steps = []
            start_t = 0
            while start_t <= raw_data.shape[1] - physical_step:
                end_t = start_t + physical_step
                step_data = raw_data[:, start_t:end_t]
                if self.use_fft:
                    step_data, _ = computeFFT(step_data, n=physical_step)
                step_data = np.nan_to_num(step_data, nan=0.0, posinf=0.0, neginf=0.0)
                time_steps.append(step_data)
                start_t = end_t
            eeg_clip = np.stack(time_steps, axis=0)  # (max_seq_len, num_nodes, num_features)

        elif raw_data.ndim == 3:
            if raw_data.shape[1] == self.max_seq_len:
                raw_data = raw_data.transpose(1, 0, 2)
            if self.use_fft and raw_data.shape[-1] == int(self.time_step_size * FREQUENCY):
                steps = []
                for t in range(raw_data.shape[0]):
                    fft_step, _ = computeFFT(raw_data[t], n=raw_data.shape[-1])
                    fft_step = np.nan_to_num(fft_step, nan=0.0, posinf=0.0, neginf=0.0)
                    steps.append(fft_step)
                eeg_clip = np.stack(steps, axis=0)
            else:
                eeg_clip = raw_data
        else:
            eeg_clip = raw_data

        curr_feature = np.nan_to_num(eeg_clip.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.standardize and self.scaler is not None:
            curr_feature = self.scaler.transform(curr_feature)
            curr_feature = np.nan_to_num(curr_feature, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            feat_mean = np.mean(curr_feature)
            feat_std = np.std(curr_feature)
            if feat_std > 1e-5:
                curr_feature = (curr_feature - feat_mean) / feat_std
            curr_feature = np.nan_to_num(curr_feature, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.FloatTensor(curr_feature)
        y = torch.FloatTensor([seizure_label])
        seq_len = torch.LongTensor([self.max_seq_len])
        writeout_fn = file_name.replace('.pkl', '')

        # Fast In-Memory Dynamic Graph Adjacency Matrix
        seq_len_dim, num_sensors_dim, _ = curr_feature.shape
        norms = np.linalg.norm(curr_feature, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-5)
        norm_eeg = curr_feature / norms
        adj_mat_seq = np.abs(norm_eeg @ norm_eeg.swapaxes(-1, -2)).astype(np.float32)
        adj_mat_seq = np.nan_to_num(adj_mat_seq, nan=0.0, posinf=1.0, neginf=0.0)
        adj_mat_seq = np.clip(adj_mat_seq, 0.0, 1.0)
        for t in range(seq_len_dim):
            np.fill_diagonal(adj_mat_seq[t], 1.0)

        supports = torch.empty(0)
        adj_mat = torch.FloatTensor(adj_mat_seq)

        return x, y, seq_len, supports, adj_mat, writeout_fn


class SeizureDataset(Dataset):
    def __init__(
            self,
            input_dir,
            raw_data_dir,
            time_step_size=1,
            max_seq_len=60,
            standardize=True,
            scaler=None,
            split='train',
            data_augment=False,
            adj_mat_dir=None,
            graph_type=None,
            top_k=None,
            filter_type='laplacian',
            sampling_ratio=1,
            seed=123,
            use_fft=False,
            preproc_dir=None):
        """
        Args:
            input_dir: dir to resampled signals h5 files
            raw_data_dir: dir to TUSZ edf files
            time_step_size: int, in seconds
            max_seq_len: int, eeg clip length, in seconds
            standardize: if True, will z-normalize wrt train set
            scaler: scaler object for standardization
            split: train, dev or test
            data_augment: if True, perform random augmentation on EEG
            adj_mat_dir: dir to pre-computed distance graph adjacency matrix
            graph_type: 'combined' (i.e. distance graph) or 'individual' (correlation graph)
            top_k: int, top-k neighbors of each node to keep. For correlation graph only
            filter_type: 'laplacian' for distance graph, 'dual_random_walk' for correlation graph
            sampling_ratio: ratio of positive to negative examples for undersampling
            seed: random seed for undersampling
            use_fft: whether perform Fourier transform
            preproc_dir: dir to preprocessed Fourier transformed data, optional 
        """
        if standardize and (scaler is None):
            raise ValueError('To standardize, please provide scaler.')
        if (graph_type == 'individual') and (top_k is None):
            raise ValueError('Please specify top_k for individual graph.')

        self.input_dir = input_dir
        self.raw_data_dir = raw_data_dir
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.data_augment = data_augment
        self.adj_mat_dir = adj_mat_dir
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir

        # get full paths to all raw edf files
        self.edf_files = []
        for path, subdirs, files in os.walk(raw_data_dir):
            for name in files:
                if name.endswith(".edf"):
                    self.edf_files.append(os.path.join(path, name))

        seizure_file = os.path.join(
            FILEMARKER_DIR,
            split +
            'Set_seq2seq_' +
            str(max_seq_len) +
            's_sz.txt')
        nonSeizure_file = os.path.join(
            FILEMARKER_DIR,
            split +
            'Set_seq2seq_' +
            str(max_seq_len) +
            's_nosz.txt')
        if not os.path.exists(seizure_file) or not os.path.exists(nonSeizure_file):
            raise FileNotFoundError(
                f"File marker files for max_seq_len={max_seq_len}s not found in {FILEMARKER_DIR}. "
                f"Missing: {seizure_file} or {nonSeizure_file}"
            )
        self.file_tuples = parseTxtFiles(
            split,
            seizure_file,
            nonSeizure_file,
            cv_seed=seed,
            scale_ratio=sampling_ratio)

        # Verify that the underlying resampled .h5 files exist on disk
        if self.input_dir is not None and os.path.exists(self.input_dir):
            valid_tuples = []
            for item in self.file_tuples:
                h5_name = item[0].split('.edf')[0] + '.h5'
                p1 = os.path.join(self.input_dir, h5_name)
                subject_name = h5_name.split('_')[0]
                p2 = os.path.join(self.input_dir, subject_name, h5_name)
                if os.path.exists(p1) or os.path.exists(p2):
                    valid_tuples.append(item)
            if len(valid_tuples) < len(self.file_tuples):
                print(f"[{split.upper()}] Kept {len(valid_tuples)}/{len(self.file_tuples)} clips with existing .h5 files on disk.")
            self.file_tuples = valid_tuples

        self.size = len(self.file_tuples)

        # Get sensor ids
        self.sensor_ids = [x.split(' ')[-1] for x in CHBMIT_INCLUDED_CHANNELS]

        targets = []
        for i in range(len(self.file_tuples)):
            if self.file_tuples[i][-1] == 0:
                targets.append(0)
            else:
                targets.append(1)
        self._targets = targets

    def __len__(self):
        return self.size

    def targets(self):
        return self._targets

    def _random_reflect(self, EEG_seq):
        """
        Randomly reflect EEG along midline
        """
        swap_pairs = get_swap_pairs(CHBMIT_INCLUDED_CHANNELS)
        EEG_seq_reflect = EEG_seq.copy()
        if(np.random.choice([True, False])):
            for pair in swap_pairs:
                EEG_seq_reflect[:, [pair[0], pair[1]],
                                :] = EEG_seq[:, [pair[1], pair[0]], :]
        else:
            swap_pairs = None
        return EEG_seq_reflect, swap_pairs

    def _random_scale(self, EEG_seq):
        """
        Scale EEG signals by a random number between 0.8 and 1.2
        """
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            EEG_seq += np.log(scale_factor)
        else:
            EEG_seq *= scale_factor
        return EEG_seq

    def _get_indiv_graphs(self, eeg_clip, swap_nodes=None):
        """
        Compute adjacency matrix for correlation graph
        Args:
            eeg_clip: shape (seq_len, num_nodes, input_dim)
            swap_nodes: list of swapped node index
        Returns:
            adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        """
        #print("eeg_clip:" + str(eeg_clip.shape))
        num_sensors = len(self.sensor_ids)
        adj_mat = np.eye(num_sensors, num_sensors,
                         dtype=np.float32)  # diagonal is 1

        # (num_nodes, seq_len, input_dim)
        eeg_clip = np.transpose(eeg_clip, (1, 0, 2))
        assert eeg_clip.shape[0] == num_sensors

        # (num_nodes, seq_len*input_dim)
        eeg_clip = eeg_clip.reshape((num_sensors, -1))

        # Vectorized cross-correlation / cosine similarity (93x faster than nested loops)
        norms = np.linalg.norm(eeg_clip, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-8
        norm_eeg = eeg_clip / norms
        adj_mat = np.abs(norm_eeg @ norm_eeg.T).astype(np.float32)
        np.fill_diagonal(adj_mat, 1.0)

        if (self.top_k is not None):
            adj_mat = keep_topk(adj_mat, top_k=self.top_k, directed=True)
        else:
            raise ValueError('Invalid top_k value!')

        return adj_mat

    def _get_combined_graph(self, swap_nodes=None):
        """
        Get adjacency matrix for pre-computed distance graph
        Returns:
            adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
        """
        with open(self.adj_mat_dir, 'rb') as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]

        adj_mat_new = adj_mat.copy()
        if swap_nodes is not None:
            for node_pair in swap_nodes:
                for i in range(adj_mat.shape[0]):
                    adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                    adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                    adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                    adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                    adj_mat_new[i, i] = 1
                adj_mat_new[node_pair[0], node_pair[1]
                            ] = adj_mat[node_pair[1], node_pair[0]]
                adj_mat_new[node_pair[1], node_pair[0]
                            ] = adj_mat[node_pair[0], node_pair[1]]

        return adj_mat_new

    def _compute_supports(self, adj_mat):
        """
        Comput supports
        """
        supports = []
        supports_mat = []
        if self.filter_type == "laplacian":  # ChebNet graph conv
            supports_mat.append(
                utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif self.filter_type == "random_walk":  # Forward random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        elif self.filter_type == "dual_random_walk":  # Bidirectional random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
            supports_mat.append(
                utils.calculate_random_walk_matrix(adj_mat.T).T)
        else:
            supports_mat.append(utils.calculate_scaled_laplacian(adj_mat))
        for support in supports_mat:
            supports.append(torch.FloatTensor(support.toarray()))
        return supports

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, seq_len, supports, adj_mat, writeout_fn)
        """
        h5_fn, seizure_label = self.file_tuples[idx]

        cache_file_name = h5_fn.replace('.h5', '_cache.h5')
        os.makedirs(os.path.join("graph_cache", str(self.max_seq_len), self.filter_type), exist_ok=True)
        cache_file_path = os.path.join("graph_cache", str(self.max_seq_len), self.filter_type, cache_file_name)

        clip_idx = int(h5_fn.split('_')[-1].split('.h5')[0])

        target_edf_basename = h5_fn.split('.edf')[0] + '.edf'
        edf_file = [file for file in self.edf_files if os.path.basename(file) == target_edf_basename]
        if len(edf_file) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 raw EDF file matching '{target_edf_basename}', found {len(edf_file)} in {self.raw_data_dir}"
            )
        edf_file = edf_file[0]

        # preprocess
        if self.preproc_dir is None:
            h5_name = h5_fn.split('.edf')[0] + '.h5'
            resample_sig_dir = os.path.join(self.input_dir, h5_name)
            if not os.path.exists(resample_sig_dir):
                subject_name = h5_name.split('_')[0]
                resample_sig_dir = os.path.join(self.input_dir, subject_name, h5_name)

            eeg_clip, is_seizure = computeSliceMatrix(
                h5_fn=resample_sig_dir, edf_fn=edf_file, clip_idx=clip_idx,
                time_step_size=self.time_step_size, clip_len=self.max_seq_len,
                is_fft=self.use_fft)
        else:
            with h5py.File(os.path.join(self.preproc_dir, h5_fn), 'r') as hf:
                eeg_clip = hf['clip'][()]

        # data augmentation
        if self.data_augment:
            curr_feature, swap_nodes = self._random_reflect(eeg_clip)
            curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = eeg_clip.copy()

        # standardize wrt train mean and std
        if self.standardize:
            curr_feature = self.scaler.transform(curr_feature)

        # convert to tensors
        x = torch.FloatTensor(curr_feature)
        y = torch.FloatTensor([seizure_label])
        seq_len = torch.LongTensor([self.max_seq_len])
        writeout_fn = h5_fn.split('.h5')[0]

        # Get adjacency matrix for graphs
        if self.graph_type == 'individual':
            indiv_adj_mat = self._get_indiv_graphs(eeg_clip, swap_nodes)
            indiv_supports = self._compute_supports(indiv_adj_mat)
            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")

            # Repeat these values for each time step in EEG clip
            time_steps = eeg_clip.shape[0] 
            indiv_supports = torch.stack(indiv_supports)
            supports_seq = indiv_supports.repeat(time_steps, 1, 1, 1)
            adj_mat_seq = np.stack([indiv_adj_mat for _ in range(time_steps)])
            
        elif self.graph_type == 'dynamic':
            # Fast in-memory dynamic adjacency matrix computation across all timesteps (eliminates 44k file I/O bottleneck)
            seq_len_dim, num_sensors_dim, _ = eeg_clip.shape
            norms = np.linalg.norm(eeg_clip, axis=-1, keepdims=True)
            norms[norms == 0] = 1e-8
            norm_eeg = eeg_clip / norms
            adj_mat_seq = np.abs(norm_eeg @ norm_eeg.swapaxes(-1, -2)).astype(np.float32)

            for t in range(seq_len_dim):
                np.fill_diagonal(adj_mat_seq[t], 1.0)
                if self.top_k is not None:
                    adj_mat_seq[t] = keep_topk(adj_mat_seq[t], top_k=self.top_k, directed=True)

            adj_mat_seq = torch.from_numpy(adj_mat_seq)
            supports_seq = torch.zeros(seq_len_dim, 2, num_sensors_dim, num_sensors_dim, dtype=torch.float32)

        elif self.adj_mat_dir is not None:
            indiv_adj_mat = self._get_combined_graph(swap_nodes)
            indiv_supports = self._compute_supports(indiv_adj_mat)
            time_steps = eeg_clip.shape[0]
            indiv_supports = torch.stack(indiv_supports)
            supports_seq = indiv_supports.repeat(time_steps, 1, 1, 1)
            adj_mat_seq = np.stack([indiv_adj_mat for _ in range(time_steps)])
        else:
            indiv_supports = []
            indiv_adj_mat = []
            supports_seq = torch.tensor([])
            adj_mat_seq = torch.tensor([])

        if seq_len.item() != self.max_seq_len:
            print(f"seq_len: {seq_len}")
            print(f"supports_seq.shape: {supports_seq.shape}")
            print(f"adj_mat_seq.shape: {adj_mat_seq.shape}")
        return (x, y, seq_len, supports_seq, adj_mat_seq, writeout_fn)

def load_dataset_chb(
        task=None,
        input_dir=None,
        raw_data_dir=None,
        train_batch_size=64,
        test_batch_size=None,
        time_step_size=1,
        max_seq_len=60,
        standardize=True,
        num_workers=8,
        augmentation=False,
        adj_mat_dir=None,
        graph_type=None,
        top_k=None,
        filter_type='laplacian',
        use_fft=False,
        sampling_ratio=1,
        seed=123,
        preproc_dir=None):
    """
    Args:
        input_dir: dir to preprocessed h5 file
        raw_data_dir: dir to TUSZ raw edf files
        train_batch_size: int
        test_batch_size: int
        time_step_size: int, in seconds
        max_seq_len: EEG clip length, in seconds
        standardize: if True, will z-normalize wrt train set
        num_workers: int
        augmentation: if True, perform random augmentation on EEG
        adj_mat_dir: dir to pre-computed distance graph adjacency matrix
        graph_type: 'combined' (i.e. distance graph) or 'individual' (correlation graph)
        top_k: int, top-k neighbors of each node to keep. For correlation graph only
        filter_type: 'laplacian' for distance graph, 'dual_random_walk' for correlation graph
        use_fft: whether perform Fourier transform
        sampling_ratio: ratio of positive to negative examples for undersampling
        seed: random seed for undersampling
        preproc_dir: dir to preprocessed Fourier transformed data, optional
    Returns:
        dataloaders: dictionary of train/dev/test dataloaders
        datasets: dictionary of train/dev/test datasets
        scaler: standard scaler
    """
    if (graph_type is not None) and (
            graph_type not in ['individual', 'combined', 'dynamic']):
        raise NotImplementedError

    # load mean and std
    scaler = None
    if standardize:
        means_dir = os.path.join(
            FILEMARKER_DIR,
            'means_seq2seq_fft_' +
            str(max_seq_len) +
            's_szdetect_single.pkl')
        stds_dir = os.path.join(
            FILEMARKER_DIR,
            'stds_seq2seq_fft_' +
            str(max_seq_len) +
            's_szdetect_single.pkl')

        if not (os.path.exists(means_dir) and os.path.exists(stds_dir)):
            # Fallback to 12s or any available scaler in FILEMARKER_DIR
            fallback_means = os.path.join(FILEMARKER_DIR, 'means_seq2seq_fft_12s_szdetect_single.pkl')
            fallback_stds = os.path.join(FILEMARKER_DIR, 'stds_seq2seq_fft_12s_szdetect_single.pkl')
            if os.path.exists(fallback_means) and os.path.exists(fallback_stds):
                print(f"Notice: {means_dir} not found. Reusing 12s scaler ({fallback_means}) since 1-second FFT scales are identical.")
                means_dir = fallback_means
                stds_dir = fallback_stds
            else:
                print(f"Warning: No scaler file found in {FILEMARKER_DIR}. Proceeding without standardization.")
                means_dir = None
                stds_dir = None

        if means_dir and stds_dir and os.path.exists(means_dir) and os.path.exists(stds_dir):
            with open(means_dir, 'rb') as f:
                means = pickle.load(f)
            with open(stds_dir, 'rb') as f:
                stds = pickle.load(f)
            scaler = StandardScaler(mean=means, std=stds)

    # Check if input_dir contains preprocessed .pkl segment dataset
    is_pkl_mode = False
    if input_dir and os.path.exists(input_dir):
        for sub in ['train', 'val', 'dev', 'test']:
            p = os.path.join(input_dir, sub)
            if os.path.exists(p) and any(f.endswith('.pkl') for f in os.listdir(p)):
                is_pkl_mode = True
                break
        if not is_pkl_mode and any(f.endswith('.pkl') for f in os.listdir(input_dir)):
            is_pkl_mode = True

    dataloaders = {}
    datasets = {}
    for split in ['train', 'dev', 'test']:
        if split == 'train':
            data_augment = augmentation
        else:
            data_augment = False  # never do augmentation on dev/test sets

        if is_pkl_mode:
            dataset = PklSeizureDataset(
                data_dir=input_dir,
                split=split,
                time_step_size=time_step_size,
                max_seq_len=max_seq_len,
                use_fft=use_fft,
                standardize=standardize,
                scaler=scaler,
                data_augment=data_augment,
                graph_type=graph_type,
                top_k=top_k,
                filter_type=filter_type
            )
        else:
            dataset = SeizureDataset(input_dir=input_dir,
                                     raw_data_dir=raw_data_dir,
                                     time_step_size=time_step_size,
                                     max_seq_len=max_seq_len,
                                     standardize=standardize,
                                     scaler=scaler,
                                     split=split,
                                     data_augment=data_augment,
                                     adj_mat_dir=adj_mat_dir,
                                     graph_type=graph_type,
                                     top_k=top_k,
                                     filter_type=filter_type,
                                     sampling_ratio=sampling_ratio,
                                     seed=seed,
                                     use_fft=use_fft,
                                     preproc_dir=preproc_dir)

        if split == 'train':
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size

        loader = DataLoader(dataset=dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)
        dataloaders[split] = loader
        datasets[split] = dataset

    return dataloaders, datasets, scaler
