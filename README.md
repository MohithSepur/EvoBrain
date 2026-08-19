# EvoBrain: Dynamic Multi-channel EEG Graph Modeling for Time-Evolving Brain Networks

Official PyTorch implementation of **EvoBrain** (*NeurIPS 2025, Spotlight*).

EvoBrain models multi-channel EEG signals as dynamic, time-evolving brain network graphs combining State Space Models (Mamba) and Graph Neural Networks (GNNs) for seizure detection, classification, and prediction.

---

## 📌 Supported Datasets

1. **TUSZ (Temple University Seizure Corpus)**: Referential 19-channel EEG montage.
   * Dataset source: [TUH EEG Seizure Corpus](https://isip.piconepress.com/projects/tuh_eeg/)
2. **CHB-MIT (Boston Children's Hospital Seizure Database)**: Longitudinal bipolar 18-channel EEG montage.
   * Dataset source: [PhysioNet CHB-MIT Database](https://physionet.org/content/chbmit/1.0.0/)

---

## ⚙️ Environment Setup

### 1. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch with CUDA Support
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install PyTorch Geometric (PyG) & Dependencies
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

### 4. Install EvoBrain Requirements
```bash
pip install -r requirements.txt
```

> **Note:** `mamba-ssm` and `causal-conv1d` install pre-compiled CUDA binaries on standard Linux x86_64 machines. No `nvcc` compiler is required at runtime.

---

## 🚀 Step-by-Step Pipeline

### Step 1: Resample Raw EEG Signals to 200 Hz
Convert raw `.edf` recordings into 200 Hz `.h5` files:

* **For TUSZ:**
  ```bash
  python ./data/resample_signals.py \
    --raw_edf_dir ./data/raw/TUSZ \
    --save_dir ./data/resampled/TUSZ \
    --dataset TUSZ
  ```

* **For CHB-MIT:**
  ```bash
  python ./data/resample_signals.py \
    --raw_edf_dir ./data/raw/CHB-MIT \
    --save_dir ./data/resampled/CHB-MIT \
    --dataset CHBMIT
  ```

*(On high-core CPU servers, you can add `--num_cores 16` to speed up preprocessing).*

---

### Step 2: Generate Dataset Markers & Splits
Extract seizure/non-seizure window markers (12s and 60s windows):

* **For TUSZ:**
  ```bash
  python generate_markers.py --dataset TUSZ --raw_data_dir ./data/raw/TUSZ
  ```

* **For CHB-MIT:**
  ```bash
  python generate_markers.py --dataset CHBMIT --raw_data_dir ./data/raw/CHB-MIT --split_mode patient
  ```

Marker text files will be stored in `data/file_markers_detection/` (TUSZ) or `data/file_markers_chb/` (CHB-MIT).

---

## 🔬 Model Training & Evaluation

### 1. Train EvoBrain

* **Train on TUSZ:**
  ```bash
  python main.py \
    --dataset TUSZ \
    --model_name evobrain \
    --task detection \
    --graph_type dynamic \
    --device cuda \
    --train_batch_size 32 \
    --test_batch_size 64 \
    --save_dir ./results
  ```

* **Train on CHB-MIT:**
  ```bash
  python main.py \
    --dataset CHBMIT \
    --model_name evobrain \
    --task detection \
    --graph_type dynamic \
    --device cuda \
    --train_batch_size 32 \
    --test_batch_size 64 \
    --save_dir ./result/CHBMIT
  ```

---

### 2. Train Baseline Models
EvoBrain supports multiple comparative architectures (`--model_name`):
* `lstm` (Standard LSTM)
* `cnnlstm` (CNN + LSTM)
* `dcrnn` (Diffusion Convolutional RNN)
* `gru_gcn` (GRU + GCN)
* `BIOT` (Biomedical Transformer)

Example running DCRNN on TUSZ:
```bash
python main.py --model_name dcrnn --dataset TUSZ --device cuda --save_dir ./results
```

---

### 3. Evaluate a Pretrained Checkpoint
To run testing on a trained checkpoint:
```bash
python main.py \
  --model_name evobrain \
  --dataset TUSZ \
  --test \
  --load_model_path ./results/TUSZ/detection/12/evobrain_dynamic_123_01/best.pth.tar \
  --device cuda
```

---

## 🖥️ Server & Hardware Configuration Tips

* **Consumer / Laptop GPUs (6GB VRAM)**: Set `--train_batch_size 32` (or `16` for long sequences) to prevent OOM errors.
* **Server GPUs (A100, RTX 3090/4090, V100)**: Scale up `--train_batch_size 64` or `128` and `--num_workers 8` or `16` for maximum GPU saturation.
* **Multi-GPU Servers**: Target a specific GPU using `--device cuda:0` or prefix commands with `CUDA_VISIBLE_DEVICES=1`.

---

## 📂 Repository Structure

```
EvoBrain/
├── args.py                     # Command-line configuration parser
├── constants.py                # Global constants and paths
├── diagnose_channels.py        # Channel diagnosis utility for EDFs
├── generate_markers.py         # Multi-dataset split & marker generator
├── main.py                     # Training, evaluation & testing pipeline
├── requirements.txt            # Python package dependencies
├── utils.py                    # Training utilities, metrics & logging
├── data/
│   ├── constants.py            # Channel montages (19 TUSZ, 18 CHB-MIT)
│   ├── compute_scalers.py      # Dataset normalization statistics calculator
│   ├── data_utils.py           # FFT, channel aliases, signal transformations
│   ├── dataloader_chb.py       # CHB-MIT PyTorch DataLoader & Dataset
│   ├── dataloader_detection.py # TUSZ PyTorch DataLoader & Dataset
│   ├── resample_signals.py     # Multiprocessing EDF -> 200Hz H5 resampler
│   ├── file_markers_chb/       # CHB-MIT split files
│   └── file_markers_detection/ # TUSZ split files
└── model/
    ├── EvoBrain.py             # EvoBrain model (Mamba + Dynamic GNN)
    ├── DCRNN.py                # Diffusion Convolutional RNN
    ├── gru_gcn.py              # GRU-GCN model
    ├── BIOT.py                 # BIOT Transformer model
    ├── lstm.py                 # LSTM baseline
    └── cnnlstm.py              # CNN-LSTM baseline
```

---

## 📜 Citation

If you find this work useful, please cite the paper:

```bibtex
@inproceedings{
  kotoge2025evobrain,
  title={EvoBrain: Dynamic Multi-Channel {EEG} Graph Modeling for Time-Evolving Brain Networks},
  author={Rikuto Kotoge and Zheng Chen and Tasuku Kimura and Yasuko Matsubara and Takufumi Yanagisawa and Haruhiko Kishima and Yasushi Sakurai},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```