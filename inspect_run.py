import os
import argparse
import json
import torch
import numpy as np
import sklearn.metrics as metrics

def print_banner(text):
    print("\n" + "=" * 65)
    print(f" {text}")
    print("=" * 65)

def inspect_run(run_dir):
    if not os.path.exists(run_dir):
        print(f"Error: Directory '{run_dir}' does not exist.")
        return

    print_banner(f"EXPERIMENT STATISTICS SUMMARY: {os.path.basename(run_dir)}")
    print(f"Path: {os.path.abspath(run_dir)}")

    # 1. Inspect args.json
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = json.load(f)
        print_banner("1. Configuration & Hyperparameters")
        print(f" - Dataset:            {args.get('dataset', 'N/A')}")
        print(f" - Task:               {args.get('task', 'N/A')}")
        print(f" - Model Architecture: {args.get('model_name', 'N/A')} (Graph: {args.get('graph_type', 'N/A')})")
        print(f" - Electrode Nodes:    {args.get('num_nodes', 'N/A')} channels")
        print(f" - Max Sequence Len:   {args.get('max_seq_len', 'N/A')} seconds")
        print(f" - Batch Size:         Train={args.get('train_batch_size', 'N/A')}, Test={args.get('test_batch_size', 'N/A')}")
        print(f" - Epochs:             {args.get('num_epochs', 'N/A')}")
        print(f" - Learning Rate:      {args.get('lr_init', 'N/A')}")
        print(f" - Random Seed:        {args.get('rand_seed', 'N/A')}")

    # 2. Inspect best.pth.tar
    best_pth = os.path.join(run_dir, "best.pth.tar")
    if os.path.exists(best_pth):
        checkpoint = torch.load(best_pth, map_location="cpu")
        print_banner("2. Checkpoint Details (best.pth.tar)")
        print(f" - Best Epoch:         Epoch {checkpoint.get('epoch', 'N/A')}")
        print(f" - Best Dev AUROC:     {checkpoint.get('metric_val', 'N/A'):.4f}" if isinstance(checkpoint.get('metric_val'), (int, float)) else f" - Best Metric: {checkpoint.get('metric_val', 'N/A')}")
        if 'model_state' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['model_state'].values())
            print(f" - Trainable Weights:  {total_params:,} parameters")

    # 3. Inspect Validation / DEV Results
    dev_npz = os.path.join(run_dir, "dev_results.npz")
    if os.path.exists(dev_npz):
        dev_data = np.load(dev_npz)
        y_true = dev_data['y_true']
        y_prob = dev_data['y_prob']
        auroc = metrics.roc_auc_score(y_true, y_prob)
        pr_auc = metrics.average_precision_score(y_true, y_prob)
        pos_cnt = int(np.sum(y_true == 1))
        neg_cnt = int(np.sum(y_true == 0))
        print_banner("3. Validation (DEV) Set Evaluation")
        print(f" - Total Samples:      {len(y_true):,} ({pos_cnt} Seizures / {neg_cnt} Non-Seizures)")
        print(f" - AUROC:              {auroc:.4f} ({auroc*100:.2f}%)")
        print(f" - PR-AUC:             {pr_auc:.4f}")

    # 4. Inspect Test Results
    test_npz = os.path.join(run_dir, "test_results.npz")
    if os.path.exists(test_npz):
        test_data = np.load(test_npz)
        y_true = test_data['y_true']
        y_prob = test_data['y_prob']
        
        pos_cnt = int(np.sum(y_true == 1))
        neg_cnt = int(np.sum(y_true == 0))
        auroc = metrics.roc_auc_score(y_true, y_prob)
        pr_auc = metrics.average_precision_score(y_true, y_prob)

        print_banner("4. Unseen TEST Set Evaluation")
        print(f" - Total Test Samples: {len(y_true):,} ({pos_cnt} Seizures / {neg_cnt} Non-Seizures)")
        print(f" - Test AUROC:         {auroc:.4f} ({auroc*100:.2f}%)")
        print(f" - Test PR-AUC:        {pr_auc:.4f}")
        
        # Threshold Performance Table
        print("\n Threshold Sensitivity / Specificity Sweep on Test Set:")
        print(" " + "-" * 60)
        print(f" {'Threshold (τ)':<15} {'Sensitivity':<15} {'Specificity':<15} {'F1-Score':<15}")
        print(" " + "-" * 60)
        for tau in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
            y_pred = (y_prob >= tau).astype(int)
            sens = metrics.recall_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
            print(f"  τ = {tau:<10.2f} {sens*100:<14.2f}% {spec*100:<14.2f}% {f1:<14.4f}")
        print(" " + "-" * 60)

    print("\n" + "=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect all statistics from a run directory.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiment result directory")
    args = parser.parse_args()
    inspect_run(args.run_dir)
