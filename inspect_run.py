import os
import argparse
import json
import torch
import numpy as np
import sklearn.metrics as metrics

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def print_cm(y_true, y_pred, title="Confusion Matrix"):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    print(f"\n [{title}] (Total: {total:,} clips)")
    print(f"                  Predicted Normal (0)   Predicted Seizure (1)")
    print(f"  Actual Normal (0):    {tn:<18,} (TN)      {fp:<18,} (FP)")
    print(f"  Actual Seizure(1):    {fn:<18,} (FN)      {tp:<18,} (TP)")
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    
    print(f"  -> Sensitivity (Recall): {sens*100:.2f}% | Specificity: {spec*100:.2f}%")
    print(f"  -> Precision:            {prec*100:.2f}% | Accuracy:    {acc*100:.2f}% | F1-Score: {f1:.4f}")

def inspect_run(run_dir):
    if not os.path.exists(run_dir):
        print(f"Error: Directory '{run_dir}' does not exist.")
        return

    print_banner(f"EXPERIMENT FULL STATISTICS: {os.path.basename(run_dir)}")
    print(f"Directory: {os.path.abspath(run_dir)}")

    # 1. Inspect args.json
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = json.load(f)
        max_seq_len = args.get('max_seq_len', 10)
        time_step = args.get('time_step_size', 1)
        window_duration = max_seq_len * time_step
        print_banner("1. Model Configuration & Window Properties")
        print(f" - Window Length:      {window_duration}s per segment (max_seq_len={max_seq_len}, step_size={time_step}s)")
        print(f" - Dataset:            {args.get('dataset', 'N/A')}")
        print(f" - Task:               {args.get('task', 'N/A')}")
        print(f" - Model Architecture: {args.get('model_name', 'N/A')} (Graph Type: {args.get('graph_type', 'N/A')})")
        print(f" - Channels (Nodes):   {args.get('num_nodes', 'N/A')} electrode channels")
        print(f" - FFT Feature Dim:    {args.get('input_dim', 'N/A')} frequency bins (use_fft={args.get('use_fft', True)})")
        print(f" - Batch Size:         Train={args.get('train_batch_size', 'N/A')}, Test={args.get('test_batch_size', 'N/A')}")
        print(f" - Total Epochs:       {args.get('num_epochs', 'N/A')}")
        print(f" - Learning Rate:      {args.get('lr_init', 'N/A')}")
        print(f" - Random Seed:        {args.get('rand_seed', 'N/A')}")

    # 2. Inspect best.pth.tar
    best_pth = os.path.join(run_dir, "best.pth.tar")
    if os.path.exists(best_pth):
        checkpoint = torch.load(best_pth, map_location="cpu")
        print_banner("2. Best Checkpoint State (best.pth.tar)")
        print(f" - Best Selected Epoch:{checkpoint.get('epoch', 'N/A')}")
        print(f" - Best Dev AUROC:     {checkpoint.get('metric_val', 'N/A'):.4f}" if isinstance(checkpoint.get('metric_val'), (int, float)) else f" - Best Metric: {checkpoint.get('metric_val', 'N/A')}")
        if 'model_state' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['model_state'].values())
            print(f" - Trainable Weights:  {total_params:,} parameters ({total_params * 4 / (1024*1024):.2f} MB in FP32)")

    # 3. Inspect Validation / DEV Results
    dev_npz = os.path.join(run_dir, "dev_results.npz")
    if os.path.exists(dev_npz):
        dev_data = np.load(dev_npz)
        y_true = dev_data['y_true']
        y_prob = dev_data['y_prob']
        y_pred = dev_data.get('y_pred', (y_prob >= 0.5).astype(int))
        auroc = metrics.roc_auc_score(y_true, y_prob)
        pr_auc = metrics.average_precision_score(y_true, y_prob)
        
        print_banner("3. Validation (DEV) Set Performance")
        print(f" - AUROC:  {auroc:.4f} ({auroc*100:.2f}%)")
        print(f" - PR-AUC: {pr_auc:.4f}")
        print_cm(y_true, y_pred, title="Validation Set Confusion Matrix")

    # 4. Inspect Test Results
    test_npz = os.path.join(run_dir, "test_results.npz")
    if os.path.exists(test_npz):
        test_data = np.load(test_npz)
        y_true = test_data['y_true']
        y_prob = test_data['y_prob']
        y_pred = test_data.get('y_pred', (y_prob >= 0.5).astype(int))
        auroc = metrics.roc_auc_score(y_true, y_prob)
        pr_auc = metrics.average_precision_score(y_true, y_prob)

        print_banner("4. Unseen TEST Set Performance")
        print(f" - Test AUROC:  {auroc:.4f} ({auroc*100:.2f}%)")
        print(f" - Test PR-AUC: {pr_auc:.4f}")
        
        print_cm(y_true, y_pred, title="Test Set Default Confusion Matrix")

        # Threshold Performance Sweep Table with Confusion Matrix values
        print("\n" + "-" * 70)
        print(" Full Test Set Threshold Sweep (Confusion Matrix & Clinical Metrics):")
        print("-" * 70)
        print(f" {'τ':<6} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<9} {'Sensitivity':<13} {'Specificity':<13} {'Precision':<11} {'F1':<7}")
        print("-" * 70)
        for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            yp = (y_prob >= tau).astype(int)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = metrics.f1_score(y_true, yp, zero_division=0)
            print(f" {tau:<6.2f} {tp:<6} {fp:<6} {fn:<6} {tn:<9} {sens*100:<12.2f}% {spec*100:<12.2f}% {prec*100:<10.2f}% {f1:<7.4f}")
        print("-" * 70)

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect full statistics and confusion matrices from a run directory.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiment result directory")
    args = parser.parse_args()
    inspect_run(args.run_dir)
