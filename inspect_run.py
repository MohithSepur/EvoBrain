import os
import argparse
import json
import numpy as np
import sklearn.metrics as metrics

def get_window_length(run_dir, args_dict=None):
    if args_dict is not None:
        max_seq_len = args_dict.get('max_seq_len')
        time_step = args_dict.get('time_step_size', 1)
        if max_seq_len is not None:
            return f"{max_seq_len * time_step}s"
    
    # Try searching args.json
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r") as f:
                args = json.load(f)
            max_seq_len = args.get('max_seq_len')
            time_step = args.get('time_step_size', 1)
            if max_seq_len is not None:
                return f"{max_seq_len * time_step}s"
        except Exception:
            pass

    # Try inferring from directory name (e.g. /12/, /60/, /10s/)
    norm_path = os.path.normpath(run_dir)
    parts = norm_path.split(os.sep)
    for part in parts:
        if part in ["10", "12", "60", "10s", "12s", "60s"]:
            return part if part.endswith("s") else f"{part}s"
    
    return "N/A"

def display_metrics_and_cm(y_true, y_pred, y_prob=None, split_name="Test"):
    print(f"\n--- {split_name} Set Evaluation Metrics ---")
    total = len(y_true)
    unique_labels = np.unique(y_true)

    if len(unique_labels) <= 2:
        cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        acc = metrics.accuracy_score(y_true, y_pred)
        bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        
        if y_prob is not None and len(unique_labels) > 1:
            try:
                auroc = metrics.roc_auc_score(y_true, y_prob)
                print(f"  AUROC:                {auroc:.4f} ({auroc*100:.2f}%)")
            except Exception:
                pass
            try:
                pr_auc = metrics.average_precision_score(y_true, y_prob)
                print(f"  PR-AUC:               {pr_auc:.4f}")
            except Exception:
                pass

        print(f"  Accuracy:             {acc*100:.2f}%")
        print(f"  Balanced Accuracy:    {bal_acc*100:.2f}%")
        print(f"  Sensitivity (Recall): {sens*100:.2f}%")
        print(f"  Specificity:          {spec*100:.2f}%")
        print(f"  Precision:            {prec*100:.2f}%")
        print(f"  F1-Score:             {f1:.4f}")

        print(f"\n--- {split_name} Set Confusion Matrix ---")
        print(f"                     Predicted Normal (0)   Predicted Seizure (1)")
        print(f"  Actual Normal (0):        {tn:<18,}  {fp:<18,}")
        print(f"  Actual Seizure (1):       {fn:<18,}  {tp:<18,}")
        print(f"  [TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,} | Total: {total:,}]")
    else:
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"  Accuracy:             {acc*100:.2f}%")
        print(f"  Weighted F1-Score:    {f1:.4f}")
        print(f"\n--- {split_name} Set Confusion Matrix ---")
        print(metrics.confusion_matrix(y_true, y_pred))

def inspect_run(run_dir):
    if not os.path.exists(run_dir):
        print(f"Error: Path '{run_dir}' does not exist.")
        return

    # Handle if path to a specific file was passed
    if os.path.isfile(run_dir):
        run_file = run_dir
        run_dir_path = os.path.dirname(run_file)
    else:
        run_file = None
        run_dir_path = run_dir

    args_dict = None
    args_path = os.path.join(run_dir_path, "args.json")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r") as f:
                args_dict = json.load(f)
        except Exception:
            pass

    window_len = get_window_length(run_dir_path, args_dict)

    print("\n" + "=" * 60)
    print(f"Run: {os.path.basename(os.path.abspath(run_dir_path))}")
    print(f"Window Length: {window_len}")
    print("=" * 60)

    found_results = False

    # Check for test results
    test_npz = os.path.join(run_dir_path, "test_results.npz") if run_file is None else (run_file if "test" in os.path.basename(run_file) else None)
    if test_npz and os.path.exists(test_npz):
        try:
            data = np.load(test_npz)
            y_true = data['y_true']
            y_prob = data.get('y_prob', None)
            y_pred = data.get('y_pred', (y_prob >= 0.5).astype(int) if y_prob is not None else None)
            display_metrics_and_cm(y_true, y_pred, y_prob, split_name="Test")
            found_results = True
        except Exception as e:
            print(f"Error reading test results: {e}")

    # Check for dev results
    dev_npz = os.path.join(run_dir_path, "dev_results.npz") if run_file is None else (run_file if "dev" in os.path.basename(run_file) else None)
    if dev_npz and os.path.exists(dev_npz):
        try:
            data = np.load(dev_npz)
            y_true = data['y_true']
            y_prob = data.get('y_prob', None)
            y_pred = data.get('y_pred', (y_prob >= 0.5).astype(int) if y_prob is not None else None)
            display_metrics_and_cm(y_true, y_pred, y_prob, split_name="Dev / Validation")
            found_results = True
        except Exception as e:
            print(f"Error reading dev results: {e}")

    if not found_results and run_file and os.path.exists(run_file):
        try:
            data = np.load(run_file)
            y_true = data['y_true']
            y_prob = data.get('y_prob', None)
            y_pred = data.get('y_pred', (y_prob >= 0.5).astype(int) if y_prob is not None else None)
            display_metrics_and_cm(y_true, y_pred, y_prob, split_name="Evaluation")
            found_results = True
        except Exception as e:
            print(f"Error reading results from {run_file}: {e}")

    if not found_results:
        print("\nNo evaluation results (.npz files) found in this directory.")

    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect run window length, evaluation metrics, and confusion matrix.")
    parser.add_argument("run_dir", type=str, nargs="?", default=None, help="Path to experiment result directory")
    parser.add_argument("--run_dir", dest="run_dir_opt", type=str, default=None, help="Path to experiment result directory (optional flag)")
    cli_args = parser.parse_args()
    
    target_dir = cli_args.run_dir or cli_args.run_dir_opt or "."
    inspect_run(target_dir)
