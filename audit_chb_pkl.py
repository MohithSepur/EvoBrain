#!/usr/bin/env python3
"""Audit pre-segmented CHB-MIT PKLs before and after EvoBrain loading.

The raw pass runs before EvoBrain's ``nan_to_num`` calls and reports malformed
samples, non-finite EEG values, shape/dtype distributions, constant channels,
content duplicates, and patient overlap between splits.  Unless
``--skip-post-loader`` is supplied, a second pass exercises the repository's
actual ``PklSeizureDataset`` and validates its six-field output contract.

Only audit trusted PKLs: unpickling data can execute arbitrary code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm


SPLITS = ("train", "dev", "test")
DEFAULT_PATIENT_REGEX = r"(?i)(chb\d{2})"


class AuditError(RuntimeError):
    """Raised when the requested audit cannot be performed safely."""


def resolve_splits(root: Path) -> dict[str, Path]:
    """Resolve train/dev/test, accepting ``val`` as the dev directory."""
    resolved: dict[str, Path] = {}
    for split in SPLITS:
        names = ("dev", "val") if split == "dev" else (split,)
        candidates = [root / name for name in names if (root / name).is_dir()]
        if not candidates:
            raise AuditError(
                f"Could not resolve split {split!r} below {root}. "
                "Expected train, dev (or val), and test directories."
            )
        resolved[split] = candidates[0].resolve()

    if len(set(resolved.values())) != len(SPLITS):
        raise AuditError("Two or more dataset splits resolve to the same directory.")
    return resolved


def list_pkls(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.rglob("*.pkl")
        if path.is_file() and not path.name.startswith(".")
    )


def extract_sample(sample: Any) -> tuple[Any, Any]:
    """Use the same raw-data/label key precedence as PklSeizureDataset."""
    if isinstance(sample, dict):
        if "X" in sample:
            raw = sample["X"]
        elif "data" in sample:
            raw = sample["data"]
        else:
            raise ValueError("dictionary contains neither 'X' nor 'data'")

        if "y" in sample:
            label = sample["y"]
        elif "label" in sample:
            label = sample["label"]
        else:
            raise ValueError("dictionary contains neither 'y' nor 'label'")
        return raw, label

    if isinstance(sample, (tuple, list)):
        if len(sample) < 2:
            raise ValueError("tuple/list contains fewer than two fields")
        return sample[0], sample[1]
    raise ValueError(f"unsupported top-level PKL type: {type(sample).__name__}")


def scalar_binary_label(value: Any) -> int:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"label contains {array.size} values")
    numeric = float(array.reshape(-1)[0])
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"label is not binary: {array.reshape(-1)[0]!r}")
    return int(numeric)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(repr(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _base_record(split: str, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "split": split,
        "file": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "readable": False,
        "label": None,
        "shape": None,
        "dtype": None,
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
        "nonfinite_count": 0,
        "finite_min": None,
        "finite_max": None,
        "max_abs": None,
        "constant_channels": None,
        "near_constant_channels": None,
        "all_zero": False,
        "file_sha256": None,
        "signal_sha256": None,
        "patient_id": None,
        "issues": [],
    }


def audit_raw_file(task: tuple[str, str, str, float | None, float | None]) -> dict[str, Any]:
    split, filename, patient_pattern, extreme_threshold, near_constant_std = task
    path = Path(filename)
    try:
        record = _base_record(split, path)
    except Exception as exc:
        return {
            "split": split,
            "file": filename,
            "readable": False,
            "issues": [f"stat_error: {type(exc).__name__}: {exc}"],
        }

    patient_match = re.search(patient_pattern, str(path))
    if patient_match:
        try:
            record["patient_id"] = patient_match.group(1).lower()
        except IndexError:
            record["issues"].append("patient_regex_has_no_capture_group")
    else:
        record["issues"].append("patient_id_not_found")

    try:
        record["file_sha256"] = sha256_file(path)
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        raw, label_value = extract_sample(sample)
        record["label"] = scalar_binary_label(label_value)
        array = np.asarray(raw)
        record["shape"] = list(array.shape)
        record["dtype"] = str(array.dtype)

        if array.size == 0:
            record["issues"].append("empty_signal")
            record["signal_sha256"] = sha256_array(array)
            record["readable"] = True
            return record

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"EEG dtype is not numeric: {array.dtype}")
        if array.ndim not in (2, 3):
            record["issues"].append(f"unexpected_ndim:{array.ndim}")

        record["signal_sha256"] = sha256_array(array)
        nan_mask = np.isnan(array)
        posinf_mask = np.isposinf(array)
        neginf_mask = np.isneginf(array)
        finite_mask = np.isfinite(array)
        record["nan_count"] = int(nan_mask.sum())
        record["posinf_count"] = int(posinf_mask.sum())
        record["neginf_count"] = int(neginf_mask.sum())
        record["nonfinite_count"] = int((~finite_mask).sum())
        if record["nan_count"]:
            record["issues"].append("raw_nan")
        if record["posinf_count"] or record["neginf_count"]:
            record["issues"].append("raw_inf")

        finite_values = array[finite_mask]
        if finite_values.size:
            finite_min = float(finite_values.min())
            finite_max = float(finite_values.max())
            max_abs = float(np.max(np.abs(finite_values)))
            record["finite_min"] = finite_min
            record["finite_max"] = finite_max
            record["max_abs"] = max_abs
            record["all_zero"] = bool(np.all(finite_values == 0)) and bool(finite_mask.all())
            if record["all_zero"]:
                record["issues"].append("all_zero")
            if extreme_threshold is not None and max_abs > extreme_threshold:
                record["issues"].append("above_configured_extreme_threshold")
        else:
            record["issues"].append("no_finite_values")

        if array.ndim >= 2:
            flattened = array.reshape(array.shape[0], -1)
            constant_count = 0
            near_constant_count = 0
            for channel in flattened:
                finite_channel = channel[np.isfinite(channel)]
                if finite_channel.size == 0:
                    continue
                channel_std = float(np.std(finite_channel, dtype=np.float64))
                if float(finite_channel.min()) == float(finite_channel.max()):
                    constant_count += 1
                if near_constant_std is not None and channel_std <= near_constant_std:
                    near_constant_count += 1
            record["constant_channels"] = constant_count
            record["near_constant_channels"] = (
                near_constant_count if near_constant_std is not None else None
            )
            if constant_count:
                record["issues"].append("constant_channel")
            if near_constant_std is not None and near_constant_count:
                record["issues"].append("near_constant_channel")

        record["readable"] = True
    except Exception as exc:
        record["issues"].append(f"read_error: {type(exc).__name__}: {exc}")
    return record


def _record_key(record: dict[str, Any]) -> tuple[str, int | None, int | None]:
    return (record["file"], record.get("size_bytes"), record.get("mtime_ns"))


def read_resume_records(path: Path) -> dict[tuple[str, int | None, int | None], dict[str, Any]]:
    records: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[_record_key(record)] = record
            except Exception as exc:
                raise AuditError(f"Invalid resume record at {path}:{line_number}: {exc}") from exc
    return records


def current_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def iter_raw_records(
    split_files: dict[str, list[Path]],
    workers: int,
    patient_regex: str,
    extreme_threshold: float | None,
    near_constant_std: float | None,
    progress_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    cached = read_resume_records(progress_path) if resume else {}
    records: list[dict[str, Any]] = []
    tasks: list[tuple[str, str, str, float | None, float | None]] = []

    for split, files in split_files.items():
        for path in files:
            cached_record = cached.get(current_key(path))
            if cached_record is not None:
                records.append(cached_record)
            else:
                tasks.append(
                    (split, str(path), patient_regex, extreme_threshold, near_constant_std)
                )

    mode = "a" if resume and progress_path.exists() else "w"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open(mode, encoding="utf-8") as progress:
        records_since_flush = 0
        if workers == 1:
            iterator: Iterable[dict[str, Any]] = map(audit_raw_file, tasks)
            for record in tqdm(iterator, total=len(tasks), desc="Raw PKL audit", unit="pkl", dynamic_ncols=True):
                records.append(record)
                progress.write(json.dumps(record, allow_nan=False) + "\n")
                records_since_flush += 1
                if records_since_flush >= 100:
                    progress.flush()
                    records_since_flush = 0
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(audit_raw_file, tasks, chunksize=16)
                for record in tqdm(iterator, total=len(tasks), desc="Raw PKL audit", unit="pkl", dynamic_ncols=True):
                    records.append(record)
                    progress.write(json.dumps(record, allow_nan=False) + "\n")
                    records_since_flush += 1
                    if records_since_flush >= 100:
                        progress.flush()
                        records_since_flush = 0
    return sorted(records, key=lambda item: (item["split"], item["file"]))


def duplicate_rows(records: list[dict[str, Any]], key: str, kind: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value:
            grouped[value].append(record)

    rows: list[dict[str, Any]] = []
    for digest, group in grouped.items():
        splits = sorted({record["split"] for record in group})
        if len(group) < 2:
            continue
        rows.append(
            {
                "kind": kind,
                "sha256": digest,
                "cross_split": len(splits) > 1,
                "splits": ";".join(splits),
                "count": len(group),
                "files": ";".join(record["file"] for record in group),
            }
        )
    return rows


def summarize_raw(
    records: list[dict[str, Any]],
    resolved: dict[str, Path],
    extreme_threshold: float | None,
    near_constant_std: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "split_directories": {split: str(path) for split, path in resolved.items()},
        "configured_extreme_abs_threshold": extreme_threshold,
        "configured_near_constant_std": near_constant_std,
        "splits": {},
    }
    anomalies = [record for record in records if record.get("issues")]

    patients_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        group = [record for record in records if record["split"] == split]
        shapes = Counter(
            repr(tuple(record["shape"]))
            for record in group
            if record.get("shape") is not None
        )
        dtypes = Counter(record["dtype"] for record in group if record.get("dtype"))
        patients = {record["patient_id"] for record in group if record.get("patient_id")}
        patients_by_split[split] = patients
        summary["splits"][split] = {
            "files": len(group),
            "positive": sum(record.get("label") == 1 for record in group),
            "negative": sum(record.get("label") == 0 for record in group),
            "invalid_label_or_unreadable": sum(record.get("label") not in (0, 1) for record in group),
            "files_with_nan": sum(record.get("nan_count", 0) > 0 for record in group),
            "nan_values": sum(record.get("nan_count", 0) for record in group),
            "files_with_inf": sum(
                record.get("posinf_count", 0) + record.get("neginf_count", 0) > 0
                for record in group
            ),
            "inf_values": sum(
                record.get("posinf_count", 0) + record.get("neginf_count", 0)
                for record in group
            ),
            "all_zero_files": sum(bool(record.get("all_zero")) for record in group),
            "files_with_constant_channels": sum(
                (record.get("constant_channels") or 0) > 0 for record in group
            ),
            "finite_min": min(
                (record["finite_min"] for record in group if record.get("finite_min") is not None),
                default=None,
            ),
            "finite_max": max(
                (record["finite_max"] for record in group if record.get("finite_max") is not None),
                default=None,
            ),
            "maximum_absolute_value": max(
                (record["max_abs"] for record in group if record.get("max_abs") is not None),
                default=None,
            ),
            "shape_histogram": dict(sorted(shapes.items())),
            "dtype_histogram": dict(sorted(dtypes.items())),
            "patient_ids": sorted(patients),
        }

    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{left}-{right}"] = sorted(patients_by_split[left] & patients_by_split[right])
    summary["patient_overlap"] = overlaps

    duplicates = duplicate_rows(records, "file_sha256", "whole_pkl")
    duplicates.extend(duplicate_rows(records, "signal_sha256", "eeg_array"))
    summary["duplicate_groups"] = {
        "total": len(duplicates),
        "cross_split": sum(bool(row["cross_split"]) for row in duplicates),
    }
    return summary, anomalies, duplicates


def audit_post_loader(root: Path, use_fft: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from data.dataloader_chb import PklSeizureDataset
    except Exception as exc:
        raise AuditError(
            "Could not import EvoBrain's PklSeizureDataset. Run this script from "
            f"the EvoBrain environment or use --skip-post-loader. Import error: {exc}"
        ) from exc

    summary: dict[str, Any] = {}
    anomalies: list[dict[str, Any]] = []
    for split in SPLITS:
        dataset = PklSeizureDataset(
            data_dir=str(root),
            split=split,
            time_step_size=1,
            max_seq_len=10,
            use_fft=use_fft,
            standardize=True,
            scaler=None,
            data_augment=False,
            graph_type="dynamic",
        )
        split_counts = Counter()
        seen_writeout: Counter[str] = Counter()
        for index in tqdm(
            range(len(dataset)),
            desc=f"EvoBrain loader {split}",
            unit="pkl",
            dynamic_ncols=True,
        ):
            file_path = dataset.files[index]
            issues: list[str] = []
            try:
                batch = dataset[index]
                if not isinstance(batch, tuple) or len(batch) != 6:
                    raise ValueError(
                        f"expected six-field tuple, got {type(batch).__name__} "
                        f"with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}"
                    )
                x, y, seq_len, supports, adjacency, writeout_fn = batch
                del supports
                if not bool(np.isfinite(x.detach().cpu().numpy()).all()):
                    issues.append("post_loader_nonfinite_x")
                if not bool(np.isfinite(adjacency.detach().cpu().numpy()).all()):
                    issues.append("post_loader_nonfinite_adjacency")
                if x.ndim != 3 or x.shape[0] != 10:
                    issues.append(f"unexpected_x_shape:{tuple(x.shape)}")
                expected_width = 100 if use_fft else 200
                if x.ndim == 3 and x.shape[-1] != expected_width:
                    issues.append(f"unexpected_feature_width:{x.shape[-1]}")
                channels = int(x.shape[1]) if x.ndim == 3 else None
                expected_adj = (10, channels, channels) if channels is not None else None
                if expected_adj is not None and tuple(adjacency.shape) != expected_adj:
                    issues.append(f"unexpected_adjacency_shape:{tuple(adjacency.shape)}")
                if int(np.asarray(y.detach().cpu()).reshape(-1)[0]) not in (0, 1):
                    issues.append("post_loader_invalid_label")
                if int(np.asarray(seq_len.detach().cpu()).reshape(-1)[0]) != 10:
                    issues.append("post_loader_invalid_seq_len")
                if not isinstance(writeout_fn, str) or not writeout_fn:
                    issues.append("empty_writeout_fn")
                else:
                    seen_writeout[writeout_fn] += 1
                split_counts["checked"] += 1
            except Exception as exc:
                issues.append(f"post_loader_error: {type(exc).__name__}: {exc}")
                split_counts["failed"] += 1

            if issues:
                anomalies.append(
                    {"stage": "post_loader", "split": split, "file": file_path, "issues": issues}
                )

        duplicate_names = {name: count for name, count in seen_writeout.items() if count > 1}
        for name, count in duplicate_names.items():
            anomalies.append(
                {
                    "stage": "post_loader",
                    "split": split,
                    "file": name,
                    "issues": [f"duplicate_writeout_fn:{count}"],
                }
            )
        summary[split] = {
            "files": len(dataset),
            "checked": split_counts["checked"],
            "failed": split_counts["failed"],
            "anomaly_records": sum(1 for row in anomalies if row["split"] == split),
            "duplicate_writeout_names": len(duplicate_names),
            "use_fft": use_fft,
        }
    return summary, anomalies


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            if isinstance(row.get("issues"), list):
                row["issues"] = ";".join(row["issues"])
            writer.writerow(row)


def print_summary(summary: dict[str, Any]) -> None:
    print("\nRaw CHB-MIT PKL audit")
    print(
        f"{'split':<8} {'files':>9} {'pos':>8} {'neg':>9} {'invalid':>9} "
        f"{'NaN files':>10} {'Inf files':>10} {'constant':>10}"
    )
    print("-" * 86)
    for split in SPLITS:
        row = summary["raw"]["splits"][split]
        print(
            f"{split:<8} {row['files']:>9,} {row['positive']:>8,} {row['negative']:>9,} "
            f"{row['invalid_label_or_unreadable']:>9,} {row['files_with_nan']:>10,} "
            f"{row['files_with_inf']:>10,} {row['files_with_constant_channels']:>10,}"
        )
    print("\nPatient overlap:", summary["raw"]["patient_overlap"])
    print("Duplicate groups:", summary["raw"]["duplicate_groups"])
    if summary.get("post_loader") is not None:
        print("Post-loader:", summary["post_loader"])
    print("Overall status:", summary["status"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw CHB-MIT PKLs and EvoBrain's post-loader tensors"
    )
    parser.add_argument(
        "--chb-pkl-dir",
        type=Path,
        required=True,
        help="Root containing train, dev/val, and test directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chb_audit"),
        help="Report directory (default: ./chb_audit)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--patient-regex",
        default=DEFAULT_PATIENT_REGEX,
        help=f"Regex with one capture group for patient ID (default: {DEFAULT_PATIENT_REGEX})",
    )
    parser.add_argument(
        "--extreme-abs-threshold",
        type=float,
        default=None,
        help="Optional raw absolute-value threshold; no threshold is assumed by default",
    )
    parser.add_argument(
        "--near-constant-std",
        type=float,
        default=None,
        help="Optional channel standard-deviation threshold; exact constants are always reported",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse unchanged records in raw_progress.jsonl",
    )
    parser.add_argument(
        "--skip-post-loader",
        action="store_true",
        help="Skip the slower second pass through PklSeizureDataset",
    )
    parser.add_argument(
        "--use-fft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EvoBrain FFT tensors in the post-loader pass (default: true)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise AuditError("--workers must be at least 1")
    if args.extreme_abs_threshold is not None and (
        not math.isfinite(args.extreme_abs_threshold) or args.extreme_abs_threshold <= 0
    ):
        raise AuditError("--extreme-abs-threshold must be a positive finite number")
    if args.near_constant_std is not None and (
        not math.isfinite(args.near_constant_std) or args.near_constant_std < 0
    ):
        raise AuditError("--near-constant-std must be a non-negative finite number")
    try:
        patient_pattern = re.compile(args.patient_regex)
    except re.error as exc:
        raise AuditError(f"Invalid --patient-regex: {exc}") from exc
    if patient_pattern.groups < 1:
        raise AuditError("--patient-regex must contain at least one capture group")

    root = args.chb_pkl_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_splits(root)
    split_files = {split: list_pkls(path) for split, path in resolved.items()}
    for split, files in split_files.items():
        if not files:
            raise AuditError(f"No PKLs found for {split} in {resolved[split]}")

    print("WARNING: Only audit PKLs you trust; pickle loading can execute code.", file=sys.stderr)
    for split in SPLITS:
        print(f"Resolved {split}: {resolved[split]} ({len(split_files[split]):,} PKLs)")

    records = iter_raw_records(
        split_files=split_files,
        workers=args.workers,
        patient_regex=args.patient_regex,
        extreme_threshold=args.extreme_abs_threshold,
        near_constant_std=args.near_constant_std,
        progress_path=output_dir / "raw_progress.jsonl",
        resume=args.resume,
    )
    raw_summary, raw_anomalies, duplicates = summarize_raw(
        records,
        resolved,
        args.extreme_abs_threshold,
        args.near_constant_std,
    )

    post_summary: dict[str, Any] | None = None
    post_anomalies: list[dict[str, Any]] = []
    if not args.skip_post_loader:
        post_summary, post_anomalies = audit_post_loader(root, args.use_fft)

    all_anomalies = [dict(record, stage="raw") for record in raw_anomalies]
    all_anomalies.extend(post_anomalies)
    cross_split_duplicates = [row for row in duplicates if row["cross_split"]]
    patient_overlap = raw_summary["patient_overlap"]
    has_patient_overlap = any(patient_overlap.values())
    post_failures = sum(row.get("failed", 0) for row in (post_summary or {}).values())
    status = "PASS" if not all_anomalies and not cross_split_duplicates and not has_patient_overlap and not post_failures else "REVIEW"

    summary = {
        "status": status,
        "raw": raw_summary,
        "post_loader": post_summary,
        "anomaly_records": len(all_anomalies),
        "cross_split_duplicate_groups": len(cross_split_duplicates),
    }
    with (output_dir / "chb_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)

    anomaly_fields = [
        "stage", "split", "file", "issues", "label", "shape", "dtype",
        "nan_count", "posinf_count", "neginf_count", "finite_min", "finite_max",
        "max_abs", "constant_channels", "near_constant_channels", "patient_id",
    ]
    duplicate_fields = ["kind", "sha256", "cross_split", "splits", "count", "files"]
    write_csv(output_dir / "chb_audit_anomalies.csv", all_anomalies, anomaly_fields)
    write_csv(output_dir / "chb_audit_duplicates.csv", duplicates, duplicate_fields)
    print_summary(summary)
    print(f"\nReports written to {output_dir}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
