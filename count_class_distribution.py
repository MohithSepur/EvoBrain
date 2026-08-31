#!/usr/bin/env python3
"""Count positive and negative seizure examples used by EvoBrain.

TUSZ counts come from EvoBrain's positive/negative marker manifests.  The
script also reports the train-balanced selection made by ``parseTxtFiles``.
When ``--tusz-input-dir`` is supplied, it additionally applies EvoBrain's HDF5
existence filter and reports the examples that can actually be loaded.

CHB-MIT counts come from every pre-segmented PKL resolved by
``PklSeizureDataset`` for train/dev/test.  PKLs are read only; malformed or
non-binary labels are reported rather than silently counted as negative.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


SPLITS = ("train", "dev", "test")
REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_TUSZ_MARKER_DIR = REPOSITORY_ROOT / "data" / "file_markers_detection"


class CountingError(RuntimeError):
    """Raised when an exact class count cannot be produced."""


@dataclass
class Counts:
    positive: int = 0
    negative: int = 0
    invalid: int = 0

    @property
    def valid(self) -> int:
        return self.positive + self.negative

    @property
    def total(self) -> int:
        return self.valid + self.invalid

    def add(self, label: int) -> None:
        if label == 1:
            self.positive += 1
        elif label == 0:
            self.negative += 1
        else:
            self.invalid += 1

    def to_dict(self) -> dict[str, int]:
        result = asdict(self)
        result.update(valid=self.valid, total=self.total)
        return result


def _read_marker_entries(path: Path, expected_label: int) -> list[tuple[str, int]]:
    if not path.is_file():
        raise CountingError(f"Required TUSZ marker file does not exist: {path}")

    entries: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                marker_name, label_text = line.rsplit(",", 1)
                label = int(label_text)
            except (ValueError, TypeError) as exc:
                raise CountingError(
                    f"Malformed marker at {path}:{line_number}: {line!r}"
                ) from exc
            if label not in (0, 1):
                raise CountingError(
                    f"Non-binary marker label at {path}:{line_number}: {label!r}"
                )
            if label != expected_label:
                raise CountingError(
                    f"Marker label/file mismatch at {path}:{line_number}: "
                    f"expected {expected_label}, found {label}"
                )
            entries.append((marker_name, label))
    return entries


def _counts_from_entries(entries: list[tuple[str, int]]) -> Counts:
    counts = Counts()
    for _, label in entries:
        counts.add(label)
    return counts


def _evobrain_marker_selection(
    split: str,
    positive: list[tuple[str, int]],
    negative: list[tuple[str, int]],
    seed: int,
    sampling_ratio: float,
) -> list[tuple[str, int]]:
    """Mirror dataloader_detection.parseTxtFiles selection order."""
    positive = list(positive)
    negative = list(negative)
    rng = np.random.RandomState(seed)
    if split == "train":
        selected_count = int(sampling_ratio * len(positive))
        positive_indices = list(range(len(positive)))
        rng.shuffle(positive_indices)
        positive = [positive[index] for index in positive_indices[:selected_count]]
        rng.shuffle(negative)
        negative = negative[:selected_count]
    selected = positive + negative
    rng.shuffle(selected)
    return selected


def _tusz_h5_exists(input_dir: Path, split: str, marker_name: str) -> bool:
    split_folder = "eval" if split == "test" else split
    base_h5 = marker_name.split(".edf")[0] + ".h5"
    return (
        (input_dir / split_folder / base_h5).exists()
        or (input_dir / base_h5).exists()
    )


def count_tusz(
    marker_dir: Path,
    max_seq_len: int,
    input_dir: Path | None,
    seed: int,
    sampling_ratio: float,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        positive_path = marker_dir / f"{split}Set_seq2seq_{max_seq_len}s_sz.txt"
        negative_path = marker_dir / f"{split}Set_seq2seq_{max_seq_len}s_nosz.txt"
        positive = _read_marker_entries(positive_path, expected_label=1)
        negative = _read_marker_entries(negative_path, expected_label=0)
        source = positive + negative
        selected = _evobrain_marker_selection(
            split, positive, negative, seed=seed, sampling_ratio=sampling_ratio
        )
        split_result: dict[str, Any] = {
            "source_markers": _counts_from_entries(source).to_dict(),
            "selected_by_loader": _counts_from_entries(selected).to_dict(),
        }
        if input_dir is not None:
            loadable = [
                entry
                for entry in selected
                if _tusz_h5_exists(input_dir, split, entry[0])
            ]
            split_result["loadable_after_h5_filter"] = _counts_from_entries(
                loadable
            ).to_dict()
        result[split] = split_result
    return result


def _resolve_chb_split(data_dir: Path, split: str) -> tuple[Path, bool]:
    candidates = [
        data_dir / split,
        data_dir / ("val" if split == "dev" else split),
        data_dir / ("dev" if split == "val" else split),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return candidate, False

    if data_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() == ".pkl"
        for path in data_dir.iterdir()
    ):
        return data_dir, True
    raise CountingError(
        f"Could not resolve CHB-MIT split {split!r} below {data_dir}. "
        "Expected train/dev/test (dev may be named val)."
    )


def _chb_files(split_dir: Path, flat_root: bool) -> list[Path]:
    paths = split_dir.iterdir() if flat_root else split_dir.rglob("*.pkl")
    return sorted(
        path
        for path in paths
        if path.is_file()
        and path.suffix.lower() == ".pkl"
        and not path.name.startswith(".")
    )


def _scalar_binary_label(value: Any) -> int:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"label has {array.size} values")
    scalar = array.reshape(-1)[0]
    numeric = float(scalar)
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"label is not binary: {scalar!r}")
    return int(numeric)


def _label_from_pkl(path: Path) -> int:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if isinstance(sample, dict):
        if "y" in sample:
            value = sample["y"]
        elif "label" in sample:
            value = sample["label"]
        else:
            raise ValueError("dictionary contains neither 'y' nor 'label'")
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        value = sample[1]
    else:
        raise ValueError("unsupported PKL structure")
    return _scalar_binary_label(value)


def count_chb(data_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    resolved: dict[str, Path] = {}
    for split in SPLITS:
        split_dir, flat_root = _resolve_chb_split(data_dir, split)
        resolved[split] = split_dir.resolve()
        files = _chb_files(split_dir, flat_root)
        counts = Counts()
        invalid_examples: list[str] = []
        for path in files:
            try:
                counts.add(_label_from_pkl(path))
            except Exception as exc:  # report bad data without hiding the other counts
                counts.invalid += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append(f"{path}: {exc}")
        result[split] = {
            **counts.to_dict(),
            "directory": str(split_dir),
            "invalid_examples": invalid_examples,
        }

    if len(set(resolved.values())) != len(SPLITS):
        warnings.append(
            "Multiple CHB-MIT splits resolve to the same directory. EvoBrain's flat-PKL "
            "fallback would reuse the same files across train/dev/test; this is split leakage."
        )
    return result, warnings


def _print_counts(title: str, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
    print(f"\n{title}")
    print(f"{'split':<8} {'stage':<27} {'positive':>10} {'negative':>10} {'invalid':>9} {'total':>10}")
    print("-" * 82)
    for split, stage, counts in rows:
        print(
            f"{split:<8} {stage:<27} {counts['positive']:>10,} "
            f"{counts['negative']:>10,} {counts['invalid']:>9,} {counts['total']:>10,}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count positive/negative train, dev, and test examples for EvoBrain"
    )
    parser.add_argument(
        "--tusz-marker-dir",
        type=Path,
        default=DEFAULT_TUSZ_MARKER_DIR,
        help=f"TUSZ marker directory (default: {DEFAULT_TUSZ_MARKER_DIR})",
    )
    parser.add_argument(
        "--tusz-input-dir",
        type=Path,
        default=None,
        help="Optional resampled TUSZ HDF5 root for exact post-filter counts",
    )
    parser.add_argument("--tusz-max-seq-len", type=int, default=10)
    parser.add_argument("--sampling-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--chb-pkl-dir",
        type=Path,
        required=True,
        help="CHB-MIT PKL root containing train/dev/test (dev may be named val)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of tables",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.tusz_max_seq_len <= 0:
        raise CountingError("--tusz-max-seq-len must be positive")
    if args.sampling_ratio < 0:
        raise CountingError("--sampling-ratio must be non-negative")

    tusz = count_tusz(
        marker_dir=args.tusz_marker_dir,
        max_seq_len=args.tusz_max_seq_len,
        input_dir=args.tusz_input_dir,
        seed=args.seed,
        sampling_ratio=args.sampling_ratio,
    )
    chb, warnings = count_chb(args.chb_pkl_dir)

    if args.json:
        print(json.dumps({"TUSZ": tusz, "CHB-MIT": chb, "warnings": warnings}, indent=2))
    else:
        tusz_rows = [
            (split, stage, counts)
            for split, stages in tusz.items()
            for stage, counts in stages.items()
        ]
        _print_counts("TUSZ", tusz_rows)
        _print_counts(
            "CHB-MIT PKL",
            [(split, "all_pkls", counts) for split, counts in chb.items()],
        )
        for split, counts in chb.items():
            print(f"  {split}: {counts['directory']}")
            for example in counts["invalid_examples"]:
                print(f"    INVALID: {example}", file=sys.stderr)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

    invalid = sum(counts["invalid"] for counts in chb.values())
    return 1 if invalid else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CountingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
