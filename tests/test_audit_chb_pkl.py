import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_chb_pkl as audit


class AuditChbPklTest(unittest.TestCase):
    def _write(self, path: Path, x: np.ndarray, y: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"X": x, "y": y}, handle)

    def test_raw_nan_and_constant_channel_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train" / "chb01_clip.pkl"
            x = np.ones((2, 8), dtype=np.float32)
            x[1, 0] = np.nan
            self._write(path, x, 1)

            record = audit.audit_raw_file(
                ("train", str(path), audit.DEFAULT_PATIENT_REGEX, None, None)
            )

            self.assertTrue(record["readable"])
            self.assertEqual(record["label"], 1)
            self.assertEqual(record["nan_count"], 1)
            self.assertEqual(record["patient_id"], "chb01")
            self.assertIn("raw_nan", record["issues"])
            self.assertIn("constant_channel", record["issues"])

    def test_cross_split_signal_duplicate_and_patient_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x = np.arange(16, dtype=np.float32).reshape(2, 8)
            paths = {
                "train": root / "train" / "chb01_a.pkl",
                "dev": root / "val" / "chb01_b.pkl",
                "test": root / "test" / "chb02_c.pkl",
            }
            for index, (split, path) in enumerate(paths.items()):
                self._write(path, x if split != "test" else x + 1, index % 2)

            resolved = audit.resolve_splits(root)
            records = [
                audit.audit_raw_file(
                    (split, str(path), audit.DEFAULT_PATIENT_REGEX, None, None)
                )
                for split, path in paths.items()
            ]
            summary, _, duplicates = audit.summarize_raw(
                records, resolved, None, None
            )

            self.assertEqual(summary["patient_overlap"]["train-dev"], ["chb01"])
            signal_groups = [
                row for row in duplicates
                if row["kind"] == "eeg_array" and row["cross_split"]
            ]
            self.assertEqual(len(signal_groups), 1)

    def test_invalid_label_is_not_silently_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test" / "chb03_bad.pkl"
            self._write(path, np.zeros((2, 8), dtype=np.float32), 2)
            record = audit.audit_raw_file(
                ("test", str(path), audit.DEFAULT_PATIENT_REGEX, None, None)
            )
            self.assertIsNone(record["label"])
            self.assertTrue(any("label is not binary" in issue for issue in record["issues"]))

    def test_actual_evobrain_post_loader_contract_is_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split_dir, patient in (
                ("train", "chb01"), ("val", "chb02"), ("test", "chb03")
            ):
                signal = np.linspace(-1.0, 1.0, 2 * 2560, dtype=np.float32).reshape(2, 2560)
                self._write(root / split_dir / f"{patient}_clip.pkl", signal, 0)

            summary, anomalies = audit.audit_post_loader(root, use_fft=True)

            self.assertEqual(anomalies, [])
            self.assertEqual(summary["train"]["checked"], 1)
            self.assertEqual(summary["dev"]["checked"], 1)
            self.assertEqual(summary["test"]["checked"], 1)


if __name__ == "__main__":
    unittest.main()
