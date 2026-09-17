import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ("numpy", "h5py"))
if AVAILABLE:
    import h5py
    import numpy as np

    from spectra_scheduler.dataset_io import discover_files, inspect_header, iter_pulses, scan_file


@unittest.skipUnless(AVAILABLE, "install the dataset extra")
class DatasetIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pulse.h5"

    def write(self, rows=7, labels=True):
        with h5py.File(self.path, "w") as f:
            data = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5)
            f["data"] = data
            f["metadata/feature_names"] = np.array(
                ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
            )
            if labels:
                f["labels"] = (np.arange(rows) % 2).reshape(-1, 1)

    def test_chunk_boundaries_and_column_labels(self):
        self.write()
        batches = list(iter_pulses(self.path, 3))
        self.assertEqual([len(b.features) for b in batches], [3, 3, 1])
        self.assertEqual([b.start for b in batches], [0, 3, 6])
        self.assertEqual(batches[-1].labels.shape, (1,))

    def test_reordered_columns_follow_metadata(self):
        self.write()
        with h5py.File(self.path, "r+") as f:
            f["data"][:] = f["data"][:][:, [1, 0, 2, 3, 4]]
            f["metadata/feature_names"][:] = [
                b"Frequency",
                b"ToA",
                b"PulseWidth",
                b"AoA",
                b"Amplitude",
            ]
        batch = next(iter_pulses(self.path))
        np.testing.assert_array_equal(batch.features[0], np.arange(5))

    def test_missing_metadata_rejected(self):
        self.write()
        with h5py.File(self.path, "r+") as f:
            del f["metadata"]
        with self.assertRaisesRegex(ValueError, "feature_names"):
            inspect_header(self.path)

    def test_duplicate_features_rejected(self):
        self.write()
        with h5py.File(self.path, "r+") as f:
            f["metadata/feature_names"][0] = b"Frequency"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inspect_header(self.path)

    def test_misaligned_labels_rejected(self):
        self.write()
        with h5py.File(self.path, "r+") as f:
            del f["labels"]
            f["labels"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "aligned"):
            inspect_header(self.path)

    def test_nonfinite_data_rejected(self):
        self.write()
        with h5py.File(self.path, "r+") as f:
            f["data"][5, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            list(iter_pulses(self.path, 3))

    def test_order_validation_across_chunks(self):
        self.write()
        with h5py.File(self.path, "r+") as f:
            f["data"][3, 0] = -1
        with self.assertRaisesRegex(ValueError, "not ordered"):
            list(iter_pulses(self.path, 3))

    def test_empty_and_unlabelled_files(self):
        self.write(rows=0, labels=False)
        stats, sample, truth, indices = scan_file(self.path, sample_rows=10)
        self.assertEqual(sample.shape, (0, 5))
        self.assertIsNone(truth)
        self.assertEqual(len(indices), 0)
        self.assertIsNone(stats["features"]["toa_us"]["mean"])

    def test_sampling_independent_of_chunk_size_and_labels(self):
        self.write(100)
        a = scan_file(self.path, 7, 15, seed=9)
        with h5py.File(self.path, "r+") as f:
            f["labels"][:] = 123
        b = scan_file(self.path, 13, 15, seed=9)
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[3], b[3])
        self.assertEqual(a[0]["rows"], 100)
        self.assertEqual(a[0]["sample_rows"], 15)

    def test_streaming_stats_match_full_array(self):
        self.write(100)
        stats, _, _, _ = scan_file(self.path, 7)
        values = np.arange(500).reshape(100, 5)
        self.assertAlmostEqual(stats["features"]["toa_us"]["mean"], values[:, 0].mean())
        self.assertAlmostEqual(stats["features"]["toa_us"]["std"], values[:, 0].std())

    def test_discovery_excludes_partial_archive_and_other_split(self):
        root = Path(self.temp.name)
        target = root / "scan/train_scan"
        target.mkdir(parents=True)
        for name in ("b.h5", "a.h5", "c.h5.incomplete"):
            (target / name).touch()
        (root / "archive").mkdir()
        (root / "archive/old.h5").touch()
        self.assertEqual([p.name for p in discover_files(root)], ["a.h5", "b.h5"])
        with self.assertRaises(ValueError):
            discover_files(root, split="all")

    def test_invalid_batch_and_sample_limits(self):
        self.write()
        with self.assertRaises(ValueError):
            list(iter_pulses(self.path, 0))
        with self.assertRaises(ValueError):
            scan_file(self.path, sample_rows=-1)
