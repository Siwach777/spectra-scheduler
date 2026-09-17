import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ("numpy", "h5py", "sklearn"))
if AVAILABLE:
    import h5py
    import numpy as np

    from spectra_scheduler.dataset_cli import main
    from spectra_scheduler.dataset_evaluation import (
        DatasetConfig,
        predict_clusters,
        run_dataset,
        score_clusters,
        transform_features,
    )


@unittest.skipUnless(AVAILABLE, "install the dataset extra")
class DatasetEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = self.root / "scan/train_scan"
        self.split.mkdir(parents=True)
        self.config = DatasetConfig(sample_rows=200, min_cluster_size=5, min_samples=3)

    def create_file(self, name="a.h5", rows=100, labels=True):
        rng = np.random.default_rng(4)
        target = np.arange(rows) % 2
        x = np.column_stack(
            (
                np.arange(rows),
                100 + 1000 * target + rng.normal(0, 0.1, rows),
                2 + 10 * target + rng.normal(0, 0.01, rows),
                90 * target + rng.normal(0, 0.1, rows),
                -80 + 40 * target,
            )
        )
        path = self.split / name
        with h5py.File(path, "w") as f:
            f["data"] = x
            f["metadata/feature_names"] = np.array(
                ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
            )
            if labels:
                f["labels"] = target
        return path

    def test_separable_fixture_clusters_correctly(self):
        self.create_file()
        report = run_dataset(self.root, self.config, evaluate=True)
        self.assertEqual(report["summary"]["scored_files"], 1)
        self.assertGreater(report["files"][0]["metrics"]["v_measure"], 0.95)
        self.assertEqual(report["files"][0]["statistics"]["sample_rows"], 100)

    def test_macro_aggregation_does_not_merge_file_local_labels(self):
        self.create_file("a.h5")
        self.create_file("b.h5", rows=60)
        report = run_dataset(self.root, self.config, evaluate=True)
        files = report["files"]
        expected = sum(f["metrics"]["v_measure"] for f in files) / 2
        self.assertEqual(report["summary"]["macro_metrics"]["v_measure"], expected)

    def test_parallel_and_sequential_reports_equal(self):
        self.create_file("a.h5")
        self.create_file("b.h5")
        serial = run_dataset(self.root, self.config, evaluate=True)
        parallel = run_dataset(self.root, self.config, evaluate=True, workers=2)
        self.assertEqual(serial, parallel)

    def test_empty_unlabelled_and_corrupt_files_visible(self):
        self.create_file("empty.h5", rows=0)
        self.create_file("unlabelled.h5", labels=False)
        (self.split / "broken.h5").write_bytes(b"not hdf5")
        report = run_dataset(self.root, self.config, evaluate=True)
        self.assertEqual(report["summary"]["empty_files"], 1)
        self.assertEqual(report["summary"]["unlabelled_files"], 1)
        self.assertEqual(report["summary"]["error_files"], 1)
        self.assertEqual(report["summary"]["scored_files"], 0)
        self.assertIsNone(report["summary"]["macro_metrics"])
        json.dumps(report, allow_nan=False)

    def test_all_noise_reports_both_conventions(self):
        self.create_file(rows=4)
        report = run_dataset(self.root, self.config, evaluate=True)
        result = report["files"][0]
        self.assertEqual(result["noise_fraction"], 1)
        self.assertEqual(result["predicted_clusters"], 0)
        self.assertEqual(result["noise_singleton_metrics"]["pairwise_recall"], 0)
        self.assertGreater(result["metrics"]["pairwise_recall"], 0)

    def test_label_changes_do_not_change_features_or_predictions(self):
        path = self.create_file()
        first = run_dataset(self.root, self.config, evaluate=True)["files"][0]
        with h5py.File(path, "r+") as f:
            f["labels"][:] = 99
        second = run_dataset(self.root, self.config, evaluate=True)["files"][0]
        for key in (
            "sample_features_sha256",
            "sample_predictions_sha256",
            "sample_indices_sha256",
            "predicted_clusters",
            "noise_fraction",
        ):
            self.assertEqual(first[key], second[key])

    def test_sampling_is_bounded_and_explicit(self):
        self.create_file(rows=100)
        report = run_dataset(self.root, replace(self.config, sample_rows=30), evaluate=True)
        self.assertEqual(report["summary"]["validated_pulses"], 100)
        self.assertEqual(report["summary"]["scored_pulses"], 30)

    def test_pair_scores_against_hand_count(self):
        result = score_clusters(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(result["pairwise_precision"], 1 / 3)
        self.assertAlmostEqual(result["pairwise_recall"], 1 / 2)
        self.assertAlmostEqual(result["pairwise_f1"], 0.4)
        with self.assertRaises(ValueError):
            score_clusters(np.array([]), np.array([]))

    def test_angle_transform_wraps_and_excludes_absolute_toa(self):
        a = np.array([[0, 100, 2, -180, -80], [1, 100, 2, 180, -80]], dtype=float)
        features = transform_features(a, "signature")
        np.testing.assert_allclose(features[0], features[1], atol=1e-12)
        b = a.copy()
        b[:, 0] += 1e6
        np.testing.assert_array_equal(features, transform_features(b, "signature"))
        np.testing.assert_array_equal(a, transform_features(a, "raw"))

    def test_raw_transform_does_not_mutate_input(self):
        values = np.arange(25, dtype=float).reshape(5, 5)
        original = values.copy()
        predict_clusters(values, replace(self.config, features="raw"))
        np.testing.assert_array_equal(values, original)

    def test_inspect_never_clusters(self):
        self.create_file()
        report = run_dataset(self.root, self.config)
        self.assertNotIn("metrics", report["files"][0])
        self.assertEqual(report["files"][0]["statistics"]["sample_rows"], 0)

    def test_profile_is_opt_in(self):
        self.create_file()
        report = run_dataset(self.root, self.config, profile=True)
        self.assertGreater(report["files"][0]["profile"]["read_pulses_per_second"], 0)

    def test_cli_writes_report_and_returns_error_for_bad_files(self):
        self.create_file()
        output = self.root / "report.json"
        with contextlib.redirect_stderr(io.StringIO()):
            result = main(["inspect", "--root", str(self.root), "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.read_text())["summary"]["valid_files"], 1)
        (self.split / "broken.h5").write_bytes(b"bad")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(["inspect", "--root", str(self.root), "--output", str(output)]), 1
            )

    def test_invalid_configuration_and_empty_selection(self):
        for kw in ({"batch_rows": 0}, {"sample_rows": 1}, {"min_cluster_size": 1}, {"seed": -1}):
            with self.assertRaises(ValueError):
                DatasetConfig(**kw)
        with self.assertRaisesRegex(ValueError, "no completed"):
            run_dataset(self.root, self.config)

    def test_cli_refuses_pulse_file_as_report_output(self):
        path = self.create_file()
        original = path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["inspect", "--root", str(self.root), "--output", str(path)])
        self.assertEqual(path.read_bytes(), original)

    def test_inspection_reports_unlabelled_files(self):
        self.create_file(labels=False)
        self.assertEqual(run_dataset(self.root, self.config)["summary"]["unlabelled_files"], 1)
