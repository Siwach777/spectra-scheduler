"""Receiver boundary, information isolation and streaming parity tests."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = importlib.util.find_spec("h5py") is not None
if AVAILABLE:
    import h5py
    import numpy as np

    from spectra_scheduler.pulse_replay import DwellAction, PulseReplay, ReplayConfig


@unittest.skipUnless(AVAILABLE, "install dataset extra")
class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pulses.h5"

    def write(self, rows, labels=True):
        with h5py.File(self.path, "w") as f:
            f["data"] = np.array(rows, dtype=float).reshape(-1, 5)
            f["metadata/feature_names"] = np.array(
                ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
            )
            if labels:
                f["labels"] = np.arange(len(rows)) % 3

    def config(self, **kwargs):
        return ReplayConfig(
            **dict(stop_us=30, max_frequency_mhz=20, bandwidth_mhz=10, retune_us=5, **kwargs)
        )

    def test_boundaries_retune_and_whole_pulse(self):
        self.write(
            [
                [0, 0, 1, 0, -20],
                [9, 5, 2, 0, -20],
                [10, 10, 1, 0, -20],
                [14, 15, 1, 0, -20],
                [15, 15, 1, 0, -20],
                [25, 15, 1, 0, -20],
                [29, 20, 1, 0, -20],
                [30, 15, 1, 0, -20],
            ]
        )
        with PulseReplay(self.path, self.config(batch_rows=2), source_mode="stare") as env:
            a = env.step(DwellAction(5, 10))
            b = env.step(DwellAction(15, 10))
            c = env.step(DwellAction(15, 10))
            np.testing.assert_array_equal(a.pulses[:, 0], [0])
            np.testing.assert_array_equal(b.pulses[:, 0], [15])
            np.testing.assert_array_equal(c.pulses[:, 0], [25])
            self.assertEqual((b.listening_start_us, b.end_us), (15, 25))
            report = env.report()
            self.assertEqual(report["truth_pulses"], 6)
            self.assertEqual(report["delivered_pulses"], 3)
            self.assertEqual(report["retuning_us"], 5)
            self.assertEqual(report["listening_us"], 25)
            self.assertTrue(env.done)
            self.assertFalse(hasattr(a, "labels"))
            with self.assertRaises(RuntimeError):
                env.step(DwellAction(5, 1))

    def test_chunk_and_schedule_independent_detection(self):
        self.write([[i, 5, 0.1, 0, -20] for i in range(30)])
        outputs = []
        for chunk, dwell in [(1, 30), (7, 10), (100, 30)]:
            with PulseReplay(
                self.path,
                self.config(batch_rows=chunk, detection_probability=0.5, seed=9),
                source_mode="stare",
            ) as env:
                parts = []
                while not env.done:
                    parts.append(env.step(DwellAction(5, dwell)).pulses)
                outputs.append(np.concatenate(parts))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[0], outputs[2])
        self.assertGreater(len(outputs[0]), 0)
        self.assertLess(len(outputs[0]), 30)

    def test_overflow_and_sensitivity(self):
        self.write([[i, 5, 0.1, 0, -i] for i in range(30)])
        with PulseReplay(
            self.path,
            self.config(max_observation_pulses=2, sensitivity_db=-4, batch_rows=3),
            source_mode="stare",
        ) as env:
            obs = env.step(DwellAction(5, 30))
            self.assertEqual(len(obs.pulses), 2)
            self.assertEqual(obs.overflow_pulses, 3)
            self.assertEqual(env.report()["detectable_pulses"], 5)
            self.assertEqual(env.report()["emitters_discovered"], 2)
            outcome = env.evaluation_outcome()
            self.assertEqual(outcome.truth_count, 30)
            self.assertEqual(outcome.eligible_count, 30)
            self.assertEqual(outcome.detectable_count, 5)
            self.assertEqual(outcome.captured_count, 5)  # Before overflow.
            self.assertEqual(outcome.first_intercept_seconds, 0)

    def test_empty_and_unlabelled(self):
        for rows in ([], [[1, 5, 1, 0, -20]]):
            self.write(rows, labels=False)
            with PulseReplay(self.path, self.config(), source_mode="stare") as env:
                obs = env.step(DwellAction(5, 30))
                self.assertEqual(len(obs.pulses), len(rows))
                self.assertIsNone(env.report()["emitters_present"])

    def test_start_offset_and_duplicate_arrivals(self):
        self.write([[i, 5, 1, 0, -20] for i in (0, 10, 10, 10, 20, 30)])
        with PulseReplay(
            self.path, self.config(start_us=10, batch_rows=2), source_mode="stare"
        ) as env:
            obs = env.step(DwellAction(5, 20))
            self.assertEqual(len(obs.pulses), 4)
            self.assertEqual(env.report()["truth_pulses"], 4)

    def test_slew_and_horizon_during_retune(self):
        self.write([[i, 5, 1, 0, -20] for i in range(30)])
        with PulseReplay(self.path, self.config(slew_mhz_per_us=0.1), source_mode="stare") as env:
            env.step(DwellAction(5, 10))
            obs = env.step(DwellAction(15, 10))
            self.assertEqual((obs.listening_start_us, obs.end_us), (30, 30))
            self.assertEqual(env.report()["retuning_us"], 20)
            outcome = env.evaluation_outcome()
            self.assertEqual(outcome.truth_count, 20)
            self.assertEqual(outcome.eligible_count, 0)
            self.assertIsNone(outcome.first_intercept_seconds)

    def test_invalid_source_action_and_config(self):
        self.write([])
        with self.assertRaises(ValueError):
            PulseReplay(self.path, self.config(), source_mode="scan")
        for kw in (
            {"batch_rows": 0},
            {"seed": -1},
            {"start_us": float("nan")},
            {"slew_mhz_per_us": 0},
            {"max_observation_pulses": 1.5},
        ):
            with self.assertRaises(ValueError):
                self.config(**kw)
        with PulseReplay(self.path, self.config(), source_mode="stare") as env:
            for action in (DwellAction(0, 1), DwellAction(5, 0), DwellAction(5, float("nan"))):
                with self.assertRaises(ValueError):
                    env.step(action)
            self.assertEqual(env.time_us, 0)

    def test_invalid_physical_pulse(self):
        self.write([[1, 5, -1, 0, -20]])
        with PulseReplay(self.path, self.config(), source_mode="stare") as env:
            with self.assertRaisesRegex(ValueError, "width"):
                env.step(DwellAction(5, 30))

    def test_label_changes_cannot_change_observations(self):
        self.write([[i, 5, 0.1, 0, -20] for i in range(30)])
        with PulseReplay(self.path, self.config(), source_mode="stare") as env:
            before = env.step(DwellAction(5, 30)).pulses
        with h5py.File(self.path, "r+") as f:
            f["labels"][:] = 999
        with PulseReplay(self.path, self.config(), source_mode="stare") as env:
            after = env.step(DwellAction(5, 30)).pulses
            self.assertEqual(env.report()["emitters_present"], 1)
        np.testing.assert_array_equal(before, after)

    def test_stream_matches_independent_in_memory_receiver(self):
        rng = np.random.default_rng(51)
        rows = np.column_stack(
            (
                np.sort(rng.uniform(0, 30, 500)),
                rng.uniform(0, 20, 500),
                rng.uniform(0.01, 2, 500),
                np.zeros(500),
                rng.uniform(-40, 0, 500),
            )
        )
        self.write(rows)
        for chunk in (1, 17, 1000):
            with PulseReplay(
                self.path, self.config(batch_rows=chunk, sensitivity_db=-20), source_mode="stare"
            ) as env:
                time = 0
                previous = None
                total = 0
                for center, dwell in [(5, 3), (15, 4), (15, 2), (5, 50)]:
                    listen = min(
                        time + (5 if previous is not None and center != previous else 0), 30
                    )
                    end = min(listen + dwell, 30)
                    expected = rows[
                        (rows[:, 0] >= listen)
                        & (rows[:, 0] < end)
                        & (rows[:, 0] + rows[:, 2] <= end)
                        & (rows[:, 1] >= center - 5)
                        & (rows[:, 1] < center + 5)
                        & (rows[:, 4] >= -20)
                    ]
                    actual = env.step(DwellAction(center, dwell))
                    np.testing.assert_array_equal(actual.pulses, expected)
                    outcome = env.evaluation_outcome()
                    self.assertEqual(outcome.captured_count, len(expected))
                    self.assertEqual(outcome.detectable_count, len(expected))
                    self.assertEqual(
                        outcome.truth_count, int(((rows[:, 0] >= time) & (rows[:, 0] < end)).sum())
                    )
                    self.assertEqual(
                        outcome.first_intercept_seconds,
                        (expected[0, 0] - time) / 1e6 if len(expected) else None,
                    )
                    total += len(expected)
                    time, previous = end, center
                self.assertEqual(env.report()["truth_pulses"], len(rows))
                self.assertEqual(env.report()["intercepted_pulses"], total)

    def test_close_releases_stream_and_rejects_steps(self):
        self.write([[i, 5, 1, 0, -20] for i in range(30)])
        env = PulseReplay(self.path, self.config(batch_rows=2), source_mode="stare")
        env.step(DwellAction(5, 1))
        env.close()
        self.assertIsNone(env._stream.gi_frame)
        env.close()
        with self.assertRaises(RuntimeError):
            env.step(DwellAction(5, 1))
