import unittest

from spectra_scheduler.change_detection import BinaryRateChangeDetector


class BinaryRateChangeDetectorTests(unittest.TestCase):
    def test_detects_sustained_rate_drop(self) -> None:
        detector = BinaryRateChangeDetector(
            reference_window_size=4,
            recent_window_size=3,
            minimum_rate_change=0.75,
        )

        changes = [detector.update(0, value) for value in [True] * 4 + [False] * 3]

        self.assertEqual(changes, [False] * 6 + [True])

    def test_tracks_each_key_independently(self) -> None:
        detector = BinaryRateChangeDetector(
            reference_window_size=2,
            recent_window_size=2,
            minimum_rate_change=1.0,
        )

        for value in [True, True, False]:
            self.assertFalse(detector.update(0, value))
            self.assertFalse(detector.update(1, True))

        self.assertTrue(detector.update(0, False))
        self.assertFalse(detector.update(1, True))

    def test_does_not_repeat_immediately_after_a_change(self) -> None:
        detector = BinaryRateChangeDetector(
            reference_window_size=2,
            recent_window_size=2,
            minimum_rate_change=1.0,
        )

        changes = [
            detector.update(0, value)
            for value in [True, True, False, False, False, False]
        ]

        self.assertEqual(changes.count(True), 1)

    def test_reset_discards_old_observations(self) -> None:
        detector = BinaryRateChangeDetector(
            reference_window_size=2,
            recent_window_size=2,
            minimum_rate_change=1.0,
        )
        detector.update(0, True)
        detector.update(0, True)
        detector.reset()

        self.assertFalse(detector.update(0, False))
        self.assertFalse(detector.update(0, False))

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            BinaryRateChangeDetector(reference_window_size=0)
        with self.assertRaises(ValueError):
            BinaryRateChangeDetector(recent_window_size=0)
        with self.assertRaises(ValueError):
            BinaryRateChangeDetector(minimum_rate_change=0.0)


if __name__ == "__main__":
    unittest.main()
