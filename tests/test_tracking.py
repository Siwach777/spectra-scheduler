import unittest

from spectra_scheduler.models import Observation, SignalMeasurement
from spectra_scheduler.tracking import SignalTracker


def measured_observation(
    time_step: int,
    band: int,
    power_dbm: float,
    pulse_width_us: float,
) -> Observation:
    return Observation(
        time_step=time_step,
        band=band,
        detections=1,
        measurements=(SignalMeasurement(power_dbm, pulse_width_us),),
    )


class SignalTrackerTests(unittest.TestCase):
    def test_associates_similar_measurements(self) -> None:
        tracker = SignalTracker()

        tracker.update(measured_observation(1, 1, -80.0, 1.0))
        tracks = tracker.update(measured_observation(3, 2, -78.0, 1.1))

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].observation_count, 2)
        self.assertEqual(tracks[0].last_band, 2)
        self.assertAlmostEqual(tracks[0].mean_power_dbm, -79.0)

    def test_separates_different_pulse_widths(self) -> None:
        tracker = SignalTracker(pulse_width_tolerance_us=0.1)

        tracker.update(measured_observation(1, 1, -80.0, 0.5))
        tracks = tracker.update(measured_observation(2, 1, -80.0, 1.0))

        self.assertEqual(len(tracks), 2)

    def test_expires_old_tracks(self) -> None:
        tracker = SignalTracker(max_age_steps=3)
        tracker.update(measured_observation(1, 1, -80.0, 1.0))

        tracks = tracker.update(Observation(time_step=5, band=0))

        self.assertEqual(tracks, ())

    def test_predicts_band_from_recent_motion(self) -> None:
        tracker = SignalTracker()
        tracker.update(measured_observation(1, 1, -80.0, 1.0))
        track = tracker.update(measured_observation(3, 2, -80.0, 1.0))[0]

        self.assertEqual(track.predicted_band(time_step=5, num_bands=6), 3)

    def test_prediction_stays_inside_spectrum(self) -> None:
        tracker = SignalTracker()
        tracker.update(measured_observation(1, 4, -80.0, 1.0))
        track = tracker.update(measured_observation(2, 5, -80.0, 1.0))[0]

        self.assertEqual(track.predicted_band(time_step=4, num_bands=6), 5)

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            SignalTracker(pulse_width_tolerance_us=0.0)
        with self.assertRaises(ValueError):
            SignalTracker(power_tolerance_db=0.0)
        with self.assertRaises(ValueError):
            SignalTracker(max_age_steps=0)


if __name__ == "__main__":
    unittest.main()
