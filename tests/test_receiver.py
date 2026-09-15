import unittest

from spectra_scheduler.models import Transmission
from spectra_scheduler.receiver import Receiver


class ReceiverTests(unittest.TestCase):
    def test_perfect_receiver_detects_visible_event(self) -> None:
        receiver = Receiver()
        event = Transmission(time_step=3, band=2, emitter_id="radar")

        record = receiver.listen(time_step=3, band=2, visible_events=[event])

        self.assertEqual(record.detected_emitters, ("radar",))
        self.assertTrue(record.observation.hit)
        self.assertEqual(record.observation.measurements[0].power_dbm, -60.0)
        self.assertFalse(record.false_alarm)

    def test_zero_detection_probability_misses_event(self) -> None:
        receiver = Receiver(detection_probability=0.0)
        event = Transmission(time_step=3, band=2, emitter_id="radar")

        record = receiver.listen(time_step=3, band=2, visible_events=[event])

        self.assertFalse(record.observation.hit)

    def test_signal_below_sensitivity_is_not_detectable(self) -> None:
        receiver = Receiver(sensitivity_dbm=-80.0)
        event = Transmission(
            time_step=3,
            band=2,
            emitter_id="weak-radar",
            power_dbm=-85.0,
        )

        record = receiver.listen(time_step=3, band=2, visible_events=[event])

        self.assertEqual(record.detectable_emitters, ())
        self.assertEqual(record.detected_emitters, ())
        self.assertFalse(record.observation.hit)

    def test_signal_at_sensitivity_is_detectable(self) -> None:
        receiver = Receiver(sensitivity_dbm=-80.0)
        event = Transmission(
            time_step=3,
            band=2,
            emitter_id="radar",
            power_dbm=-80.0,
        )

        record = receiver.listen(time_step=3, band=2, visible_events=[event])

        self.assertEqual(record.detectable_emitters, ("radar",))
        self.assertEqual(record.detected_emitters, ("radar",))

    def test_false_alarm_is_reported_on_empty_band(self) -> None:
        receiver = Receiver(false_alarm_probability=1.0)

        record = receiver.listen(time_step=3, band=2, visible_events=[])

        self.assertTrue(record.observation.hit)
        self.assertEqual(len(record.observation.measurements), 1)
        self.assertTrue(record.false_alarm)
        self.assertEqual(record.detected_emitters, ())

    def test_noise_is_repeatable_for_same_event(self) -> None:
        receiver = Receiver(detection_probability=0.5, noise_std_db=3.0, seed=18)
        event = Transmission(time_step=6, band=1, emitter_id="radar")

        first = receiver.listen(time_step=6, band=1, visible_events=[event])
        second = receiver.listen(time_step=6, band=1, visible_events=[event])

        self.assertEqual(first, second)

    def test_rejects_invalid_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            Receiver(detection_probability=1.1)
        with self.assertRaises(ValueError):
            Receiver(false_alarm_probability=-0.1)
        with self.assertRaises(ValueError):
            Receiver(noise_std_db=-0.1)
        with self.assertRaises(ValueError):
            Receiver(retune_steps=-1)
        with self.assertRaises(ValueError):
            Receiver(tuning_speed_bands_per_step=0)

    def test_retuning_duration_grows_with_band_distance(self) -> None:
        receiver = Receiver(retune_steps=1, tuning_speed_bands_per_step=2)

        self.assertEqual(receiver.retune_duration(2, 2), 0)
        self.assertEqual(receiver.retune_duration(0, 1), 1)
        self.assertEqual(receiver.retune_duration(0, 2), 1)
        self.assertEqual(receiver.retune_duration(0, 3), 2)
        self.assertEqual(receiver.retune_duration(0, 5), 3)


if __name__ == "__main__":
    unittest.main()
