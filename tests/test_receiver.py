import unittest

from spectra_scheduler.models import Transmission
from spectra_scheduler.receiver import Receiver


class ReceiverTests(unittest.TestCase):
    def test_perfect_receiver_detects_visible_event(self) -> None:
        receiver = Receiver()
        event = Transmission(time_step=3, band=2, emitter_id="radar")

        observation = receiver.listen(time_step=3, band=2, visible_events=[event])

        self.assertEqual(observation.detected_emitters, ("radar",))
        self.assertTrue(observation.hit)
        self.assertFalse(observation.false_alarm)

    def test_zero_detection_probability_misses_event(self) -> None:
        receiver = Receiver(detection_probability=0.0)
        event = Transmission(time_step=3, band=2, emitter_id="radar")

        observation = receiver.listen(time_step=3, band=2, visible_events=[event])

        self.assertFalse(observation.hit)

    def test_false_alarm_is_reported_on_empty_band(self) -> None:
        receiver = Receiver(false_alarm_probability=1.0)

        observation = receiver.listen(time_step=3, band=2, visible_events=[])

        self.assertTrue(observation.hit)
        self.assertTrue(observation.false_alarm)
        self.assertEqual(observation.detected_emitters, ())

    def test_noise_is_repeatable_for_same_event(self) -> None:
        receiver = Receiver(detection_probability=0.5, seed=18)
        event = Transmission(time_step=6, band=1, emitter_id="radar")

        first = receiver.listen(time_step=6, band=1, visible_events=[event])
        second = receiver.listen(time_step=6, band=1, visible_events=[event])

        self.assertEqual(first, second)

    def test_rejects_invalid_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            Receiver(detection_probability=1.1)
        with self.assertRaises(ValueError):
            Receiver(false_alarm_probability=-0.1)


if __name__ == "__main__":
    unittest.main()
