import math
import unittest

from spectra_scheduler.models import Observation, Transmission


class ObservationTests(unittest.TestCase):
    def test_public_observation_contains_no_truth_labels(self) -> None:
        observation = Observation(time_step=2, band=1, detections=1)

        self.assertTrue(observation.hit)
        self.assertFalse(hasattr(observation, "detected_emitters"))
        self.assertFalse(hasattr(observation, "false_alarm"))

    def test_rejects_negative_detection_count(self) -> None:
        with self.assertRaises(ValueError):
            Observation(time_step=2, band=1, detections=-1)

    def test_transmission_requires_finite_power(self) -> None:
        with self.assertRaises(ValueError):
            Transmission(2, 1, "radar", power_dbm=math.inf)


if __name__ == "__main__":
    unittest.main()
