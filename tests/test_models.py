import math
import unittest

from spectra_scheduler.models import Observation, SignalMeasurement, Transmission


class ObservationTests(unittest.TestCase):
    def test_public_observation_contains_no_truth_labels(self) -> None:
        observation = Observation(time_step=2, band=1, detections=1)

        self.assertTrue(observation.hit)
        self.assertTrue(observation.listening)
        self.assertFalse(hasattr(observation, "detected_emitters"))
        self.assertFalse(hasattr(observation, "false_alarm"))

    def test_observation_can_include_anonymous_signal_measurements(self) -> None:
        measurement = SignalMeasurement(power_dbm=-78.5)
        observation = Observation(
            time_step=2,
            band=1,
            detections=1,
            measurements=(measurement,),
        )

        self.assertEqual(observation.measurements, (measurement,))
        self.assertFalse(hasattr(measurement, "emitter_id"))

    def test_measurements_cannot_outnumber_detections(self) -> None:
        with self.assertRaises(ValueError):
            Observation(
                time_step=2,
                band=1,
                measurements=(SignalMeasurement(power_dbm=-78.5),),
            )

    def test_rejects_negative_detection_count(self) -> None:
        with self.assertRaises(ValueError):
            Observation(time_step=2, band=1, detections=-1)

    def test_retuning_observation_cannot_report_a_detection(self) -> None:
        with self.assertRaises(ValueError):
            Observation(time_step=2, band=1, detections=1, listening=False)

    def test_transmission_requires_finite_power(self) -> None:
        with self.assertRaises(ValueError):
            Transmission(2, 1, "source", power_dbm=math.inf)

    def test_signal_measurement_requires_finite_power(self) -> None:
        with self.assertRaises(ValueError):
            SignalMeasurement(power_dbm=math.inf)

    def test_pulse_width_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            Transmission(2, 1, "source", pulse_width_us=0.0)
        with self.assertRaises(ValueError):
            SignalMeasurement(power_dbm=-70.0, pulse_width_us=0.0)


if __name__ == "__main__":
    unittest.main()
