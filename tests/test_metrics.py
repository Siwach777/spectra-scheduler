import unittest

from spectra_scheduler.emitters import FrequencyHoppingEmitter, PeriodicEmitter
from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.schedulers import RoundRobinScheduler
from spectra_scheduler.simulation import Simulation


class MetricsTests(unittest.TestCase):
    def test_calculates_metrics_from_small_trace(self) -> None:
        simulation = Simulation(
            num_bands=3,
            duration=6,
            emitters=(
                PeriodicEmitter("fixed", band=0, period=2),
                FrequencyHoppingEmitter("hopper", bands=(1, 2), period=1),
            ),
        )

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.total_transmissions, 9)
        self.assertEqual(metrics.detected_transmissions, 3)
        self.assertAlmostEqual(metrics.interception_ratio, 1 / 3)
        self.assertEqual(metrics.hit_rate, 0.5)
        self.assertEqual(metrics.detected_emitters, 2)
        self.assertEqual(metrics.total_emitters, 2)
        self.assertEqual(metrics.mean_first_detection_delay, 2.0)

    def test_empty_scenario_has_zero_metrics(self) -> None:
        simulation = Simulation(num_bands=2, duration=4, emitters=())

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.total_transmissions, 0)
        self.assertEqual(metrics.interception_ratio, 0.0)
        self.assertEqual(metrics.mean_first_detection_delay, 0.0)


if __name__ == "__main__":
    unittest.main()
