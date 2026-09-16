import unittest

from spectra_scheduler.emitters import (
    FrequencyHoppingEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
)
from spectra_scheduler.metrics import calculate_metrics, calculate_track_metrics
from spectra_scheduler.receiver import Receiver
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
        self.assertEqual(metrics.eligible_transmissions, 3)
        self.assertEqual(metrics.detectable_transmissions, 3)
        self.assertEqual(metrics.sensitivity_misses, 0)
        self.assertEqual(metrics.sensitivity_loss_rate, 0.0)
        self.assertEqual(metrics.detected_transmissions, 3)
        self.assertEqual(metrics.probability_of_detection, 1.0)
        self.assertAlmostEqual(metrics.interception_ratio, 1 / 3)
        self.assertEqual(metrics.false_alarms, 0)
        self.assertEqual(metrics.false_alarm_rate, 0.0)
        self.assertEqual(metrics.hit_rate, 0.5)
        self.assertEqual(metrics.detected_emitters, 2)
        self.assertEqual(metrics.total_emitters, 2)
        self.assertEqual(metrics.emitter_discovery_ratio, 1.0)
        self.assertEqual(metrics.mean_first_detection_delay, 2.0)
        self.assertEqual(metrics.total_emitter_changes, 0)
        self.assertEqual(metrics.reacquisition_ratio, 0.0)
        self.assertEqual(metrics.mean_reacquisition_delay, 0.0)
        self.assertEqual(metrics.max_band_gap, 2)

    def test_empty_scenario_has_zero_metrics(self) -> None:
        simulation = Simulation(num_bands=2, duration=4, emitters=())

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.total_transmissions, 0)
        self.assertEqual(metrics.interception_ratio, 0.0)
        self.assertEqual(metrics.emitter_discovery_ratio, 0.0)
        self.assertEqual(metrics.mean_first_detection_delay, 0.0)

    def test_separates_missed_detections_from_false_alarms(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=4,
            emitters=(PeriodicEmitter("fixed", band=0, period=2),),
            receiver=Receiver(detection_probability=0.0, false_alarm_probability=1.0),
        )

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.eligible_transmissions, 2)
        self.assertEqual(metrics.detected_transmissions, 0)
        self.assertEqual(metrics.probability_of_detection, 0.0)
        self.assertEqual(metrics.false_alarms, 2)
        self.assertEqual(metrics.false_alarm_rate, 1.0)
        self.assertEqual(metrics.hit_rate, 0.5)

    def test_measures_detection_after_an_emitter_changes_mode(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=6,
            emitters=(
                ModeSwitchingEmitter(
                    first_mode=PeriodicEmitter("changing", band=0, period=1),
                    second_mode=PeriodicEmitter("changing", band=1, period=1),
                    switch_time=2,
                ),
            ),
        )

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.total_emitter_changes, 1)
        self.assertEqual(metrics.reacquired_changes, 1)
        self.assertEqual(metrics.reacquisition_ratio, 1.0)
        self.assertEqual(metrics.mean_reacquisition_delay, 1.0)

    def test_counts_tuned_signals_below_receiver_sensitivity(self) -> None:
        simulation = Simulation(
            num_bands=1,
            duration=4,
            emitters=(
                PeriodicEmitter("weak", band=0, period=1, power_dbm=-95.0),
            ),
            receiver=Receiver(sensitivity_dbm=-90.0),
        )

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.eligible_transmissions, 4)
        self.assertEqual(metrics.detectable_transmissions, 0)
        self.assertEqual(metrics.sensitivity_misses, 4)
        self.assertEqual(metrics.sensitivity_loss_rate, 1.0)
        self.assertEqual(metrics.probability_of_detection, 0.0)

    def test_weak_signal_is_an_opportunity_for_a_false_alarm(self) -> None:
        simulation = Simulation(
            num_bands=1,
            duration=3,
            emitters=(
                PeriodicEmitter("weak", band=0, period=1, power_dbm=-95.0),
            ),
            receiver=Receiver(
                sensitivity_dbm=-90.0,
                false_alarm_probability=1.0,
            ),
        )

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.false_alarms, 3)
        self.assertEqual(metrics.false_alarm_rate, 1.0)

    def test_counts_time_lost_while_retuning(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=4,
            emitters=(PeriodicEmitter("fixed", band=0, period=1),),
            receiver=Receiver(retune_steps=1),
        )

        metrics = calculate_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.listening_steps, 1)
        self.assertEqual(metrics.retuning_steps, 3)
        self.assertEqual(metrics.retuning_fraction, 0.75)
        self.assertEqual(metrics.eligible_transmissions, 1)

    def test_measures_clean_signal_association(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=4,
            emitters=(
                PeriodicEmitter(
                    "first",
                    band=0,
                    period=2,
                    power_dbm=-70.0,
                    pulse_width_us=0.5,
                ),
                PeriodicEmitter(
                    "second",
                    band=1,
                    period=2,
                    phase=1,
                    power_dbm=-82.0,
                    pulse_width_us=1.5,
                ),
            ),
        )

        metrics = calculate_track_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.assigned_measurements, 4)
        self.assertEqual(metrics.confirmed_tracks, 2)
        self.assertEqual(metrics.mixed_tracks, 0)
        self.assertEqual(metrics.association_purity, 1.0)
        self.assertEqual(metrics.pairwise_precision, 1.0)
        self.assertEqual(metrics.pairwise_recall, 1.0)
        self.assertEqual(metrics.pairwise_f1, 1.0)
        self.assertEqual(metrics.mean_tracks_per_emitter, 1.0)

    def test_detects_ambiguous_signal_association(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=4,
            emitters=(
                PeriodicEmitter("first", band=0, period=2, power_dbm=-75.0),
                PeriodicEmitter(
                    "second",
                    band=1,
                    period=2,
                    phase=1,
                    power_dbm=-75.0,
                ),
            ),
        )

        metrics = calculate_track_metrics(simulation.run(RoundRobinScheduler()))

        self.assertEqual(metrics.confirmed_tracks, 1)
        self.assertEqual(metrics.mixed_tracks, 1)
        self.assertEqual(metrics.association_purity, 0.5)
        self.assertAlmostEqual(metrics.pairwise_precision, 1 / 3)
        self.assertEqual(metrics.pairwise_recall, 1.0)
        self.assertEqual(metrics.pairwise_f1, 0.5)


if __name__ == "__main__":
    unittest.main()
