import unittest

from spectra_scheduler.emitters import FrequencyHoppingEmitter, PeriodicEmitter
from spectra_scheduler.models import Observation
from spectra_scheduler.receiver import Receiver
from spectra_scheduler.schedulers import (
    AdaptiveDwellScheduler,
    DwellSweepScheduler,
    RandomScheduler,
    PeriodAwareScheduler,
    RevisitOnHitScheduler,
    RoundRobinScheduler,
    ShuffledSweepScheduler,
    SlidingWindowUcbScheduler,
    UcbScheduler,
)
from spectra_scheduler.simulation import Simulation


class SchedulerTests(unittest.TestCase):
    def test_round_robin_visits_bands_in_order(self) -> None:
        scheduler = RoundRobinScheduler(start_band=1)
        scheduler.reset(num_bands=3)

        bands = [scheduler.choose_band(step) for step in range(5)]

        self.assertEqual(bands, [1, 2, 0, 1, 2])

    def test_dwell_sweep_stays_before_moving_to_next_band(self) -> None:
        scheduler = DwellSweepScheduler(dwell_steps=2, start_band=1)
        scheduler.reset(num_bands=3)

        bands = [scheduler.choose_band(step) for step in range(7)]

        self.assertEqual(bands, [1, 1, 2, 2, 0, 0, 1])

    def test_dwell_sweep_requires_positive_dwell(self) -> None:
        scheduler = DwellSweepScheduler(dwell_steps=0)

        with self.assertRaises(ValueError):
            scheduler.reset(num_bands=3)

    def test_adaptive_dwell_extends_after_a_hit(self) -> None:
        scheduler = AdaptiveDwellScheduler(
            minimum_dwell_steps=2,
            hit_extension_steps=2,
            maximum_dwell_steps=4,
        )
        scheduler.reset(num_bands=2)

        selected_bands: list[int] = []
        for time_step in range(5):
            band = scheduler.choose_band(time_step)
            selected_bands.append(band)
            scheduler.observe(Observation(time_step, band, int(time_step == 0)))

        self.assertEqual(selected_bands, [0, 0, 0, 0, 1])

    def test_adaptive_dwell_does_not_count_retuning_as_listening(self) -> None:
        scheduler = AdaptiveDwellScheduler(minimum_dwell_steps=1)
        scheduler.reset(num_bands=2)

        first_band = scheduler.choose_band(time_step=0)
        scheduler.observe(Observation(0, first_band))
        second_band = scheduler.choose_band(time_step=1)
        scheduler.observe(Observation(1, second_band, listening=False))

        self.assertEqual(scheduler.choose_band(time_step=2), second_band)

    def test_adaptive_dwell_rejects_maximum_below_minimum(self) -> None:
        scheduler = AdaptiveDwellScheduler(
            minimum_dwell_steps=3,
            maximum_dwell_steps=2,
        )

        with self.assertRaises(ValueError):
            scheduler.reset(num_bands=2)

    def test_random_scheduler_restarts_from_same_seed(self) -> None:
        scheduler = RandomScheduler(seed=42)
        scheduler.reset(num_bands=4)
        first_run = [scheduler.choose_band(step) for step in range(8)]

        scheduler.reset(num_bands=4)
        second_run = [scheduler.choose_band(step) for step in range(8)]

        self.assertEqual(first_run, second_run)

    def test_shuffled_sweep_visits_every_band_once_per_cycle(self) -> None:
        scheduler = ShuffledSweepScheduler(seed=6)
        scheduler.reset(num_bands=4)

        bands = [scheduler.choose_band(step) for step in range(8)]

        self.assertEqual(set(bands[:4]), {0, 1, 2, 3})
        self.assertEqual(set(bands[4:]), {0, 1, 2, 3})

    def test_shuffled_sweep_is_repeatable(self) -> None:
        scheduler = ShuffledSweepScheduler(seed=6)
        scheduler.reset(num_bands=4)
        first_run = [scheduler.choose_band(step) for step in range(8)]

        scheduler.reset(num_bands=4)
        second_run = [scheduler.choose_band(step) for step in range(8)]

        self.assertEqual(first_run, second_run)

    def test_revisit_scheduler_returns_to_a_band_after_a_hit(self) -> None:
        simulation = Simulation(
            num_bands=3,
            duration=6,
            emitters=(PeriodicEmitter("fixed", band=1, period=2, phase=1),),
        )

        result = simulation.run(RevisitOnHitScheduler())

        self.assertEqual([item.band for item in result.observations], [0, 1, 1, 2, 0, 1])

    def test_ucb_scheduler_tries_every_band_before_using_scores(self) -> None:
        scheduler = UcbScheduler()
        scheduler.reset(num_bands=2)

        first_band = scheduler.choose_band(time_step=0)
        scheduler.observe(Observation(time_step=0, band=first_band, detections=1))
        second_band = scheduler.choose_band(time_step=1)
        scheduler.observe(Observation(time_step=1, band=second_band))
        third_band = scheduler.choose_band(time_step=2)

        self.assertEqual([first_band, second_band, third_band], [0, 1, 0])

    def test_ucb_retries_a_band_after_retuning(self) -> None:
        scheduler = UcbScheduler()
        scheduler.reset(num_bands=2)

        first_band = scheduler.choose_band(time_step=0)
        scheduler.observe(Observation(0, first_band))
        second_band = scheduler.choose_band(time_step=1)
        scheduler.observe(Observation(1, second_band, listening=False))

        self.assertEqual(scheduler.choose_band(time_step=2), second_band)

    def test_sliding_ucb_forgets_old_observations(self) -> None:
        scheduler = SlidingWindowUcbScheduler(window_size=2, exploration=0.0)
        scheduler.reset(num_bands=2)

        selected_bands: list[int] = []
        detections = [1, 0, 0, 0, 0]
        for time_step, detected in enumerate(detections):
            band = scheduler.choose_band(time_step)
            selected_bands.append(band)
            scheduler.observe(Observation(time_step, band, detected))

        self.assertEqual(selected_bands, [0, 1, 0, 0, 1])

    def test_sliding_ucb_requires_a_positive_window(self) -> None:
        scheduler = SlidingWindowUcbScheduler(window_size=0)

        with self.assertRaises(ValueError):
            scheduler.reset(num_bands=2)

    def test_period_aware_scheduler_probes_then_returns_when_due(self) -> None:
        scheduler = PeriodAwareScheduler(probe_steps=4)
        scheduler.reset(num_bands=3)

        selected_bands: list[int] = []
        hit_steps = {0, 3, 6}
        for time_step in range(7):
            band = scheduler.choose_band(time_step)
            selected_bands.append(band)
            detections = int(band == 0 and time_step in hit_steps)
            scheduler.observe(Observation(time_step, band, detections))

        self.assertEqual(selected_bands, [0, 0, 0, 0, 1, 2, 0])

    def test_period_aware_scheduler_limits_unvisited_time(self) -> None:
        scheduler = PeriodAwareScheduler(probe_steps=10, max_band_gap=2)
        scheduler.reset(num_bands=3)

        selected_bands: list[int] = []
        for time_step in range(5):
            band = scheduler.choose_band(time_step)
            selected_bands.append(band)
            detections = int(time_step == 0)
            scheduler.observe(Observation(time_step, band, detections))

        self.assertEqual(selected_bands, [0, 0, 1, 2, 0])

    def test_period_aware_scheduler_retries_after_retuning(self) -> None:
        scheduler = PeriodAwareScheduler()
        scheduler.reset(num_bands=2)

        first_band = scheduler.choose_band(time_step=0)
        scheduler.observe(Observation(0, first_band))
        second_band = scheduler.choose_band(time_step=1)
        scheduler.observe(Observation(1, second_band, listening=False))

        self.assertEqual(scheduler.choose_band(time_step=2), second_band)


class SimulationTests(unittest.TestCase):
    def test_receiver_detects_only_events_in_selected_band(self) -> None:
        simulation = Simulation(
            num_bands=3,
            duration=6,
            emitters=(
                PeriodicEmitter("fixed", band=0, period=2),
                FrequencyHoppingEmitter("hopper", bands=(1, 2), period=1),
            ),
        )

        result = simulation.run(RoundRobinScheduler())

        self.assertEqual([item.band for item in result.observations], [0, 1, 2, 0, 1, 2])
        self.assertEqual(
            [item.detected_emitters for item in result.detection_records],
            [("fixed",), (), (), (), ("hopper",), ("hopper",)],
        )
        self.assertEqual(len(result.transmissions), 9)

    def test_rejects_an_invalid_scheduler_choice(self) -> None:
        class InvalidScheduler:
            def reset(self, num_bands: int) -> None:
                pass

            def choose_band(self, time_step: int) -> int:
                return 99

            def observe(self, observation: object) -> None:
                pass

        simulation = Simulation(num_bands=3, duration=2, emitters=())

        with self.assertRaises(ValueError):
            simulation.run(InvalidScheduler())

    def test_band_changes_consume_receiver_retuning_steps(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=5,
            emitters=(),
            receiver=Receiver(retune_steps=1),
        )

        result = simulation.run(RoundRobinScheduler())

        self.assertEqual(
            [observation.listening for observation in result.observations],
            [True, False, False, False, False],
        )

    def test_longer_band_change_takes_more_retuning_steps(self) -> None:
        class FarJumpScheduler:
            def reset(self, num_bands: int) -> None:
                pass

            def choose_band(self, time_step: int) -> int:
                return 0 if time_step == 0 else 5

            def observe(self, observation: Observation) -> None:
                pass

        simulation = Simulation(
            num_bands=6,
            duration=5,
            emitters=(),
            receiver=Receiver(tuning_speed_bands_per_step=2),
        )

        result = simulation.run(FarJumpScheduler())

        self.assertEqual(
            [observation.listening for observation in result.observations],
            [True, False, False, False, True],
        )

    def test_precomputed_truth_can_be_reused(self) -> None:
        simulation = Simulation(
            num_bands=2,
            duration=4,
            emitters=(PeriodicEmitter("fixed", band=0, period=2),),
        )
        truth = simulation.generate_truth()

        first = simulation.run(RoundRobinScheduler(), truth=truth)
        second = simulation.run(RoundRobinScheduler(start_band=1), truth=truth)

        self.assertIs(first.transmissions, truth)
        self.assertIs(second.transmissions, truth)


if __name__ == "__main__":
    unittest.main()
