import unittest

from spectra_scheduler.emitters import FrequencyHoppingEmitter, PeriodicEmitter
from spectra_scheduler.models import Observation
from spectra_scheduler.schedulers import (
    RandomScheduler,
    RevisitOnHitScheduler,
    RoundRobinScheduler,
    UcbScheduler,
)
from spectra_scheduler.simulation import Simulation


class SchedulerTests(unittest.TestCase):
    def test_round_robin_visits_bands_in_order(self) -> None:
        scheduler = RoundRobinScheduler(start_band=1)
        scheduler.reset(num_bands=3)

        bands = [scheduler.choose_band(step) for step in range(5)]

        self.assertEqual(bands, [1, 2, 0, 1, 2])

    def test_random_scheduler_restarts_from_same_seed(self) -> None:
        scheduler = RandomScheduler(seed=42)
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
        scheduler.observe(Observation(time_step=0, band=first_band, detected_emitters=("a",)))
        second_band = scheduler.choose_band(time_step=1)
        scheduler.observe(Observation(time_step=1, band=second_band))
        third_band = scheduler.choose_band(time_step=2)

        self.assertEqual([first_band, second_band, third_band], [0, 1, 0])


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
            [item.detected_emitters for item in result.observations],
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


if __name__ == "__main__":
    unittest.main()
