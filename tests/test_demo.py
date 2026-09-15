import unittest

from spectra_scheduler.comparison import (
    run_comparison,
    run_repeated_comparison,
)
from spectra_scheduler.scenarios import (
    SCENARIO_NAMES,
    build_acquisition_scenario,
    build_change_scenario,
    build_comparison_scenario,
    build_scenario,
    build_tracking_scenario,
)


class DemoTests(unittest.TestCase):
    def test_runs_each_strategy_on_the_same_number_of_transmissions(self) -> None:
        results = run_comparison()

        self.assertEqual(
            set(results),
            {
                "round-robin",
                "dwell-sweep",
                "adaptive-dwell",
                "random",
                "shuffled-sweep",
                "revisit-on-hit",
                "ucb-bandit",
                "sliding-ucb",
                "bayesian-band",
                "change-aware",
                "transition-band",
                "track-aware",
                "period-aware",
            },
        )
        totals = {metrics.total_transmissions for metrics in results.values()}
        self.assertEqual(len(totals), 1)
        self.assertGreater(totals.pop(), 0)

    def test_same_seed_reproduces_comparison(self) -> None:
        self.assertEqual(run_comparison(seed=17), run_comparison(seed=17))

    def test_scenario_seed_changes_emitter_timing(self) -> None:
        first_truth = build_comparison_scenario(seed=2).generate_truth()
        second_truth = build_comparison_scenario(seed=3).generate_truth()

        self.assertNotEqual(first_truth, second_truth)

    def test_scenario_contains_entry_exit_and_mode_change(self) -> None:
        scenario = build_comparison_scenario(seed=5)
        truth = scenario.generate_truth()
        search_times = [event.time_step for event in truth if event.emitter_id == "search"]
        burst_times = [event.time_step for event in truth if event.emitter_id == "burst"]
        tracking_events = [event for event in truth if event.emitter_id == "tracking"]
        scanner_events = [event for event in truth if event.emitter_id == "scanner"]

        self.assertTrue(search_times)
        self.assertLess(max(search_times), 36)
        self.assertTrue(burst_times)
        self.assertGreaterEqual(min(burst_times), 20)
        self.assertTrue(all(event.band == 4 for event in tracking_events if event.time_step < 30))
        self.assertTrue(all(event.band == 0 for event in tracking_events if event.time_step >= 30))
        scanner_bands = [event.band for event in scanner_events]
        self.assertTrue(scanner_bands)
        self.assertTrue(
            all(abs(first - second) == 1 for first, second in zip(scanner_bands, scanner_bands[1:]))
        )
        self.assertEqual(scenario.receiver.sensitivity_dbm, -90.0)
        self.assertGreater(scenario.receiver.noise_std_db, 0.0)
        self.assertEqual(scenario.receiver.retune_steps, 1)
        self.assertEqual(scenario.receiver.tuning_speed_bands_per_step, 2)
        self.assertGreater(len({event.power_dbm for event in truth}), 1)

    def test_repeated_comparison_summarizes_each_strategy(self) -> None:
        summary = run_repeated_comparison(runs=4, start_seed=8)

        self.assertEqual(set(summary), set(run_comparison()))
        self.assertTrue(all(stats.runs == 4 for stats in summary.values()))
        self.assertTrue(
            all(0.0 <= stats.mean_interception_ratio <= 1.0 for stats in summary.values())
        )
        self.assertTrue(
            all(0.0 <= stats.mean_emitter_discovery_ratio <= 1.0 for stats in summary.values())
        )
        self.assertTrue(
            all(0.0 <= stats.mean_reacquisition_ratio <= 1.0 for stats in summary.values())
        )
        self.assertTrue(
            all(0.0 <= stats.mean_sensitivity_loss_rate <= 1.0 for stats in summary.values())
        )
        self.assertTrue(
            all(0.0 <= stats.mean_retuning_fraction <= 1.0 for stats in summary.values())
        )
        self.assertTrue(all(stats.mean_reacquisition_delay >= 0 for stats in summary.values()))
        self.assertTrue(all(stats.mean_max_band_gap >= 0 for stats in summary.values()))

    def test_repeated_comparison_requires_a_run(self) -> None:
        with self.assertRaises(ValueError):
            run_repeated_comparison(runs=0)

    def test_parallel_comparison_matches_sequential_result(self) -> None:
        sequential = run_repeated_comparison(runs=8, start_seed=11)
        parallel = run_repeated_comparison(runs=8, start_seed=11, workers=2)

        self.assertEqual(parallel, sequential)

    def test_repeated_comparison_requires_a_worker(self) -> None:
        with self.assertRaises(ValueError):
            run_repeated_comparison(runs=2, workers=0)

    def test_focused_scenarios_isolate_their_target_behavior(self) -> None:
        acquisition = build_acquisition_scenario(seed=3)
        tracking = build_tracking_scenario(seed=3)
        change = build_change_scenario(seed=3)

        self.assertEqual(len({event.emitter_id for event in acquisition.generate_truth()}), 4)
        tracking_bands = [event.band for event in tracking.generate_truth()]
        self.assertTrue(tracking_bands)
        self.assertTrue(
            all(
                abs(first - second) == 1
                for first, second in zip(tracking_bands, tracking_bands[1:])
            )
        )
        self.assertEqual(len(change.generate_emitter_changes()), 1)

    def test_named_scenarios_can_run_repeated_comparisons(self) -> None:
        for scenario in SCENARIO_NAMES:
            results = run_repeated_comparison(runs=2, scenario=scenario)
            self.assertEqual(set(results), set(run_comparison(scenario=scenario)))

    def test_rejects_unknown_scenario(self) -> None:
        with self.assertRaises(ValueError):
            build_scenario("missing")


if __name__ == "__main__":
    unittest.main()
