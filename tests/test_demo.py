import unittest

from spectra_scheduler.comparison import (
    build_comparison_scenario,
    run_comparison,
    run_repeated_comparison,
)


class DemoTests(unittest.TestCase):
    def test_runs_each_strategy_on_the_same_number_of_transmissions(self) -> None:
        results = run_comparison()

        self.assertEqual(
            set(results),
            {
                "round-robin",
                "random",
                "shuffled-sweep",
                "revisit-on-hit",
                "ucb-bandit",
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
        truth = build_comparison_scenario(seed=5).generate_truth()
        search_times = [event.time_step for event in truth if event.emitter_id == "search"]
        burst_times = [event.time_step for event in truth if event.emitter_id == "burst"]
        tracking_events = [event for event in truth if event.emitter_id == "tracking"]

        self.assertTrue(search_times)
        self.assertLess(max(search_times), 36)
        self.assertTrue(burst_times)
        self.assertGreaterEqual(min(burst_times), 20)
        self.assertTrue(all(event.band == 4 for event in tracking_events if event.time_step < 30))
        self.assertTrue(all(event.band == 0 for event in tracking_events if event.time_step >= 30))

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


if __name__ == "__main__":
    unittest.main()
