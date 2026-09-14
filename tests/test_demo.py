import unittest

from spectra_scheduler.comparison import run_comparison


class DemoTests(unittest.TestCase):
    def test_runs_each_strategy_on_the_same_number_of_transmissions(self) -> None:
        results = run_comparison()

        self.assertEqual(
            set(results),
            {"round-robin", "random", "revisit-on-hit", "ucb-bandit"},
        )
        totals = {metrics.total_transmissions for metrics in results.values()}
        self.assertEqual(len(totals), 1)
        self.assertGreater(totals.pop(), 0)

    def test_same_seed_reproduces_comparison(self) -> None:
        self.assertEqual(run_comparison(seed=17), run_comparison(seed=17))


if __name__ == "__main__":
    unittest.main()
