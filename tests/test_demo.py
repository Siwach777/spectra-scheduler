import unittest

from spectra_scheduler.demo import run_demo


class DemoTests(unittest.TestCase):
    def test_runs_each_strategy_on_the_same_number_of_transmissions(self) -> None:
        results = run_demo()

        self.assertEqual(set(results), {"round-robin", "random", "revisit-on-hit"})
        totals = {metrics.total_transmissions for metrics in results.values()}
        self.assertEqual(len(totals), 1)
        self.assertGreater(totals.pop(), 0)


if __name__ == "__main__":
    unittest.main()
