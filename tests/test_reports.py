import csv
import json
import tempfile
import unittest
from pathlib import Path

from spectra_scheduler.reports import (
    REPORT_SCHEMA_VERSION,
    build_experiment_report,
    write_experiment_report,
)


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = build_experiment_report(
            runs=2,
            start_seed=4,
            scenario="crowded",
        )

    def test_builds_reproducible_report_without_runtime_metadata(self) -> None:
        repeated = build_experiment_report(
            runs=2,
            start_seed=4,
            scenario="crowded",
        )

        self.assertEqual(self.report, repeated)
        self.assertEqual(self.report.schema_version, REPORT_SCHEMA_VERSION)
        self.assertEqual(self.report.scenario, "crowded")
        self.assertEqual(self.report.start_seed, 4)
        self.assertEqual(self.report.runs, 2)

    def test_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_experiment_report(
                self.report,
                Path(directory) / "nested" / "report.json",
            )
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(data["scenario"], "crowded")
        self.assertIn("track-aware", data["strategy_results"])
        self.assertIn("mean_pairwise_f1", data["track_association"])
        self.assertNotIn("timestamp", data)

    def test_writes_csv_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_experiment_report(
                self.report,
                Path(directory) / "report.csv",
            )
            with path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))

        strategy_rows = [row for row in rows if row["record_type"] == "strategy"]
        association_rows = [
            row for row in rows if row["record_type"] == "track-association"
        ]
        self.assertEqual(len(strategy_rows), len(self.report.strategy_results))
        self.assertEqual(len(association_rows), 1)
        self.assertEqual(association_rows[0]["name"], "track-aware")

    def test_rejects_unknown_report_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_experiment_report(
                    self.report,
                    Path(directory) / "report.txt",
                )


if __name__ == "__main__":
    unittest.main()
