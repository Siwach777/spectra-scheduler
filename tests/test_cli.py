import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from spectra_scheduler.cli import main, parse_args


class CliTests(unittest.TestCase):
    def test_parses_report_options(self) -> None:
        args = parse_args(
            [
                "--scenario",
                "crowded",
                "--runs",
                "3",
                "--seed",
                "7",
                "--workers",
                "2",
                "--association",
                "--output",
                "result.json",
            ]
        )

        self.assertEqual(args.scenario, "crowded")
        self.assertEqual(args.runs, 3)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.workers, 2)
        self.assertTrue(args.association)
        self.assertEqual(args.output, "result.json")

    def test_writes_report_from_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            console = StringIO()
            with redirect_stdout(console):
                exit_code = main(
                    [
                        "--scenario",
                        "crowded",
                        "--runs",
                        "2",
                        "--output",
                        str(output_path),
                    ]
                )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(data["scenario"], "crowded")
        self.assertIn("Track association", console.getvalue())
        self.assertIn("Report written to", console.getvalue())


if __name__ == "__main__":
    unittest.main()
