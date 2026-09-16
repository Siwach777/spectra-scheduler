import json
import tempfile
import unittest
from pathlib import Path

from spectra_scheduler.comparison import (
    run_repeated_comparison,
    run_repeated_track_evaluation,
)
from spectra_scheduler.scenario_io import (
    build_scenario_from_definition,
    load_scenario_file,
    scenario_name,
)


def example_definition() -> dict[str, object]:
    return {
        "name": "file-example",
        "num_bands": 5,
        "duration": 30,
        "receiver": {
            "detection_probability": 0.9,
            "false_alarm_probability": 0.02,
            "seed_offset": 20,
        },
        "emitters": [
            {
                "type": "periodic",
                "emitter_id": "fixed",
                "band": 1,
                "period": 4,
                "phase": "random",
                "power_dbm": -75.0,
                "pulse_width_us": 0.8,
            },
            {
                "type": "frequency-hopping",
                "emitter_id": "hopper",
                "bands": [0, 3, 4],
                "period": 3,
                "phase": "random",
                "power_dbm": -82.0,
                "pulse_width_us": 1.2,
            },
            {
                "type": "windowed",
                "start_time": 10,
                "emitter": {
                    "type": "jittered-periodic",
                    "emitter_id": "late",
                    "band": 2,
                    "period": 5,
                    "jitter": 1,
                    "phase": 0,
                    "seed_offset": 7,
                },
            },
        ],
    }


class ScenarioIoTests(unittest.TestCase):
    def test_builds_supported_emitters_from_definition(self) -> None:
        scenario = build_scenario_from_definition(example_definition(), seed=3)

        self.assertEqual(scenario.num_bands, 5)
        self.assertEqual(scenario.duration, 30)
        self.assertEqual(len(scenario.emitters), 3)
        self.assertEqual(scenario.receiver.seed, 23)
        self.assertGreater(len(scenario.generate_truth()), 0)

    def test_file_scenario_is_seeded_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps(example_definition()), encoding="utf-8")

            first = load_scenario_file(path, seed=5)
            repeated = load_scenario_file(path, seed=5)
            changed = load_scenario_file(path, seed=6)

        self.assertEqual(first.generate_truth(), repeated.generate_truth())
        self.assertNotEqual(first.generate_truth(), changed.generate_truth())

    def test_reads_scenario_name(self) -> None:
        self.assertEqual(scenario_name(example_definition()), "file-example")
        self.assertEqual(scenario_name({}, fallback="fallback"), "fallback")

    def test_file_scenario_runs_parallel_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(json.dumps(example_definition()), encoding="utf-8")

            sequential = run_repeated_comparison(runs=4, scenario_file=str(path))
            parallel = run_repeated_comparison(
                runs=4,
                workers=2,
                scenario_file=str(path),
            )
            association = run_repeated_track_evaluation(
                runs=4,
                workers=2,
                scenario_file=str(path),
            )

        self.assertEqual(parallel, sequential)
        self.assertEqual(association.runs, 4)

    def test_rejects_unknown_emitter_type(self) -> None:
        definition = example_definition()
        definition["emitters"] = [{"type": "missing"}]

        with self.assertRaisesRegex(ValueError, "unknown emitter type"):
            build_scenario_from_definition(definition)

    def test_rejects_unknown_root_field(self) -> None:
        definition = example_definition()
        definition["unexpected"] = True

        with self.assertRaisesRegex(ValueError, "unknown scenario fields"):
            build_scenario_from_definition(definition)


if __name__ == "__main__":
    unittest.main()
