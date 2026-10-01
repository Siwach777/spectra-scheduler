"""Automated verification suite for web GUI backend adapter."""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from spectra_scheduler.metrics import calculate_metrics  # noqa: E402
from spectra_scheduler.schedulers import BayesianBandScheduler, UcbScheduler  # noqa: E402
from web.api import (  # noqa: E402
    PRESETS,
    SCHEDULER_REGISTRY,
    build_scenario_instance,
    create_learned_hit_scheduler,
    create_rl_scheduler,
    extract_cognitive_state,
    get_available_scenarios,
    get_available_schedulers,
    run_simulation,
)
from web.server import SpectraConsoleHandler  # noqa: E402


class TestWebAPI(unittest.TestCase):
    def test_gui_metrics_match_native_simulator(self):
        for scenario in get_available_scenarios():
            for key in ("round-robin", "track-aware", "bayesian", "period-aware"):
                with self.subTest(scenario=scenario["id"], policy=key):
                    sim = build_scenario_instance(scenario["id"], seed=13)
                    native = calculate_metrics(sim.run(SCHEDULER_REGISTRY[key][0]()))
                    payload = run_simulation(scenario["id"], "round-robin", key, seed=13)
                    metrics = payload["active"]["data"]["metrics"]
                    for name, value in asdict(native).items():
                        self.assertEqual(metrics[name], value)

    def test_all_non_neural_policy_traces_cover_every_tick(self):
        for scenario in get_available_scenarios():
            for key in SCHEDULER_REGISTRY:
                if key in ("rl-mlp", "learned-hit", "timing-trained"):
                    continue
                with self.subTest(scenario=scenario["id"], policy=key):
                    payload = run_simulation(scenario["id"], "round-robin", key, seed=9)
                    steps = payload["active"]["data"]["steps"]
                    self.assertEqual([s["tick"] for s in steps], list(range(scenario["duration"])))
                    self.assertTrue(all(0 <= s["rx_band"] < scenario["bands"] for s in steps))

    def test_missing_models_never_fall_back(self):
        with patch("web.api.ARTIFACTS_DIR", Path("/tmp/spectra-no-model-artifacts")):
            with self.assertRaises(ValueError):
                create_rl_scheduler()
            with self.assertRaises(ValueError):
                create_learned_hit_scheduler()
            states = {s["id"]: s for s in get_available_schedulers()}
            self.assertFalse(states["rl-mlp"]["available"])
            self.assertFalse(states["learned-hit"]["available"])

    def test_bayesian_state_is_not_presented_as_ucb(self):
        policy = BayesianBandScheduler()
        policy.reset(6)
        state = extract_cognitive_state(policy, 6, 0)
        self.assertEqual(len(state["beliefs"]), 6)
        self.assertEqual(state["ucb_scores"], [])
        policy = UcbScheduler()
        policy.reset(6)
        self.assertEqual(len(extract_cognitive_state(policy, 6, 0)["ucb_scores"]), 6)

    def test_random_policies_use_request_seed(self):
        for key in ("random", "shuffled-sweep"):
            first = run_simulation("crowded", key, key, seed=31)
            repeated = run_simulation("crowded", key, key, seed=31)
            other = run_simulation("crowded", key, key, seed=32)
            self.assertEqual(first, repeated)
            self.assertEqual(first["baseline"]["data"], first["active"]["data"])
            self.assertNotEqual(first["active"]["data"]["steps"], other["active"]["data"]["steps"])

    def test_zero_baseline_uses_absolute_delta(self):
        payload = run_simulation("frequency-agile", "round-robin", "track-aware", seed=42)
        self.assertIsNone(payload["deltas"]["interception_gain_pct"])
        self.assertGreater(payload["deltas"]["interception_delta_pp"], 0)

    def test_scenarios_metadata(self):
        scenarios = get_available_scenarios()
        self.assertGreaterEqual(len(scenarios), 6)
        ids = [s["id"] for s in scenarios]
        self.assertIn("frequency-agile", ids)
        self.assertIn("spatial-scan", ids)
        self.assertIn("change", ids)
        self.assertIn("crowded", ids)

    def test_schedulers_metadata(self):
        schedulers = get_available_schedulers()
        self.assertGreaterEqual(len(schedulers), 6)
        ids = [s["id"] for s in schedulers]
        self.assertIn("round-robin", ids)
        self.assertIn("track-aware", ids)
        self.assertIn("change-aware", ids)
        self.assertIn("bayesian", ids)

    def test_presets(self):
        self.assertIn("agile-hopper", PRESETS)
        self.assertIn("spatial-radar", PRESETS)
        self.assertIn("mode-switch", PRESETS)
        self.assertIn("crowded-battlefield", PRESETS)
        self.assertIn("neural-mlp", PRESETS)
        self.assertIn("baseline-duel", PRESETS)

    def test_run_simulation_neural_mlp(self):
        if not next(s["available"] for s in get_available_schedulers() if s["id"] == "rl-mlp"):
            self.skipTest("No compatible frozen DQN artifact is installed")
        res = run_simulation(
            scenario_id="crowded",
            baseline_id="round-robin",
            active_id="rl-mlp",
            seed=42,
        )
        self.assertIn("active", res)
        self.assertIn("baseline", res)
        self.assertIn("deltas", res)
        cog = res["active"]["data"]["final_cognitive_state"]
        self.assertIn("q_values", cog)
        self.assertEqual(len(cog["q_values"]), 8)

    def test_run_simulation_baselines(self):
        res = run_simulation(
            scenario_id="frequency-agile",
            baseline_id="random",
            active_id="revisit-on-hit",
            seed=42,
        )
        self.assertIn("active", res)
        self.assertIn("baseline", res)
        self.assertEqual(res["active"]["id"], "revisit-on-hit")
        self.assertEqual(res["baseline"]["id"], "random")

    def test_run_simulation_frequency_agile(self):
        res = run_simulation(
            scenario_id="frequency-agile",
            baseline_id="round-robin",
            active_id="track-aware",
            seed=42,
        )
        self.assertIn("scenario", res)
        self.assertIn("truth_grid", res)
        self.assertIn("baseline", res)
        self.assertIn("active", res)
        self.assertIn("deltas", res)

        duration = res["scenario"]["duration"]
        self.assertEqual(len(res["truth_grid"]), duration)
        self.assertEqual(len(res["baseline"]["data"]["steps"]), duration)
        self.assertEqual(len(res["active"]["data"]["steps"]), duration)

        # Baseline metrics checks
        b_metrics = res["baseline"]["data"]["metrics"]
        self.assertIn("interception_ratio", b_metrics)
        self.assertIn("probability_of_detection", b_metrics)
        self.assertIn("retuning_fraction", b_metrics)

        # Active metrics checks
        a_metrics = res["active"]["data"]["metrics"]
        self.assertIn("interception_ratio", a_metrics)

    def test_run_simulation_perturbation(self):
        res = run_simulation(
            scenario_id="change",
            baseline_id="bayesian",
            active_id="change-aware",
            seed=7,
            perturbation="frequency-hop",
        )
        self.assertEqual(res["scenario"]["id"], "change")
        self.assertGreater(len(res["truth_grid"]), 0)


class TestSpectraConsoleHTTP(unittest.TestCase):
    server: Any = None
    thread: Any = None
    port: int = 0

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SpectraConsoleHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.shutdown()
            cls.server.server_close()

    def test_http_get_index(self):
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            body = resp.read().decode("utf-8")
            self.assertIn("Spectra Scheduler", body)

    def test_http_get_presets(self):
        url = f"http://127.0.0.1:{self.port}/api/presets"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("agile-hopper", data)

    def test_invalid_parameters_return_json_without_running(self):
        for payload in (
            [1],
            {"seed": -1},
            {"sensitivity_dbm": -999},
            {"perturbation": "unknown"},
            {"active": "unknown"},
        ):
            with self.subTest(payload=payload):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/api/run",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(error.exception.code, 400)
                self.assertIn("error", json.loads(error.exception.read()))

    def test_invalid_parameters_do_not_start_simulation(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/run",
            data=json.dumps({"seed": -1}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with patch("web.server.run_simulation") as simulation:
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(error.exception.code, 400)
            simulation.assert_not_called()

    def test_busy_timing_scheduler_returns_service_unavailable(self):
        from web.timing_backend import TimingBusyError

        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/run",
            data=json.dumps({"active": "timing-trained"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with patch("web.server.run_simulation", side_effect=TimingBusyError("Busy")):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(error.exception.code, 503)
            self.assertEqual(json.loads(error.exception.read())["error"], "Busy")

    def test_http_post_run(self):
        url = f"http://127.0.0.1:{self.port}/api/run"
        payload = json.dumps(
            {
                "scenario": "frequency-agile",
                "baseline": "round-robin",
                "active": "track-aware",
                "seed": 42,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("deltas", data)
            self.assertIn("active", data)
            self.assertIn("baseline", data)


if __name__ == "__main__":
    unittest.main()
