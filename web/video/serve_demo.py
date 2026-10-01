"""Run the actual GUI with the verified temporal-forecasting/RL video scenario."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from spectra_scheduler.experiments.timing_report import reporting_world
from spectra_scheduler.grouped_policy import load_grouped
from spectra_scheduler.schedulers import DwellSweepScheduler
from web.api import simulate_single_scheduler
from web.server import SpectraConsoleHandler
from web.video.collect_demo import RecordedPolicy, digest

OUT = ROOT / "artifacts/demo-video"
SOURCE = json.loads((OUT / "measured-run.json").read_text())
SLOT = threading.Lock()


class ConsolePolicy(RecordedPolicy):
    def accept_action(self, step, predicted, index):
        action = super().accept_action(step, predicted, index)
        self.console_state = {
            "decision_tick": step, "dwell_ticks": action.dwell_steps,
            "forecast_ticks": 10,
            "timing_forecasts": [{"band": b, "expected_detections": float(row[:10].sum())}
                                 for b, row in enumerate(predicted)],
            "decision_reason": f"RL selected band {action.band} for {action.dwell_steps} listening ticks",
        }
        return action


def run(model, actor, config):
    sim, world_seed = reporting_world("spatial-scan", 16007)
    truth = sim.generate_truth()
    baseline = simulate_single_scheduler(sim, lambda: DwellSweepScheduler(dwell_steps=50), truth)

    def policy():
        p = ConsolePolicy(model, actor, config)
        p.model_provenance = {
            "artifact": SOURCE["provenance"]["checkpoint"],
            "sha256": SOURCE["provenance"]["checkpoint_sha256"], "device": "cuda",
            "algorithm": SOURCE["provenance"]["algorithm"],
        }
        return p

    active = simulate_single_scheduler(sim, policy, truth)
    for data, expected in [(baseline, SOURCE["baseline"]), (active, SOURCE["adaptive"])]:
        for key, value in expected["metrics"].items():
            if data["metrics"][key] != value:
                raise ValueError(f"GUI replay differs from verified evaluation: {key}")
    grid = [[] for _ in range(sim.duration)]
    for tx in truth:
        grid[tx.time_step].append(asdict(tx))
    b_ratio = baseline["metrics"]["interception_ratio"]
    a_ratio = active["metrics"]["interception_ratio"]
    payload = {
        "scenario": {"id": "spatial-scan", "num_bands": sim.num_bands, "duration": sim.duration,
                     "seed": 16007, "world_seed": world_seed, "total_transmissions": len(truth),
                     "sensitivity_dbm": sim.receiver.sensitivity_dbm,
                     "retune_steps": sim.receiver.retune_steps, "perturbation": None},
        "truth_grid": grid,
        "baseline": {"id": "round-robin-50", "name": "Round robin (50-tick dwell)", "data": baseline},
        "active": {"id": "temporal-rl", "name": "Temporal forecasting + RL", "data": active},
        "deltas": {"interception_gain_pct": (a_ratio / b_ratio - 1) * 100,
                   "interception_delta_pp": (a_ratio - b_ratio) * 100,
                   "baseline_interception": b_ratio, "active_interception": a_ratio},
        "provenance": SOURCE["provenance"],
    }
    (OUT / "gui-run.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU inference fallback is disabled")
    torch.set_num_threads(1)
    checkpoint = ROOT / SOURCE["provenance"]["checkpoint"]
    if digest(checkpoint) != SOURCE["provenance"]["checkpoint_sha256"]:
        raise ValueError("The selected trained RL checkpoint has changed")
    model, actor, config, _ = load_grouped(checkpoint)
    run(model, actor, config)

    class Handler(SpectraConsoleHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/api/scenarios":
                self._send_json([{"id": "spatial-scan", "name": "Spatial scan",
                                  "bands": 8, "duration": 512, "sensitivity_dbm": -90,
                                  "retune_steps": 1}])
            elif self.path == "/api/schedulers":
                self._send_json([
                    {"id": "round-robin-50", "name": "Round robin (50-tick dwell)", "available": True},
                    {"id": "temporal-rl", "name": "Temporal forecasting + RL", "available": True,
                     "artifact": SOURCE["provenance"]["checkpoint"]},
                ])
            elif self.path == "/api/presets":
                self._send_json({"temporal-rl-video": {
                    "id": "temporal-rl-video", "name": "Temporal forecasting + RL",
                    "scenario": "spatial-scan", "baseline": "round-robin-50", "active": "temporal-rl",
                    "seed": 16007,
                    "description": "Selected spatial-scan example. A trained temporal forecaster and RL policy use past receiver observations to schedule the band and dwell.",
                }})
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/api/run":
                return super().do_POST()
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length < 16384:
                    raise ValueError("Invalid request size")
                params = json.loads(self.rfile.read(length))
                expected = {"scenario": "spatial-scan", "baseline": "round-robin-50",
                            "active": "temporal-rl", "seed": 16007, "sensitivity_dbm": -90,
                            "retune_steps": 1, "perturbation": None}
                if any(params.get(key) != value for key, value in expected.items()):
                    raise ValueError("This recording server exposes the verified demonstration preset only.")
                with SLOT:
                    self._send_json(run(model, actor, config))
            except Exception as error:
                self._send_error_json(str(error))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Verified RL demo: http://127.0.0.1:{args.port}/?preset=temporal-rl-video&view=dual", flush=True)
    print("Actual CUDA evaluation: 37 versus 10 true detections. Every Run recomputes the episode.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
