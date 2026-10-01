"""Reproduce one favorable frozen reporting example for the explanatory video."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from spectra_scheduler.experiments.calibrated_timing import configure_public_detection
from spectra_scheduler.experiments.timing_report import reporting_world
from spectra_scheduler.grouped_policy import GroupedTimingPolicy, load_grouped
from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.schedulers import DwellSweepScheduler
from spectra_scheduler.simulation import SimulationEpisode, configure_scheduler


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RecordedPolicy(GroupedTimingPolicy):
    def accept_action(self, step, predicted, index):
        # Save the exact pre-action forecast, without changing the selected action.
        self.video_prediction = predicted.tolist()
        return super().accept_action(step, predicted, index)


def collect(sim, policy, truth):
    configure_scheduler(sim, policy)
    if hasattr(policy, "set_detection_probability"):
        configure_public_detection(sim, policy)
    episode = SimulationEpisode(sim, truth=truth)
    steps, decisions = [], []
    while episode.time_step < sim.duration:
        tick = episode.time_step
        macro = hasattr(policy, "choose_action")
        action = policy.choose_action(tick) if macro else policy.choose_band(tick)
        forecast = policy.forecast(tick, action) if macro else None
        observations = episode.step_action(action) if macro else (episode.step(action),)
        decisions.append({
            "tick": tick, "band": action.band if macro else action,
            "dwell": action.dwell_steps if macro else 1,
            "forecast": asdict(forecast) if forecast is not None else None,
            "rates": getattr(policy, "video_prediction", None),
        })
        for obs, record in zip(observations, episode.records[-len(observations):], strict=True):
            policy.observe(obs)
            steps.append({"tick": obs.time_step, "band": obs.band,
                          "listening": obs.listening, "captured": len(record.detected_emitters),
                          "false_alarm": bool(record.false_alarm)})
    return {"metrics": asdict(calculate_metrics(episode.result())),
            "steps": steps, "decisions": decisions}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural validation; CPU fallback is disabled")
    torch.set_num_threads(1)
    directory = ROOT / "artifacts/grouped-mpc-grid-report-v1"
    checkpoint = directory / "seed-1/best.pt"
    report_path = directory / "comparison.json"
    report = json.loads(report_path.read_text())
    frozen = report["frozen"]["checkpoints"]["1"]
    if digest(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("Selected RL checkpoint changed since frozen reporting")
    model, actor, config, metadata = load_grouped(checkpoint)
    sim, world_seed = reporting_world("spatial-scan", 16007)
    truth = sim.generate_truth()
    baseline = collect(sim, DwellSweepScheduler(dwell_steps=50), truth)
    adaptive = collect(sim, RecordedPolicy(model, actor, config), truth)
    expected = next(r for r in report["results"] if r["scenario"] == "spatial-scan" and r["seed"] == 16007)
    for data, name in [(baseline, "round-robin-50"), (adaptive, "grouped-1")]:
        counts = expected["policies"][name]["evaluation"]["counts"]
        if (data["metrics"]["detected_transmissions"], len(truth)) != (counts["captured"], counts["truth"]):
            raise ValueError(f"Live CUDA replay differs from frozen reporting: {name}")
    b = baseline["metrics"]["detected_transmissions"]
    a = adaptive["metrics"]["detected_transmissions"]
    if (a, b) != (37, 10):
        raise ValueError(f"Expected 37 / 10 genuine detections, received {a} / {b}")
    output = {
        "scenario": "spatial-scan", "reporting_seed": 16007, "world_seed": world_seed,
        "duration": sim.duration, "bands": sim.num_bands,
        "receiver": asdict(sim.receiver), "baseline_dwell": 50,
        "truth": [asdict(tx) for tx in truth], "baseline": baseline, "adaptive": adaptive,
        "provenance": {"checkpoint": str(checkpoint.relative_to(ROOT)),
                       "checkpoint_sha256": digest(checkpoint),
                       "forecaster_sha256": digest(directory / "forecaster.pt"),
                       "report": str(report_path.relative_to(ROOT)), "report_sha256": digest(report_path),
                       "device": torch.cuda.get_device_name(0),
                       "algorithm": "Temporal timing forecaster + trajectory-trained RLOO action policy",
                       "selection": "Illustrative favorable example selected from existing frozen reporting; not an aggregate benchmark",
                       "policy": asdict(config)},
    }
    path = ROOT / "artifacts/demo-video/measured-run.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({"captures": {"round_robin": b, "temporal_rl": a}, "truth": len(truth),
                      "multiple": a / b, "duration": sim.duration, "saved": str(path)}, indent=2))


if __name__ == "__main__":
    main()
