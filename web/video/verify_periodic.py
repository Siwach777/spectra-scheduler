"""Verify the exact periodic GUI setup and save its real export for recording."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["SPECTRA_TIMING_CHECKPOINT"] = str(ROOT / "artifacts/timing-refine-v1/seed-0/best.pt")

import torch

from web.api import run_simulation

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for neural validation")
torch.set_num_threads(1)
run = run_simulation("periodic-scan", "dwell-sweep-50", "timing-trained", seed=42)
b, a = run["baseline"]["data"]["metrics"], run["active"]["data"]["metrics"]
model = run["active"]["data"]["model"]
assert (model["training_seed"], model["selected_epoch"]) == (0, 24), model
assert (a["detected_transmissions"], b["detected_transmissions"]) == (62, 13), (a, b)
assert a["total_transmissions"] == b["total_transmissions"] == 86
assert a["emitter_discovery_ratio"] == b["emitter_discovery_ratio"] == 1.0
path = ROOT / "artifacts/demo-video/periodic-run.json"
path.write_text(json.dumps(run, indent=2, allow_nan=False))
print(json.dumps({"model": model, "scenario": run["scenario"],
                  "trained_captures": a["detected_transmissions"],
                  "baseline_captures": b["detected_transmissions"],
                  "trained_capture_pct": a["interception_ratio"] * 100,
                  "baseline_capture_pct": b["interception_ratio"] * 100,
                  "capture_multiple": a["detected_transmissions"] / b["detected_transmissions"],
                  "discovery_both": a["emitter_discovery_ratio"], "saved": str(path)}, indent=2))
