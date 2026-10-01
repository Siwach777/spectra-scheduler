"""Local CUDA timing scheduler adapter; no training or fallback weights."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class TimingUnavailableError(ValueError):
    pass


class TimingBusyError(RuntimeError):
    pass


_model_lock = threading.Lock()
_run_slot = threading.BoundedSemaphore(1)
_cached = {}


def checkpoint_path(artifacts):
    configured = os.environ.get("SPECTRA_TIMING_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    portable = Path(artifacts) / "timing-console-v1" / "best.pt"
    return portable if portable.is_file() else (
        Path(artifacts) / "timing-refine-v1" / "seed-0" / "best.pt")


def _snapshot(artifacts):
    path = checkpoint_path(artifacts)
    if not path.is_file():
        raise TimingUnavailableError("The trained timing checkpoint is not installed.")
    try:
        import torch
    except ImportError as error:
        raise TimingUnavailableError("Start the console with the CUDA Python environment.") from error
    if not torch.cuda.is_available():
        raise TimingUnavailableError("CUDA is required to run the trained timing scheduler.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    frozen = path.parent / "frozen.json"
    if path.name == "best.pt" and frozen.is_file():
        if json.loads(frozen.read_text())["checkpoint_sha256"] != digest:
            raise TimingUnavailableError("The trained checkpoint does not match its selected artifact.")
    return path, digest


def model_status(artifacts):
    try:
        path, digest = _snapshot(artifacts)
        details = {
            "artifact": str(path.relative_to(artifacts)) if path.is_relative_to(artifacts) else path.name,
            "sha256": digest, "device": "cuda",
        }
        with _model_lock:
            if _cached.get("identity", ())[:2] == (str(path), digest):
                details.update(_cached["provenance"])
        return {"available": True, **details}
    except (ValueError, OSError, RuntimeError) as error:
        return {"available": False, "artifact": None, "reason": str(error)}


@contextmanager
def timing_run_slot(required):
    if required and not _run_slot.acquire(blocking=False):
        raise TimingBusyError("The trained scheduler is running another comparison. Try again shortly.")
    try:
        yield
    finally:
        if required:
            _run_slot.release()


def create_scheduler(artifacts):
    path, digest = _snapshot(artifacts)
    captured = os.environ.get("SPECTRA_CUDA_GRAPH") == "1"
    identity = (str(path), digest, captured)
    with _model_lock:
        if _cached.get("identity") != identity:
            import torch
            from spectra_scheduler.timing_belief import BeliefPolicyConfig
            from spectra_scheduler.timing_ensemble import load_predictor

            torch.set_num_threads(1)
            model, metadata = load_predictor(path)
            model_kind = "count ensemble" if hasattr(model, "models") else "timing forecaster"
            raw_policy = dict(metadata.get("policy", {
                "dwells": (1, 10, 50), "revisit": 512, "probe": 10,
                "exploration": 0.0, "retune_cost": 0.0, "switch_margin": 0.0,
            }))
            raw_policy["dwells"] = tuple(raw_policy["dwells"])
            config = BeliefPolicyConfig(**raw_policy)
            if captured:
                from spectra_scheduler.timing_runtime import CapturedTimingPredictor
                model = CapturedTimingPredictor(model)
            if config.dwells != (1, 10, 50):
                raise TimingUnavailableError("The selected timing checkpoint uses a different scan menu.")
            provenance = {
                "artifact": str(path.relative_to(artifacts)) if path.is_relative_to(artifacts) else path.name,
                "sha256": digest, "device": "cuda", "selected_epoch": metadata.get("epoch"),
                "training_seed": metadata.get("seed"), "policy": raw_policy,
                "training_epochs": metadata.get("arguments", {}).get("epochs"),
                "cuda_graph": captured,
                "model_kind": model_kind,
            }
            _cached.clear()
            _cached.update(identity=identity, model=model, config=config,
                           provenance=provenance)
        model, config, provenance = _cached["model"], _cached["config"], dict(_cached["provenance"])

    from spectra_scheduler.timing_planner import CalibratedTimingPlannerPolicy

    class ConsoleTimingPolicy(CalibratedTimingPlannerPolicy):
        def reset(self, bands):
            if captured and self.model.bands != bands:
                from spectra_scheduler.timing_runtime import CapturedTimingPredictor
                self.model = CapturedTimingPredictor(self.model.model, bands=bands)
            super().reset(bands)
            self.console_state = {}
            self.inference_seconds = 0.0

        def choose_action(self, step):
            before = time.perf_counter()
            action = super().choose_action(step)
            self.inference_seconds += time.perf_counter() - before
            return action

        def select(self, step, predicted):
            coverage = self.coverage_action(step) is not None
            action = super().select(step, predicted)
            horizon = min(predicted.shape[1], self.horizon - step)
            self.console_state = {
                "decision_tick": step,
                "dwell_ticks": action.dwell_steps,
                "timing_forecasts": [{"band": band, "expected_detections": float(row[:horizon].sum())}
                                     for band, row in enumerate(predicted)],
                "forecast_ticks": horizon,
                "decision_reason": (f"Checking band {action.band}" if coverage else
                                    f"Listening on band {action.band} for {action.dwell_steps} ticks"),
            }
            return action

    scheduler = ConsoleTimingPolicy(model, config, coverage=True)
    from spectra_scheduler.planner_native import runtime_details
    provenance.update(runtime_details())
    scheduler.model_provenance = provenance
    return scheduler
