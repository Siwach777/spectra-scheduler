"""Framework-independent lifecycle with transactional epoch checkpoints."""

import json
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .storage import checkpoint_path, fingerprint, run_lock, verify_artifacts, write_json


class Learner(Protocol):
    """Adapters own batches, architecture, optimization, RNG and serialization.

    train_epoch must be deterministic from restored state and epoch number if exact
    resume is claimed. validate must not modify the learner's training/RNG state.
    Generic artifacts have fixed filenames but adapter-specific encodings.
    """

    def configuration(self) -> dict: ...
    def train_epoch(self, epoch: int, progress) -> dict: ...
    def save_state(self, path: Path) -> None: ...
    def load_state(self, path: Path) -> None: ...
    def export_policy(self, path: Path) -> None: ...
    def validate(self, policy_path: Path) -> dict: ...


@dataclass(frozen=True)
class RunConfig:
    epochs: int = 1
    selection: tuple[str, ...] = ("summary", "predictor", "interception_ratio", "mean")
    maximize: bool = True
    min_delta: float = 0.0

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive total budget")
        if not self.selection or any(not isinstance(k, str) or not k for k in self.selection):
            raise ValueError("selection must be a nonempty metric key path")
        if (
            type(self.maximize) is not bool
            or not math.isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError("invalid selection direction or minimum improvement")


def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def run_experiment(directory, learner: Learner, config=None, *, resume=False, stream=None):
    """Validate initialization, then train/validate/commit epochs; resume latest commit.

    One atomic state.json publishes both latest and best references. Failure before
    publication leaves the previous epoch authoritative. Orphan directories are
    retained for inspection. Epoch budget can grow on resume; semantic config cannot.
    """
    cfg = config or RunConfig()
    root = Path(directory).resolve()
    stream = stream if stream is not None else sys.stdout
    semantic = asdict(cfg)
    semantic.pop("epochs")
    expected = _normalized({"version": 1, "learner": learner.configuration(), "run": semantic})
    begin, last_update = perf_counter(), 0.0

    def progress(status, **values):
        nonlocal last_update
        now = perf_counter()
        if status == "training" and now - last_update < 1:
            return
        last_update = now
        value = {"status": status, "elapsed_seconds": now - begin, **values}
        write_json(root / "progress.json", value)
        print(
            f"{status}: " + ", ".join(f"{k}={v}" for k, v in values.items()),
            file=stream,
            flush=True,
        )

    with run_lock(root):
        settings = root / "run.json"
        state_path = root / "state.json"
        if resume:
            if not settings.exists() or not state_path.exists():
                raise ValueError("resume requires a committed run")
            if json.loads(settings.read_text()) != expected:
                raise ValueError("resume configuration differs from the committed run")
            state = json.loads(state_path.read_text())
            if state.get("version") != 1:
                raise ValueError("unsupported experiment state version")
            verify_artifacts(root, state["latest"])
            verify_artifacts(root, state["best"])
            if cfg.epochs < state["latest"]["epoch"]:
                raise ValueError("total epoch budget is below completed epochs")
            learner.load_state(checkpoint_path(root, state["latest"], "training.bin"))
            start = state["latest"]["epoch"] + 1
        else:
            if any(p.name != "run.lock" for p in root.iterdir()):
                raise ValueError("new experiment requires an empty run directory")
            write_json(settings, expected)
            state, start = None, 0
        checkpoints = root / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        try:
            for epoch in range(start, cfg.epochs + 1):
                training = (
                    {}
                    if epoch == 0
                    else learner.train_epoch(
                        epoch, lambda epoch=epoch, **v: progress("training", epoch=epoch, **v)
                    )
                )
                # Separate immutable files let one pointer publish a complete generation.
                target = Path(tempfile.mkdtemp(prefix=f"epoch-{epoch:06d}-", dir=checkpoints))
                progress("validating", epoch=epoch)
                learner.export_policy(target / "policy.bin")
                report = learner.validate(target / "policy.bin")
                score = report
                for key in cfg.selection:
                    score = score[key]
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                ):
                    raise ValueError("selection metric must be finite and defined")
                # Save after validation: adapters must preserve training state during validation.
                learner.save_state(target / "training.bin")
                write_json(target / "validation.json", report)
                write_json(target / "training.json", training)
                artifacts = {
                    str((target / name).relative_to(root)): fingerprint(target / name)
                    for name in ("policy.bin", "training.bin", "validation.json", "training.json")
                }
                entry = {
                    "epoch": epoch,
                    "score": score,
                    "directory": str(target.relative_to(root)),
                    "artifacts": artifacts,
                }
                best = state["best"] if state else entry
                difference = score - best["score"]
                if (difference if cfg.maximize else -difference) > cfg.min_delta:
                    best = entry
                state = {"version": 1, "latest": entry, "best": best}
                write_json(state_path, state)
                progress("checkpointed", epoch=epoch, score=score, best_score=best["score"])
            progress("complete", epoch=state["latest"]["epoch"], best_score=state["best"]["score"])
            return state
        except BaseException as error:
            progress(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=type(error).__name__,
                detail=str(error),
            )
            raise
