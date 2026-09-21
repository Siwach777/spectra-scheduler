import io
import json

import pytest

from spectra_scheduler.experiments import RunConfig, run_experiment
from spectra_scheduler.experiments.storage import checkpoint_path, run_lock, write_json


class ToyLearner:
    """Independent JSON learner proves that the lifecycle does not require Torch."""

    def __init__(self, scores=(1, 3, 2), failure=None):
        self.epoch = 0
        self.scores = scores
        self.failure = failure

    def configuration(self):
        return {"adapter": "toy", "scores": self.scores}

    def train_epoch(self, epoch, progress):
        self.epoch = epoch
        progress(batches=1)
        if self.failure == "training":
            raise KeyboardInterrupt("stop")
        return {"loss": 1 / epoch}

    def save_state(self, path):
        write_json(path, {"epoch": self.epoch})

    def load_state(self, path):
        self.epoch = json.loads(path.read_text())["epoch"]

    def export_policy(self, path):
        self.save_state(path)

    def validate(self, path):
        if self.failure == "validation" and self.epoch:
            raise RuntimeError("validation failed")
        return {"score": self.scores[self.epoch]}


def run(path, learner=None, epochs=2, resume=False, **kwargs):
    return run_experiment(
        path,
        learner or ToyLearner(),
        RunConfig(epochs, selection=("score",), **kwargs),
        resume=resume,
        stream=io.StringIO(),
    )


def test_best_latest_and_resume(tmp_path):
    state = run(tmp_path, epochs=1)
    assert state["best"]["epoch"] == 1
    resumed = run(tmp_path, resume=True)
    assert resumed["best"] == state["best"]
    assert resumed["latest"]["epoch"] == 2
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "complete"
    assert run(tmp_path, resume=True) == resumed
    with pytest.raises(ValueError, match="empty"):
        run(tmp_path)


@pytest.mark.parametrize(
    "failure,error", [("training", KeyboardInterrupt), ("validation", RuntimeError)]
)
def test_failure_preserves_committed_checkpoint_and_releases_lock(tmp_path, failure, error):
    with pytest.raises(error):
        run(tmp_path, ToyLearner(failure=failure))
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["latest"]["epoch"] == 0
    assert run(tmp_path, resume=True)["latest"]["epoch"] == 2


def test_corruption_config_drift_and_lock_rejected(tmp_path):
    state = run(tmp_path, epochs=1)
    with pytest.raises(ValueError, match="configuration"):
        run(tmp_path, ToyLearner(scores=(1, 2, 3)), resume=True)
    with run_lock(tmp_path), pytest.raises(RuntimeError, match="another writer"):
        run(tmp_path, resume=True)
    path = checkpoint_path(tmp_path, state["latest"], "training.bin")
    path.write_text("broken")
    with pytest.raises(ValueError, match="integrity"):
        run(tmp_path, resume=True)


def test_undefined_selection_and_minimize(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        run(tmp_path / "bad", ToyLearner(scores=(None, 1, 2)))
    assert not (tmp_path / "bad" / "state.json").exists()
    state = run(tmp_path / "min", ToyLearner(scores=(3, 2, 1)), maximize=False)
    assert state["best"]["score"] == 1


def test_atomic_json_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"ok": True})
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"ok": True}
    assert list(tmp_path.iterdir()) == [path]
