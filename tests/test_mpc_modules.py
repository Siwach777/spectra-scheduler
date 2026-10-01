"""Module boundaries, compatibility exports and command/checkpoint wiring."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler import mpc_training, neural_mpc  # noqa: E402
from spectra_scheduler.mpc import (  # noqa: E402
    checkpoints,
    config,
    data,
    evaluation,
    learning,
    model,
    observation,
    scheduler,
    search,
    trainer,
)


def test_compatibility_exports_are_the_same_implementations():
    assert neural_mpc.NeuralMPCModel is model.NeuralMPCModel
    assert neural_mpc.NeuralMPCScheduler is scheduler.NeuralMPCScheduler
    assert neural_mpc.ObservationEncoder is observation.ObservationEncoder
    assert neural_mpc.MCTS is search.MCTS
    assert neural_mpc.TrainConfig is config.TrainConfig
    assert neural_mpc.train_model is learning.train_model
    assert neural_mpc.save_model is checkpoints.save_model
    assert mpc_training.Config is config.Config
    assert mpc_training.Replay is data.Replay
    assert mpc_training.collect_worker is data.collect_worker
    assert mpc_training.batch_loss is learning.batch_loss
    assert mpc_training.search_batch is search.search_batch
    assert mpc_training.evaluate_run is evaluation.evaluate_run
    assert mpc_training.run is trainer.run


def test_provenance_covers_all_implementation_modules():
    hashes = checkpoints.implementation_hashes()
    package = Path(checkpoints.__file__).parent
    assert {f"mpc/{p.name}" for p in package.glob("*.py")} <= hashes.keys()
    assert {"neural_mpc.py", "mpc_training.py", "rl_scenarios.py"} <= hashes.keys()
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Neural MPC training requires CUDA")
def test_legacy_pretraining_command_still_saves_loadable_model(tmp_path):
    output = tmp_path / "demo.pt"
    neural_mpc.main(["train", "--episodes", "1", "--epochs", "1", "--output", str(output)])
    assert isinstance(checkpoints.load_model(output), model.NeuralMPCModel)
