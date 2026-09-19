"""Foreground rendering and collection progress leave training data unchanged."""

from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pytest

pytest.importorskip("torch")

from spectra_scheduler.mpc import data  # noqa: E402
from spectra_scheduler.mpc.config import Config  # noqa: E402
from spectra_scheduler.mpc.model import NeuralMPCModel  # noqa: E402
from spectra_scheduler.mpc.progress import LiveProgress  # noqa: E402


def test_plain_progress_has_stage_counts_losses_and_completion():
    stream = StringIO()
    progress = LiveProgress(stream)
    progress.update("learner", 0, 10)
    progress.update("learner", 10, 10, "reward=0.02")
    progress.message("checkpoint saved")
    output = stream.getvalue()
    assert "0/10 (0%)" in output
    assert "10/10 (100%)" in output
    assert "reward=0.02" in output
    assert "checkpoint saved" in output
    assert "\r" not in output


def test_terminal_progress_finishes_line():
    class Terminal(StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    progress = LiveProgress(stream)
    progress.update("updates", 0, 1)
    progress.update("updates", 1, 1)
    assert "\r" in stream.getvalue()
    assert stream.getvalue().endswith("\n")


def test_parallel_collection_reports_completion_and_preserves_seed_order(monkeypatch):
    def actor(payload):
        return [{"seed": seed} for seed in payload[2]]

    monkeypatch.setattr(data, "collect_worker", actor)
    updates = []
    with ThreadPoolExecutor(2) as pool:
        records = data.collect(
            NeuralMPCModel(),
            Config(workers=2),
            [3, 1, 9, 2],
            "train",
            pool,
            progress=lambda done, total: updates.append((done, total)),
        )
    assert [r["seed"] for r in records] == [3, 1, 9, 2]
    assert updates[0] == (0, 480)
    assert updates[-1] == (480, 480)


def test_single_actor_reports_intermediate_steps():
    updates = []
    data.collect(
        NeuralMPCModel(),
        Config(workers=1),
        [0],
        "train",
        policy_only=True,
        progress=lambda done, total: updates.append((done, total)),
    )
    assert updates[0] == (0, 120)
    assert updates[-1] == (120, 120)
    assert (60, 120) in updates
