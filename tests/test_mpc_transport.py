"""Retained actor data must not retain Torch multiprocessing storage handles."""

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.mpc.config import Config  # noqa: E402
from spectra_scheduler.mpc.data import Replay, collect  # noqa: E402
from spectra_scheduler.mpc.model import NeuralMPCModel  # noqa: E402


@pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor-limit regression")
def test_spawn_collection_retains_replay_without_descriptor_growth():
    import resource

    original = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = min(128, original[0])
    if len(list(Path("/proc/self/fd").iterdir())) + 32 >= limit:
        pytest.skip("test process already near descriptor limit")
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, original[1]))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = NeuralMPCModel()
        cfg = Config(workers=2, simulations=1)
        replay = Replay(128)
        with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
            for turn in range(4):
                seeds = list(range(turn * 16, (turn + 1) * 16))
                batch = collect(model, cfg, seeds, "train", pool, policy_only=True)
                replay.extend(batch)
                if turn == 0:
                    expected = collect(model, cfg, seeds, "train", policy_only=True)
                    for left, right in zip(batch, expected, strict=True):
                        for key in ("features", "actions", "rewards", "policies", "values"):
                            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                    baseline = len(list(Path("/proc/self/fd").iterdir()))
                else:
                    assert len(list(Path("/proc/self/fd").iterdir())) <= baseline + 8
                assert all(not e["features"].is_shared() for e in replay.episodes)
        assert len(replay.episodes) == 64  # 320 retained tensor storages, only 128 FDs allowed.
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original)
