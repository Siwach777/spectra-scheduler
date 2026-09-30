"""Search targets, terminal boundaries, isolation, updates and restart behavior."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.mpc.search import MCTS, MCTSNode  # noqa: E402
from spectra_scheduler.mpc_training import (  # noqa: E402
    Config,
    Replay,
    batch_loss,
    collect,
    evaluate_run,
    run,
    search_batch,
    value_targets,
)
from spectra_scheduler.neural_mpc import NeuralMPCModel  # noqa: E402


class ToyModel:
    def predict(self, states):
        if states.ndim == 1:
            states = states[None]
        return torch.zeros(len(states), 8), torch.full((len(states),), 100.0)

    def dynamics(self, states, actions):
        return states, (actions == 3).float()


def test_batched_search_reward_and_terminal_bootstrap():
    cfg = Config(simulations=64)
    visits, values = search_batch(
        ToyModel(), torch.zeros(2, 128), [1, 1], cfg, np.random.default_rng(0)
    )
    assert np.all(visits.argmax(-1) == 3)
    np.testing.assert_allclose(visits.sum(-1), 1)
    # Huge predicted future value must not be backed up beyond episode end.
    assert np.all((values >= 0) & (values <= 1))


class MacroToyModel:
    def predict(self, states):
        if states.ndim == 1:
            states = states[None]
        logits = torch.zeros(len(states), 24)
        logits[:, 5] = 100.0  # Prefer an eight-tick action when it is legal.
        return logits, torch.full((len(states),), 100.0)

    def dynamics(self, states, actions):
        return states, torch.ones(len(actions))


def test_macro_search_masks_bands_and_durations_at_physical_horizon():
    cfg = Config(simulations=24, depth=2, device="cpu")
    visits, values = search_batch(
        MacroToyModel(), torch.zeros(3, 128), [1, 4, 8],
        cfg, np.random.default_rng(0), num_bands=2,
    )
    assert visits.shape == (3, 24)
    assert np.all(visits[:, 6:] == 0)  # Bands two through seven are unavailable.
    assert np.all(visits[0, 1::3] == 0)
    assert np.all(visits[0, 2::3] == 0)
    assert np.all(visits[1, 2::3] == 0)
    assert visits[2, 5] > 0
    np.testing.assert_allclose(visits.sum(-1), 1)
    assert np.all(values <= (1 - cfg.gamma**8) / (1 - cfg.gamma))


def test_macro_backup_discounts_by_elapsed_ticks():
    search = MCTS(MacroToyModel(), 1, gamma=0.5)
    root = MCTSNode()
    child = MCTSNode(duration=4, elapsed=4)
    child.reward = 1.875  # Four unit rewards discounted by 0.5.
    search._backprop([root, child], 8.0)
    assert child.value == 8.0
    assert root.value == 1.875 + 0.5**4 * 8.0

    short = MCTSNode(0.5, duration=1)
    long = MCTSNode(0.5, duration=8)
    for candidate in (short, long):
        candidate.visit_count = 1
        candidate.value_sum = 8.0
    root.children = {1: long, 0: short}
    assert search._select_child(root)[0] == 0


def test_n_step_targets_do_not_bootstrap_terminal():
    result = value_targets(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0]), 0.5, 2)
    torch.testing.assert_close(result, torch.tensor([9.5, 3.5, 3.0]))


def test_replay_capacity_and_split_isolation():
    replay = Replay(2)
    replay.extend([{"split": "train", "seed": i} for i in range(3)])
    assert [e["seed"] for e in replay.episodes] == [1, 2]
    with pytest.raises(ValueError):
        replay.extend([{"split": "validation"}])


def test_collection_targets_and_all_networks_receive_gradients():
    torch.set_num_threads(1)
    model = NeuralMPCModel()
    cfg = Config(workers=1, episodes=2, simulations=2, batch_size=2)
    episodes = collect(model, cfg, [0, 1], "train", explore=True)
    assert episodes[0]["policies"].shape == (120, 8)
    torch.testing.assert_close(episodes[0]["policies"].sum(-1), torch.ones(120))
    loss, terms = batch_loss(
        model, NeuralMPCModel(), episodes, cfg, np.random.default_rng(0), "cpu"
    )
    assert np.isfinite(list(terms.values())).all()
    loss.backward()
    for network in (model.representation, model._dynamics, model._prediction):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MPC training requires CUDA")
def test_resume_matches_uninterrupted_training(tmp_path):
    cfg = Config(
        iterations=1,
        episodes=1,
        workers=1,
        batch_size=2,
        updates=1,
        simulations=1,
        validation_episodes=1,
        validation_every=1,
        threads=1,
    )
    run(cfg, tmp_path / "resumed")
    with pytest.raises(FileExistsError):
        run(cfg, tmp_path / "resumed")
    cfg = replace(cfg, iterations=2)
    run(cfg, tmp_path / "resumed", resume=True)
    run(cfg, tmp_path / "continuous")
    resumed = torch.load(tmp_path / "resumed/training.pt", weights_only=True)
    continuous = torch.load(tmp_path / "continuous/training.pt", weights_only=True)
    assert resumed["iteration"] == 2
    assert [e["seed"] for e in resumed["replay"]] == [0, 1]
    for key in resumed["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
    report = evaluate_run(tmp_path / "resumed", episodes=1)
    assert report["split"] == "test"
    assert "shift/search" in report["metrics"]
    assert all(e["split"] == "validation" for e in resumed["probes"])


def test_invalid_configuration():
    with pytest.raises(ValueError):
        Config(updates=0)


def test_fixed_batch_optimization_reduces_loss():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model, target = NeuralMPCModel(), NeuralMPCModel()
    cfg = Config(workers=1, simulations=2)
    batch = collect(model, cfg, [4, 5], "train", explore=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for _ in range(15):
        loss, _ = batch_loss(model, target, batch, cfg, np.random.default_rng(0), "cpu")
        losses.append(float(loss.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
