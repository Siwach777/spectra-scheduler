"""Exploration, auxiliary grounding, reusable buffers and legacy artifact checks."""

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.mpc.checkpoints import load_model  # noqa: E402
from spectra_scheduler.mpc.config import Config, saved_config  # noqa: E402
from spectra_scheduler.mpc.data import collect  # noqa: E402
from spectra_scheduler.mpc.learning import BatchWorkspace, batch_loss  # noqa: E402
from spectra_scheduler.mpc.model import NeuralMPCModel  # noqa: E402
from spectra_scheduler.mpc.search import MCTS, MCTSNode  # noqa: E402


def test_temporal_exploration_reduces_switching_without_forcing_evaluation():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = NeuralMPCModel()
    cfg = Config(workers=1, simulations=8)
    held = collect(model, cfg, [0, 1], "train", explore=True)
    for episode in held:
        blocks = episode["actions"].reshape(-1, cfg.exploration_hold)
        assert torch.all(blocks == blocks[:, :1])
    normal = collect(model, replace(cfg, exploration_hold=1), [0, 1], "train", explore=True)
    assert sum(float(e["features"][:, 9].sum()) for e in held) > sum(
        float(e["features"][:, 9].sum()) for e in normal
    )
    a = collect(model, cfg, [10000], "validation")
    b = collect(model, replace(cfg, exploration_hold=1), [10000], "validation")
    torch.testing.assert_close(a[0]["actions"], b[0]["actions"], rtol=0, atol=0)


def test_workspace_reuses_storage_caches_targets_and_auxiliary_head_learns():
    torch.set_num_threads(1)
    model = NeuralMPCModel()
    cfg = Config(workers=1, simulations=1)
    episodes = collect(model, cfg, [0, 1], "train", explore=True)
    workspace = BatchWorkspace()
    original = workspace.prepare(episodes, cfg, "cpu")
    pointers = {key: tensor.data_ptr() for key, tensor in original.items()}
    cache = episodes[0]["_returns"]
    second = workspace.prepare(episodes, cfg, "cpu")
    assert pointers == {key: tensor.data_ptr() for key, tensor in second.items()}
    assert episodes[0]["_returns"] is cache
    loss, terms = batch_loss(
        model, model, episodes, cfg, np.random.default_rng(0), "cpu", workspace
    )
    loss.backward()
    assert terms["observation"] > 0
    assert model._observation[0].weight.grad.abs().sum() > 0
    assert model.representation.gru.weight_ih_l0.grad.abs().sum() > 0


def test_old_checkpoint_loads_without_new_head_or_search_semantics(tmp_path):
    model = NeuralMPCModel(observation_head=False)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "version": 2,
            "state_dict": model.state_dict(),
            "config": {"step_dim": 29, "hidden_size": 128, "max_bands": 8},
        },
        path,
    )
    restored = load_model(path)
    assert restored._observation is None
    assert not saved_config({"seed": 0}).normalize_search
    assert saved_config({"seed": 0}).exploration_hold == 1
    for old, new in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_normalized_selection_is_invariant_to_common_q_offset():
    search = MCTS(NeuralMPCModel(), 2)
    root = MCTSNode()
    root.children = {i: MCTSNode(0.5) for i in range(2)}
    for i, node in root.children.items():
        node.visit_count = 2
        node.reward = -3 + 0.1 * i
    before, _ = search._select_child(root, [-3, -2.9])
    for node in root.children.values():
        node.reward += 10
    after, _ = search._select_child(root, [7, 7.1])
    assert before == after == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_workspace_and_gru_packing():
    torch.set_num_threads(1)
    cfg = Config(workers=1, simulations=1, device="cuda")
    model = NeuralMPCModel()
    episodes = collect(model, cfg, [0, 1], "train", explore=True)
    model = model.cuda()
    target = copy.deepcopy(model).eval()
    model.representation.gru.flatten_parameters()
    target.representation.gru.flatten_parameters()
    workspace = BatchWorkspace()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            loss, terms = batch_loss(
                model, target, episodes, cfg, np.random.default_rng(0), "cuda", workspace
            )
            loss.backward()
            assert np.isfinite(list(terms.values())).all()
    assert not any("contiguous chunk" in str(w.message) for w in caught)
    assert workspace.host["features"].is_pinned()
