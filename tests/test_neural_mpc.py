"""Planning correctness and end-to-end model training smoke checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.neural_mpc import (  # noqa: E402
    MCTS,
    NeuralMPCScheduler,
    TrainConfig,
    collect_demonstrations,
    load_model,
    save_model,
    train_model,
)


class RewardModel:
    def predict(self, state):
        return torch.zeros(1, 8), torch.zeros(1)

    def dynamics(self, state, action):
        return state, (action == 1).float()


def test_search_uses_edge_reward_and_masks_bands():
    search = MCTS(RewardModel(), 2, num_simulations=40, max_depth=2)
    action, visits = search.search(torch.zeros(128))
    assert action == 1
    assert visits.shape == (2,)
    assert visits.sum() == pytest.approx(1)
    assert visits[1] > visits[0]


def test_train_roundtrip_and_reset(tmp_path):
    torch.set_num_threads(1)
    cfg = TrainConfig(demo_episodes_per_policy=1, demo_policies=("adaptive-dwell",), epochs=1)
    episodes = collect_demonstrations(cfg)
    assert episodes[0][0]["num_bands"] == 8
    assert all(-0.25 <= row["reward"] <= 1 for row in episodes[0])
    losses = []
    model = train_model(episodes, cfg, lambda _, loss, __: losses.append(loss))
    assert np.isfinite(losses).all()
    path = save_model(model, tmp_path / "model.pt")
    restored = load_model(path)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)
    policy = NeuralMPCScheduler(model=restored, num_simulations=0)
    policy.reset(6)
    first = policy.choose_band(0)
    assert 0 <= first < 6
    policy.reset(6)
    assert policy.choose_band(0) == first


def test_invalid_search_budget():
    with pytest.raises(ValueError):
        MCTS(RewardModel(), 2, num_simulations=0)
