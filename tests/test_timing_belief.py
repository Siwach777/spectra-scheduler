"""Focused causal-history and counterfactual-label regressions; CPU unit checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.models import Observation, SignalMeasurement  # noqa: E402
from spectra_scheduler.timing_belief import (  # noqa: E402
    BeliefConfig,
    TimingBeliefNetwork,
    TimingHistory,
    belief_loss,
    validated_retune_table,
)


def test_ring_preserves_exact_timing_and_distinguishes_miss_from_unobserved():
    history = TimingHistory(2, 4)
    for step in range(6):
        history.observe(Observation(step, step % 2, detections=int(step == 4),
                                    listening=step != 3))
    x = history.encode().copy()
    np.testing.assert_equal(x[0, 0], [1, 0, 1, 0])
    np.testing.assert_equal(x[1, 0], [0, 0, 0, 1])
    np.testing.assert_equal(x[0, 1], [0, 0, 1, 0])
    with pytest.raises(ValueError, match="contiguous"):
        history.observe(Observation(8, 0))


def test_all_band_loss_has_finite_gradients_and_permutation_equivariance():
    torch.set_num_threads(1)
    model = TimingBeliefNetwork(BeliefConfig(history=32, max_period=16, width=8))
    x = torch.zeros(2, 3, 3, 32)
    x[:, :, 0] = torch.randint(2, (2, 3, 32))
    x[:, :, 1] = (torch.rand(2, 3, 32) > 0.8) * x[:, :, 0]
    x[:, :, 2] = x[:, :, 1] * 0.6
    permutation = torch.tensor([2, 0, 1])
    predicted = model(x)
    torch.testing.assert_close(model(x[:, permutation]), predicted[:, permutation])
    target = torch.zeros_like(predicted)
    target[:, 1, ::8] = 1
    valid = torch.ones(2, 80)
    loss = belief_loss(model, x, target, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.gate[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_collector_labels_match_shared_receiver_and_inputs_are_pre_action(tmp_path):
    from spectra_scheduler.experiments.timing_study import _collect_world

    shard = _collect_world((0, "periodic-scan", 12,
                            BeliefConfig(history=32, max_period=16), tmp_path, None))
    with np.load(shard["path"]) as x:
        assert not x["history"][0].any()
        assert x["future"].shape == (shard["samples"], 8, 80)
        assert x["history"].shape == (shard["samples"], 8, 3, 32)
        assert x["valid"][0].all()
        # No observed pulse can appear without an actual listening exposure.
        assert (x["history"][:, :, 1] <= 4 * x["history"][:, :, 0]).all()
        assert (x["future"].sum(-1) > 0).any()


def test_power_is_a_receiver_measurement_not_a_truth_identity():
    history = TimingHistory(1, 4)
    history.observe(Observation(0, 0, 1, measurements=(SignalMeasurement(-75),)))
    assert history.encode()[0, 2, -1] == pytest.approx(0.625)


def test_short_forecast_horizon_has_valid_count_loss():
    model = TimingBeliefNetwork(BeliefConfig(history=16, future=2, max_period=8, width=4))
    x = torch.zeros(1, 2, 3, 16)
    loss = belief_loss(model, x, torch.zeros(1, 2, 2), torch.ones(1, 2))
    loss.backward()
    assert torch.isfinite(loss)


def test_public_retune_horizon_is_checked_before_action_scoring():
    np.testing.assert_array_equal(validated_retune_table([[0, 30], [30, 0]], 80, (1, 10, 50)),
                                  [[0, 30], [30, 0]])
    with pytest.raises(ValueError, match="forecast horizon"):
        validated_retune_table([[0, 31], [31, 0]], 80, (1, 10, 50))
    for table in ([[0, 0.5], [1, 0]], [[0, -1], [1, 0]], [[0, np.nan], [1, 0]], [[0, 1, 2]]):
        with pytest.raises(ValueError, match="retune table"):
            validated_retune_table(table, 80, (1, 10, 50))
