"""Joint timing evidence regressions. Small CPU checks are not training runs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.joint_belief import JointTimingNetwork  # noqa: E402
from spectra_scheduler.timing_belief import (  # noqa: E402
    BeliefConfig,
    TimingBeliefNetwork,
    belief_loss,
)


def test_joint_model_preserves_band_permutation_and_frozen_anchor():
    torch.set_num_threads(1)
    model = JointTimingNetwork(
        TimingBeliefNetwork(BeliefConfig(history=32, future=16, max_period=16, width=8))
    )
    x = torch.zeros(2, 3, 3, 32)
    for band in range(3):
        x[:, band, 0, band::3] = 1
        x[:, band, 1, band::9] = 1
        x[:, band, 2, band::9] = 0.625
    permutation = torch.tensor([2, 0, 1])
    prediction = model(x)
    torch.testing.assert_close(model(x[:, permutation]), prediction[:, permutation])
    loss = belief_loss(model, x, torch.zeros_like(prediction), torch.ones(2, 16))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.period_gate[0].weight.grad.abs().sum() > 0
    assert model.band_gate[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_other_band_observations_change_joint_forecasts_and_empty_histories_are_finite():
    model = JointTimingNetwork(
        TimingBeliefNetwork(BeliefConfig(history=32, future=16, max_period=16, width=8))
    )
    x = torch.zeros(1, 3, 3, 32)
    before = model(x)
    x[:, 1, 0, ::4] = 1
    x[:, 1, 1, ::8] = 1
    x[:, 1, 2, ::8] = 0.625
    after = model(x)
    assert torch.isfinite(before).all() and (before > 0).all()
    assert not torch.allclose(before[:, 0], after[:, 0])
    with pytest.raises(ValueError, match="history"):
        model(x[..., :-1])


def test_collection_copy_trims_final_partial_block():
    from spectra_scheduler.experiments.joint_study import copy_valid_rows

    source = np.arange(4000).reshape(2000, 2)
    destination = np.empty((1366, 2), dtype=source.dtype)
    copy_valid_rows(destination, source, len(destination))
    np.testing.assert_array_equal(destination, source[:1366])
    with pytest.raises(ValueError, match="extent"):
        copy_valid_rows(destination, source, 2000)
