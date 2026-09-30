"""Arithmetic and schema unit checks; no neural model validation or training."""

import pytest
import torch
from torch import nn

from spectra_scheduler.timing_belief import BeliefConfig
from spectra_scheduler.timing_ensemble import TimingCountEnsemble


class CountSource(nn.Module):
    def __init__(self, value, config=None):
        super().__init__()
        self.value = value
        self.config = config or BeliefConfig()

    def forward(self, history):
        return torch.full((len(history), 8, self.config.future), self.value)


def test_convex_counts_preserve_shape_and_use_identical_history():
    history = torch.zeros(2, 8, 3, 288)
    before = history.clone()
    model = TimingCountEnsemble([CountSource(1.), CountSource(3.)], [0.75, 0.25])
    assert torch.equal(model(history), torch.full((2, 8, 80), 1.5))
    assert torch.equal(history, before)
    assert not model.training


@pytest.mark.parametrize("weights", ([1, 1], [1, -1], [float("nan"), 1], [0.5]))
def test_invalid_mixture_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        TimingCountEnsemble([CountSource(1.), CountSource(2.)], weights)


def test_incompatible_history_schemas_are_rejected():
    with pytest.raises(ValueError, match="schemas"):
        TimingCountEnsemble([CountSource(1.), CountSource(2., BeliefConfig(history=144))],
                            [0.5, 0.5])
