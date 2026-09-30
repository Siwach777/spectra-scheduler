"""Unit checks of trajectory advantages and legal-action gradient boundaries."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.grouped_policy import ActionResidual, leave_one_out  # noqa: E402


def test_leave_one_out_uses_only_other_trajectories_of_the_same_world():
    returns = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    advantages = leave_one_out(returns)
    np.testing.assert_allclose(advantages, [[-0.15, 0, 0.15], [0.15, 0, -0.15]], atol=1e-7)
    shifted = returns + np.array([[12.0], [-5.0]])
    np.testing.assert_allclose(leave_one_out(shifted), advantages, atol=1e-7)
    np.testing.assert_allclose(leave_one_out(2 * returns), 2 * advantages, atol=1e-7)
    with pytest.raises(ValueError):
        leave_one_out(np.zeros((3, 1)))


def test_initial_policy_preserves_prior_and_forced_probes_have_zero_policy_gradient():
    torch.set_num_threads(1)
    actor = ActionResidual(features=7, hidden=16)
    features = torch.randn(2, 5, 7).half()
    prior = torch.tensor([[0.0, 1.0, 3.0, 2.0, -1.0], [1.0, 1.0, 2.0, 2.0, 3.0]])
    legal = torch.tensor([[True] * 5, [False, True, False, False, False]])
    logits = actor(features, prior, legal)
    assert logits[0].argmax() == prior[0].argmax()
    assert logits[1].argmax() == 1
    log_probability = logits.log_softmax(-1)
    (-log_probability[1, 1]).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in actor.parameters())
    actor.zero_grad()
    logits = actor(features, prior, legal)
    (-logits.log_softmax(-1)[0, 2]).backward()
    assert actor.layers[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
