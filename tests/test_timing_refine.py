"""Count-loss unit checks; these do not train or validate a neural model."""

import torch

from spectra_scheduler.experiments.timing_refine import refinement_loss


def batch_for(future, valid=None):
    return {"history": torch.empty(2, 2, 3, 1), "future": future,
            "valid": torch.ones(2, 80) if valid is None else valid,
            "anchor": torch.ones_like(future)}


def test_native_windows_penalize_shifted_pulse_and_have_finite_gradients():
    target = torch.full((2, 2, 80), 0.01)
    target[..., 9] = 1
    accurate = target.clone().requires_grad_()
    shifted = target.clone()
    shifted[..., 9], shifted[..., 10] = 0.01, 1
    shifted.requires_grad_()
    batch = batch_for(target)
    a = refinement_loss(accurate, batch, anchor_weight=0)
    b = refinement_loss(shifted, batch, anchor_weight=0)
    assert b > a
    b.backward()
    assert torch.isfinite(shifted.grad).all()
    assert shifted.grad[..., 9].abs().min() > 0


def test_invalid_future_suffix_is_masked_in_all_losses():
    target = torch.ones(2, 2, 80)
    valid = torch.zeros(2, 80)
    valid[:, :12] = 1
    first = target.clone()
    second = first.clone()
    second[..., 12:] = 100
    first.requires_grad_()
    second.requires_grad_()
    batch = batch_for(target, valid)
    a, b = refinement_loss(first, batch), refinement_loss(second, batch)
    assert torch.equal(a, b)
    b.backward()
    assert not second.grad[..., 12:].any()


def test_poisson_anchor_penalty_is_minimized_at_frozen_prediction():
    target = torch.ones(2, 2, 80)
    batch = batch_for(target)
    same, changed = target.clone(), target * 2
    anchor_same = refinement_loss(same, batch) - refinement_loss(same, batch, anchor_weight=0)
    anchor_changed = (refinement_loss(changed, batch)
                      - refinement_loss(changed, batch, anchor_weight=0))
    assert anchor_same == 0
    assert anchor_changed > 0
