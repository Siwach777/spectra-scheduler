"""Exact distillation direction, causal teacher dispatch and forced-action checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.experiments.mopd_study import (  # noqa: E402
    episode_weights,
    reverse_kl,
    teacher_scores,
)
from spectra_scheduler.grouped_policy import ActionResidual  # noqa: E402
from spectra_scheduler.mopd_policy import ExpandedActionResidual  # noqa: E402


def test_expanded_head_preserves_original_then_can_reverse_large_prior_gap():
    original = ActionResidual(features=4, hidden=8)
    with torch.no_grad():
        original.layers[-1].weight.normal_()
    expanded = ExpandedActionResidual(original)
    features = torch.randn(2, 24, 4)
    prior = torch.randn(2, 24)
    legal = torch.ones(2, 24, dtype=torch.bool)
    torch.testing.assert_close(
        expanded(features, prior, legal), original(features, prior, legal), rtol=0, atol=0
    )
    features = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    prior = torch.tensor([[20.0, 0.0]])
    legal = torch.ones_like(prior, dtype=torch.bool)
    with torch.no_grad():
        expanded.correction.weight[0, 0] = 50.0
    assert expanded(features, prior, legal).argmax() == 1


def test_reverse_kl_matches_full_distribution_and_freezes_teacher():
    student = torch.tensor([[0.3, -0.7, 0.8]], requires_grad=True)
    teacher = torch.tensor([[1.4, 0.2, -0.1]], requires_grad=True)
    legal = torch.ones_like(student, dtype=torch.bool)
    expected = (student.softmax(-1) * (student.log_softmax(-1) - teacher.log_softmax(-1))).sum(-1)
    actual = reverse_kl(student, teacher, legal)
    torch.testing.assert_close(actual, expected)
    forward = (teacher.softmax(-1) * (teacher.log_softmax(-1) - student.log_softmax(-1))).sum(-1)
    assert not torch.allclose(actual, forward)
    actual.sum().backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_illegal_and_forced_actions_have_no_distillation_gradient():
    student = torch.tensor([[3.0, 8.0, -1.0], [3.0, 8.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, -50.0, 2.0], [1.0, -50.0, 2.0]])
    legal = torch.tensor([[True, False, True], [False, True, False]])
    loss = reverse_kl(student, teacher, legal)
    assert loss[1] == 0
    loss.sum().backward()
    assert student.grad[0, 1] == 0
    assert student.grad[1].abs().sum() == 0
    with pytest.raises(ValueError):
        reverse_kl(student, teacher, torch.zeros_like(legal))


def test_episode_weights_balance_short_and_long_student_trajectories():
    episode = np.array([0, 0, 1, 1, 1, 1])
    legal = np.ones((6, 24), bool)
    legal[1, 1:] = False
    weights = episode_weights(episode, legal, 2)
    assert weights[1] == 0
    np.testing.assert_allclose([weights[episode == i].sum() for i in (0, 1)], [0.5, 0.5])


def test_teacher_dispatch_only_changes_target_scores():
    class Teacher(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(offset))

        def forward(self, features, prior, legal):
            return features[..., 0] + prior + self.offset

    teachers = [Teacher(1.0), Teacher(7.0)]
    features = torch.randn(3, 24, 2)
    prior = torch.randn(3, 24)
    before = features.clone()
    legal = torch.ones(3, 24, dtype=torch.bool)
    targets = teacher_scores(teachers, features, prior, legal, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(
        targets, features[..., 0] + prior + torch.tensor([7.0, 1.0, 7.0])[:, None]
    )
    torch.testing.assert_close(features, before)
    assert not targets.requires_grad
    assert all(t.offset.grad is None for t in teachers)
