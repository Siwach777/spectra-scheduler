import numpy as np

from spectra_scheduler.adaptive_scan import AdaptiveDwellScheduler, ThompsonBandScheduler
from spectra_scheduler.models import Observation


def test_thompson_discount_advances_during_retuning_without_new_evidence():
    policy = ThompsonBandScheduler(discount=0.5)
    policy.reset(2)
    policy.observe(Observation(0, 0, detections=1))
    policy.observe(Observation(1, 1, listening=False))
    np.testing.assert_allclose(policy.alpha, [1.5, 1])
    np.testing.assert_allclose(policy.beta, [3, 3])


def test_thompson_seed_is_repeatable_and_native_dwell_includes_all_feedback():
    first = AdaptiveDwellScheduler("thompson", 50, seed=41)
    repeated = AdaptiveDwellScheduler("thompson", 50, seed=41)
    first.reset(8)
    repeated.reset(8)
    for step in range(20):
        action = first.choose_action(step)
        assert action == repeated.choose_action(step)
        assert action.dwell_steps == 50
        observation = Observation(step, action.band, detections=1)
        first.observe(observation)
        repeated.observe(observation)
    assert np.isclose(first.learner.alpha.sum(), 28)
