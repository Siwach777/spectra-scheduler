from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spectra_scheduler.pulse_replay import PulseObservation
from spectra_scheduler.models import Observation
from spectra_scheduler.timing_belief import BeliefConfig, TimingBeliefNetwork, TimingHistory
from spectra_scheduler.timing_replay import TimingReplayPolicy


def adapter():
    policy = object.__new__(TimingReplayPolicy)
    history = TimingHistory(2, 8)
    policy.scheduler = SimpleNamespace(history=history, observe=history.observe)
    policy.delivered_counts = np.zeros(2)
    policy._selected, policy.start_us = 1, 0
    return policy


def test_pdw_feedback_keeps_retune_mask_and_causal_timestamp_counts():
    policy = adapter()
    pulses = np.array([[2100, 10, 5, 20, -40], [2300, 10, 5, 20, -50]])
    policy.observe_pulses(PulseObservation(0, 2000, 4000, 10, pulses, 0))
    history = policy.scheduler.history
    assert history.time == 4
    np.testing.assert_array_equal(history.values[1, 0, :4], [0, 0, 1, 1])
    np.testing.assert_array_equal(history.values[1, 1, :4], [0, 0, 2, 0])
    assert not history.values[:, 2].any()
    assert not history.values[0].any()
    assert policy._selected is None


def test_dataset_amplitude_does_not_become_synthetic_dbm():
    first, repeated = adapter(), adapter()
    pulses = np.array([[100, 10, 5, 20, -40]])
    first.observe_pulses(PulseObservation(0, 0, 1000, 10, pulses, 0))
    pulses[:, 4] = 1000
    repeated.observe_pulses(PulseObservation(0, 0, 1000, 10, pulses, 0))
    np.testing.assert_array_equal(first.scheduler.history.values, repeated.scheduler.history.values)


def test_fractional_tick_feedback_is_rejected():
    with pytest.raises(ValueError, match="aligned"):
        adapter().observe_pulses(PulseObservation(0, 100, 1000, 10, np.empty((0, 5)), 0))


def test_uncapped_history_preserves_delivered_pulse_count():
    capped, raw = TimingHistory(1, 8), TimingHistory(1, 8, count_limit=None)
    observation = Observation(0, 0, 17, True)
    capped.observe(observation)
    raw.observe(observation)
    assert capped.values[0, 1, 0] == 4
    assert raw.values[0, 1, 0] == 17
    np.testing.assert_array_equal(raw.values[:, 0], capped.values[:, 0])


def test_missing_power_does_not_apply_the_learned_sensitivity_gate():
    # CPU structural check; experiment inference is required to run on CUDA.
    model = TimingBeliefNetwork(BeliefConfig(history=8, future=8, max_period=4, width=2))
    history = torch.zeros(1, 2, 3, 8)
    history[:, :, 0] = 1
    history[:, 0, 1, ::2] = 2
    with torch.no_grad():
        before = model(history, power_available=False)
        gated = model(history)
        model.quality_threshold.fill_(100)
        after = model(history, power_available=False)
        suppressed = model(history)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.allclose(gated, suppressed)


def test_receiver_rate_memory_uses_only_delivered_counts_and_listening_exposure():
    policy = adapter()
    np.testing.assert_array_equal(policy.causal_rates(), [-1, -1])
    pulses = np.array([[2100, 10, 5, 20, -40], [2300, 10, 5, 20, -50]])
    policy.observe_pulses(PulseObservation(0, 2000, 4000, 10, pulses, 0))
    np.testing.assert_allclose(policy.causal_rates(), [-1, 2.2 / 6])


def test_unknown_rate_prior_preserves_default_and_known_empty_band_is_remembered():
    model = TimingBeliefNetwork(BeliefConfig(history=8, future=8, max_period=4, width=2))
    history = torch.zeros(1, 2, 3, 8)
    with torch.no_grad():
        default = model(history, power_available=False)
        unknown = model(history, power_available=False, rate_prior=torch.full((1, 2), -1.0))
        known = model(history, power_available=False, rate_prior=torch.zeros(1, 2))
    torch.testing.assert_close(default, unknown, rtol=0, atol=0)
    assert not known.any()
    assert (default > 0).all()
