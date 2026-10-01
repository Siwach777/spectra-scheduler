from types import SimpleNamespace

import numpy as np
import pytest

from spectra_scheduler.pulse_replay import PulseObservation
from spectra_scheduler.timing_belief import TimingHistory
from spectra_scheduler.timing_replay import TimingReplayPolicy


def adapter():
    policy = object.__new__(TimingReplayPolicy)
    history = TimingHistory(2, 8)
    policy.scheduler = SimpleNamespace(history=history, observe=history.observe)
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
