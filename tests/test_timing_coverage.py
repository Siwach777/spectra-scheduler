"""Coverage scheduling checks with public history and supplied forecasts only."""

import numpy as np
import pytest

from spectra_scheduler.simulation import SyntheticAction
from spectra_scheduler.timing_belief import BeliefPolicyConfig, TimingHistory
from spectra_scheduler.models import Observation, SignalMeasurement
from spectra_scheduler.timing_coverage import (
    AcquisitionConfig,
    AcquisitionTimingPlanner,
    RecoveryConfig,
    RecoveryTimingPlanner,
)


def policy():
    scheduler = object.__new__(RecoveryTimingPlanner)
    scheduler.config = BeliefPolicyConfig(dwells=(1, 10, 50), revisit=512, probe=10)
    scheduler.recovery = RecoveryConfig(revisit=96, dwell=50, fraction=0.15)
    scheduler.history = TimingHistory(3, 1)
    scheduler.history.last_listen[:] = [30, 40, 50]
    scheduler.history.hits[:] = [8, 0, 0]
    scheduler.history.current_band = 0
    scheduler.retune = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    scheduler.horizon, scheduler.bands = 512, 3
    scheduler.coverage, scheduler.empirical = True, False
    scheduler.recovery_ticks = scheduler.decisions = 0
    scheduler._pending_recovery = None
    return scheduler


def test_recovery_uses_weak_evidence_and_charges_retuning_once():
    scheduler = policy()
    action = scheduler.coverage_action(150)
    assert action == SyntheticAction(1, 50)
    assert scheduler.recovery_ticks == 0
    assert scheduler.coverage_action(150) == action
    scheduler.history.time = 150
    scheduler.accept_action(150, np.zeros((3, 80)), action)
    assert scheduler.recovery_ticks == 52
    assert scheduler.coverage_action(160) is None


def test_initial_coverage_is_preserved_and_no_oracle_state_required():
    scheduler = policy()
    scheduler.history.last_listen[2] = -1
    assert scheduler.coverage_action(150) == SyntheticAction(2, 10)
    assert scheduler._pending_recovery is None


def test_recovery_cost_is_clipped_at_public_episode_end():
    scheduler = policy()
    scheduler.recovery_ticks = 70
    assert scheduler.coverage_action(510) == SyntheticAction(1, 50)
    assert scheduler._pending_recovery[2] == 2


def test_forecast_opportunity_cost_rejects_expensive_probe():
    scheduler = policy()
    action = scheduler.coverage_action(150)
    values = np.ones((3, 3))
    values[0, 0] = 10
    assert scheduler.filter_coverage_action(action, values) is None
    assert scheduler._pending_recovery is None
    action = scheduler.coverage_action(150)
    values[1, 2] = 9.5
    assert scheduler.filter_coverage_action(action, values) == action


def test_invalid_recovery_settings_fail():
    with pytest.raises(ValueError):
        RecoveryConfig(fraction=1.1)
    with pytest.raises(ValueError):
        RecoveryConfig(revisit=0)


def acquisition_policy():
    scheduler = object.__new__(AcquisitionTimingPlanner)
    source = policy()
    scheduler.__dict__.update(source.__dict__)
    scheduler.acquisition = AcquisitionConfig(fraction=0.1)
    scheduler.reliable_hits = np.array([3, 0, 0])
    scheduler.power_threshold = 0.5
    scheduler.acquisition_ticks = 0
    scheduler._pending_acquisition = None
    scheduler.history.time = 150
    return scheduler


def test_acquisition_extends_already_selected_band_and_charges_only_added_time():
    scheduler = acquisition_policy()
    values = np.zeros((3, 3))
    values[1, 0], values[1, 1] = 10, 9.7
    action = scheduler.filter_coverage_action(None, values)
    assert action == SyntheticAction(1, 10)
    scheduler.accept_action(150, np.zeros((3, 80)), action)
    assert scheduler.acquisition_ticks == 9
    scheduler.acquisition_ticks = 45
    assert scheduler.filter_coverage_action(None, values) is None


def test_acquisition_preserves_first_signal_search_and_rejects_expensive_extension():
    scheduler = acquisition_policy()
    values = np.zeros((3, 3))
    values[1, 0], values[1, 1] = 10, 9.7
    scheduler.reliable_hits[:] = 0
    assert scheduler.filter_coverage_action(None, values) is None
    scheduler.reliable_hits[0] = 3
    values[1, 1] = 5
    assert scheduler.filter_coverage_action(None, values) is None
    forced = SyntheticAction(2, 10)
    assert scheduler.filter_coverage_action(forced, values) == forced


def test_acquisition_evidence_uses_measured_power_instead_of_raw_hit_flag():
    scheduler = acquisition_policy()
    scheduler.observe(Observation(150, 1, 1, measurements=(SignalMeasurement(-95),)))
    assert scheduler.history.hits[1] == 1
    assert scheduler.reliable_hits[1] == 0
    scheduler.observe(Observation(151, 1, 1, measurements=(SignalMeasurement(-60),)))
    assert scheduler.reliable_hits[1] == 1
