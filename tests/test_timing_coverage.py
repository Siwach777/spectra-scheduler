"""Coverage scheduling checks with public history and supplied forecasts only."""

import numpy as np
import pytest

from spectra_scheduler.simulation import SyntheticAction
from spectra_scheduler.timing_belief import BeliefPolicyConfig, TimingHistory
from spectra_scheduler.timing_coverage import RecoveryConfig, RecoveryTimingPlanner


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
