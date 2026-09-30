"""CPU checks of forecast planning; no neural training or validation."""

from functools import lru_cache

import numpy as np
import pytest

from spectra_scheduler.simulation import SyntheticAction
from spectra_scheduler.timing_belief import BeliefPolicyConfig, TimingHistory
from spectra_scheduler.timing_planner import (
    ForecastPlannerWorkspace,
    TimingPlannerPolicy,
    first_actions,
)


def exhaustive_values(prediction, retune, dwells, horizon, current):
    prefix = np.pad(prediction[:, :horizon].cumsum(1), ((0, 0), (1, 0)))

    @lru_cache(None)
    def visit(tick, previous):
        if tick >= horizon:
            return 0.0
        return max(
            value(tick, previous, band, dwell)
            for band in range(len(prediction))
            for dwell in dwells
        )

    def value(tick, previous, band, dwell):
        delay = 0 if previous < 0 else retune[previous, band]
        start = min(horizon, tick + delay)
        end = min(horizon, start + dwell)
        return prefix[band, end] - prefix[band, start] + visit(end, band)

    return np.asarray(
        [[value(0, current, band, dwell) for dwell in dwells] for band in range(len(prediction))]
    )


def test_batched_values_match_exhaustive_variable_horizons():
    rng = np.random.default_rng(8)
    prediction = rng.random((4, 3, 10))
    retune = rng.integers(1, 4, size=(4, 3, 3))
    retune[:, np.arange(3), np.arange(3)] = 0
    dwells = (1, 3, 5)
    horizons, current = np.array([1, 4, 7, 10]), np.array([-1, 2, 0, 1])
    workspace = ForecastPlannerWorkspace(4, 3, 10, dwells)
    bands, selected_dwells = first_actions(prediction, current, retune, horizons, dwells, workspace)
    for index in range(4):
        expected = exhaustive_values(
            prediction[index], retune[index], dwells, horizons[index], current[index]
        )
        assert np.allclose(workspace.action_values[index], expected, rtol=1e-12, atol=1e-12)
        assert np.isclose(
            expected[bands[index], dwells.index(selected_dwells[index])], expected.max()
        )


def test_native_planner_can_wait_to_preserve_future_opportunity():
    prediction = np.zeros((1, 2, 80))
    prediction[0, 0, 0] = 0.4
    prediction[0, 1, 3] = 1.0
    retune = np.array([[[0, 2], [2, 0]]])
    workspace = ForecastPlannerWorkspace(1, 2, 80)
    bands, dwells = first_actions(prediction, [0], retune, [80], workspace=workspace)
    assert (bands[0], dwells[0]) == (0, 1)
    assert np.isclose(workspace.action_values.max(), 1.4)
    # Remaining on band zero for a long native dwell loses the next opportunity.
    assert np.isclose(workspace.action_values[0, 0, 1], 0.4)


def test_workspace_reuse_clears_previous_values_and_clips_end():
    retune = np.array([[[0, 2], [2, 0]]])
    workspace = ForecastPlannerWorkspace(1, 2, 80)
    prediction = np.ones((1, 2, 80))
    first_actions(prediction, [0], retune, [80], workspace=workspace)
    prediction.fill(0)
    prediction[0, 1, 0] = 100
    first_actions(prediction, [0], retune, [1], workspace=workspace)
    assert not workspace.action_values.any()
    assert workspace.state_values[0, 0, 0] == 0
    assert not workspace.state_values[:, 1:].any()


def test_public_horizon_validation():
    with pytest.raises(ValueError, match="forecast horizon"):
        first_actions(np.zeros((1, 2, 50)), [0], np.array([[[0, 2], [2, 0]]]), [50])


def test_accept_action_accounts_for_retune_and_remaining_window():
    # Exercise forecast accounting with supplied forecasts, without a neural model.
    policy = object.__new__(TimingPlannerPolicy)
    policy.config = BeliefPolicyConfig(dwells=(1, 10, 50))
    policy.history = TimingHistory(2, 1)
    policy.history.time, policy.history.current_band = 77, 0
    policy.bands, policy.horizon, policy.decisions = 2, 80, 0
    policy.empirical = False
    policy.retune = np.array([[0, 2], [2, 0]])
    predicted = np.full((2, 80), 0.5)
    action = SyntheticAction(1, 10)
    assert policy.accept_action(77, predicted, action) == action
    forecast = policy.forecast(77, action)
    assert forecast.hit_probability == 0.5
    assert forecast.intercept_delay_seconds == 0.002
    assert np.isclose(forecast.interception_ratio, 1 / 6)
