"""Statistical policy/interface checks; no neural training or CPU neural validation."""

from types import SimpleNamespace

import numpy as np
import pytest

from spectra_scheduler.whittle_policy import (
    DiscoverySwitch,
    ScanStrategyConfig,
    WhittleScanScheduler,
    estimate_dynamics,
    whittle_curve,
)


def observation(tick, band, hit=False, listening=True):
    return SimpleNamespace(time_step=tick, band=band, hit=hit, listening=listening)


def test_independent_chain_index_is_immediate_occupancy():
    curve, indexable = whittle_curve(0.2, 0.2, points=33, subsidies=65)
    assert indexable
    assert np.allclose(curve, np.linspace(0, 1, 33), atol=1 / 64)
    assert not curve.flags.writeable


def test_sparse_transition_fit_respects_elapsed_gaps():
    states = np.tile([0, 0, 1, 1], 8)
    short = estimate_dynamics(list(zip(np.arange(32), states, strict=True)))
    long = estimate_dynamics(list(zip(np.arange(32) * 10, states, strict=True)))
    # Even sampling gaps cannot identify the sign of the chain's persistence.
    assert abs(long[1] - long[0]) > abs(short[1] - short[0])
    assert estimate_dynamics([(0, 0)]) == (0.02, 0.9)


def test_phase_switch_needs_actual_coverage_and_ignores_dead_ticks():
    switch = DiscoverySwitch(ScanStrategyConfig(strategy="adaptive", ewma_span=4), 2)
    switch.observe(observation(0, 0, True))
    switch.observe(observation(1, 0, True))
    switch.observe(observation(2, 1, True, listening=False))
    assert switch.exploring(3)
    switch.observe(observation(3, 1))
    assert not switch.exploring(4)
    assert switch.switched_at == 4


def test_patience_restarts_on_new_declared_hit_and_reset_is_clean():
    cfg = ScanStrategyConfig(strategy="phased", patience=4)
    policy = WhittleScanScheduler(cfg)
    policy.set_retune_table([[0, 1], [1, 0]])
    policy.set_episode_horizon(20)
    policy.reset(2)
    for tick in range(6):
        policy.observe(observation(tick, tick % 2, tick == 3))
    assert policy.switch.exploring(6)
    assert not policy.switch.exploring(7)
    policy.reset(2)
    assert policy.time == 0 and not policy.switch.seen.any()
    assert policy.switch.switched_at is None
    assert not any(policy.samples)


def test_golden_decision_is_idempotent_and_observations_are_contiguous():
    policy = WhittleScanScheduler(ScanStrategyConfig(strategy="golden", dwell=10))
    policy.set_retune_table(np.ones((8, 8), dtype=int) - np.eye(8, dtype=int))
    policy.set_episode_horizon(512)
    policy.reset(8)
    assert policy.choose_action(0) == policy.choose_action(0)
    assert policy.sequence == 1
    assert policy.choose_action(0).dwell_steps == 10
    with pytest.raises(ValueError, match="contiguous"):
        policy.observe(observation(1, 0))


def test_correlated_index_matches_paper_closed_form_branches():
    p01, p11, discount = 0.1, 0.8, 0.99
    curve, indexable = whittle_curve(p01, p11)
    beliefs = np.linspace(0, 1, len(curve))
    stationary = p01 / (1 + p01 - p11)
    middle = (beliefs >= stationary) & (beliefs < p11)
    expected = beliefs[middle] / (1 - discount * p11 + discount * beliefs[middle])
    assert indexable
    assert np.allclose(curve[middle], expected, atol=1 / 63 + 0.005)
    outside = (beliefs <= p01) | (beliefs >= p11)
    assert np.allclose(curve[outside], beliefs[outside], atol=1 / 63 + 0.005)


def test_signed_dynamics_recovers_negative_correlation():
    rng = np.random.default_rng(10)
    states = np.zeros(512, dtype=int)
    for tick in range(1, len(states)):
        states[tick] = rng.random() < (0.2 if states[tick - 1] else 0.8)
    p01, p11 = estimate_dynamics(list(zip(np.arange(512), states, strict=True)))
    assert p01 > p11
    assert abs(p01 - 0.8) < 0.1 and abs(p11 - 0.2) < 0.1


def test_markov_belief_conditions_hits_and_drifts_through_retuning():
    policy = WhittleScanScheduler(ScanStrategyConfig(belief_mode="markov"))
    policy.set_observation_probabilities(0.9, 0.1)
    policy.reset(2)
    policy.p01[:] = 0.1
    policy.p11[:] = 0.8
    policy.belief[:] = 1 / 3
    policy.observe(observation(0, 0, hit=True))
    expected = 0.1 + 0.7 * ((0.9 / 3) / (0.9 / 3 + 0.1 * 2 / 3))
    assert policy.belief[0] == pytest.approx(expected)
    assert policy.belief[1] == pytest.approx(1 / 3)
    policy.observe(observation(1, 0, hit=True, listening=False))
    assert policy.belief[0] == pytest.approx(0.1 + 0.7 * expected)
    assert len(policy.samples[0]) == 1


def test_noisy_independent_chain_index_is_detectable_occupancy():
    curve, indexable = whittle_curve(0.2, 0.2, detection=0.9, false_alarm=0.1)
    assert indexable
    assert np.allclose(curve, 0.9 * np.linspace(0, 1, len(curve)), atol=1 / 63)
