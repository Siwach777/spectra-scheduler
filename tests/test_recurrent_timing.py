"""Physical dwell bookkeeping and causal timing checks without neural execution."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from spectra_scheduler.experiments.recurrent_assess import SampledRecurrentScheduler
from spectra_scheduler.models import Observation
from spectra_scheduler.recurrent_context import PhysicalContext
from spectra_scheduler.recurrent_env import SpectrumEnv, encode_context
from spectra_scheduler.rl import Context


@pytest.mark.parametrize("physical", [False, True])
def test_sampled_hold_counts_listening_ticks_once(physical):
    # Start inside an already selected hold; no neural policy or sampling is needed.
    scheduler = SampledRecurrentScheduler(SimpleNamespace())
    scheduler.context = Context(8)
    scheduler.pending = None
    scheduler.remaining = 4
    scheduler.held_band = 3
    scheduler.physical_contract = physical
    expected = 4
    for step, listening in enumerate((False, False, True, True)):
        assert scheduler.choose_band(step) == 3
        if not physical:
            expected -= 1
        assert scheduler.remaining == expected
        scheduler.observe(Observation(step, 3, listening=listening))
        if physical and listening:
            expected -= 1
        assert scheduler.remaining == expected
    assert scheduler.remaining == (2 if physical else 0)


def test_multiscale_context_preserves_old_ages_and_observed_edges():
    context = PhysicalContext(2)
    context.observe(Observation(0, 0))
    context.observe(Observation(1, 0, detections=1))
    context.observe(Observation(2, 0, detections=1))
    context.observe(Observation(80, 0))
    context.observe(Observation(81, 0, detections=1))
    assert context.period[0] == 80
    assert context.intervals[0] == 1
    context.observe(Observation(120, 0, detections=1))
    assert context.onset[0] == 81  # A hit after an unobserved gap is censored.
    early, late = context.encode(200), context.encode(1000)
    assert early[0, 3] == late[0, 3] == 1  # Legacy age has saturated.
    assert late[0, 22] > early[0, 22]  # 512 ms age retains the distinction.
    assert np.isfinite(late).all()
    assert (late >= 0).all() and (late <= 1).all()


def test_multiscale_macro_matches_every_causal_tick_and_discount():
    from spectra_scheduler.action_contract import DWELL_STEPS

    kwargs = dict(physical_contract=True, observation_version=2, gamma=0.999)
    macro = SpectrumEnv(dwell_steps=DWELL_STEPS, **kwargs)
    tick = SpectrumEnv(dwell_steps=(1,), **kwargs)
    macro.reset(seed=3)
    tick.reset(seed=3)
    state, reward, _, _, info = macro.step(2)  # Band zero, 50 listening ticks.
    total = 0.0
    for step in range(50):
        tick_state, value, _, _, _ = tick.step(0)
        total += 0.999**step * value
    np.testing.assert_array_equal(state, tick_state)
    assert state.shape == (256,)
    assert macro.observation_space.contains(state)
    assert reward == pytest.approx(total)
    assert info == {"elapsed_steps": 50}


def test_multiscale_context_keeps_smaller_band_validity_mask():
    state = encode_context(PhysicalContext(2), 0)
    assert state.shape == (256,)
    assert state[-8:].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
