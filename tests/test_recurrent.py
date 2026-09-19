"""Optional training-stack tests; core environment does not require PyTorch."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

import torch
from gymnasium.utils.env_checker import check_env
from sb3_contrib import RecurrentPPO

from spectra_scheduler.recurrent_env import RecurrentScheduler, SpectrumEnv, encode_context
from spectra_scheduler.rl import Context
from spectra_scheduler.rl_scenarios import procedural_scenario
from spectra_scheduler.schedulers import DwellSweepScheduler
from spectra_scheduler.simulation import SimulationEpisode


def test_gym_contract_and_truth_boundary():
    env = SpectrumEnv()
    check_env(env, skip_render_check=True)
    state, info = env.reset(seed=11)
    assert state.shape == (176,)
    assert info == {}
    assert env.observation_space.contains(state)
    for step in range(120):
        state, reward, terminal, truncated, info = env.step(step % 8)
        assert env.observation_space.contains(state)
        assert np.isfinite(reward)
        assert terminal == (step == 119)
        assert not truncated
        assert info == {}
    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_engine_matches_batch_results():
    sim = procedural_scenario(4, "validation")
    expected = sim.run(DwellSweepScheduler())
    session = SimulationEpisode(sim)
    for observation in expected.observations:
        assert session.step(observation.band) == observation
    assert session.result() == expected


def test_recurrent_state_reset_and_invalid_band_mask():
    torch.set_num_threads(1)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        SpectrumEnv(),
        n_steps=8,
        batch_size=8,
        policy_kwargs={"lstm_hidden_size": 16, "net_arch": [16]},
        seed=0,
        device="cpu",
    )
    sim = procedural_scenario(4, "validation")
    a = sim.run(RecurrentScheduler(model))
    b = sim.run(RecurrentScheduler(model))
    assert a == b
    assert all(0 <= observation.band < 6 for observation in a.observations)
    context = Context(6)
    state = encode_context(context, 0)
    assert state[-8:].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        RecurrentScheduler(model).reset(9)


def test_recurrent_training_updates_weights():
    torch.set_num_threads(1)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        SpectrumEnv(),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"lstm_hidden_size": 16, "net_arch": [16]},
        seed=0,
        device="cpu",
    )
    before = [p.detach().clone() for p in model.policy.parameters()]
    model.learn(16)
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.policy.parameters(), strict=True)
    )
