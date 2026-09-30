"""Optional training-stack tests; core environment does not require PyTorch."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

import torch
from gymnasium.utils.env_checker import check_env
from sb3_contrib import RecurrentPPO

from spectra_scheduler.action_contract import DWELL_STEPS
from spectra_scheduler.emitters import SpatialScanningEmitter
from spectra_scheduler.recurrent_env import RecurrentScheduler, SpectrumEnv, encode_context
from spectra_scheduler.rl import Context
from spectra_scheduler.rl_scenarios import physical_scenario, procedural_scenario
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


def test_physical_time_advantages_and_terminal_mask():
    from gymnasium.spaces import Box, Discrete

    from spectra_scheduler.recurrent_smdp import DurationRolloutBuffer

    buffer = DurationRolloutBuffer(
        2, Box(0, 1, shape=(1,)), Discrete(2), (2, 1, 1, 4), device="cpu", gamma=0.5, gae_lambda=1
    )
    buffer.rewards[:, 0] = [1, 2]
    buffer.elapsed_steps[:, 0] = [2, 1]
    buffer.compute_returns_and_advantage(torch.tensor([8.0]), np.array([False]))
    np.testing.assert_allclose(buffer.returns[:, 0], [2.5, 6])
    buffer.compute_returns_and_advantage(torch.tensor([8.0]), np.array([True]))
    np.testing.assert_allclose(buffer.returns[:, 0], [1.5, 2])
    buffer.episode_starts[1, 0] = 1
    buffer.compute_returns_and_advantage(torch.tensor([8.0]), np.array([True]))
    np.testing.assert_allclose(buffer.returns[:, 0], [1, 2])


def test_dwell_action_matches_physical_receiver_steps():
    macro, tick = SpectrumEnv(dwell_steps=(1, 4, 8)), SpectrumEnv()
    macro.reset(seed=3)
    tick.reset(seed=3)
    state, reward, done, truncated, info = macro.step(3 * 3 + 2)
    expected = 0.0
    for i in range(8):
        tick_state, value, _, _, _ = tick.step(3)
        expected += 0.99**i * value
    np.testing.assert_array_equal(state, tick_state)
    assert reward == pytest.approx(expected)
    assert info == {"elapsed_steps": 8}
    assert not done and not truncated
    for _ in range(27):
        macro.step(3 * 3 + 1)
    _, _, done, truncated, info = macro.step(3 * 3 + 2)
    assert done and not truncated
    assert info == {"elapsed_steps": 4}


def test_physical_training_world_and_post_retune_dwell():
    worlds = [physical_scenario(seed, "train") for seed in range(8)]
    assert all(
        any(isinstance(e, SpatialScanningEmitter) for e in world.emitters) for world in worlds
    )
    assert all(world.generate_truth() for world in worlds)
    env = SpectrumEnv(dwell_steps=DWELL_STEPS, physical_contract=True)
    env.reset(seed=3)
    env.step(0)
    before = env.episode.time_step
    delay = env.episode.simulation.receiver.retune_duration(0, 7)
    _, _, done, _, info = env.step(7 * len(DWELL_STEPS))
    assert not done
    assert delay > 0
    assert info["elapsed_steps"] == delay + 1
    assert env.episode.time_step - before == delay + 1
    assert [item.observation.listening for item in env.episode.records[before:]] == [
        False
    ] * delay + [True]
