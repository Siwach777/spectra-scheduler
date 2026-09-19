import json
from dataclasses import replace

import numpy as np
import pytest

from spectra_scheduler.learning_cli import write_json
from spectra_scheduler.models import Observation
from spectra_scheduler.rl import (
    RL_FEATURES,
    Adam,
    Context,
    QNetwork,
    Replay,
    RewardConfig,
    RLModel,
    RLScheduler,
    TrainConfig,
    TrainingScheduler,
    double_q_targets,
    train,
)
from spectra_scheduler.rl_benchmark import benchmark, paired_delta
from spectra_scheduler.rl_cli import main
from spectra_scheduler.rl_scenarios import procedural_scenario, scenario_fingerprint


def test_context_only_observations_and_retuning_not_negative_listen():
    context = Context(3)
    before = context.encode(0)
    context.observe(Observation(0, 1, listening=False))
    after = context.encode(1)
    assert context.history.visits == [0, 0, 0]
    assert after[1, 0] == before[1, 0]
    assert after[1, 12] == 0
    context.observe(Observation(1, 1, detections=1))
    assert context.encode(2)[1, 0] == pytest.approx(2 / 3)
    assert context.encode(2).shape == (3, len(RL_FEATURES))
    assert np.isfinite(context.encode(2)).all()
    with pytest.raises(ValueError):
        context.encode(1)


def test_reward_is_feedback_only_and_accounts_for_coverage():
    context = Context(2)
    hit = Observation(0, 1, detections=1)
    context.observe(hit)
    assert RewardConfig().compute(hit, context) == pytest.approx(1 - 0.05 / 48)
    retune = Observation(1, 0, listening=False)
    context.observe(retune)
    assert RewardConfig().compute(retune, context) < -0.05
    with pytest.raises(ValueError):
        RewardConfig(coverage=float("nan"))


def test_analytic_gradients_against_finite_differences():
    network = QNetwork(7, 4)
    x = np.random.default_rng(3).normal(size=(5, len(RL_FEATURES)))
    targets = np.asarray([-0.4, 0.2, 1.8, -0.7, 0.8])
    _, gradients = network.loss_gradients(x, targets)
    for parameter, gradient in zip(network.parameters, gradients, strict=True):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = network.loss_gradients(x, targets)[0]
            parameter[index] = original - 1e-6
            minus = network.loss_gradients(x, targets)[0]
            parameter[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-6)


def test_adam_reduces_supervised_loss():
    network = QNetwork(0, 8)
    x = np.ones((10, len(RL_FEATURES)))
    targets = np.ones(10)
    initial = network.loss_gradients(x, targets)[0]
    optimizer = Adam(network, 0.01)
    for _ in range(50):
        _, gradients = network.loss_gradients(x, targets)
        optimizer.update(gradients)
    assert network.loss_gradients(x, targets)[0] < initial * 0.1


def test_double_q_action_selection_and_terminal_mask():
    class Fixed:
        def __init__(self, values):
            self.values = np.asarray(values)

        def predict(self, _):
            return self.values

    values = double_q_targets(
        Fixed([[1, 4], [8, 1]]),
        Fixed([[10, 2], [3, 20]]),
        None,
        np.asarray([1.0, 2.0]),
        np.asarray([False, True]),
        0.5,
    )
    assert values.tolist() == [2.0, 2.0]


def test_replay_ring_and_terminal_transition():
    replay = Replay(4, 6)
    config = TrainConfig(episodes=1, replay_capacity=8, batch_size=2, warmup=8)
    network = QNetwork()
    agent = TrainingScheduler(
        network,
        network.copy(),
        replay,
        Adam(network, 0.001),
        np.random.default_rng(0),
        config,
        RewardConfig(),
    )
    agent.duration = 3
    agent.reset(6)
    for step in range(3):
        band = agent.choose_band(step)
        agent.observe(Observation(step, band))
    assert replay.done[:3].tolist() == [False, False, True]
    state = np.zeros((6, len(RL_FEATURES)))
    for _ in range(10):
        replay.add(state, 1, 0.0, state, False)
    assert replay.count == 13
    assert len(replay.states) == 4


def test_no_handwritten_dwell_override_and_frozen_inference():
    class PickBand:
        index = 0

        def predict(self, x):
            values = np.zeros(len(x))
            values[self.index] = 1
            self.index = (self.index + 1) % len(x)
            return values

    policy = RLScheduler(PickBand())
    policy.reset(3)
    assert policy.choose_band(0) == 0
    policy.observe(Observation(0, 0, listening=False))
    assert policy.choose_band(1) == 1  # Can change even while retuning.
    with pytest.raises(ValueError):
        policy.observe(Observation(1, 0))
    network = QNetwork(1)
    before = [a.copy() for a in network.parameters]
    procedural_scenario(9, "validation").run(RLScheduler(network))
    for a, b in zip(before, network.parameters, strict=True):
        np.testing.assert_array_equal(a, b)


def test_worlds_repeat_and_randomize_structure_and_split():
    first = procedural_scenario(4, "train")
    assert scenario_fingerprint(first) == scenario_fingerprint(procedural_scenario(4, "train"))
    assert first.emitters != procedural_scenario(5, "train").emitters
    assert first.emitters != procedural_scenario(4, "validation").emitters
    assert procedural_scenario(4, "test", True).num_bands == 8


@pytest.fixture
def small_model():
    return train(
        TrainConfig(
            episodes=3, hidden=8, warmup=16, batch_size=8, replay_capacity=64, target_every=20
        )
    )


def test_training_repeatable_and_artifact_validation(small_model, tmp_path):
    other = train(TrainConfig(**small_model.training["config"]))
    assert small_model.fingerprint == other.fingerprint
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(small_model.network.parameters, QNetwork(0, 8).parameters, strict=True)
    )
    path = tmp_path / "model.json"
    write_json(path, small_model.to_dict())
    assert RLModel.load(path).fingerprint == small_model.fingerprint
    data = small_model.to_dict()
    data["weights"][0][0][0] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="weights"):
        RLModel.load(path)


def test_benchmark_determinism_pairing_controls_and_validation(small_model):
    before = small_model.fingerprint
    report = benchmark([small_model], runs=2, suites=("randomized",))
    assert report == benchmark([small_model], runs=2, suites=("randomized",))
    assert small_model.fingerprint == before
    suite = report["suites"]["randomized"]
    assert len(suite["mean_metrics"]) == 15
    assert "untrained-0" in suite["paired_interception_delta"]["rl-0"]
    assert len(suite["episodes"]) == 2
    assert suite["training_seed_variability"]["stddev"] is None
    assert paired_delta([1, 2], [1, 2])["approximate_95pct_ci"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        benchmark([small_model], split="train")
    with pytest.raises(ValueError):
        benchmark([small_model, small_model])
    with pytest.raises(ValueError):
        replace(TrainConfig(), episodes=0)


def test_cli_protects_input_and_validates_config(small_model, tmp_path):
    path = tmp_path / "model.json"
    write_json(path, small_model.to_dict())
    assert main(["benchmark", "--models", str(path), "--output", str(path)]) == 1
    assert RLModel.load(path).fingerprint == small_model.fingerprint
    assert main(["train", "--episodes", "0", "--output", str(path)]) == 1


def test_validation_test_namespaces_are_distinct(small_model):
    # Verify protocol separation without running the actual final benchmark.
    validation = benchmark([small_model], runs=1, suites=("mixed",), split="validation")
    testing = benchmark([small_model], runs=1, suites=("mixed",), split="test")
    a = validation["suites"]["mixed"]["episodes"][0]
    b = testing["suites"]["mixed"]["episodes"][0]
    assert a["scenario_seed"] != b["scenario_seed"]
    assert a["scenario_sha256"] != b["scenario_sha256"]


def test_small_training_can_learn_a_direct_action():
    # Optimizer sanity: a repeatedly rewarded action's Q-value rises.
    network = QNetwork(0, 8)
    optimizer = Adam(network, 0.01)
    context = Context(2)
    context.observe(Observation(0, 1, detections=1))
    features = context.encode(1)
    x = np.repeat(features, 16, axis=0)
    targets = np.repeat([0.0, 1.0], 16)
    for _ in range(200):
        _, gradients = network.loss_gradients(x, targets)
        optimizer.update(gradients)
    assert network.predict(features)[1] > network.predict(features)[0] + 0.8
