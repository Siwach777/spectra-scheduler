# Optional learner tests run in .venv-rl; the core package remains NumPy-only.
# ruff: noqa: E402
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra_scheduler.forecast_model import (
    ForecastNetwork,
    ModelConfig,
    PredictorPolicy,
    load_predictor,
    optimization_step,
    prediction_loss,
    save_predictor,
)
from spectra_scheduler.pulse_replay import ReplayConfig
from spectra_scheduler.replay_env import InterfaceConfig, ReplayEnv


def configuration(tmp_path):
    env = ReplayEnv(
        tmp_path / "unused.h5",
        ReplayConfig(stop_us=20, max_frequency_mhz=20, bandwidth_mhz=10, retune_us=1),
        InterfaceConfig(bands=2, dwell_us=(2, 4)),
    )
    return ModelConfig(
        features=20, actions=4, history_steps=3, hidden=8, time_bins=4
    ), env.specification()


def test_censoring_loss_has_gradients_and_masks_empty_ratio(tmp_path):
    torch.set_num_threads(1)
    config, _ = configuration(tmp_path)
    model = ForecastNetwork(config)
    batch = {
        "time_bins": 4,
        "history": np.zeros((3, 3, 20), np.float32),
        "action": np.array([0, 1, 2], np.int64),
        "time_class": np.array([0, 2, 4], np.int64),
        "ratio": np.zeros(3, np.float32),
        "ratio_valid": np.zeros(3, np.bool_),
    }
    loss, components = prediction_loss(model, batch)
    assert components["ratio"] == 0
    loss.backward()
    assert model.time_head.weight.grad.abs().sum() > 0
    assert model.ratio_head.weight.grad.abs().sum() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    batch["ratio_valid"][:] = True
    before = model.ratio_head.weight.detach().clone()
    optimization_step(model, torch.optim.Adam(model.parameters(), lr=0.01), batch)
    assert not torch.equal(before, model.ratio_head.weight)
    batch["action"][0] = -1
    with pytest.raises(ValueError, match="action"):
        prediction_loss(model, batch)


def test_artifact_inference_and_reset_parity(tmp_path):
    config, spec = configuration(tmp_path)
    torch.manual_seed(1)
    model = ForecastNetwork(config)
    path = tmp_path / "predictor.pt"
    save_predictor(path, model, spec, ("a" * 64,))
    restored, saved_spec, hashes = load_predictor(path)
    assert hashes == ("a" * 64,)
    assert json.dumps(saved_spec, sort_keys=True) == json.dumps(spec, sort_keys=True)
    inputs = torch.zeros((1, 3, 20))
    for a, b in zip(model(inputs), restored(inputs), strict=True):
        torch.testing.assert_close(a, b)
    policy = PredictorPolicy(path)
    obs = np.zeros(20, np.float32)
    obs[-1] = -1
    policy.reset(spec, 0)
    first = policy.act(obs)
    assert 0 <= first.action < 4
    assert 0 <= first.forecast.hit_probability <= 1
    assert first.forecast.intercept_delay_seconds is None or (
        0 <= first.forecast.intercept_delay_seconds <= 4e-6
    )
    policy.reset(spec, 1)
    assert first == policy.act(obs)
    spec["feature_version"] = -1
    with pytest.raises(ValueError, match="incompatible"):
        policy.reset(spec, 0)


def test_batched_inference_equals_single_sequence(tmp_path):
    config, _ = configuration(tmp_path)
    model = ForecastNetwork(config).eval()
    inputs = torch.randn(4, 3, 20)
    batch = model(inputs)
    for i in range(4):
        single = model(inputs[i : i + 1])
        torch.testing.assert_close(batch[0][i : i + 1], single[0])
        torch.testing.assert_close(batch[1][i : i + 1], single[1])


def test_model_dimensions_and_nonfinite_artifact_rejected(tmp_path):
    config, spec = configuration(tmp_path)
    model = ForecastNetwork(config)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 2, 20))
    with torch.no_grad():
        model.time_head.weight[0, 0] = float("nan")
    path = tmp_path / "bad.pt"
    save_predictor(path, model, spec)
    with pytest.raises(ValueError, match="nonfinite"):
        load_predictor(path)


def test_collection_loss_checkpoint_and_benchmark_integration(tmp_path):
    h5py = pytest.importorskip("h5py")
    from contextlib import closing

    from spectra_scheduler.forecast_model import predictor_spec
    from spectra_scheduler.policy_benchmark import PolicySpec, benchmark_policies, make_plan
    from spectra_scheduler.replay_evaluation import ReferencePolicy
    from spectra_scheduler.replay_training import BatchConfig, UniformActionPolicy, training_batches

    directory = tmp_path / "stare" / "train_stare"
    directory.mkdir(parents=True)
    path = directory / "tiny.h5"
    with h5py.File(path, "w") as f:
        f["data"] = np.array([[i, 5, 0.1, 0, -20] for i in range(20)], dtype=float)
        f["metadata/feature_names"] = np.array(
            ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
        )
    config, spec = configuration(tmp_path)
    model = ForecastNetwork(config)
    receiver = ReplayConfig(**spec["receiver"])
    interface = InterfaceConfig(**spec["interface"])
    plan = make_plan(tmp_path, "train", max_files=1, seeds=(0,))
    with closing(
        training_batches(
            tmp_path, plan, UniformActionPolicy, receiver, interface, BatchConfig(2, 3, 4)
        )
    ) as batches:
        loss, _ = prediction_loss(model, next(batches))
        assert torch.isfinite(loss)
    checkpoint = tmp_path / "reference.pt"
    hashes = tuple(r["sha256"] for r in plan["recordings"])
    save_predictor(checkpoint, model, spec, hashes)
    policies = [PolicySpec("sweep", ReferencePolicy, "default"), predictor_spec(checkpoint)]
    report = benchmark_policies(
        tmp_path, plan, policies, baseline="sweep", receiver=receiver, interface=interface
    )
    assert report["summary"]["predictor"]["prediction.coverage"]["mean"] == 1
    assert report["results"][0]["policies"]["predictor"]["complete"]
    # Registered benchmark cannot silently load replacement checkpoint bytes.
    save_predictor(checkpoint, ForecastNetwork(config), spec, hashes)
    with pytest.raises(ValueError, match="content changed"):
        policies[1].factory()
