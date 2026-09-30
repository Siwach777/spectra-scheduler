"""Supervised replay adapter; shared runner has no knowledge of this learner."""

import copy
import math
from contextlib import closing
from dataclasses import asdict, dataclass
from functools import partial
from time import perf_counter

import numpy as np
import torch

from ..forecast_model import (
    ForecastNetwork,
    ModelConfig,
    optimization_step,
    predictor_spec,
    save_predictor,
)
from ..policy_benchmark import PolicySpec, benchmark_policies, validate_plan
from ..pulse_replay import ReplayConfig
from ..replay_env import InterfaceConfig, ReplayEnv
from ..replay_evaluation import ReferencePolicy
from ..replay_training import BatchConfig, UniformActionPolicy, training_batches
from .storage import fingerprint, load_torch, save_torch


@dataclass(frozen=True)
class PredictorConfig:
    seed: int = 0
    learning_rate: float = 0.001
    batch_size: int = 64
    history_steps: int = 8
    hidden: int = 64
    time_bins: int = 16
    max_batches_per_epoch: int | None = None
    encoder: str = "gru"

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be uint32")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        BatchConfig(self.batch_size, self.history_steps, self.time_bins)
        if type(self.hidden) is not int or self.hidden < 1:
            raise ValueError("hidden size must be a positive integer")
        if self.encoder not in ("gru", "mlp", "tcn"):
            raise ValueError("unknown predictor encoder")
        if self.max_batches_per_epoch is not None and (
            type(self.max_batches_per_epoch) is not int or self.max_batches_per_epoch < 1
        ):
            raise ValueError("batch budget must be positive or None")


class PredictorLearner:
    def __init__(
        self,
        root,
        train_plan,
        validation_plan,
        config=None,
        receiver=None,
        interface=None,
        *,
        device="cuda",
        threads=1,
        validation_workers=1,
        cache=None,
    ):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be positive")
        if type(validation_workers) is not int or validation_workers < 1:
            raise ValueError("validation workers must be positive")
        if train_plan.get("split") != "train" or validation_plan.get("split") != "val":
            raise ValueError("learner requires separate train and val plans; test is forbidden")
        self.train_plan, self.validation_plan = (
            copy.deepcopy(train_plan),
            copy.deepcopy(validation_plan),
        )
        self.root, self.config = root, config or PredictorConfig()
        paths = validate_plan(root, self.train_plan)
        validate_plan(root, self.validation_plan)
        train_hashes = {r["sha256"] for r in self.train_plan["recordings"]}
        if train_hashes.intersection(r["sha256"] for r in self.validation_plan["recordings"]):
            raise ValueError("training and validation content overlap")
        self.training_hashes = tuple(sorted(train_hashes))
        self.receiver, self.interface = receiver or ReplayConfig(), interface or InterfaceConfig()
        self.specification = ReplayEnv(paths[0], self.receiver, self.interface).specification()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.set_num_threads(threads)
        torch.manual_seed(self.config.seed)
        self.model = ForecastNetwork(
            ModelConfig(
                features=self.specification["observation_size"],
                actions=self.specification["action_count"],
                history_steps=self.config.history_steps,
                hidden=self.config.hidden,
                time_bins=self.config.time_bins,
                bands=self.interface.bands,
                encoder=self.config.encoder,
            )
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, fused=self.device.type == "cuda"
        )
        self.threads, self.validation_workers = threads, validation_workers
        self.samples = self.updates = 0
        self.cache = None
        if cache is not None:
            from .cache import CachedBatches, cache_configuration

            self.cache = CachedBatches(cache)
            expected = cache_configuration(
                self.train_plan,
                self.receiver,
                self.interface,
                self.config.history_steps,
                self.config.time_bins,
            )
            import json

            if self.cache.manifest["configuration"] != json.loads(json.dumps(expected)):
                raise ValueError("training cache differs from learner configuration")
            if self.device.type != "cuda":
                raise ValueError("cached training requires CUDA")

    def configuration(self):
        from pathlib import Path

        package = Path(__file__).resolve().parent.parent
        names = (
            "experiments/predictor.py",
            "experiments/study.py",
            "experiments/cache.py",
            "forecast_model.py",
            "replay_training.py",
            "replay_env.py",
            "replay_features.py",
            "pulse_replay.py",
            "evaluation_contract.py",
            "policy_benchmark.py",
            "replay_evaluation.py",
            "dataset_io.py",
        )
        return {
            "adapter": "supervised-replay-predictor-v1",
            "config": asdict(self.config),
            "specification": self.specification,
            "train_plan": self.train_plan,
            "validation_plan": self.validation_plan,
            "behavior": "uniform-all-actions;epoch-shuffled-files-and-seeds",
            "implementation": {name: fingerprint(package / name) for name in names},
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "device": str(self.device),
            "threads": self.threads,
            "cache": self.cache.manifest if self.cache else None,
        }

    def train_epoch(self, epoch, progress):
        started = perf_counter()
        plan = copy.deepcopy(self.train_plan)
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch]))
        rng.shuffle(plan["recordings"])
        # Deterministic distinct episode randomness; no validation/test trajectories.
        offset = int(rng.integers(0, 2**32))
        plan["seeds"] = [(seed + offset) % 2**32 for seed in plan["seeds"]]
        batch_cfg = BatchConfig(
            self.config.batch_size, self.config.history_steps, self.config.time_bins
        )
        totals = torch.zeros(2, device=self.device)
        samples = batches = 0
        iterator = (
            self.cache.batches(
                self.config.batch_size,
                np.random.SeedSequence([self.config.seed, epoch]),
                self.device,
                self.config.max_batches_per_epoch,
            )
            if self.cache
            else training_batches(
                self.root, plan, UniformActionPolicy, self.receiver, self.interface, batch_cfg
            )
        )
        with closing(iterator):
            for batch in iterator:
                losses = optimization_step(
                    self.model, self.optimizer, batch, validated=self.cache is not None
                )
                count = len(batch["action"])
                # Accumulate on device; synchronize once at the epoch boundary.
                totals[0] += losses["timing"] * count
                totals[1] += losses["ratio"] * count
                samples += count
                batches += 1
                progress(batches=batches, samples=samples)
                if (
                    self.config.max_batches_per_epoch
                    and batches >= self.config.max_batches_per_epoch
                ):
                    break
        validate_plan(self.root, self.train_plan)
        if not samples:
            raise ValueError("training generated no examples")
        self.samples += samples
        self.updates += batches
        timing, ratio = (totals / samples).cpu().tolist()
        return {
            "samples": samples,
            "batches": batches,
            "total_samples": self.samples,
            "total_updates": self.updates,
            "mean_timing_loss": timing,
            "batch_weighted_ratio_loss": ratio,
            "elapsed_seconds": perf_counter() - started,
            "samples_per_second": samples / (perf_counter() - started),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
            ),
        }

    def save_state(self, path):
        save_torch(
            path,
            {
                "version": 1,
                "configuration": self.configuration(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "samples": self.samples,
                "updates": self.updates,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            },
        )

    def load_state(self, path):
        import json

        saved = load_torch(path)
        if saved.get("version") != 1 or json.dumps(
            saved["configuration"], sort_keys=True
        ) != json.dumps(self.configuration(), sort_keys=True):
            raise ValueError("incompatible predictor training checkpoint")
        self.model.load_state_dict(saved["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("checkpoint model contains nonfinite parameters")
        self.optimizer.load_state_dict(saved["optimizer"])
        self.samples, self.updates = saved["samples"], saved["updates"]
        torch.set_rng_state(saved["rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def export_policy(self, path):
        save_predictor(path, self.model, self.specification, self.training_hashes)

    def validate(self, policy_path):
        # Loading independent inference models must not consume the learner's RNG.
        rng = torch.get_rng_state()
        try:
            dwell = min(1, len(self.interface.dwell_us) - 1)
            policies = [
                PolicySpec(
                    "sweep", partial(ReferencePolicy, "sweep", dwell), f"dwell-index={dwell}"
                ),
                PolicySpec(
                    "random", partial(ReferencePolicy, "random", dwell), f"dwell-index={dwell}"
                ),
                predictor_spec(policy_path, threads=self.threads, device=str(self.device)),
                predictor_spec(
                    policy_path,
                    name="constant-coverage",
                    threads=self.threads,
                    device=str(self.device),
                    constant_predictions=True,
                ),
                predictor_spec(
                    policy_path,
                    name="observed-rate",
                    threads=self.threads,
                    device=str(self.device),
                    observed_rate=True,
                ),
            ]
            report = benchmark_policies(
                self.root,
                self.validation_plan,
                policies,
                baseline="sweep",
                workers=self.validation_workers,
                receiver=self.receiver,
                interface=self.interface,
                inference_batch_size=16 if self.device.type == "cuda" else 1,
            )
            capture = report["summary"]["predictor"]["interception_ratio"]["mean"]
            discovery = report["summary"]["predictor"]["discovery_fraction"]["mean"]
            report["selection"] = {
                "capture_discovery_harmonic": None
                if capture is None or discovery is None
                else 2 * capture * discovery / max(capture + discovery, 1e-12),
            }
            return report
        finally:
            torch.set_rng_state(rng)
