"""CUDA evaluation of frozen recurrent policies on independent procedural worlds."""

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..recurrent_cli import load_checked
from ..recurrent_env import RecurrentScheduler, encode_context
from ..rl import RewardConfig
from ..rl_benchmark import SUITES, benchmark
from ..schedulers import AdaptiveDwellScheduler, DwellSweepScheduler
from .storage import fingerprint, write_json


class SampledRecurrentScheduler(RecurrentScheduler):
    """Seeded categorical execution of the actual stochastic PPO policy."""

    def __init__(self, model, seed=0):
        super().__init__(model)
        self.seed = seed

    def reset(self, num_bands):
        super().reset(num_bands)
        self.generator = torch.Generator(device=self.model.device).manual_seed(self.seed)

    @torch.inference_mode()
    def choose_band(self, time_step):
        if self.pending is not None:
            raise ValueError("previous action has no observation")
        if self.remaining:
            if not self.physical_contract:
                self.remaining -= 1
            self.pending = time_step, self.held_band
            return self.held_band
        observation = encode_context(self.context, time_step)
        tensor, _ = self.model.policy.obs_to_tensor(observation)
        distribution, self.state = self.model.policy.get_distribution(
            tensor,
            self.state,
            torch.tensor([self.episode_start], device=self.model.device, dtype=torch.float32),
        )
        logits = distribution.distribution.logits[
            0, : self.context.history.num_bands * len(self.dwell_steps)
        ]
        action = int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator).item())
        band, dwell = divmod(action, len(self.dwell_steps))
        self.remaining = (
            self.dwell_steps[dwell] if self.physical_contract else self.dwell_steps[dwell] - 1
        )
        self.held_band = band
        self.episode_start = False
        self.pending = time_step, band
        return band


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20000)
    args = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise ValueError("CUDA inference required")
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model, metadata = load_checked(args.run_dir / "best.zip", "cuda")
    initial, initial_metadata = load_checked(args.run_dir / "untrained.zip", "cuda")
    if model.physical_contract != initial.physical_contract:
        raise ValueError("trained and initial policies use different action timing contracts")
    policies = {
        "ppo-greedy": lambda: RecurrentScheduler(model),
        "ppo-sampled": lambda: SampledRecurrentScheduler(model),
        "untrained-greedy": lambda: RecurrentScheduler(initial),
        "untrained-sampled": lambda: SampledRecurrentScheduler(initial),
        "dwell-4": partial(DwellSweepScheduler, dwell_steps=4),
        "dwell-8": partial(DwellSweepScheduler, dwell_steps=8),
        "adaptive-long": partial(
            AdaptiveDwellScheduler,
            minimum_dwell_steps=4,
            hit_extension_steps=4,
            maximum_dwell_steps=16,
        ),
    }
    report = benchmark(
        [],
        runs=args.runs,
        seed=args.seed,
        split="validation",
        num_bands=8,
        physical_worlds=model.physical_contract,
        suites=("randomized", "receiver-shift", "periodic-scan")
        if model.physical_contract else SUITES,
        profile=True,
        reward=RewardConfig(**metadata["config"]["reward"]),
        extra_policies=policies,
        extra_metadata={
            "trained": metadata,
            "untrained": initial_metadata,
            "additional_controls": "sweep dwells 4/8; adaptive minimum4/extension4/max16",
        },
        progress=lambda suite, means: print(
            json.dumps(
                {
                    "suite": suite,
                    "capture": {
                        name: values["interception_ratio"] for name, values in means.items()
                    },
                }
            ),
            flush=True,
        ),
    )
    for suite in report["suites"].values():
        for candidate in ("ppo-greedy", "ppo-sampled"):
            for reference in suite["mean_metrics"]:
                if reference == candidate:
                    continue
                deltas = np.array(
                    [
                        row["metrics"][candidate]["interception_ratio"]
                        - row["metrics"][reference]["interception_ratio"]
                        for row in suite["episodes"]
                    ]
                )
                rng = np.random.default_rng(0)
                means = np.array([rng.choice(deltas, size=len(deltas)).mean() for _ in range(2000)])
                suite["paired_interception_delta"][candidate][reference][
                    "bootstrap_95_interval"
                ] = np.quantile(means, [0.025, 0.975]).tolist()
    report["scope"] = (
        "development validation, one trained seed; stochastic execution uses "
        "fixed policy seed zero; no test data or operational-performance claim"
    )
    report["assessment_sha256"] = fingerprint(__file__)
    torch.cuda.synchronize()
    report["runtime"] = {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "latency_definition": "mean choose_band time per physical tick, including held dwell",
        "reward_definition": "common legacy 24-tick coverage debt for all benchmark policies",
    }
    write_json(args.run_dir / "comparison.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
