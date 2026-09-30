"""Paired CUDA comparison of existing PPO and MPC policies on physical worlds."""

import argparse
from functools import partial
from pathlib import Path

import torch

from ..mpc.checkpoints import load_model
from ..mpc.config import REWARD
from ..mpc.scheduler import NeuralMPCScheduler
from ..recurrent_cli import load_checked
from ..recurrent_env import RecurrentScheduler
from ..rl_benchmark import benchmark
from ..schedulers import AdaptiveDwellScheduler, DwellSweepScheduler
from .storage import fingerprint, write_json


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-run", type=Path, required=True)
    parser.add_argument("--mpc-run", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    if min(args.runs, args.simulations) < 1 or args.seed < 0:
        parser.error("runs/simulations must be positive and seed nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural policy comparison")
    torch.set_num_threads(1)
    ppo_path = args.ppo_run / "best.zip"
    mpc_path = args.mpc_run / "best.pt"
    ppo, ppo_metadata = load_checked(ppo_path, "cuda")
    mpc = load_model(mpc_path, torch.device("cuda"))
    if not ppo.physical_contract or not mpc.physical_contract:
        raise ValueError("both checkpoints must use post-retune physical dwell timing")
    mpc_metadata = torch.load(mpc_path, map_location="cpu", weights_only=True)["metadata"]
    saved = mpc_metadata["config"]
    policies = {
        "recurrent-ppo": lambda: RecurrentScheduler(ppo),
        "mpc-search": partial(
            NeuralMPCScheduler,
            model=mpc,
            num_simulations=args.simulations,
            gamma=saved["gamma"],
            depth=saved["depth"],
            normalize_search=saved["normalize_search"],
            device="cuda",
        ),
        "mpc-policy": partial(
            NeuralMPCScheduler,
            model=mpc,
            num_simulations=0,
            gamma=saved["gamma"],
            device="cuda",
        ),
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
        suites=("randomized", "receiver-shift", "periodic-scan"),
        num_bands=8,
        physical_worlds=True,
        reward=REWARD,
        extra_policies=policies,
        extra_metadata={
            "ppo": {"sha256": fingerprint(ppo_path), "metadata": ppo_metadata},
            "mpc": {"sha256": fingerprint(mpc_path), "metadata": mpc_metadata},
            "search_simulations": args.simulations,
            "training_rewards_differ": ppo_metadata["config"]["reward"] != vars(REWARD),
        },
        progress=lambda suite, means: print(
            suite,
            {name: means[name]["interception_ratio"] for name in policies},
            flush=True,
        ),
    )
    report["scope"] = "paired physical validation worlds; test split untouched"
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
