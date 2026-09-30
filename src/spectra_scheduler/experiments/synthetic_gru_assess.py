"""Paired synthetic GRU assessment across required emitter scenarios."""

import argparse
from functools import partial
from pathlib import Path

from spectra_scheduler.experiments.storage import write_json
from spectra_scheduler.policy_benchmark import PolicySpec, benchmark_synthetic
from spectra_scheduler.schedulers import (
    AdaptiveDwellScheduler,
    DwellSweepScheduler,
    PeriodAwareScheduler,
    RoundRobinScheduler,
    ShuffledSweepScheduler,
)
from spectra_scheduler.synthetic_forecast import synthetic_gru_spec


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sensitivity-dbm", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    if args.runs < 1 or args.seed < 0 or args.seed + args.runs >= 2**32:
        parser.error("invalid run count or seed range")
    policies = [
        PolicySpec("sweep", RoundRobinScheduler, "round-robin"),
        PolicySpec("shuffled", ShuffledSweepScheduler, "shuffled-sweep"),
        PolicySpec("period-aware", PeriodAwareScheduler, "period-aware"),
        PolicySpec("dwell-8", partial(DwellSweepScheduler, dwell_steps=8), "dwell-sweep:8"),
        PolicySpec("dwell-32", partial(DwellSweepScheduler, dwell_steps=32), "dwell-sweep:32"),
        PolicySpec(
            "adaptive-long",
            partial(
                AdaptiveDwellScheduler,
                minimum_dwell_steps=4,
                hit_extension_steps=4,
                maximum_dwell_steps=32,
            ),
            "adaptive-dwell:4:4:32",
        ),
    ]
    policies.extend(
        synthetic_gru_spec(path, f"gru-{index}") for index, path in enumerate(args.checkpoint)
    )
    policies.extend(
        synthetic_gru_spec(args.checkpoint[0], f"gru-{mode}", ablation=mode)
        for mode in ("constant", "observed-rate", "no-timing")
    )
    report = benchmark_synthetic(
        policies,
        baseline="sweep",
        split=args.split,
        seeds=tuple(range(args.seed, args.seed + args.runs)),
        workers=1,
        sensitivity_dbm=args.sensitivity_dbm,
    )
    report["acceptance_criteria"] = {
        "capture": (
            "mean interception ratio at least the best of dwell-32, adaptive-long, "
            "and observed-rate on each required scenario"
        ),
        "discovery": "mean emitter discovery ratio at least dwell-32 on each scenario",
        "timing": "restricted timing MAE reported with forecast and event coverage",
        "selection": "validation only; test used after frozen model selection",
    }
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
