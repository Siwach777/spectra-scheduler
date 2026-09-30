"""Paired replay diagnostics for passband placement and dwell choices.

This compares simple, observation-only controls. It measures whether a grid
constrains these controls; it is not an estimate of an optimal policy's value.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from spectra_scheduler.dataset_io import discover_files
from spectra_scheduler.experiments.storage import fingerprint, write_json
from spectra_scheduler.pulse_replay import DwellAction, PulseReplay, ReplayConfig
from spectra_scheduler.replay_env import InterfaceConfig, tune_centers


@dataclass(frozen=True)
class GridCase:
    name: str
    centers: int
    short_us: float
    long_us: float
    policy: str


CASES = (
    GridCase("sweep-1ms", 8, 1_000, 1_000, "sweep"),
    GridCase("sweep-5ms", 8, 5_000, 5_000, "sweep"),
    GridCase("sweep-10ms", 8, 10_000, 10_000, "sweep"),
    GridCase("sweep-25ms", 8, 25_000, 25_000, "sweep"),
    GridCase("sweep-50ms", 8, 50_000, 50_000, "sweep"),
    GridCase("rate-8-s1-l50", 8, 1_000, 50_000, "rate"),
    GridCase("rate-8-s1-l25", 8, 1_000, 25_000, "rate"),
    GridCase("rate-8-s5-l50", 8, 5_000, 50_000, "rate"),
    GridCase("rate-8-s5-l25", 8, 5_000, 25_000, "rate"),
    GridCase("rate-8-s10-l50", 8, 10_000, 50_000, "rate"),
    GridCase("rate-9-s1-l50", 9, 1_000, 50_000, "rate"),
    GridCase("rate-12-s1-l50", 12, 1_000, 50_000, "rate"),
    GridCase("rate-12-s5-l50", 12, 5_000, 50_000, "rate"),
    GridCase("rate-16-s1-l50", 16, 1_000, 50_000, "rate"),
    GridCase("rate-24-s1-l50", 24, 1_000, 50_000, "rate"),
)
REFERENCE = "rate-8-s1-l50"


def center_grid(receiver: ReplayConfig, count: int) -> np.ndarray:
    return tune_centers(receiver, InterfaceConfig(bands=count))


def evaluate_case(path: Path, receiver: ReplayConfig, case: GridCase) -> dict:
    centers = center_grid(receiver, case.centers)
    rates = np.zeros(len(centers), dtype=np.float64)
    last_visit = np.full(len(centers), -np.inf, dtype=np.float64)
    seen = np.zeros(len(centers), dtype=bool)
    visits = np.zeros(len(centers), dtype=np.int32)
    step = 0
    with PulseReplay(path, receiver, source_mode="stare") as replay:
        while not replay.done:
            if case.policy == "sweep":
                choice, dwell = step % len(centers), case.long_us
            else:
                unseen = np.flatnonzero(~seen)
                if len(unseen):
                    choice, dwell = int(unseen[0]), case.short_us
                else:
                    ages = replay.time_us - last_visit
                    oldest = int(np.argmax(ages))
                    if ages[oldest] >= 500_000:
                        choice, dwell = oldest, case.short_us
                    else:
                        choice, dwell = int(np.argmax(rates)), case.long_us
            observation = replay.step(DwellAction(float(centers[choice]), dwell))
            step += 1
            listening_us = observation.end_us - observation.listening_start_us
            if listening_us > 0:
                observed_rate = len(observation.pulses) * 1_000_000 / listening_us
                rates[choice] = (
                    observed_rate if not seen[choice] else 0.5 * (rates[choice] + observed_rate)
                )
                seen[choice] = True
                last_visit[choice] = observation.end_us
                visits[choice] += 1
        report = replay.report()
    return {
        key: report[key]
        for key in (
            "truth_pulses",
            "intercepted_pulses",
            "delivered_pulses",
            "emitters_present",
            "emitters_discovered",
            "mean_discovery_delay_us",
            "retuning_us",
            "listening_us",
            "steps",
        )
    } | {"centers_visited": int(np.count_nonzero(visits))}


def evaluate_recording(job):
    path, receiver = job
    return {
        "path": str(path),
        "cases": {case.name: evaluate_case(path, receiver, case) for case in CASES},
    }


def summarize(rows: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(rows), size=(2_000, len(rows)))
    result = {}
    baseline = [row["cases"][REFERENCE] for row in rows]
    for case in CASES:
        values = [row["cases"][case.name] for row in rows]
        truth = np.asarray([value["truth_pulses"] for value in values], dtype=np.float64)
        captured = np.asarray([value["intercepted_pulses"] for value in values], dtype=np.float64)
        control = np.asarray([value["intercepted_pulses"] for value in baseline], dtype=np.float64)
        baseline_truth = np.asarray([value["truth_pulses"] for value in baseline])
        if not np.array_equal(truth, baseline_truth):
            raise ValueError("paired grid cases have different truth denominators")
        total = truth.sum()
        if not total:
            raise ValueError("selected validation recordings have no in-range pulses")
        boot_truth = truth[samples].sum(axis=1)
        boot_delta = (captured[samples].sum(axis=1) - control[samples].sum(axis=1)) / np.maximum(
            boot_truth, 1
        )
        discovery = [(value["emitters_discovered"], value["emitters_present"]) for value in values]
        known = [pair for pair in discovery if pair[0] is not None and pair[1] is not None]
        result[case.name] = {
            "grid": asdict(case),
            "pooled_capture": float(captured.sum() / total),
            "pooled_capture_delta_vs_reference": float((captured.sum() - control.sum()) / total),
            "paired_recording_bootstrap_95": np.quantile(boot_delta, [0.025, 0.975]).tolist(),
            "mean_recording_capture": float(np.mean(captured[truth > 0] / truth[truth > 0])),
            "pooled_discovery": (
                sum(pair[0] for pair in known) / sum(pair[1] for pair in known)
                if known and sum(pair[1] for pair in known)
                else None
            ),
            "mean_retuning_us": float(np.mean([v["retuning_us"] for v in values])),
            "mean_centers_visited": float(np.mean([v["centers_visited"] for v in values])),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--files", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/action-grid/validation.json")
    )
    args = parser.parse_args(argv)
    if args.files < 1 or args.workers < 1 or args.seed < 0:
        parser.error("files/workers must be positive and seed nonnegative")
    files = discover_files(args.root, "stare", "val")
    if len(files) < args.files:
        parser.error("fewer completed validation recordings than requested")
    selection = np.sort(
        np.random.default_rng(args.seed).choice(len(files), args.files, replace=False)
    )
    selected = [files[int(index)] for index in selection]
    receiver = ReplayConfig()
    jobs = [(path, receiver) for path in selected]
    if args.workers == 1:
        rows = [evaluate_recording(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            rows = list(pool.map(evaluate_recording, jobs))
    report = {
        "schema_version": 1,
        "scope": "paired validation; observation-only controls; no optimality claim",
        "selection_seed": args.seed,
        "reference": REFERENCE,
        "receiver": asdict(receiver),
        "implementation_sha256": fingerprint(__file__),
        "recordings": rows,
        "summary": summarize(rows, args.seed),
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(rows),
                "summary": {
                    name: {
                        "capture": value["pooled_capture"],
                        "discovery": value["pooled_discovery"],
                    }
                    for name, value in report["summary"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
