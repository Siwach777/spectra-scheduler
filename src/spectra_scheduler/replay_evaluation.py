"""Shared inference runner and paired recording-level comparison, without training."""

import argparse
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import copy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from .dataset_io import discover_files
from .evaluation_contract import Decision, EvaluationAccumulator
from .pulse_replay import ReplayConfig
from .replay_env import InterfaceConfig, ReplayEnv


class Policy(Protocol):
    def reset(self, specification: dict, seed: int) -> None: ...
    def act(self, observation: np.ndarray) -> int | Decision: ...


class ReferencePolicy:
    """Simple execution checks, not learned competitors or performance targets."""

    def __init__(self, kind="sweep", dwell_index=1):
        if kind not in ("sweep", "random"):
            raise ValueError("unknown reference policy")
        self.kind, self.dwell_index = kind, dwell_index

    def reset(self, specification, seed):
        cfg = specification["interface"]
        self.bands, self.dwells = cfg["bands"], len(cfg["dwell_us"])
        if not 0 <= self.dwell_index < self.dwells:
            raise ValueError("dwell index out of range")
        self.rng = np.random.default_rng(seed)
        self.step = 0

    def act(self, observation):
        band = (
            self.step % self.bands if self.kind == "sweep" else int(self.rng.integers(self.bands))
        )
        self.step += 1
        return band * self.dwells + self.dwell_index


def evaluate_policy(path, policy: Policy, receiver=None, interface=None):
    """Models own their inference device/state; this runner never imports a ML framework."""
    receiver = receiver if receiver is not None else ReplayConfig()
    interface = interface if interface is not None else InterfaceConfig()
    begin = perf_counter()
    inference_seconds = maximum_latency = reward = 0.0
    steps = 0
    evaluation = EvaluationAccumulator()
    with ReplayEnv(path, receiver, interface) as env:
        observation = env.reset()
        spec = env.specification()
        policy.reset(spec, receiver.seed)
        while True:
            t = perf_counter()
            decision = policy.act(observation)
            action = decision.action if isinstance(decision, Decision) else decision
            forecast = decision.forecast if isinstance(decision, Decision) else None
            latency = perf_counter() - t
            inference_seconds += latency
            maximum_latency = max(maximum_latency, latency)
            transition = env.step(action)
            evaluation.add(env.evaluation_outcome(), transition.reward, forecast)
            reward += transition.reward
            observation = transition.observation
            steps += 1
            if transition.terminated:
                break
        report = env.metrics()
    figures = evaluation.report()
    figures["sensitivity"] = {
        "threshold": receiver.sensitivity_db,
        "unit": "dataset_db",
        "status": "configured" if receiver.sensitivity_db is not None else "disabled",
        "calibrated_dbm": False,
    }
    figures["false_alarm_status"] = "not_modelled"
    report.update(
        evaluation=figures,
        specification=spec,
        reward_sum=reward,
        elapsed_seconds=perf_counter() - begin,
        mean_policy_latency_seconds=inference_seconds / steps,
        max_policy_latency_seconds=maximum_latency,
    )
    return report


def evaluate_policy_batch(jobs, factory, interface):
    """Bounded concurrent episodes; one shared model and batched inference if supported.

    Each decision's latency includes the complete batch inference wait, not divided
    throughput masquerading as single-action latency. Receiver semantics are unchanged.
    """
    template = factory()
    begin = perf_counter()
    with ExitStack() as stack:
        states = []
        for path, receiver in jobs:
            env = stack.enter_context(ReplayEnv(path, receiver, interface))
            observation = env.reset()
            policy = copy(template) if hasattr(template, "act_batch") else factory()
            spec = env.specification()
            policy.reset(spec, receiver.seed)
            states.append(
                dict(
                    env=env,
                    policy=policy,
                    observation=observation,
                    receiver=receiver,
                    spec=spec,
                    accumulator=EvaluationAccumulator(),
                    reward=0.0,
                    seconds=0.0,
                    maximum=0.0,
                    steps=0,
                    done=False,
                )
            )
        while active := [s for s in states if not s["done"]]:
            start = perf_counter()
            if hasattr(template, "act_batch"):
                decisions = template.act_batch(
                    [s["policy"] for s in active], [s["observation"] for s in active]
                )
            else:
                decisions = [s["policy"].act(s["observation"]) for s in active]
            latency = perf_counter() - start
            for state, decision in zip(active, decisions, strict=True):
                action = decision.action if isinstance(decision, Decision) else decision
                forecast = decision.forecast if isinstance(decision, Decision) else None
                transition = state["env"].step(action)
                state["accumulator"].add(
                    state["env"].evaluation_outcome(), transition.reward, forecast
                )
                state["reward"] += transition.reward
                state["observation"] = transition.observation
                state["done"] = transition.terminated
                state["seconds"] += latency
                state["maximum"] = max(latency, state["maximum"])
                state["steps"] += 1
        reports = []
        for state in states:
            figures = state["accumulator"].report()
            sensitivity = state["receiver"].sensitivity_db
            figures["sensitivity"] = {
                "threshold": sensitivity,
                "unit": "dataset_db",
                "status": "configured" if sensitivity is not None else "disabled",
                "calibrated_dbm": False,
            }
            figures["false_alarm_status"] = "not_modelled"
            report = state["env"].metrics()
            report.update(
                evaluation=figures,
                specification=state["spec"],
                reward_sum=state["reward"],
                elapsed_seconds=perf_counter() - begin,
                mean_policy_latency_seconds=state["seconds"] / state["steps"],
                max_policy_latency_seconds=state["maximum"],
                inference_batch_size=len(jobs),
            )
            reports.append(report)
        return reports


def _evaluate_pair(job):
    path, receiver, interface, dwell_index = job
    return {
        name: evaluate_policy(path, ReferencePolicy(name, dwell_index), receiver, interface)
        for name in ("sweep", "random")
    }


def benchmark(
    root,
    split="val",
    max_files=3,
    workers=1,
    receiver=None,
    interface=None,
    dwell_index=1,
):
    receiver = receiver if receiver is not None else ReplayConfig()
    interface = interface if interface is not None else InterfaceConfig()
    if max_files < 1 or workers < 1:
        raise ValueError("max-files and workers must be positive")
    files = discover_files(Path(root), "stare", split)[:max_files]
    if not files:
        raise ValueError("no completed stare files in selected split")
    jobs = [(p, receiver, interface, dwell_index) for p in files]
    if workers == 1:
        results = [_evaluate_pair(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(files)), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = list(pool.map(_evaluate_pair, jobs))
    differences = np.array(
        [
            r["sweep"]["delivery_fraction"] - r["random"]["delivery_fraction"]
            for r in results
            if r["sweep"]["delivery_fraction"] is not None
        ]
    )
    interval = None
    if len(differences) > 1:
        rng = np.random.default_rng(receiver.seed)
        # Bounded bootstrap workspace: avoid allocating resamples x all recordings.
        means = np.empty(2000)
        for i in range(len(means)):
            means[i] = rng.choice(differences, size=len(differences), replace=True).mean()
        interval = np.quantile(means, [0.025, 0.975]).tolist()
    return dict(
        schema_version=1,
        split=split,
        receiver=asdict(receiver),
        interface=asdict(interface),
        paired_files=len(differences),
        mean_delivery_difference=float(differences.mean()) if len(differences) else None,
        bootstrap_95_interval=interval,
        comparison="sweep minus random",
        results=results,
    )


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-us", type=float, default=10_000_000)
    parser.add_argument("--dwell-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        result = benchmark(
            args.root,
            args.split,
            args.max_files,
            args.workers,
            ReplayConfig(seed=args.seed, stop_us=args.stop_us),
            dwell_index=args.dwell_index,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=args.output.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Evaluated {len(result['results'])} paired recordings; report: {args.output}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
