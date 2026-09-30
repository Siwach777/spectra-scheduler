"""Same-origin specialist RL followed by full-action on-policy distillation.

Scenario labels dispatch frozen teachers during training only. The deployed
student is one ordinary GroupedTimingPolicy receiving causal observations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import resource
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..grouped_policy import leave_one_out, load_grouped
from ..mopd_policy import ExpandedActionResidual, save_mopd
from ..scenarios import REQUIREMENT_SCENARIOS, build_scenario
from .grouped_study import optimize, trajectories
from .storage import fingerprint, load_torch, run_lock, write_json
from .timing_report import reporting_world


def reverse_kl(student_logits, teacher_logits, legal):
    """Exact KL(student || teacher) on the shared legal action support."""
    if student_logits.shape != teacher_logits.shape or student_logits.shape != legal.shape:
        raise ValueError("student, teacher and legal action shapes differ")
    if legal.dtype != torch.bool or not legal.any(-1).all():
        raise ValueError("each state requires at least one legal action")
    student_logp = student_logits.masked_fill(~legal, -1e9).log_softmax(-1)
    teacher_logp = teacher_logits.detach().masked_fill(~legal, -1e9).log_softmax(-1)
    return (student_logp.exp() * (student_logp - teacher_logp)).masked_fill(~legal, 0).sum(-1)


@torch.no_grad()
def teacher_scores(teachers, features, prior, legal, dispatch):
    """Score identical student states; dispatch never becomes a student feature."""
    if dispatch.ndim != 1 or len(dispatch) != len(features):
        raise ValueError("one teacher dispatch per state required")
    if (dispatch < 0).any() or (dispatch >= len(teachers)).any():
        raise ValueError("unknown teacher dispatch")
    scores = torch.empty_like(prior)
    for index, teacher in enumerate(teachers):
        chosen = dispatch == index
        if chosen.any():
            scores[chosen] = teacher(features[chosen], prior[chosen], legal[chosen])
    return scores.detach()


def episode_weights(episode, legal, episodes):
    """Equal-world objective across free decisions, excluding mandatory probes."""
    free = legal.sum(-1) > 1
    counts = np.bincount(episode[free], minlength=episodes)
    weights = free / np.maximum(counts[episode], 1) / episodes
    return weights.astype(np.float32)


def training_world(job):
    scenario, seed, stage = job
    digest = hashlib.sha256(f"mopd-scan-v1:train:{stage}:{scenario}:{seed}".encode()).digest()
    return build_scenario(scenario, int.from_bytes(digest[:4], "little"))


def select(model, actor, config, scenario=None):
    domains = REQUIREMENT_SCENARIOS if scenario is None else (scenario,)
    jobs = [(domain, seed) for domain in domains for seed in range(2000, 2012)]
    worlds = [reporting_world(domain, seed)[0] for domain, seed in jobs]
    returns, _, evaluations, _ = trajectories(model, actor.eval(), config, worlds)
    by_scenario = {}
    for index, domain in enumerate(domains):
        rows = evaluations[index * 12 : (index + 1) * 12]
        by_scenario[domain] = {
            "capture": float(
                np.mean(
                    [r["interception_ratio"] for r in rows if r["interception_ratio"] is not None]
                )
            ),
            "discovery": float(np.mean([r["discovery"]["emitter_discovery_ratio"] for r in rows])),
        }
    return {
        "score": float(returns.mean()),
        "by_scenario": by_scenario,
        "seeds": list(range(2000, 2012)),
    }


def distill(student, teachers, optimizer, storage, dispatch, episodes, args, generator):
    weights = episode_weights(storage["episode"], storage["legal"], episodes)
    chosen = np.flatnonzero(weights > 0)
    if not len(chosen):
        return {"reverse_kl": 0.0, "free_states": 0}
    state_dispatch = dispatch[storage["episode"]]
    # Frozen teacher targets are reused within this iteration, but states are
    # recollected from the updated student before the next iteration.
    targets = np.empty_like(storage["prior"])
    for start in range(0, len(chosen), args.batch_size):
        take = chosen[start : start + args.batch_size]
        features = torch.from_numpy(storage["features"][take].astype(np.float32)).cuda()
        prior = torch.from_numpy(storage["prior"][take]).cuda()
        legal = torch.from_numpy(storage["legal"][take]).cuda()
        domain = torch.from_numpy(state_dispatch[take]).cuda()
        targets[take] = teacher_scores(teachers, features, prior, legal, domain).cpu().numpy()
    totals = torch.zeros(2, device="cuda")
    for _ in range(args.update_epochs):
        order = generator.permutation(chosen)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), args.batch_size):
            take = order[start : start + args.batch_size]
            features = torch.from_numpy(storage["features"][take].astype(np.float32)).cuda()
            prior = torch.from_numpy(storage["prior"][take]).cuda()
            legal = torch.from_numpy(storage["legal"][take]).cuda()
            target = torch.from_numpy(targets[take]).cuda()
            weight = torch.from_numpy(weights[take]).cuda()
            logits = student(features, prior, legal)
            kl = reverse_kl(logits, target, legal)
            loss = (kl * weight).sum()
            torch._assert_async(torch.isfinite(loss), "nonfinite MOPD reverse KL")
            loss.backward()
            totals += torch.stack((loss.detach(), weight.sum()))
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        torch._assert_async(torch.isfinite(norm), "nonfinite MOPD gradient")
        optimizer.step()
    values = totals.cpu().tolist()
    return {
        "reverse_kl": values[0] / max(values[1], 1e-12),
        "free_states": len(chosen),
        "teacher_state_counts": np.bincount(
            state_dispatch[chosen], minlength=len(teachers)
        ).tolist(),
    }


def _metadata(semantic, seed, iteration, selection, stage, scenario=None):
    return {
        "semantic": semantic,
        "seed": seed,
        "iteration": iteration,
        "selection": selection,
        "stage": stage,
        "scenario": scenario,
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--specialist-iterations", type=int, default=64)
    parser.add_argument("--student-iterations", type=int, default=32)
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--entropy-weight", type=float, default=0.0001)
    parser.add_argument("--selection-every", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--report-seed", type=int, default=20000)
    parser.add_argument("--report-runs", type=int, default=100)
    args = parser.parse_args(arguments)
    if (
        min(
            args.specialist_iterations,
            args.student_iterations,
            args.worlds,
            args.workers,
            args.update_epochs,
            args.selection_every,
            args.report_runs,
        )
        < 1
    ):
        parser.error("positive budgets required")
    if args.batch_size < 256 or args.group < 2 or args.worlds < 3:
        parser.error("batch >=256, group >=2 and at least three worlds required")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("unique nonnegative training seeds required")
    reporting_seeds = list(range(args.report_seed, args.report_seed + args.report_runs))
    if set(reporting_seeds) & set(range(2000, 2012)):
        parser.error("reporting seeds overlap selection")
    if not torch.cuda.is_available():
        raise RuntimeError("MOPD neural acting, training and validation require CUDA")
    torch.set_num_threads(2)
    os.environ.update(
        {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    )
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    model, old_head, config, initial_metadata = load_grouped(args.initial_actor)
    initial = ExpandedActionResidual(old_head).cuda().eval()
    del old_head
    if len(config.dwells) * 8 != 24:
        raise ValueError("MOPD experiment requires the native 24-action menu")
    model.requires_grad_(False).eval()
    initial.requires_grad_(False).eval()
    forecaster = args.initial_actor.parent.parent / "forecaster.pt"
    semantic = {
        "version": 1,
        "algorithm": (
            "same-origin domain-specialist grouped RL then "
            "full-action on-policy reverse-KL distillation"
        ),
        "initial_actor_sha256": fingerprint(args.initial_actor),
        "initial_forecaster_sha256": fingerprint(forecaster),
        "initial_selected_iteration": initial_metadata["iteration"],
        "action_architecture": (
            "old bounded residual + zero-initialized unbounded linear correction "
            "+ global prior scale clamped [.05,2]"
        ),
        "policy": asdict(config),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "return": "episode true capture + 0.05 * emitter discovery",
        "distillation": (
            "exact KL(student || dispatched frozen teacher) over all legal actions; "
            "equal-world mean across free decisions"
        ),
        "teacher_dispatch": (
            "training-only scenario membership; never included in actor features or runtime policy"
        ),
        "state_collection": (
            "fresh stochastic student rollout before each distillation iteration; "
            "no teacher rollouts in integration"
        ),
        "selection_seeds": list(range(2000, 2012)),
        "reporting_seeds": reporting_seeds,
        "papers": ["https://arxiv.org/abs/2606.30406", "https://arxiv.org/abs/2609.32577"],
        "source_sha256": {
            str(path): fingerprint(path)
            for path in (
                Path(__file__),
                Path(__file__).parent / "grouped_study.py",
                Path(__file__).parent.parent / "grouped_policy.py",
                Path(__file__).parent.parent / "scenarios.py",
            )
        },
    }
    with run_lock(args.run_dir):
        if (args.run_dir / "config.json").exists():
            raise ValueError("use a fresh MOPD training directory")
        shutil.copyfile(forecaster, args.run_dir / "forecaster.pt")
        shutil.copyfile(args.initial_actor, args.run_dir / "initial-actor.pt")
        write_json(args.run_dir / "config.json", semantic)
        initial_selection = select(model, initial, config)
        write_json(args.run_dir / "initial-selection.json", initial_selection)
        with ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for seed in args.seeds:
                torch.manual_seed(seed)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                numpy_generator = np.random.default_rng(seed)
                teachers, frozen_teachers = [], {}
                for scenario in REQUIREMENT_SCENARIOS:
                    actor = copy.deepcopy(initial).requires_grad_(True)
                    optimizer = torch.optim.AdamW(
                        actor.parameters(), lr=args.learning_rate, fused=True
                    )
                    directory = args.run_dir / f"teacher-{seed}-{scenario}"
                    directory.mkdir()
                    history = []
                    first = select(model, actor, config, scenario)
                    best_score, best_iteration = first["score"], 0
                    save_mopd(
                        directory / "best.pt",
                        actor,
                        config,
                        _metadata(semantic, seed, 0, first, "specialist", scenario),
                    )
                    for iteration in range(1, args.specialist_iterations + 1):
                        jobs = [
                            (
                                scenario,
                                seed * 1000000 + (iteration - 1) * args.worlds + index,
                                "specialist",
                            )
                            for index in range(args.worlds)
                        ]
                        worlds = list(pool.map(training_world, jobs))
                        actor.eval()
                        returns, storage, _, collection = trajectories(
                            model,
                            actor,
                            config,
                            worlds,
                            group=args.group,
                            generator=generator,
                            record=True,
                        )
                        actor.train()
                        losses = optimize(
                            actor,
                            optimizer,
                            storage,
                            leave_one_out(returns),
                            len(worlds) * args.group,
                            max(w.duration for w in worlds),
                            args,
                            numpy_generator,
                        )
                        del storage
                        row = {"iteration": iteration, **collection, "losses": losses}
                        if (
                            iteration % args.selection_every == 0
                            or iteration == args.specialist_iterations
                        ):
                            result = select(model, actor, config, scenario)
                            row["selection"] = result
                            if result["score"] > best_score:
                                best_score, best_iteration = result["score"], iteration
                                save_mopd(
                                    directory / "best.pt",
                                    actor,
                                    config,
                                    _metadata(
                                        semantic, seed, iteration, result, "specialist", scenario
                                    ),
                                )
                            print(
                                json.dumps(
                                    {
                                        "stage": "specialist",
                                        "seed": seed,
                                        "scenario": scenario,
                                        **row,
                                    }
                                ),
                                flush=True,
                            )
                        history.append(row)
                        write_json(
                            directory / "progress.json",
                            {
                                "history": history,
                                "best_iteration": best_iteration,
                                "best_score": best_score,
                            },
                        )
                    selected = load_torch(directory / "best.pt")
                    actor.load_state_dict(selected["actor"])
                    actor.requires_grad_(False).eval()
                    teachers.append(actor)
                    frozen_teachers[scenario] = {
                        "path": str(directory / "best.pt"),
                        "sha256": fingerprint(directory / "best.pt"),
                        "selected_iteration": best_iteration,
                        "initial_selection": first,
                        "selected_selection": selected["metadata"]["selection"],
                    }
                    write_json(directory / "frozen.json", frozen_teachers[scenario])
                    print(
                        json.dumps(
                            {
                                "stage": "teacher-frozen",
                                "seed": seed,
                                "scenario": scenario,
                                "initial_score": first["score"],
                                "selected_score": best_score,
                                "best_iteration": best_iteration,
                            }
                        ),
                        flush=True,
                    )
                student = copy.deepcopy(initial).requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    student.parameters(), lr=args.learning_rate, fused=True
                )
                directory = args.run_dir / f"seed-{seed}"
                directory.mkdir()
                best_score, best_iteration = initial_selection["score"], 0
                history = []
                save_mopd(
                    directory / "best.pt",
                    student,
                    config,
                    _metadata(semantic, seed, 0, initial_selection, "student"),
                )
                for iteration in range(1, args.student_iterations + 1):
                    dispatch = np.array(
                        [(iteration * args.worlds + index) % 3 for index in range(args.worlds)],
                        dtype=np.int64,
                    )
                    jobs = [
                        (
                            REQUIREMENT_SCENARIOS[int(domain)],
                            seed * 1000000 + (iteration - 1) * args.worlds + index,
                            "student",
                        )
                        for index, domain in enumerate(dispatch)
                    ]
                    worlds = list(pool.map(training_world, jobs))
                    student.eval()
                    _, storage, _, collection = trajectories(
                        model, student, config, worlds, generator=generator, record=True
                    )
                    student.train()
                    losses = distill(
                        student,
                        teachers,
                        optimizer,
                        storage,
                        dispatch,
                        len(worlds),
                        args,
                        numpy_generator,
                    )
                    del storage
                    row = {"iteration": iteration, **collection, "losses": losses}
                    if (
                        iteration % args.selection_every == 0
                        or iteration == args.student_iterations
                    ):
                        result = select(model, student, config)
                        row["selection"] = result
                        metadata = _metadata(semantic, seed, iteration, result, "student")
                        save_mopd(directory / "latest.pt", student, config, metadata)
                        if result["score"] > best_score:
                            best_score, best_iteration = result["score"], iteration
                            save_mopd(directory / "best.pt", student, config, metadata)
                        print(json.dumps({"stage": "student", "seed": seed, **row}), flush=True)
                    history.append(row)
                    write_json(
                        directory / "progress.json",
                        {
                            "history": history,
                            "best_iteration": best_iteration,
                            "best_score": best_score,
                        },
                    )
                frozen = {
                    "checkpoint_sha256": fingerprint(directory / "best.pt"),
                    "best_iteration": best_iteration,
                    "best_score": best_score,
                    "forecaster_sha256": fingerprint(args.run_dir / "forecaster.pt"),
                    "teachers": frozen_teachers,
                    "selection_seeds": semantic["selection_seeds"],
                    "reporting_seeds": reporting_seeds,
                }
                write_json(directory / "frozen.json", frozen)
                del teachers, actor, optimizer, student
        if (
            fingerprint(args.initial_actor) != semantic["initial_actor_sha256"]
            or fingerprint(forecaster) != semantic["initial_forecaster_sha256"]
        ):
            raise ValueError("initial artifacts changed during MOPD study")
        write_json(
            args.run_dir / "profile.json",
            {
                "training_seconds": time.perf_counter() - began,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
                "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
