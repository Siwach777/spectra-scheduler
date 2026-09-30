"""CUDA grouped full-trajectory policy optimization from a trained timing model.

RLOO supplies the advantage baseline; a clipped update borrows DAPO's asymmetric
trust region. A constant physical horizon replaces trajectory-length scaling.
This is a scheduling adaptation, not a reproduction of an LLM paper.
"""

from __future__ import annotations

import argparse
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

from ..grouped_policy import ActionResidual, GroupedTimingPolicy, leave_one_out
from ..metrics import calculate_metrics
from ..scenarios import REQUIREMENT_SCENARIOS, build_scenario
from ..timing_belief import BeliefPolicyConfig, load_belief
from .storage import fingerprint, run_lock, save_torch, write_json
from .timing_report import BatchedEpisode, reporting_world


def training_world(job):
    scenario, seed = job
    key = f"grouped-scan-v1:train:{scenario}:{seed}".encode()
    world_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return build_scenario(scenario, world_seed)


@torch.inference_mode()
def trajectories(model, actor, config, worlds, *, group=1, generator=None, record=False):
    states = [
        BatchedEpisode(world, GroupedTimingPolicy(model, actor, config))
        for world in worlds
        for _ in range(group)
    ]
    count = len(states)
    actions_count = worlds[0].num_bands * len(config.dwells)
    host = torch.empty((count, worlds[0].num_bands, 3, model.config.history), pin_memory=True)
    histories = host.numpy()
    device = torch.empty_like(host, device="cuda")
    capacity = sum(world.duration for world in worlds) * group
    storage = None
    if record:
        required = capacity * actions_count * actor.features * 2
        if required > 1024**3:
            raise MemoryError(f"trajectory features exceed the one GiB bound: {required}")
        storage = {
            "features": np.empty((capacity, actions_count, actor.features), np.float16),
            "prior": np.empty((capacity, actions_count), np.float32),
            "legal": np.empty((capacity, actions_count), bool),
            "action": np.empty(capacity, np.int64),
            "logp": np.empty(capacity, np.float32),
            "episode": np.empty(capacity, np.int64),
        }
    position = 0
    entropy_sum, sampled = 0.0, 0
    while True:
        active = [
            i
            for i, state in enumerate(states)
            if state.episode.time_step < state.simulation.duration
        ]
        if not active:
            break
        for row, i in enumerate(active):
            histories[row] = states[i].scheduler.history.encode()
        device[: len(active)].copy_(host[: len(active)], non_blocking=True)
        predicted = model(device[: len(active)]).cpu().numpy()
        values = [
            states[i].scheduler.action_features(states[i].episode.time_step, predicted[row])
            for row, i in enumerate(active)
        ]
        features = np.stack([v[0] for v in values])
        prior = np.stack([v[1] for v in values])
        legal = np.stack([v[2] for v in values])
        logits = actor(
            torch.from_numpy(features).to("cuda"),
            torch.from_numpy(prior).to("cuda"),
            torch.from_numpy(legal).to("cuda"),
        )
        distribution = logits.softmax(-1)
        action = (
            torch.multinomial(distribution, 1, generator=generator)[:, 0]
            if generator is not None
            else logits.argmax(-1)
        )
        log_probs = logits.log_softmax(-1)
        action_array = action.cpu().numpy()
        if record:
            end = position + len(active)
            storage["features"][position:end] = features
            storage["prior"][position:end] = prior
            storage["legal"][position:end] = legal
            storage["action"][position:end] = action_array
            storage["logp"][position:end] = log_probs.gather(1, action[:, None])[:, 0].cpu().numpy()
            storage["episode"][position:end] = active
        position += len(active)
        free = torch.from_numpy(legal.sum(-1) > 1).to("cuda")
        entropy_sum += float((-(distribution * log_probs).sum(-1) * free).sum())
        sampled += int((legal.sum(-1) > 1).sum())
        for row, i in enumerate(active):
            state = states[i]
            step = state.episode.time_step
            chosen = state.scheduler.accept_action(step, predicted[row], action_array[row])
            state.advance(chosen, state.scheduler.forecast(step, chosen))
    returns = np.empty(count, np.float32)
    evaluations = []
    for i, state in enumerate(states):
        metrics = calculate_metrics(state.episode.result())
        returns[i] = metrics.interception_ratio + 0.05 * metrics.emitter_discovery_ratio
        if not record:
            evaluations.append(state.report())
    if storage is not None:
        storage = {name: value[:position] for name, value in storage.items()}
    return (
        returns.reshape(-1, group),
        storage,
        evaluations,
        {
            "decisions": position,
            "physical_steps": sum(s.simulation.duration for s in states),
            "mean_free_action_entropy": entropy_sum / max(sampled, 1),
            "mean_return": float(returns.mean()),
            "effective_groups": int((returns.reshape(-1, group).std(-1) > 1e-7).sum()),
        },
    )


def optimize(actor, optimizer, storage, advantages, episodes, horizon, args, generator):
    """Accumulate each complete trajectory's decision sum with a constant divisor."""
    flat_advantage = advantages.reshape(-1)[storage["episode"]]
    count = len(flat_advantage)
    totals = np.zeros(4)
    normalizer = episodes * horizon
    for _ in range(args.update_epochs):
        permutation = generator.permutation(count)
        optimizer.zero_grad(set_to_none=True)
        accumulated = torch.zeros(4, device="cuda")
        for start in range(0, count, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            x = torch.from_numpy(storage["features"][indices].astype(np.float32)).to("cuda")
            prior = torch.from_numpy(storage["prior"][indices]).to("cuda")
            legal = torch.from_numpy(storage["legal"][indices]).to("cuda")
            action = torch.from_numpy(storage["action"][indices]).to("cuda")
            previous = torch.from_numpy(storage["logp"][indices]).to("cuda")
            advantage = torch.from_numpy(flat_advantage[indices]).to("cuda")
            logits = actor(x, prior, legal)
            logp = logits.log_softmax(-1)
            current = logp.gather(1, action[:, None])[:, 0]
            ratio = (current - previous).exp()
            objective = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.28) * advantage)
            reference_logp = prior.masked_fill(~legal, -1e9).log_softmax(-1)
            probability = logp.exp()
            kl = (probability * (logp - reference_logp)).sum(-1)
            entropy = -(probability * logp).sum(-1)
            loss = (
                -objective + args.kl_weight * kl - args.entropy_weight * entropy
            ).sum() / normalizer
            torch._assert_async(torch.isfinite(loss), "nonfinite grouped policy loss")
            loss.backward()
            accumulated += (
                torch.stack(
                    (
                        objective.sum().detach(),
                        kl.sum().detach(),
                        entropy.sum().detach(),
                        (ratio - 1).abs().sum().detach(),
                    )
                )
                / count
            )
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch._assert_async(torch.isfinite(norm), "nonfinite grouped policy gradient")
        optimizer.step()
        totals += accumulated.cpu().numpy()
    return dict(
        zip(
            ("surrogate", "kl_from_initial", "entropy", "mean_ratio_change"),
            (totals / args.update_epochs).tolist(),
            strict=True,
        )
    )


def selection(model, actor, config):
    jobs = [(scenario, seed) for scenario in REQUIREMENT_SCENARIOS for seed in range(2000, 2012)]
    worlds = [reporting_world(scenario, seed)[0] for scenario, seed in jobs]
    returns, _, evaluations, _ = trajectories(model, actor, config, worlds)
    by_scenario = {}
    for i, scenario in enumerate(REQUIREMENT_SCENARIOS):
        rows = evaluations[i * 12 : (i + 1) * 12]
        by_scenario[scenario] = {
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


def save_actor(path, actor, config, metadata):
    save_torch(
        path,
        {
            "version": 1,
            "features": actor.features,
            "hidden": actor.hidden,
            "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "policy": asdict(config),
            "metadata": metadata,
        },
    )


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--entropy-weight", type=float, default=0.0001)
    parser.add_argument("--selection-every", type=int, default=8)
    parser.add_argument("--dwells", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--probe", type=int, default=8)
    parser.add_argument("--report-seed", type=int, default=14000)
    parser.add_argument("--report-runs", type=int, default=100)
    args = parser.parse_args(arguments)
    if (
        min(args.iterations, args.worlds, args.workers, args.update_epochs, args.selection_every)
        < 1
    ):
        parser.error("positive training sizes required")
    if args.batch_size < 256 or args.group < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("batch >=256, group >=2 and unique seeds required")
    if (
        not args.dwells or min(args.dwells) < 1
        or len(set(args.dwells)) != len(args.dwells)
        or args.probe not in args.dwells or args.report_runs < 1
    ):
        parser.error("positive unique dwells, an available probe and positive report runs required")
    reporting_seeds = list(range(args.report_seed, args.report_seed + args.report_runs))
    if set(reporting_seeds) & set(range(2000, 2012)):
        parser.error("reporting seeds overlap checkpoint selection")
    torch.set_num_threads(2)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    os.environ.update(
        {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    )
    started = time.perf_counter()
    config = BeliefPolicyConfig(
        dwells=tuple(args.dwells), probe=args.probe, revisit=256, exploration=0.02
    )
    semantic = {
        "version": 1,
        "algorithm": "RLOO advantage with asymmetrically clipped policy updates",
        "initial_forecaster_sha256": fingerprint(args.checkpoint),
        "policy": asdict(config),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "return": "episode true-capture ratio + 0.05 * emitter discovery",
        "advantage": "leave-one-out return baseline within the same procedural world",
        "normalization": (
            "constant episodes * maximum physical horizon; "
            "no group std or trajectory length division"
        ),
        "truth_boundary": (
            "truth used only for training returns and evaluation; "
            "actor receives causal measurements"
        ),
        "selection_seeds": list(range(2000, 2012)),
        "reporting_seeds": reporting_seeds,
        "papers": [
            "https://arxiv.org/abs/2402.14740",
            "https://arxiv.org/abs/2503.20783",
            "https://arxiv.org/abs/2503.14476",
        ],
        "source_sha256": {
            str(p): fingerprint(p)
            for p in (Path(__file__), Path(__file__).parent.parent / "grouped_policy.py")
        },
    }
    with run_lock(args.run_dir):
        if (args.run_dir / "config.json").exists():
            raise ValueError("use a fresh training directory")
        shutil.copyfile(args.checkpoint, args.run_dir / "forecaster.pt")
        write_json(args.run_dir / "config.json", semantic)
        model, _ = load_belief(args.run_dir / "forecaster.pt")
        model.requires_grad_(False)
        with ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for seed in args.seeds:
                torch.manual_seed(seed)
                actor = ActionResidual(features=model.config.future + 20).cuda()
                optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, fused=True)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                numpy_generator = np.random.default_rng(seed)
                directory = args.run_dir / f"seed-{seed}"
                directory.mkdir()
                history = []
                initial = selection(model, actor.eval(), config)
                best_score, best_iteration = initial["score"], 0
                metadata = {
                    "semantic": semantic,
                    "seed": seed,
                    "iteration": 0,
                    "selection": initial,
                }
                save_actor(directory / "untrained.pt", actor, config, metadata)
                save_actor(directory / "best.pt", actor, config, metadata)
                print(json.dumps({"seed": seed, "iteration": 0, "selection": initial}), flush=True)
                for iteration in range(1, args.iterations + 1):
                    began = time.perf_counter()
                    jobs = [
                        (
                            REQUIREMENT_SCENARIOS[(iteration * args.worlds + i) % 3],
                            seed * 100000 + (iteration - 1) * args.worlds + i,
                        )
                        for i in range(args.worlds)
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
                    collection_seconds = time.perf_counter() - began
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
                    row = {
                        "iteration": iteration,
                        **collection,
                        "losses": losses,
                        "collection_seconds": collection_seconds,
                        "iteration_seconds": time.perf_counter() - began,
                    }
                    if iteration % args.selection_every == 0 or iteration == args.iterations:
                        result = selection(model, actor.eval(), config)
                        row["selection"] = result
                        metadata = {
                            "semantic": semantic,
                            "seed": seed,
                            "iteration": iteration,
                            "selection": result,
                        }
                        save_actor(directory / "latest.pt", actor, config, metadata)
                        if result["score"] > best_score:
                            best_score, best_iteration = result["score"], iteration
                            save_actor(directory / "best.pt", actor, config, metadata)
                        save_torch(
                            directory / "optimizer.pt",
                            {
                                "optimizer": optimizer.state_dict(),
                                "iteration": iteration,
                                "actor": actor.state_dict(),
                            },
                        )
                    history.append(row)
                    write_json(
                        directory / "progress.json",
                        {
                            "seed": seed,
                            "history": history,
                            "best_iteration": best_iteration,
                            "best_score": best_score,
                            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                            "peak_host_rss_bytes": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            * 1024,
                        },
                    )
                    print(json.dumps({"seed": seed, **row}), flush=True)
                frozen = {
                    "checkpoint_sha256": fingerprint(directory / "best.pt"),
                    "best_iteration": best_iteration,
                    "best_score": best_score,
                    "forecaster_sha256": fingerprint(args.run_dir / "forecaster.pt"),
                    "selection_seeds": semantic["selection_seeds"],
                    "reporting_seeds": semantic["reporting_seeds"],
                }
                write_json(directory / "frozen.json", frozen)
    print(
        json.dumps(
            {
                "training_seconds": time.perf_counter() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
