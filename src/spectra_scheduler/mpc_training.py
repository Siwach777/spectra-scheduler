"""Search-driven recurrent world-model learning and resumable training.

CPU actors batch tree expansions across independent worlds. The learner batches
full recurrent histories on CPU or CUDA; policy labels come from search visits,
never from a teacher. All training signals use receiver observations only.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.neural_mpc import (
    MAX_BANDS,
    MCTS,
    MCTSNode,
    NeuralMPCModel,
    ObservationEncoder,
    load_model,
    save_model,
)
from spectra_scheduler.rl import Context, RewardConfig
from spectra_scheduler.rl_scenarios import GENERATOR_VERSION, procedural_scenario
from spectra_scheduler.simulation import SimulationEpisode

VERSION = 2
REWARD = RewardConfig(coverage=0.2)


@dataclass(frozen=True)
class Config:
    iterations: int = 200
    episodes: int = 24
    workers: int = 6
    batch_size: int = 32
    updates: int = 100
    replay_capacity: int = 1000
    simulations: int = 32
    depth: int = 5
    unroll: int = 5
    td_steps: int = 10
    gamma: float = 0.97
    lr: float = 3e-4
    ema: float = 0.99
    seed: int = 0
    validation_episodes: int = 8
    validation_every: int = 5
    threads: int = 2
    device: str = "cpu"

    def __post_init__(self):
        for name in (
            "iterations",
            "episodes",
            "workers",
            "batch_size",
            "updates",
            "replay_capacity",
            "simulations",
            "depth",
            "unroll",
            "td_steps",
            "validation_episodes",
            "validation_every",
            "threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.gamma < 1 or not 0 <= self.ema < 1 or not 0 < self.lr < 1:
            raise ValueError("invalid discount, EMA or learning rate")
        if self.seed < 0 or self.device not in ("cpu", "cuda"):
            raise ValueError("invalid seed or device")


@torch.inference_mode()
def search_batch(model, states, remaining, cfg, rng, explore=False):
    """One leaf per world per inference batch; finite-horizon discounted backups."""
    if len(states) != len(remaining) or not len(states) or min(remaining) < 1:
        raise ValueError("one positive remaining horizon per root is required")
    helper = MCTS(
        model, MAX_BANDS, num_simulations=cfg.simulations, gamma=cfg.gamma, max_depth=cfg.depth
    )
    logits, _ = model.predict(states)
    priors = logits.softmax(-1).cpu().numpy()
    roots = []
    for i, state in enumerate(states):
        root = MCTSNode(1.0)
        root.latent_state = state
        p = priors[i]
        if explore:
            p = 0.75 * p + 0.25 * rng.dirichlet(np.full(MAX_BANDS, 0.3))
        root.children = {a: MCTSNode(float(p[a])) for a in range(MAX_BANDS)}
        roots.append(root)
    for _ in range(cfg.simulations):
        paths, pending, actions, parent_states = [], [], [], []
        for i, root in enumerate(roots):
            node, path = root, [root]
            limit = min(cfg.depth, int(remaining[i]))
            while node.expanded and node.children and len(path) - 1 < limit:
                action, node = helper._select_child(node)
                path.append(node)
            paths.append(path)
            if not node.expanded:
                pending.append(i)
                actions.append(action)
                parent_states.append(path[-2].latent_state)
        if pending:
            ns, rewards = model.dynamics(
                torch.stack(parent_states), torch.tensor(actions, device=states.device)
            )
            child_logits, estimates = model.predict(ns)
            probabilities = child_logits.softmax(-1).cpu().numpy()
            for j, i in enumerate(pending):
                leaf = paths[i][-1]
                leaf.latent_state = ns[j]
                leaf.reward = float(rewards[j].clamp(-0.25, 1))
                leaf.children = {a: MCTSNode(float(probabilities[j, a])) for a in range(MAX_BANDS)}
                # Store the network estimate separately from backed-up values.
            estimates = {i: float(estimates[j]) for j, i in enumerate(pending)}
        else:
            estimates = {}
        for i, path in enumerate(paths):
            steps_left = int(remaining[i]) - len(path) + 1
            if steps_left <= 0:
                value = 0.0
            elif i in estimates:
                value = estimates[i]
            else:
                _, v = model.predict(path[-1].latent_state)
                value = float(v.item())
            bound = (1 - cfg.gamma ** max(0, steps_left)) / (1 - cfg.gamma)
            value = float(np.clip(value, -0.25 * bound, bound))
            helper._backprop(path, value)
    visits = np.array(
        [[c.visit_count for c in r.children.values()] for r in roots], dtype=np.float32
    )
    visits /= visits.sum(axis=1, keepdims=True)
    # Root value is the mean backed-up return, not a privileged simulator target.
    return visits, np.array([r.value for r in roots], dtype=np.float32)


def collect_worker(payload):
    """Spawn-safe actor: independent CPU model and deterministic per-job RNG."""
    weights, cfg_dict, seeds, split, explore, shifted, policy_only = payload
    cfg = Config(**cfg_dict)
    torch.set_num_threads(1)
    model = NeuralMPCModel()
    model.load_state_dict(weights)
    model.eval()
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, seeds[0], int(explore)]))
    sims = [procedural_scenario(s, split, shifted, MAX_BANDS) for s in seeds]
    envs = [SimulationEpisode(s) for s in sims]
    encoders = [ObservationEncoder(MAX_BANDS) for _ in seeds]
    contexts = [Context(MAX_BANDS) for _ in seeds]
    hidden = model.representation.initial_state(len(seeds))
    records = [
        {
            "features": [],
            "actions": [],
            "rewards": [],
            "policies": [],
            "values": [],
            "seed": seed,
            "split": split,
        }
        for seed in seeds
    ]
    with torch.inference_mode():
        for step in range(sims[0].duration):
            state = hidden[0]
            if policy_only:
                logits, v = model.predict(state)
                policies = logits.softmax(-1).numpy()
                values = v.numpy()
            else:
                policies, values = search_batch(
                    model, state, [s.duration - step for s in sims], cfg, rng, explore
                )
            features = []
            for i, env in enumerate(envs):
                p = policies[i]
                # High early-episode exploration, sharper late-episode actions.
                sampling = p if step < 30 else p**2 / (p**2).sum()
                action = int(rng.choice(MAX_BANDS, p=sampling)) if explore else int(p.argmax())
                obs = env.step(action)
                feat = encoders[i].encode_step(obs).astype(np.float32)
                encoders[i].update(obs)
                contexts[i].observe(obs)
                reward = REWARD.compute(obs, contexts[i])
                features.append(feat)
                row = records[i]
                for key, value in (
                    ("features", feat),
                    ("actions", action),
                    ("rewards", reward),
                    ("policies", p),
                    ("values", values[i]),
                ):
                    row[key].append(value)
            x = torch.from_numpy(np.stack(features)).unsqueeze(1)
            _, hidden = model.representation(x, hidden)
    for record, env in zip(records, envs, strict=True):
        for key in ("features", "actions", "rewards", "policies", "values"):
            record[key] = torch.tensor(
                np.array(record[key]), dtype=torch.long if key == "actions" else torch.float32
            )
        record["metrics"] = asdict(calculate_metrics(env.result()))
    return records


class Replay:
    """Bounded episode replay; completed episodes never cross reset boundaries."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.episodes = []

    def extend(self, episodes):
        if any(e["split"] != "train" for e in episodes):
            raise ValueError("validation/test episodes cannot enter replay")
        self.episodes = (self.episodes + episodes)[-self.capacity :]

    def sample(self, size, rng):
        if not self.episodes:
            raise ValueError("empty replay")
        return [self.episodes[i] for i in rng.integers(len(self.episodes), size=size)]


def value_targets(rewards, values, gamma, steps):
    """N-step observed rewards + stored search value, zero bootstrap at terminal."""
    result = torch.zeros_like(rewards)
    for offset in range(min(steps, len(rewards))):
        result[: len(rewards) - offset] += gamma**offset * rewards[offset:]
    if steps < len(rewards):
        result[:-steps] += gamma**steps * values[steps:]
    return result


def batch_loss(model, target, episodes, cfg, rng, device):
    """Batched full-history representation; masked multi-step latent rollouts."""
    # Collection groups have equal length; replay currently uses 120-step train worlds.
    features = torch.stack([e["features"] for e in episodes]).to(device)
    actions = torch.stack([e["actions"] for e in episodes]).to(device)
    rewards = torch.stack([e["rewards"] for e in episodes]).to(device)
    policies = torch.stack([e["policies"] for e in episodes]).to(device)
    returns = torch.stack(
        [value_targets(e["rewards"], e["values"], cfg.gamma, cfg.td_steps) for e in episodes]
    ).to(device)
    b, length, _ = features.shape
    h = model.representation.initial_state(b).to(device)
    latent, _ = model.representation(features, h)
    states = torch.cat([h.transpose(0, 1), latent], dim=1)
    with torch.no_grad():
        target_latent, _ = target.representation(features, h)
        target_states = torch.cat([h.transpose(0, 1), target_latent], dim=1)
    starts = torch.tensor(rng.integers(length, size=b), device=device)
    rows = torch.arange(b, device=device)
    state = states[rows, starts]
    terms = {key: torch.zeros((), device=device) for key in ("policy", "value", "reward", "latent")}
    counts = dict.fromkeys(terms, 0)
    for k in range(cfg.unroll + 1):
        pos = starts + k
        valid = pos < length
        idx = pos.clamp(max=length - 1)
        logits, estimates = model.predict(state)
        if valid.any():
            terms["policy"] += (-(policies[rows, idx] * logits.log_softmax(-1)).sum(-1))[
                valid
            ].sum()
            terms["value"] += F.smooth_l1_loss(
                estimates[valid], returns[rows, idx][valid], reduction="sum"
            )
            counts["policy"] += int(valid.sum())
            counts["value"] += int(valid.sum())
        if k == cfg.unroll:
            break
        next_state, predicted_reward = model.dynamics(state, actions[rows, idx])
        if valid.any():
            terms["reward"] += F.mse_loss(
                predicted_reward[valid], rewards[rows, idx][valid], reduction="sum"
            )
            target_state = target_states[rows, (pos + 1).clamp(max=length)]
            terms["latent"] += ((next_state - target_state) ** 2).mean(-1)[valid].sum()
            counts["reward"] += int(valid.sum())
            counts["latent"] += int(valid.sum())
        state = next_state
    terms = {key: value / max(1, counts[key]) for key, value in terms.items()}
    loss = terms["policy"] + terms["value"] + terms["reward"] + 0.5 * terms["latent"]
    return loss, {k: float(v.detach()) for k, v in terms.items()}


def collect(model, cfg, seeds, split, pool=None, explore=False, shifted=False, policy_only=False):
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    groups = [
        list(map(int, x)) for x in np.array_split(seeds, min(cfg.workers, len(seeds))) if len(x)
    ]
    jobs = [(weights, asdict(cfg), group, split, explore, shifted, policy_only) for group in groups]
    batches = map(collect_worker, jobs) if pool is None else pool.map(collect_worker, jobs)
    return [episode for batch in batches for episode in batch]


def validate(model, cfg, pool, split="validation"):
    results = {}
    for shifted in (False, True):
        for policy_only in (False, True):
            name = ("shift" if shifted else "randomized") + (
                "/policy" if policy_only else "/search"
            )
            episodes = collect(
                model,
                cfg,
                list(range(10000, 10000 + cfg.validation_episodes)),
                split,
                pool,
                shifted=shifted,
                policy_only=policy_only,
            )
            results[name] = {
                "reward": float(np.mean([float(e["rewards"].mean()) for e in episodes])),
                **{
                    key: float(np.mean([e["metrics"][key] for e in episodes]))
                    for key in ("interception_ratio", "emitter_discovery_ratio", "max_band_gap")
                },
            }
            with torch.no_grad():
                _, errors = batch_loss(
                    model, model, episodes, cfg, np.random.default_rng(0), cfg.device
                )
            results[name]["prediction_losses"] = errors
    return results


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def probe_reward_error(model, episodes, device):
    """Fixed held-out trajectories: comparable one-step reward MSE across iterations."""
    x = torch.stack([e["features"] for e in episodes]).to(device)
    hidden = model.representation.initial_state(len(episodes)).to(device)
    latent, _ = model.representation(x, hidden)
    before = torch.cat([hidden.transpose(0, 1), latent[:, :-1]], dim=1)
    actions = torch.stack([e["actions"] for e in episodes]).to(device)
    actual = torch.stack([e["rewards"] for e in episodes]).to(device)
    _, predicted = model.dynamics(before.flatten(0, 1), actions.flatten())
    return float(F.mse_loss(predicted, actual.flatten()))


def save_training(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(cfg, directory, resume=False, initial=None):
    """Synchronous actor/learner iterations, restartable at committed boundaries."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(cfg, directory, resume, initial)


def _run_locked(cfg, directory, resume, initial):
    checkpoint = directory / "training.pt"
    if resume and initial:
        raise ValueError("resume and initialization are mutually exclusive")
    if not resume and (checkpoint.exists() or (directory / "config.json").exists()):
        raise FileExistsError("existing run: use --resume or a new directory")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or expose the GPU")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = (load_model(initial) if initial else NeuralMPCModel()).to(cfg.device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    replay = Replay(cfg.replay_capacity)
    iteration, best, history = 0, -float("inf"), []
    probes = None
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["version"] != VERSION or saved["generator"] != GENERATOR_VERSION:
            raise ValueError("incompatible training checkpoint")
        old, new = dict(saved["config"]), asdict(cfg)
        for key in ("iterations", "device", "threads", "workers"):
            old.pop(key)
            new.pop(key)
        if old != new:
            raise ValueError("resume configuration differs from saved training semantics")
        model.load_state_dict(saved["model"])
        target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        replay.extend(saved["replay"])
        iteration, best, history = saved["iteration"], saved["best"], saved["history"]
        probes = saved["probes"]
        if cfg.iterations < iteration:
            raise ValueError("requested iterations precede saved checkpoint")
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if cfg.device == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    atomic_json(directory / "config.json", asdict(cfg))
    atomic_json(
        directory / "environment.json",
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": cfg.device,
            "generator_version": GENERATOR_VERSION,
            "source_sha256": {
                name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("mpc_training.py", "neural_mpc.py", "rl_scenarios.py")
            },
        },
    )
    pool = (
        ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn"))
        if cfg.workers > 1
        else None
    )
    try:
        if not history:
            probes = collect(
                model,
                cfg,
                list(range(20000, 20000 + cfg.validation_episodes)),
                "validation",
                pool,
                policy_only=True,
            )
            initial_validation = validate(model, cfg, pool)
            best = np.mean(
                [v["reward"] for k, v in initial_validation.items() if k.endswith("/search")]
            )
            history.append(
                {
                    "iteration": 0,
                    "validation": initial_validation,
                    "probe_reward_mse": probe_reward_error(model, probes, cfg.device),
                }
            )
            save_model(
                model,
                directory / "best.pt",
                {"iteration": 0, "validation": initial_validation, "config": asdict(cfg)},
            )
            print(json.dumps(history[-1]), flush=True)
        for index in range(iteration, cfg.iterations):
            start = perf_counter()
            model.eval()
            seeds = list(range(index * cfg.episodes, (index + 1) * cfg.episodes))
            episodes = collect(model, cfg, seeds, "train", pool, explore=True)
            replay.extend(episodes)
            collection_seconds = perf_counter() - start
            model.train()
            torch.set_num_threads(cfg.threads)
            losses = []
            for _ in range(cfg.updates):
                batch = replay.sample(cfg.batch_size, rng)
                loss, terms = batch_loss(model, target, batch, cfg, rng, cfg.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for slow, fast in zip(target.parameters(), model.parameters(), strict=True):
                        slow.lerp_(fast, 1 - cfg.ema)
                losses.append(terms)
            row = {
                "iteration": index + 1,
                "episodes_seen": (index + 1) * cfg.episodes,
                "replay_episodes": len(replay.episodes),
                "collection_seconds": collection_seconds,
                "loss": {key: float(np.mean([x[key] for x in losses])) for key in losses[0]},
                "collection_reward": float(np.mean([float(e["rewards"].mean()) for e in episodes])),
            }
            if (index + 1) % cfg.validation_every == 0 or index + 1 == cfg.iterations:
                model.eval()
                row["validation"] = validate(model, cfg, pool)
                score = np.mean(
                    [v["reward"] for k, v in row["validation"].items() if k.endswith("/search")]
                )
                if score > best:
                    best = float(score)
                    save_model(
                        model,
                        directory / "best.pt",
                        {
                            "iteration": index + 1,
                            "validation": row["validation"],
                            "config": asdict(cfg),
                        },
                    )
            row["iteration_seconds"] = perf_counter() - start
            row["probe_reward_mse"] = probe_reward_error(model, probes, cfg.device)
            history.append(row)
            save_model(
                model, directory / "latest.pt", {"iteration": index + 1, "config": asdict(cfg)}
            )
            save_training(
                checkpoint,
                {
                    "version": VERSION,
                    "generator": GENERATOR_VERSION,
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "target": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "replay": replay.episodes,
                    "probes": probes,
                    "iteration": index + 1,
                    "best": float(best),
                    "history": history,
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else [],
                },
            )
            atomic_json(
                directory / "progress.json",
                {"history": history, "best_validation_reward": float(best)},
            )
            print(json.dumps(row), flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    return history


def evaluate_run(directory, split="test", episodes=30, workers=1):
    """Evaluate best checkpoint with its saved search settings, without learning."""
    directory = Path(directory)
    checkpoint = directory / "best.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    settings = dict(metadata["config"])
    settings.update(device="cpu", workers=workers, validation_episodes=episodes)
    cfg = Config(**settings)
    model = load_model(checkpoint)
    torch.set_num_threads(1)
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) if workers > 1 else None
    try:
        metrics = validate(model, cfg, pool, split)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    report = {
        "split": split,
        "seeds": list(range(10000, 10000 + episodes)),
        "config": asdict(cfg),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_iteration": metadata["iteration"],
        "metrics": metrics,
    }
    atomic_json(directory / f"evaluation-{split}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial", help="Optional demonstration-pretrained .pt model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best.pt; no training")
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    for name, value in asdict(Config()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), default=argparse.SUPPRESS, type=type(value)
        )
    args = vars(parser.parse_args())
    directory, resume, initial = args.pop("run_dir"), args.pop("resume"), args.pop("initial")
    evaluate, split = args.pop("evaluate"), args.pop("evaluation_split")
    if evaluate:
        if resume or initial:
            parser.error("evaluation cannot resume or initialize training")
        print(
            json.dumps(
                evaluate_run(
                    directory, split, args.get("validation_episodes", 30), args.get("workers", 1)
                )
            )
        )
        return
    if resume:
        saved = torch.load(Path(directory) / "training.pt", map_location="cpu", weights_only=True)
        args = {**saved["config"], **args}
    run(Config(**args), directory, resume, initial)


if __name__ == "__main__":
    main()
