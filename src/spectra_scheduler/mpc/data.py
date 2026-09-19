"""Demonstration collection, parallel search actors and bounded episode replay."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.rl import Context, RewardConfig
from spectra_scheduler.rl_scenarios import procedural_scenario
from spectra_scheduler.simulation import SimulationEpisode

from .config import MAX_BANDS, REWARD, Config, TrainConfig
from .model import NeuralMPCModel
from .observation import ObservationEncoder
from .search import search_batch


def collect_demonstrations(
    config: TrainConfig,
    progress: Any = None,
) -> list[list[dict]]:
    """Collect demonstration episodes from baseline scheduling policies.

    Runs each baseline policy on procedurally generated training
    scenarios and records the full observation / action / reward trace.
    """
    from spectra_scheduler.comparison import scheduler_factories
    from spectra_scheduler.rl_scenarios import procedural_scenario

    reward_fn = RewardConfig(coverage=0.2)
    all_episodes: list[list[dict]] = []

    for policy_name in config.demo_policies:
        for seed in range(config.demo_episodes_per_policy):
            sim = procedural_scenario(seed, "train", num_bands=MAX_BANDS)

            factories = scheduler_factories(seed)
            if policy_name not in factories:
                raise ValueError(f"Unknown demonstration policy: {policy_name}")

            policy = factories[policy_name]()
            episode = _run_and_record(sim, policy, reward_fn, config.gamma)
            all_episodes.append(episode)

            if progress is not None:
                progress(policy_name, seed, len(all_episodes))

    return all_episodes


def _run_and_record(
    sim: Any,
    policy: Any,
    reward_fn: RewardConfig,
    gamma: float,
) -> list[dict]:
    """Execute one episode and record transitions."""
    from spectra_scheduler.simulation import SimulationEpisode

    ep = SimulationEpisode(sim)
    policy.reset(sim.num_bands)
    encoder = ObservationEncoder(sim.num_bands)
    context = Context(sim.num_bands)

    transitions: list[dict] = []
    for t in range(sim.duration):
        band = policy.choose_band(t)
        obs = ep.step(band)

        # Encode BEFORE update (causal)
        step_feat = encoder.encode_step(obs)
        context.observe(obs)
        reward = reward_fn.compute(obs, context)

        transitions.append(
            {
                "band_idx": band,
                "step_features": step_feat,
                "action": band,
                "reward": reward,
                "num_bands": sim.num_bands,
            }
        )

        encoder.update(obs)
        policy.observe(obs)

    # Discounted returns
    G = 0.0
    for i in reversed(range(len(transitions))):
        G = transitions[i]["reward"] + gamma * G
        transitions[i]["return"] = G

    return transitions


def collect_worker(payload, progress=None):
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
            if progress is not None:
                progress((step + 1) * len(seeds), sims[0].duration * len(seeds))
    for record, env in zip(records, envs, strict=True):
        for key in ("features", "actions", "rewards", "policies", "values"):
            record[key] = torch.tensor(
                np.array(record[key]), dtype=torch.long if key == "actions" else torch.float32
            )
        record["metrics"] = asdict(calculate_metrics(env.result()))
    return records


def collect(
    model,
    cfg,
    seeds,
    split,
    pool=None,
    explore=False,
    shifted=False,
    policy_only=False,
    progress=None,
):
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    groups = [
        list(map(int, x)) for x in np.array_split(seeds, min(cfg.workers, len(seeds))) if len(x)
    ]
    jobs = [(weights, asdict(cfg), group, split, explore, shifted, policy_only) for group in groups]
    total = len(seeds) * (180 if shifted else 120)
    if progress:
        progress(0, total)
    if pool is None:
        batches, completed = [], 0
        for job in jobs:

            def report(done, _total, offset=completed):
                if progress:
                    progress(offset + done, total)

            batch = collect_worker(job, report if progress else None)
            batches.append(batch)
            completed += len(batch) * (180 if shifted else 120)
    else:
        futures = {pool.submit(collect_worker, job): i for i, job in enumerate(jobs)}
        pending, batches, completed = set(futures), [None] * len(jobs), 0
        while pending:
            finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                batch = future.result()
                batches[futures[future]] = batch
                completed += len(batch) * (180 if shifted else 120)
            if progress:
                progress(completed, total)
    # Keep submission order: completion timing must not change replay sampling.
    return [episode for batch in batches for episode in batch]


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
