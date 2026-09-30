"""Batched CUDA planning with bounded, causal band-and-dwell trajectories."""

from dataclasses import asdict

import numpy as np
import torch

from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.rl import Context
from spectra_scheduler.rl_scenarios import physical_scenario, procedural_scenario
from spectra_scheduler.simulation import SimulationEpisode

from .config import MAX_BANDS, REWARD, STEP_FEATURE_DIM
from .coverage import coverage_probe_action
from .macro import MacroObservationEncoder
from .search import search_batch


@torch.inference_mode()
def collect_macros(
    model, cfg, seeds, split, explore=False, shifted=False, policy_only=False, progress=None
):
    """One GPU model serves all worlds; no replicated CUDA process contexts.

    A batch advances one completed macro per live world. The environments remain
    CPU objects; all representation, dynamics and policy inference is batched on
    the model device. No future simulator state enters the network.
    """
    device = next(model.parameters()).device
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, seeds[0], int(explore)]))
    sims = [
        physical_scenario(int(seed), split, shifted)
        if cfg.physical_contract
        else procedural_scenario(int(seed), split, shifted, MAX_BANDS)
        for seed in seeds
    ]
    retune_tables = [
        np.fromiter(
            (
                sim.receiver.retune_duration(a, b)
                for a in range(MAX_BANDS)
                for b in range(MAX_BANDS)
            ),
            dtype=np.int64,
            count=MAX_BANDS * MAX_BANDS,
        ).reshape(MAX_BANDS, MAX_BANDS)
        for sim in sims
    ]
    envs = [SimulationEpisode(sim) for sim in sims]
    encoders = [
        MacroObservationEncoder(MAX_BANDS, include_elapsed=cfg.physical_elapsed_feature)
        for _ in seeds
    ]
    if model.step_dim != STEP_FEATURE_DIM + int(cfg.physical_elapsed_feature):
        raise ValueError("physical macro feature contract differs from model")
    contexts = [Context(MAX_BANDS) for _ in seeds]
    count, capacity = len(seeds), max(sim.duration for sim in sims)
    arrays = {
        "features": np.empty((count, capacity, model.step_dim), np.float32),
        "actions": np.empty((count, capacity), np.int64),
        "elapsed": np.empty((count, capacity), np.int64),
        "rewards": np.empty((count, capacity), np.float32),
        "raw_rewards": np.empty((count, capacity), np.float32),
        "policies": np.empty((count, capacity, model.num_actions), np.float32),
        "values": np.empty((count, capacity), np.float32),
        "prior_bands": np.empty((count, capacity), np.int64),
    }
    lengths = np.zeros(count, np.int64)
    hidden = model.representation.initial_state(count)
    host = torch.empty((count, 1, model.step_dim), pin_memory=device.type == "cuda")
    features = host.numpy()[:, 0]
    staging = torch.empty_like(host, device=device)
    total = sum(sim.duration for sim in sims)
    iteration = seeds[0] // cfg.episodes
    anneal = min(1.0, iteration / cfg.exploration_decay_iterations)
    temperature = 1.0 + anneal * (cfg.final_temperature - 1.0)
    durations = np.tile(model.dwell_steps, MAX_BANDS)
    action_bands = np.repeat(np.arange(MAX_BANDS), len(model.dwell_steps))
    if progress:
        progress(0, total)
    while True:
        active = [i for i, env in enumerate(envs) if env.time_step < sims[i].duration]
        if not active:
            break
        indices = torch.as_tensor(active, device=device)
        remaining = np.array([sims[i].duration - envs[i].time_step for i in active])
        previous = np.array(
            [-1 if envs[i].previous_band is None else envs[i].previous_band for i in active],
            dtype=np.int64,
        )
        timing = [retune_tables[i] for i in active]
        state = hidden[0].index_select(0, indices)
        forced = [
            coverage_probe_action(
                encoders[i].encoder.last_visit_step,
                envs[i].time_step,
                int(previous[row]),
                timing[row],
                model.dwell_steps,
                int(remaining[row]),
                cfg.coverage_probe_limit,
            )
            for row, i in enumerate(active)
        ]
        planned_actions = np.full(len(active), -1, dtype=np.int64)
        if policy_only:
            logits, value = model.predict(state)
            needed = np.broadcast_to(durations, (len(active), len(durations))).copy()
            if cfg.physical_contract:
                for row, band in enumerate(previous):
                    if band >= 0:
                        needed[row] += timing[row][band, action_bands]
            mask = torch.as_tensor(needed <= remaining[:, None], device=device)
            logits = logits.masked_fill(~mask, -torch.inf)
            policies = logits.softmax(-1).cpu().numpy()
            values = value.cpu().numpy()
        else:
            policies = np.zeros((len(active), model.num_actions), dtype=np.float32)
            values = np.empty(len(active), dtype=np.float32)
            free = [row for row, action in enumerate(forced) if action is None]
            if free:
                free_indices = torch.as_tensor(free, device=device)
                free_policies, free_values, free_actions = search_batch(
                    model,
                    state.index_select(0, free_indices),
                    remaining[free],
                    cfg,
                    rng,
                    explore,
                    current_bands=previous[free],
                    retune_tables=[timing[row] for row in free],
                    return_actions=True,
                )
                policies[free] = free_policies
                values[free] = free_values
                planned_actions[free] = free_actions
            if len(free) < len(active):
                _, root_values = model.predict(state)
                root_values = root_values.cpu().numpy()
                for row, action in enumerate(forced):
                    if action is not None:
                        values[row] = root_values[row]
        for row, action in enumerate(forced):
            if action is not None:
                policies[row].fill(0)
                policies[row, action] = 1.0
                planned_actions[row] = action
        for row, i in enumerate(active):
            probabilities = policies[row]
            if planned_actions[row] >= 0 and cfg.search_method == "gumbel":
                action = int(planned_actions[row])
            elif explore:
                sampling = probabilities ** (1.0 / temperature)
                sampling /= sampling.sum()
                action = int(rng.choice(model.num_actions, p=sampling))
            else:
                action = int(probabilities.argmax())
            band, dwell = divmod(action, len(model.dwell_steps))
            requested_listening = model.dwell_steps[dwell]
            reward = 0.0
            raw_reward = 0.0
            duration = listens = 0
            while (listens if cfg.physical_contract else duration) < requested_listening:
                obs = envs[i].step(band)
                encoders[i].observe(obs)
                contexts[i].observe(obs)
                tick_reward = REWARD.compute(obs, contexts[i])
                reward += cfg.gamma**duration * tick_reward
                raw_reward += tick_reward
                duration += 1
                listens += int(obs.listening)
                if envs[i].time_step == sims[i].duration:
                    break
            feature = encoders[i].finish()
            features[row] = feature
            position = lengths[i]
            for key, value in (
                ("features", feature),
                ("actions", action),
                ("elapsed", duration),
                ("rewards", reward),
                ("raw_rewards", raw_reward),
                ("policies", probabilities),
                ("values", values[row]),
                ("prior_bands", previous[row]),
            ):
                arrays[key][i, position] = value
            lengths[i] += 1
        batch = len(active)
        staging[:batch].copy_(host[:batch], non_blocking=device.type == "cuda")
        _, next_hidden = model.representation(staging[:batch], hidden.index_select(1, indices))
        hidden.index_copy_(1, indices, next_hidden)
        # The next search transfers results to host before this pinned buffer is reused.
        if progress:
            progress(sum(env.time_step for env in envs), total)
    return [
        {
            **{
                key: torch.from_numpy(value[i, : lengths[i]].copy())
                for key, value in arrays.items()
            },
            "seed": int(seed),
            "split": split,
            "retune_table": torch.from_numpy(retune_tables[i].copy()),
            "metrics": asdict(calculate_metrics(envs[i].result())),
        }
        for i, seed in enumerate(seeds)
    ]
