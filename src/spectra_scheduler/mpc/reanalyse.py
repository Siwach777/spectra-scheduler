"""Refresh stored MPC search targets from causal replay histories."""

from __future__ import annotations

import numpy as np
import torch

from .config import MAX_BANDS
from .search import search_batch


@torch.inference_mode()
def refresh_replay_targets(
    target,
    replay,
    cfg,
    rng: np.random.Generator,
    episode_count: int,
    *,
    world_batch: int = 16,
    state_batch: int = 64,
) -> dict[str, float | int]:
    """Uniformly replan old states; preserve real actions and observations.

    The EMA target representation encodes only the observed prefix before each
    saved decision. Batched finite-horizon search then replaces stale policy
    visit distributions and root values. Actual actions, elapsed physical time
    and receiver rewards are never rewritten. N-step returns are invalidated so
    the learner recomputes them from current values on its next sample.
    """
    if type(episode_count) is not int or episode_count < 0:
        raise ValueError("episode_count must be a nonnegative integer")
    if world_batch < 1 or state_batch < 1:
        raise ValueError("reanalysis batch sizes must be positive")
    if episode_count == 0:
        return {
            "episodes": 0,
            "decisions": 0,
            "physical_steps": 0,
            "mean_policy_l1_change": 0.0,
            "mean_value_absolute_change": 0.0,
        }
    if not replay.episodes:
        raise ValueError("cannot reanalyse empty replay")
    device = next(target.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MPC reanalysis requires CUDA")
    if tuple(target.dwell_steps) != tuple(cfg.dwell_steps):
        raise ValueError("target and training action spaces differ")
    if target.num_actions != MAX_BANDS * len(cfg.dwell_steps):
        raise ValueError("unexpected MPC action count")
    target.eval()
    chosen = rng.choice(
        len(replay.episodes), size=min(episode_count, len(replay.episodes)), replace=False
    )
    totals = {
        "episodes": 0,
        "decisions": 0,
        "physical_steps": 0,
        "policy_l1_sum": 0.0,
        "value_absolute_sum": 0.0,
    }
    search_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
    for offset in range(0, len(chosen), world_batch):
        episodes = [replay.episodes[int(i)] for i in chosen[offset : offset + world_batch]]
        lengths = [len(episode["actions"]) for episode in episodes]
        if any(
            length < 1 or "elapsed" not in episode
            for length, episode in zip(lengths, episodes, strict=True)
        ):
            raise ValueError("reanalysis requires nonempty macro replay episodes")
        if any(
            episode["policies"].shape != (length, target.num_actions)
            for episode, length in zip(episodes, lengths, strict=True)
        ):
            raise ValueError("replay policy dimensions do not match target model")
        maximum = max(lengths)
        host = torch.zeros(
            (len(episodes), maximum, target.step_dim),
            dtype=torch.float32,
            pin_memory=True,
        )
        for row, episode in enumerate(episodes):
            host[row, : lengths[row]].copy_(episode["features"])
        features = host.to(device, non_blocking=True)
        initial = target.representation.initial_state(len(episodes))
        latent, _ = target.representation(features, initial)
        states = torch.cat([initial.transpose(0, 1), latent], dim=1)
        state_bank = torch.cat([states[row, :length] for row, length in enumerate(lengths)], dim=0)
        horizons = np.concatenate(
            [
                np.cumsum(episode["elapsed"].numpy()[::-1], dtype=np.int64)[::-1]
                for episode in episodes
            ]
        )
        if cfg.physical_contract:
            if any(
                "prior_bands" not in episode or "retune_table" not in episode
                for episode in episodes
            ):
                raise ValueError("physical replay lacks retune timing provenance")
            previous = np.concatenate([episode["prior_bands"].numpy() for episode in episodes])
            timing = [
                episode["retune_table"].numpy()
                for episode in episodes
                for _ in range(len(episode["actions"]))
            ]
        else:
            previous = timing = None
        count = len(horizons)
        refreshed_policies = np.empty((count, target.num_actions), dtype=np.float32)
        refreshed_values = np.empty(count, dtype=np.float32)
        for start in range(0, count, state_batch):
            stop = min(start + state_batch, count)
            policies, values = search_batch(
                target,
                state_bank[start:stop],
                horizons[start:stop],
                cfg,
                search_rng,
                explore=False,
                current_bands=None if previous is None else previous[start:stop],
                retune_tables=None if timing is None else timing[start:stop],
            )
            refreshed_policies[start:stop] = policies
            refreshed_values[start:stop] = values
        cursor = 0
        for episode, length in zip(episodes, lengths, strict=True):
            policy = torch.from_numpy(refreshed_policies[cursor : cursor + length])
            value = torch.from_numpy(refreshed_values[cursor : cursor + length])
            totals["policy_l1_sum"] += float((episode["policies"] - policy).abs().sum(dim=1).sum())
            totals["value_absolute_sum"] += float((episode["values"] - value).abs().sum())
            episode["policies"].copy_(policy)
            episode["values"].copy_(value)
            episode.pop("_returns", None)
            episode.pop("_return_config", None)
            totals["episodes"] += 1
            totals["decisions"] += length
            totals["physical_steps"] += int(episode["elapsed"].sum())
            cursor += length
    decisions = max(1, totals["decisions"])
    return {
        "episodes": totals["episodes"],
        "decisions": totals["decisions"],
        "physical_steps": totals["physical_steps"],
        "mean_policy_l1_change": totals["policy_l1_sum"] / decisions,
        "mean_value_absolute_change": totals["value_absolute_sum"] / decisions,
    }
