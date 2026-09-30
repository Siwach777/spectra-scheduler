"""Demonstration optimization and search-driven multi-step training losses."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import GRU_HIDDEN, MAX_BANDS, STEP_FEATURE_DIM, TrainConfig
from .model import NeuralMPCModel


def train_model(
    episodes: list[list[dict]],
    config: TrainConfig,
    progress: Any = None,
) -> NeuralMPCModel:
    """Pre-train the Neural-MPC model on demonstration episodes."""
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Neural MPC training requires an available CUDA device")
    if not episodes or config.epochs < 1 or config.unroll_steps < 1:
        raise ValueError("nonempty demonstrations and positive epochs/unroll required")
    torch.manual_seed(config.seed)

    model = NeuralMPCModel(
        step_dim=STEP_FEATURE_DIM,
        hidden_size=GRU_HIDDEN,
        max_bands=MAX_BANDS,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):
        rng = np.random.default_rng(config.seed + epoch)
        order = rng.permutation(len(episodes))

        epoch_loss = 0.0
        n_trained = 0

        for idx in order:
            ep = episodes[idx]
            if len(ep) < config.unroll_steps + 2:
                continue

            loss = _train_on_episode(model, optimizer, ep, config, device)
            epoch_loss += loss
            n_trained += 1

        avg = epoch_loss / max(n_trained, 1)
        if progress is not None:
            progress(epoch, avg, n_trained)

    return model


def _train_on_episode(
    model: NeuralMPCModel,
    optimizer: torch.optim.Optimizer,
    episode: list[dict],
    config: TrainConfig,
    device: torch.device,
) -> float:
    """Train on a single episode.  Returns scalar loss."""
    T = len(episode)
    num_bands = episode[0]["num_bands"]
    K = config.unroll_steps

    # Build tensors
    feats = torch.tensor(
        np.array([t["step_features"] for t in episode]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # (1, T, D)

    actions = torch.tensor(
        [t["action"] for t in episode],
        dtype=torch.long,
        device=device,
    )
    rewards = torch.tensor(
        [t["reward"] for t in episode],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.tensor(
        [t["return"] for t in episode],
        dtype=torch.float32,
        device=device,
    )

    # --- Forward through GRU (full episode, with gradients) ----------------
    hidden = model.representation.initial_state(1).to(device)
    gru_out, _ = model.representation(feats, hidden)
    latent = gru_out.squeeze(0)  # (T, H)

    # States for prediction:  state_before_step[t] = state BEFORE obs_t
    # i.e.  h_0 = initial_state,  h_t = latent[t-1]
    initial_flat = hidden.squeeze(0).squeeze(0)  # (H,)
    states = torch.cat([initial_flat.unsqueeze(0), latent[:-1]], dim=0)  # (T, H)

    # --- Direct policy & value losses (trains GRU + prediction) ------------
    all_logits, all_values = model.predict(states)  # (T, B), (T,)
    policy_loss = F.cross_entropy(all_logits[:, :num_bands], actions)
    value_loss = F.mse_loss(all_values, returns)

    # --- Dynamics rollout losses (trains dynamics + prediction) -------------
    dyn_loss = torch.tensor(0.0, device=device)
    rew_loss = torch.tensor(0.0, device=device)
    n_unrolls = 0

    max_start = T - K - 1
    if max_start > 0:
        n_starts = min(8, max_start)
        starts = torch.randperm(max_start, device=device)[:n_starts]
        state = states[starts]
        for k in range(K):
            idx = starts + k
            ns, pr = model.dynamics(state, actions[idx])
            dyn_loss = dyn_loss + F.mse_loss(ns, states[idx + 1].detach())
            rew_loss = rew_loss + F.mse_loss(pr, rewards[idx])
            # Prediction must work on imagined states, not only GRU states.
            logits, values = model.predict(ns)
            dyn_loss = dyn_loss + F.cross_entropy(logits[:, :num_bands], actions[idx + 1])
            dyn_loss = dyn_loss + F.mse_loss(values, returns[idx + 1])
            n_unrolls += 1
            state = ns

    if n_unrolls > 0:
        dyn_loss = dyn_loss / n_unrolls
        rew_loss = rew_loss / n_unrolls

    # --- Total loss --------------------------------------------------------
    total = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.dynamics_loss_weight * dyn_loss
        + config.reward_loss_weight * rew_loss
    )

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return total.item()


def value_targets(rewards, values, gamma, steps):
    """N-step observed rewards + stored search value, zero bootstrap at terminal."""
    result = torch.zeros_like(rewards)
    for offset in range(min(steps, len(rewards))):
        result[: len(rewards) - offset] += gamma**offset * rewards[offset:]
    if steps < len(rewards):
        result[:-steps] += gamma**steps * values[steps:]
    return result


def macro_value_targets(rewards, values, elapsed, gamma, steps):
    """N-step physical-time returns with bootstraps at macro boundaries."""
    if not (len(rewards) == len(values) == len(elapsed)):
        raise ValueError("macro reward, value and duration lengths differ")
    if torch.any(elapsed < 1):
        raise ValueError("macro durations must be positive")
    length = len(rewards)
    result = torch.zeros_like(rewards)
    discount = torch.ones_like(rewards)
    for offset in range(min(steps, length)):
        end = length - offset
        result[:end] += discount[:end] * rewards[offset:]
        discount[:end] *= gamma ** elapsed[offset:].to(dtype=rewards.dtype)
    if steps < length:
        result[:-steps] += discount[:-steps] * values[steps:]
    return result


class BatchWorkspace:
    """Reusable host/device staging buffers; cached n-step targets per episode."""

    def __init__(self):
        self.signature = None
        self.host, self.device_buffers = {}, {}
        self.transfer = None
        self.macro_signature = None
        self.macro_capacity = 0
        self.macro_host, self.macro_device = {}, {}
        self.macro_transfer = None

    def prepare(self, episodes, cfg, device):
        if "elapsed" in episodes[0]:
            return self._prepare_macro(episodes, cfg, device)
        device = torch.device(device)
        signature = (len(episodes), len(episodes[0]["actions"]), str(device))
        if self.transfer is not None:
            self.transfer.synchronize()  # Host staging must outlive its asynchronous copy.
        if signature != self.signature:
            self.signature = signature
            for key in ("features", "actions", "rewards", "policies", "returns"):
                source = episodes[0]["rewards" if key == "returns" else key]
                self.host[key] = torch.empty(
                    (len(episodes), *source.shape),
                    dtype=source.dtype,
                    pin_memory=device.type == "cuda",
                )
                self.device_buffers[key] = (
                    torch.empty_like(self.host[key], device=device)
                    if device.type == "cuda"
                    else self.host[key]
                )
        for e in episodes:
            cache_key = (cfg.gamma, cfg.td_steps)
            if e.get("_return_config") != cache_key:
                e["_returns"] = value_targets(e["rewards"], e["values"], *cache_key)
                e["_return_config"] = cache_key
        for key, buffer in self.host.items():
            torch.stack([e["_returns" if key == "returns" else key] for e in episodes], out=buffer)
            if device.type == "cuda":
                self.device_buffers[key].copy_(buffer, non_blocking=True)
        if device.type == "cuda":
            if self.transfer is None:
                self.transfer = torch.cuda.Event()
            self.transfer.record()
        return self.device_buffers

    def _prepare_macro(self, episodes, cfg, device):
        device = torch.device(device)
        count = len(episodes)
        length = max(len(e["actions"]) for e in episodes)
        if length < 1:
            raise ValueError("macro batch contains an empty episode")
        action_dim = episodes[0]["policies"].shape[-1]
        signature = (count, action_dim, str(device))
        if self.macro_transfer is not None:
            self.macro_transfer.synchronize()
        if signature != self.macro_signature or length > self.macro_capacity:
            capacity = (
                length if signature != self.macro_signature
                else max(length, self.macro_capacity)
            )
            self.macro_signature = signature
            self.macro_capacity = capacity
            shapes = {
                "features": (count, capacity, episodes[0]["features"].shape[-1]),
                "actions": (count, capacity),
                "rewards": (count, capacity),
                "policies": (count, capacity, action_dim),
                "values": (count, capacity),
                "returns": (count, capacity),
                "elapsed": (count, capacity),
                "lengths": (count,),
            }
            dtypes = {
                "features": torch.float32, "actions": torch.long,
                "rewards": torch.float32, "policies": torch.float32,
                "values": torch.float32, "returns": torch.float32,
                "elapsed": torch.long, "lengths": torch.long,
            }
            self.macro_host = {
                key: torch.empty(shape, dtype=dtypes[key], pin_memory=device.type == "cuda")
                for key, shape in shapes.items()
            }
            self.macro_device = (
                {
                    key: torch.empty_like(value, device=device)
                    for key, value in self.macro_host.items()
                }
                if device.type == "cuda" else self.macro_host
            )
        for buffer in self.macro_host.values():
            buffer.zero_()
        for i, episode in enumerate(episodes):
            size = len(episode["actions"])
            self.macro_host["lengths"][i] = size
            cache_key = (cfg.gamma, cfg.td_steps)
            if episode.get("_return_config") != cache_key:
                episode["_returns"] = macro_value_targets(
                    episode["rewards"], episode["values"], episode["elapsed"], *cache_key
                )
                episode["_return_config"] = cache_key
            for key in ("features", "actions", "rewards", "policies", "values", "elapsed"):
                self.macro_host[key][i, :size].copy_(episode[key])
            self.macro_host["returns"][i, :size].copy_(episode["_returns"])
        if device.type == "cuda":
            for key in self.macro_host:
                source = self.macro_host[key]
                target = self.macro_device[key]
                if key != "lengths":
                    source, target = source[:, :length], target[:, :length]
                target.copy_(source, non_blocking=True)
            if self.macro_transfer is None:
                self.macro_transfer = torch.cuda.Event()
            self.macro_transfer.record()
        return {
            key: value if key == "lengths" else value[:, :length]
            for key, value in self.macro_device.items()
        }


def batch_loss(model, target, episodes, cfg, rng, device, workspace=None):
    """Batched full-history representation; masked multi-step latent rollouts."""
    if "elapsed" in episodes[0]:
        return _macro_batch_loss(model, target, episodes, cfg, rng, device, workspace)
    # Collection groups have equal length; replay currently uses 120-step train worlds.
    workspace = workspace or BatchWorkspace()
    buffers = workspace.prepare(episodes, cfg, device)
    features, actions, rewards, policies, returns = (
        buffers[key] for key in ("features", "actions", "rewards", "policies", "returns")
    )
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
    terms = {
        key: torch.zeros((), device=device)
        for key in ("policy", "value", "reward", "latent", "observation")
    }
    counts = {key: torch.zeros((), device=device) for key in terms}
    for k in range(cfg.unroll + 1):
        pos = starts + k
        valid = (pos < length).float()
        idx = pos.clamp(max=length - 1)
        logits, estimates = model.predict(state)
        terms["policy"] += (-(policies[rows, idx] * logits.log_softmax(-1)).sum(-1) * valid).sum()
        terms["value"] += (
            F.smooth_l1_loss(estimates, returns[rows, idx], reduction="none") * valid
        ).sum()
        counts["policy"] += valid.sum()
        counts["value"] += valid.sum()
        if k == cfg.unroll:
            break
        next_state, predicted_reward = model.dynamics(state, actions[rows, idx])
        terms["reward"] += (((predicted_reward - rewards[rows, idx]) ** 2) * valid).sum()
        target_state = target_states[rows, (pos + 1).clamp(max=length)]
        terms["latent"] += (((next_state - target_state) ** 2).mean(-1) * valid).sum()
        counts["reward"] += valid.sum()
        counts["latent"] += valid.sum()
        observation_logits = model.predict_observation(next_state)
        if observation_logits is not None:
            observed = features[rows, idx, MAX_BANDS : MAX_BANDS + 2]
            terms["observation"] += (
                F.binary_cross_entropy_with_logits(
                    observation_logits, observed, reduction="none"
                ).mean(-1)
                * valid
            ).sum()
            counts["observation"] += valid.sum()
        state = next_state
    terms = {key: value / counts[key].clamp_min(1) for key, value in terms.items()}
    loss = (
        terms["policy"]
        + terms["value"]
        + terms["reward"]
        + 0.5 * terms["latent"]
        + cfg.observation_loss_weight * terms["observation"]
    )
    values = torch.stack(list(terms.values())).detach().cpu().tolist()
    return loss, dict(zip(terms, values, strict=True))


def _macro_batch_loss(model, target, episodes, cfg, rng, device, workspace=None):
    """Multi-step loss over variable-length band-and-dwell decision histories."""
    workspace = workspace or BatchWorkspace()
    buffers = workspace.prepare(episodes, cfg, device)
    features, actions, rewards, policies, returns, lengths = (
        buffers[key] for key in
        ("features", "actions", "rewards", "policies", "returns", "lengths")
    )
    batch, padded_length, _ = features.shape
    hidden = model.representation.initial_state(batch).to(device)
    latent, _ = model.representation(features, hidden)
    states = torch.cat([hidden.transpose(0, 1), latent], dim=1)
    with torch.no_grad():
        target_latent, _ = target.representation(features, hidden)
        target_states = torch.cat([hidden.transpose(0, 1), target_latent], dim=1)
    starts = torch.as_tensor(
        [int(rng.integers(len(episode["actions"]))) for episode in episodes],
        dtype=torch.long, device=device,
    )
    rows = torch.arange(batch, device=device)
    state = states[rows, starts]
    names = ("policy", "value", "reward", "latent", "observation")
    terms = {key: torch.zeros((), device=device) for key in names}
    counts = {key: torch.zeros((), device=device) for key in names}
    for offset in range(cfg.unroll + 1):
        pos = starts + offset
        valid = (pos < lengths).to(features.dtype)
        idx = pos.clamp(max=padded_length - 1)
        logits, estimates = model.predict(state)
        terms["policy"] += (
            -(policies[rows, idx] * logits.log_softmax(-1)).sum(-1) * valid
        ).sum()
        terms["value"] += (
            F.smooth_l1_loss(estimates, returns[rows, idx], reduction="none") * valid
        ).sum()
        counts["policy"] += valid.sum()
        counts["value"] += valid.sum()
        if offset == cfg.unroll:
            break
        next_state, predicted_reward = model.dynamics(state, actions[rows, idx])
        terms["reward"] += (
            (predicted_reward - rewards[rows, idx]).square() * valid
        ).sum()
        target_state = target_states[rows, (pos + 1).clamp(max=padded_length)]
        terms["latent"] += (
            (next_state - target_state).square().mean(-1) * valid
        ).sum()
        counts["reward"] += valid.sum()
        counts["latent"] += valid.sum()
        observation_logits = model.predict_observation(next_state)
        if observation_logits is not None:
            observed = features[rows, idx, MAX_BANDS : MAX_BANDS + 2]
            terms["observation"] += (
                F.binary_cross_entropy_with_logits(
                    observation_logits, observed, reduction="none"
                ).mean(-1) * valid
            ).sum()
            counts["observation"] += valid.sum()
        state = next_state
    terms = {key: value / counts[key].clamp_min(1) for key, value in terms.items()}
    loss = (
        terms["policy"] + terms["value"] + terms["reward"]
        + 0.5 * terms["latent"]
        + cfg.observation_loss_weight * terms["observation"]
    )
    scalars = torch.stack(list(terms.values())).detach().cpu().tolist()
    return loss, dict(zip(terms, scalars, strict=True))
