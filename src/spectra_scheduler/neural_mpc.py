"""Neural Model Predictive Control (Neural-MPC) Scheduler.

MuZero-inspired cognitive radar scheduler for Electronic Support combining:

1. **Representation Network** (GRU) — encodes observation history into a
   compact latent state that captures RF environment dynamics.
2. **Dynamics Network** (MLP) — learned state-transition model that predicts
   the next latent state and immediate reward given a scheduling action.
3. **Prediction Network** (dual MLP heads) — outputs a policy prior over
   bands and a scalar state value from any latent state.
4. **Monte Carlo Tree Search** — uses the learned model to simulate
   action sequences and selects the action with the highest visit count.
5. **Closed-loop replanning** — only the first MCTS action is executed;
   after observing the real outcome the world model state is updated and
   the full search is re-run from scratch.

Training pipeline:
  - Collect demonstration episodes from baseline scheduling policies.
  - Pre-train the world model with policy imitation, value prediction,
    dynamics consistency and reward prediction losses.
  - Search-driven training is available in spectra_scheduler.mpc_training.

Usage::

    # Train
    .venv-rl/bin/python -m spectra_scheduler.neural_mpc train \\
        --episodes 200 --epochs 50 --output artifacts/neural-mpc.pt

    # Benchmark
    .venv-rl/bin/python -m spectra_scheduler.neural_mpc benchmark \\
        --model artifacts/neural-mpc.pt --runs 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectra_scheduler.models import Observation
from spectra_scheduler.rl import Context, RewardConfig

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

MAX_BANDS: int = 8
"""Upper bound on the number of frequency bands (zero-padded when fewer)."""

GRU_HIDDEN: int = 128
"""Latent state dimensionality for the GRU representation network."""

STEP_FEATURE_DIM: int = MAX_BANDS + 5 + 2 * MAX_BANDS
"""Per-step GRU input: band one-hot (8) + obs features (5) + context (16) = 29."""

DEFAULT_MCTS_SIMS: int = 50
"""Number of MCTS simulations per planning step."""

DEFAULT_GAMMA: float = 0.97
"""Discount factor for value targets and MCTS backpropagation."""

DEFAULT_COVERAGE_LIMIT: int = 20
"""Hard coverage constraint: force visit if a band is this many steps overdue."""

MODEL_VERSION: int = 2
"""Schema version for serialised model checkpoints."""


# ═══════════════════════════════════════════════════════════════════════
#  OBSERVATION ENCODER  (custom — no reuse of rl.Context)
# ═══════════════════════════════════════════════════════════════════════


class ObservationEncoder:
    """Tracks per-band RF observation statistics and encodes GRU inputs.

    Maintains exponentially-smoothed hit rates, visit/hit ages, signal
    fingerprints and recent outcome history for each frequency band.
    Produces a fixed-size feature vector at every time step suitable for
    the representation network's GRU.
    """

    def __init__(self, num_bands: int) -> None:
        assert 1 <= num_bands <= MAX_BANDS
        self.num_bands = num_bands

        # Per-band statistics --------------------------------------------------
        self.hit_ema = np.full(num_bands, 0.25)  # optimistic prior
        self.ema_decay = 0.92
        self.visit_count = np.zeros(num_bands, dtype=np.int64)
        self.hit_count = np.zeros(num_bands, dtype=np.int64)
        self.last_visit_step = np.full(num_bands, -1, dtype=np.int64)
        self.last_hit_step = np.full(num_bands, -1, dtype=np.int64)
        self.last_was_hit = np.zeros(num_bands, dtype=np.float64)

        # Signal fingerprints (power / pulse-width running means) --------------
        self.mean_power = np.full(num_bands, -70.0)
        self.mean_pw = np.full(num_bands, 1.0)
        self.sig_count = np.zeros(num_bands, dtype=np.int64)

        # Global tracking ------------------------------------------------------
        self.current_band: int = 0
        self.total_steps: int = 0
        self.total_hits: int = 0
        self.total_listens: int = 0

    # ------------------------------------------------------------------

    def update(self, obs: Observation) -> None:
        """Incorporate a new observation into the encoder state."""
        band = obs.band
        self.total_steps += 1

        if not obs.listening:
            return  # retuning step — no information gained

        self.total_listens += 1
        self.visit_count[band] += 1
        self.last_visit_step[band] = obs.time_step
        hit = float(obs.hit)

        if obs.hit:
            self.hit_count[band] += 1
            self.last_hit_step[band] = obs.time_step
            self.last_was_hit[band] = 1.0
            self.total_hits += 1

            # Update signal fingerprint
            if obs.measurements:
                m = obs.measurements[0]
                n = self.sig_count[band]
                self.mean_power[band] = (self.mean_power[band] * n + m.power_dbm) / (n + 1)
                self.mean_pw[band] = (self.mean_pw[band] * n + m.pulse_width_us) / (n + 1)
                self.sig_count[band] = n + 1
        else:
            self.last_was_hit[band] = 0.0

        # Exponential moving average hit rate
        self.hit_ema[band] = self.ema_decay * self.hit_ema[band] + (1.0 - self.ema_decay) * hit
        self.current_band = band

    # ------------------------------------------------------------------

    def encode_step(self, obs: Observation) -> np.ndarray:
        """Encode a single observation into a fixed-size GRU input vector.

        Layout (29-dim for MAX_BANDS=8)::

            [0..7]   band one-hot
            [8]      hit indicator
            [9]      listening indicator
            [10]     normalised detection count
            [11]     normalised power (shifted & scaled)
            [12]     normalised pulse width
            [13..20] per-band smoothed hit rates (padded to MAX_BANDS)
            [21..28] per-band normalised visit ages (padded to MAX_BANDS)

        The context fields (13-28) reflect encoder state *before* this
        observation is incorporated, which is correct for causal modelling.
        """
        features = np.zeros(STEP_FEATURE_DIM, dtype=np.float64)

        # Band one-hot
        features[obs.band] = 1.0

        # Observation features
        features[MAX_BANDS] = float(obs.hit)
        features[MAX_BANDS + 1] = float(obs.listening)
        features[MAX_BANDS + 2] = min(obs.detections, 5) / 5.0
        if obs.measurements:
            m = obs.measurements[0]
            features[MAX_BANDS + 3] = (m.power_dbm + 90.0) / 60.0  # → ~[0, 1]
            features[MAX_BANDS + 4] = min(m.pulse_width_us, 10.0) / 10.0
        # else: zeros (no measurement available)

        # Per-band context summary
        offset_ema = MAX_BANDS + 5
        offset_age = offset_ema + MAX_BANDS
        features[offset_ema : offset_ema + self.num_bands] = self.hit_ema
        for b in range(self.num_bands):
            if self.last_visit_step[b] >= 0:
                age = obs.time_step - self.last_visit_step[b]
            else:
                age = obs.time_step + 1  # never visited
            features[offset_age + b] = min(age, 30) / 30.0

        return features


# ═══════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION  (custom multi-objective design)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RewardFunction:
    """Multi-objective reward shaping for Neural-MPC training.

    Rewards interceptions, discovery of new signal sources, and
    sustained tracking; penalises lost observation time (retuning)
    and neglected frequency bands.
    """

    hit_reward: float = 1.0
    """Base reward for a successful interception."""

    miss_penalty: float = -0.05
    """Small penalty for listening on a band with no activity."""

    retune_penalty: float = -0.15
    """Cost of a retuning step (lost observation opportunity)."""

    discovery_bonus: float = 0.3
    """Extra reward for the first interception on a band."""

    tracking_bonus: float = 0.2
    """Reward for consecutive interceptions on the same band."""

    coverage_scale: float = -0.03
    """Per-band penalty weight when a band exceeds the coverage threshold."""

    coverage_threshold: int = 15
    """Steps before the progressive coverage penalty activates."""

    def compute(self, obs: Observation, encoder: ObservationEncoder) -> float:
        """Compute the scalar reward for a single observation step."""
        reward = 0.0

        if not obs.listening:
            return self.retune_penalty

        if obs.hit:
            reward += self.hit_reward
            # First-ever hit on this band
            if encoder.hit_count[obs.band] <= 1:
                reward += self.discovery_bonus
            # Consecutive hits (tracking continuity)
            if encoder.last_was_hit[obs.band] > 0.5:
                reward += self.tracking_bonus
        else:
            reward += self.miss_penalty

        # Progressive coverage penalty for neglected bands
        for b in range(encoder.num_bands):
            if encoder.last_visit_step[b] >= 0:
                age = obs.time_step - encoder.last_visit_step[b]
            else:
                age = obs.time_step + 1
            if age > self.coverage_threshold:
                excess = (age - self.coverage_threshold) / float(self.coverage_threshold)
                reward += self.coverage_scale * excess

        return reward


# ═══════════════════════════════════════════════════════════════════════
#  NEURAL NETWORKS  (PyTorch)
# ═══════════════════════════════════════════════════════════════════════


class RepresentationNetwork(nn.Module):
    """GRU encoder: observation feature sequence → latent state.

    Processes the per-step feature vector through a small MLP projection
    followed by a single-layer GRU.  The hidden state of the GRU is the
    latent representation used by the dynamics and prediction networks.
    """

    def __init__(self, step_dim: int = STEP_FEATURE_DIM, hidden_size: int = GRU_HIDDEN) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.input_proj = nn.Sequential(
            nn.Linear(step_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
        )
        self.gru = nn.GRU(48, hidden_size, batch_first=True)

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Zero-initialised GRU hidden state: shape (1, B, H)."""
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of encoded observations.

        Args:
            x: (B, T, step_dim) feature sequence.
            hidden: (1, B, H) initial hidden state.

        Returns:
            gru_output: (B, T, H) latent states after each step.
            final_hidden: (1, B, H).
        """
        projected = self.input_proj(x)  # (B, T, 48)
        return self.gru(projected, hidden)


class DynamicsNetwork(nn.Module):
    """Learned state transition: (latent, action) → (next_latent, reward).

    Given the current latent state and a one-hot action encoding, predicts
    the next latent state and the immediate scalar reward.  This is the
    "imagined simulator" used during MCTS rollouts.
    """

    def __init__(self, hidden_size: int = GRU_HIDDEN, max_bands: int = MAX_BANDS) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_proj = nn.Linear(max_bands, 16)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(256, hidden_size)
        self.reward_head = nn.Linear(256, 1)

    def forward(
        self, state: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward.

        Args:
            state: (B, H).
            action_onehot: (B, max_bands).

        Returns:
            next_state: (B, H).
            reward: (B,).
        """
        action_emb = self.action_proj(action_onehot)  # (B, 16)
        x = torch.cat([state, action_emb], dim=-1)  # (B, H+16)
        features = self.trunk(x)  # (B, 256)
        next_state = self.state_head(features)  # (B, H)
        reward = self.reward_head(features).squeeze(-1)  # (B,)
        return next_state, reward


class PredictionNetwork(nn.Module):
    """Dual-head prediction: latent state → (policy logits, value).

    The policy head outputs unnormalised log-probabilities over bands
    (used as MCTS prior).  The value head estimates the expected
    discounted return from the current state.
    """

    def __init__(self, hidden_size: int = GRU_HIDDEN, max_bands: int = MAX_BANDS) -> None:
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_bands),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: state (B, H).  Returns: policy_logits (B, max_bands), value (B,)."""
        return self.policy_head(state), self.value_head(state).squeeze(-1)


class NeuralMPCModel(nn.Module):
    """Combined MuZero-style model wrapping representation, dynamics and prediction."""

    def __init__(
        self,
        step_dim: int = STEP_FEATURE_DIM,
        hidden_size: int = GRU_HIDDEN,
        max_bands: int = MAX_BANDS,
    ) -> None:
        super().__init__()
        self.max_bands = max_bands
        self.representation = RepresentationNetwork(step_dim, hidden_size)
        self._dynamics = DynamicsNetwork(hidden_size, max_bands)
        self._prediction = PredictionNetwork(hidden_size, max_bands)

    # --- convenience wrappers that handle one-hot conversion ---------------

    def dynamics(
        self, state: torch.Tensor, action_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """state: (B, H), action_idx: (B,) long → next_state (B, H), reward (B,)."""
        onehot = F.one_hot(action_idx, self.max_bands).float().to(state.device)
        return self._dynamics(state, onehot)

    def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """state: (B, H) or (H,) → policy_logits (B, max_bands), value (B,)."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self._prediction(state)


# ═══════════════════════════════════════════════════════════════════════
#  MONTE CARLO TREE SEARCH
# ═══════════════════════════════════════════════════════════════════════


class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = ("prior", "visit_count", "value_sum", "reward", "latent_state", "children")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.reward: float = 0.0
        self.latent_state: torch.Tensor | None = None
        self.children: dict[int, MCTSNode] = {}

    @property
    def expanded(self) -> bool:
        return self.latent_state is not None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Monte Carlo Tree Search with PUCT selection and neural rollouts.

    At each planning step the search builds a tree of imagined futures
    using the learned dynamics model.  Actions are selected at internal
    nodes via the PUCT formula (balancing exploitation of Q-values with
    exploration proportional to the policy prior).  Leaf nodes are
    evaluated by the prediction network's value head.

    After all simulations the action at the root with the highest visit
    count is selected — this is the standard MuZero action-selection
    strategy.
    """

    def __init__(
        self,
        model: NeuralMPCModel,
        num_bands: int,
        *,
        num_simulations: int = DEFAULT_MCTS_SIMS,
        c_puct: float = 1.5,
        gamma: float = DEFAULT_GAMMA,
        max_depth: int = 5,
        device: torch.device | None = None,
    ) -> None:
        if num_simulations < 1 or max_depth < 1 or not 1 <= num_bands <= MAX_BANDS:
            raise ValueError("positive search budget/depth and valid band count required")
        self.model = model
        self.num_bands = num_bands
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.max_depth = max_depth
        self.device = device or torch.device("cpu")

    @torch.inference_mode()
    def search(self, root_state: torch.Tensor) -> tuple[int, np.ndarray]:
        """Run MCTS and return *(best_action, visit_distribution)*.

        Args:
            root_state: 1-D tensor of shape ``(hidden_size,)``.

        Returns:
            best_action: integer band index.
            visit_dist: array of shape ``(num_bands,)`` with normalised
                visit fractions (useful as a training target).
        """
        # Expand root -------------------------------------------------------
        root = MCTSNode(prior=1.0)
        root.latent_state = root_state

        policy_logits, root_value = self.model.predict(root_state)
        priors = F.softmax(policy_logits[0, : self.num_bands], dim=-1)
        for a in range(self.num_bands):
            root.children[a] = MCTSNode(prior=priors[a].item())
        root.visit_count = 1
        root.value_sum = root_value.item()

        # Simulations -------------------------------------------------------
        for _ in range(self.num_simulations):
            node = root
            search_path: list[MCTSNode] = [node]
            actions_taken: list[int] = []

            # SELECT — descend using PUCT until we hit an unexpanded leaf
            while node.expanded and node.children and len(actions_taken) < self.max_depth:
                action, child = self._select_child(node)
                actions_taken.append(action)
                search_path.append(child)
                node = child

            # EXPAND — use dynamics to materialise the leaf
            parent = search_path[-2] if len(search_path) >= 2 else root
            leaf = search_path[-1]
            action = actions_taken[-1] if actions_taken else 0

            if parent.latent_state is not None and not leaf.expanded:
                act_t = torch.tensor([action], dtype=torch.long, device=self.device)
                ns, rew = self.model.dynamics(parent.latent_state.unsqueeze(0), act_t)
                leaf.latent_state = ns.squeeze(0)
                leaf.reward = float(np.clip(rew.item(), -0.25, 1.0))

                child_logits, leaf_value = self.model.predict(leaf.latent_state)
                child_priors = F.softmax(child_logits[0, : self.num_bands], dim=-1)
                for a in range(self.num_bands):
                    if a not in leaf.children:
                        leaf.children[a] = MCTSNode(prior=child_priors[a].item())
                value = leaf_value.item()
            else:
                _, estimate = self.model.predict(leaf.latent_state)
                value = estimate.item()

            # BACKPROPAGATE
            self._backprop(search_path, value)

        # Action selection by visit count ------------------------------------
        visits = np.zeros(self.num_bands)
        for a, ch in root.children.items():
            visits[a] = ch.visit_count
        best = int(np.argmax(visits))
        total = visits.sum()
        dist = visits / total if total > 0 else np.ones(self.num_bands) / self.num_bands
        return best, dist

    # ------------------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        total = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total + 1)

        best_score = -math.inf
        best_action = 0
        best_child = next(iter(node.children.values()))

        for action, child in node.children.items():
            q = child.reward + self.gamma * child.value if child.visit_count > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backprop(self, path: list[MCTSNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = node.reward + self.gamma * value


# ═══════════════════════════════════════════════════════════════════════
#  SCHEDULER  (implements Scheduler protocol)
# ═══════════════════════════════════════════════════════════════════════


class NeuralMPCScheduler:
    """Neural-MPC scheduler using MCTS planning with a learned world model.

    Conforms to the project's ``Scheduler`` protocol::

        reset(num_bands: int) -> None
        choose_band(time_step: int) -> int
        observe(observation: Observation) -> None

    When a trained model is loaded the scheduler uses MCTS to select
    actions.  Without a model it falls back to a lightweight Bayesian
    heuristic so that the class is always usable (e.g. during initial
    data collection before training).
    """

    def __init__(
        self,
        model: NeuralMPCModel | None = None,
        model_path: str | Path | None = None,
        num_simulations: int = DEFAULT_MCTS_SIMS,
        coverage_limit: int = 0,
        gamma: float = DEFAULT_GAMMA,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._num_simulations = num_simulations
        self._coverage_limit = coverage_limit
        self._gamma = gamma

        # Load model
        self._model: NeuralMPCModel | None = model
        if model_path is not None and self._model is None:
            self._model = load_model(model_path, self._device)
        if self._model is not None:
            self._model.eval()

        # Episode state (initialised in reset)
        self._num_bands: int = 0
        self._encoder: ObservationEncoder | None = None
        self._mcts: MCTS | None = None
        self._hidden: torch.Tensor | None = None
        self._current_band: int = 0
        self._band_ages: np.ndarray = np.array([])
        self._retry_band: int | None = None

    # --- Scheduler protocol -----------------------------------------------

    def reset(self, num_bands: int) -> None:
        """Prepare for a new simulation episode."""
        self._num_bands = num_bands
        self._encoder = ObservationEncoder(num_bands)
        self._current_band = 0
        self._band_ages = np.zeros(num_bands, dtype=np.int64)
        self._retry_band = None

        if self._model is not None:
            h = self._model.representation.initial_state(1).to(self._device)
            self._hidden = h
            self._mcts = MCTS(
                self._model,
                num_bands,
                num_simulations=max(1, self._num_simulations),
                gamma=self._gamma,
                device=self._device,
            )

    def choose_band(self, time_step: int) -> int:
        """Select band to monitor at *time_step* using MCTS planning."""
        # No cold-start, retry or coverage override for learned policies.
        if self._model is not None:
            latent = self._hidden.squeeze(0).squeeze(0)
            if self._num_simulations == 0:
                with torch.inference_mode():
                    logits, _ = self._model.predict(latent)
                return int(logits[0, : self._num_bands].argmax())
            action, _ = self._mcts.search(latent)
            return action
        # Retry after retuning
        if self._retry_band is not None:
            band = self._retry_band
            self._retry_band = None
            return band

        # Cold start: visit each band once to seed the world model
        if time_step < self._num_bands:
            return time_step % self._num_bands

        # Hard coverage constraint
        worst = int(np.argmax(self._band_ages))
        if self._band_ages[worst] >= self._coverage_limit:
            return worst

        # MCTS planning
        if self._model is not None and self._hidden is not None and self._mcts is not None:
            latent = self._hidden.squeeze(0).squeeze(0)  # (H,)
            action, _dist = self._mcts.search(latent)
            return action

        # Fallback heuristic (no model loaded)
        return self._fallback(time_step)

    def observe(self, observation: Observation) -> None:
        """Update world model state with a new observation."""
        assert self._encoder is not None

        # Track band ages
        if observation.listening:
            self._band_ages += 1
            self._band_ages[observation.band] = 0
            self._current_band = observation.band
        else:
            self._band_ages += 1
            self._retry_band = observation.band

        # Encode step features BEFORE updating encoder (causal)
        step_features = self._encoder.encode_step(observation)

        # Update encoder state
        self._encoder.update(observation)

        # Update GRU hidden state
        if self._model is not None and self._hidden is not None:
            with torch.inference_mode():
                x = (
                    torch.tensor(
                        step_features,
                        dtype=torch.float32,
                        device=self._device,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # (1, 1, D)
                proj = self._model.representation.input_proj(x)
                _, self._hidden = self._model.representation.gru(proj, self._hidden)

    # --- Fallback heuristic -----------------------------------------------

    def _fallback(self, time_step: int) -> int:
        """Simple Bayesian-style fallback when no model is available."""
        assert self._encoder is not None
        scores = self._encoder.hit_ema[: self._num_bands].copy()
        ages = self._band_ages / max(self._coverage_limit, 1)
        scores += 0.15 * ages  # exploration bonus
        return int(np.argmax(scores))


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for world-model pre-training."""

    demo_episodes_per_policy: int = 200
    demo_policies: tuple[str, ...] = (
        "adaptive-dwell",
        "track-aware",
        "transition-band",
        "bayesian-band",
        "period-aware",
    )
    epochs: int = 50
    batch_size: int = 8  # episodes per gradient step
    lr: float = 3e-4
    gamma: float = DEFAULT_GAMMA
    unroll_steps: int = 5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    dynamics_loss_weight: float = 0.5
    reward_loss_weight: float = 0.5
    device: str = "cpu"
    seed: int = 0


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


def train_model(
    episodes: list[list[dict]],
    config: TrainConfig,
    progress: Any = None,
) -> NeuralMPCModel:
    """Pre-train the Neural-MPC model on demonstration episodes."""
    device = torch.device(config.device)
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


# ═══════════════════════════════════════════════════════════════════════
#  MODEL I/O
# ═══════════════════════════════════════════════════════════════════════


def save_model(
    model: NeuralMPCModel, path: str | Path, metadata: dict[str, Any] | None = None
) -> Path:
    """Save model weights and metadata to a ``.pt`` checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    # Compute fingerprint
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode())
        h.update(state[key].cpu().numpy().tobytes())
    fingerprint = h.hexdigest()[:16]

    checkpoint: dict[str, Any] = {
        "version": MODEL_VERSION,
        "state_dict": state,
        "fingerprint": fingerprint,
        "config": {
            "step_dim": STEP_FEATURE_DIM,
            "hidden_size": GRU_HIDDEN,
            "max_bands": MAX_BANDS,
        },
    }
    if metadata:
        checkpoint["metadata"] = metadata

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)

    # Write sidecar JSON
    sidecar = path.with_suffix(".json")
    info = {
        "version": MODEL_VERSION,
        "fingerprint": fingerprint,
        "config": checkpoint["config"],
    }
    if metadata:
        info["metadata"] = metadata
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    temporary.replace(sidecar)

    return path


def load_model(path: str | Path, device: torch.device | None = None) -> NeuralMPCModel:
    """Load a Neural-MPC model from a ``.pt`` checkpoint."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    version = checkpoint.get("version", 0)
    if version != MODEL_VERSION:
        raise ValueError(f"Model version {version} != expected {MODEL_VERSION}")

    cfg = checkpoint["config"]
    model = NeuralMPCModel(
        step_dim=cfg["step_dim"],
        hidden_size=cfg["hidden_size"],
        max_bands=cfg["max_bands"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
#  BENCHMARK UTILITIES
# ═══════════════════════════════════════════════════════════════════════


def _benchmark(
    model_path: str | Path,
    runs: int = 20,
    start_seed: int = 10000,
    device: str = "cpu",
    simulations: int = 50,
    suites: tuple = ("randomized",),
    output: str = "reports/generated/neural-mpc.json",
) -> None:
    """Benchmark the Neural-MPC scheduler against baselines."""
    from spectra_scheduler.learning_cli import write_json
    from spectra_scheduler.rl_benchmark import benchmark

    model = load_model(model_path, torch.device(device))
    report = benchmark(
        [],
        runs=runs,
        seed=start_seed,
        suites=suites,
        num_bands=MAX_BANDS,
        reward=RewardConfig(coverage=0.2),
        profile=True,
        extra_policies={
            "neural-mpc": lambda: NeuralMPCScheduler(
                model=model, num_simulations=simulations, device=device
            ),
            "policy-only": lambda: NeuralMPCScheduler(
                model=model, num_simulations=0, device=device
            ),
        },
        extra_metadata={
            "sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            "simulations": simulations,
            "depth": 5,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    write_json(Path(output), report)
    print(f"Paired benchmark saved to {output}")
    return

# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    """Command-line interface for training and benchmarking Neural-MPC."""
    parser = argparse.ArgumentParser(
        prog="spectra_scheduler.neural_mpc",
        description="Neural-MPC: MuZero-style cognitive radar scheduler",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    p_train = sub.add_parser("train", help="Pre-train the world model")
    p_train.add_argument(
        "--episodes", type=int, default=200, help="Demo episodes per baseline policy (default: 200)"
    )
    p_train.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--threads", type=int, default=2)
    p_train.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_train.add_argument("--output", required=True, help="Output path for .pt checkpoint")

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Benchmark against baselines")
    p_bench.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    p_bench.add_argument("--runs", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=10000)
    p_bench.add_argument("--simulations", type=int, default=50)
    p_bench.add_argument(
        "--suites", nargs="+", default=["randomized", "receiver-shift", "tracking"]
    )
    p_bench.add_argument("--output", default="reports/generated/neural-mpc.json")
    p_bench.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args(argv)
    torch.set_num_threads(getattr(args, "threads", 1))

    if args.command == "train":
        cfg = TrainConfig(
            demo_episodes_per_policy=args.episodes,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )

        print(
            f"[1/3] Collecting demonstrations "
            f"({len(cfg.demo_policies)} policies × {cfg.demo_episodes_per_policy} seeds)..."
        )

        def demo_progress(policy: str, seed: int, total: int) -> None:
            if seed % 50 == 0:
                print(f"  {policy}: seed {seed} ({total} episodes collected)")

        t0 = time.time()
        episodes = collect_demonstrations(cfg, progress=demo_progress)
        t1 = time.time()
        print(f"  Collected {len(episodes)} episodes in {t1 - t0:.1f}s")

        print(f"\n[2/3] Training world model ({cfg.epochs} epochs)...")

        def train_progress(epoch: int, loss: float, n: int) -> None:
            if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                print(f"  epoch {epoch:3d}/{cfg.epochs}  loss={loss:.4f}  episodes={n}")

        model = train_model(episodes, cfg, progress=train_progress)
        t2 = time.time()
        print(f"  Training complete in {t2 - t1:.1f}s")

        print(f"\n[3/3] Saving checkpoint to {args.output}")
        save_model(
            model,
            args.output,
            metadata={
                "demo_episodes": len(episodes),
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "policies": list(cfg.demo_policies),
                "seed": cfg.seed,
                "gamma": cfg.gamma,
                "training_split": "train",
                "num_bands": MAX_BANDS,
                "reward": {"hit": 1.0, "retuning": 0.05, "coverage": 0.2},
            },
        )
        print("  Done.")

    elif args.command == "benchmark":
        print(f"Benchmarking {args.model} ({args.runs} validation seeds)...\n")
        _benchmark(
            args.model,
            runs=args.runs,
            start_seed=args.seed,
            device=args.device,
            simulations=args.simulations,
            suites=tuple(args.suites),
            output=args.output,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
