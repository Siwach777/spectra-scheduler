"""Recurrent representation, latent dynamics and policy/value networks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GRU_HIDDEN, MAX_BANDS, STEP_FEATURE_DIM


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
        return self.gru.weight_ih_l0.new_zeros(1, batch_size, self.hidden_size)

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
        observation_head: bool = True,
    ) -> None:
        super().__init__()
        self.max_bands = max_bands
        self.representation = RepresentationNetwork(step_dim, hidden_size)
        self._dynamics = DynamicsNetwork(hidden_size, max_bands)
        self._prediction = PredictionNetwork(hidden_size, max_bands)
        self._observation = (
            nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 2))
            if observation_head
            else None
        )

    def predict_observation(self, state):
        """Predict hit/listening logits from a transitioned latent state."""
        return self._observation(state) if self._observation is not None else None

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
