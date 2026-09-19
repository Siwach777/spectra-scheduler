"""Adapter from learned model/search to the simulator Scheduler protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spectra_scheduler.models import Observation

from .checkpoints import load_model
from .config import DEFAULT_GAMMA, DEFAULT_MCTS_SIMS
from .model import NeuralMPCModel
from .observation import ObservationEncoder
from .search import MCTS


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
