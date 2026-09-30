"""Adapter from learned model/search to the simulator Scheduler protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spectra_scheduler.models import Observation

from .checkpoints import load_model
from .config import DEFAULT_GAMMA, DEFAULT_MCTS_SIMS, STEP_FEATURE_DIM, Config
from .coverage import coverage_probe_action
from .macro import MacroObservationEncoder
from .model import NeuralMPCModel
from .observation import ObservationEncoder
from .search import MCTS, search_batch


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
        depth: int = 5,
        normalize_search: bool = True,
        coverage_probe_limit: int = 0,
        search_method: str = "puct",
        gumbel_candidates: int = 8,
        gumbel_q_scale: float = 2.0,
    ) -> None:
        self._device = torch.device(device)
        self._num_simulations = num_simulations
        self._coverage_limit = coverage_limit
        self._gamma = gamma
        self._depth = depth
        self._normalize_search = normalize_search
        self._coverage_probe_limit = coverage_probe_limit
        self._search_method = search_method
        self._gumbel_candidates = gumbel_candidates
        self._gumbel_q_scale = gumbel_q_scale

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
        self._macro_encoder: MacroObservationEncoder | None = None
        self._macro_band = 0
        self._macro_left = 0
        self._macro_previous_band = -1
        self._retune_table = None
        self._episode_horizon: int | None = None
        self._rng = np.random.default_rng(0)

    def set_episode_horizon(self, steps: int) -> None:
        """Supply the public episode length for finite-horizon planning."""
        if type(steps) is not int or steps < 1:
            raise ValueError("episode horizon must be a positive integer")
        self._episode_horizon = steps

    def set_retune_table(self, table) -> None:
        values = np.asarray(table, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != values.shape[1] or (values < 0).any():
            raise ValueError("retune table must be a square nonnegative matrix")
        self._retune_table = values

    # --- Scheduler protocol -----------------------------------------------

    def reset(self, num_bands: int) -> None:
        """Prepare for a new simulation episode."""
        self._num_bands = num_bands
        self._encoder = ObservationEncoder(num_bands)
        self._current_band = 0
        self._band_ages = np.zeros(num_bands, dtype=np.int64)
        self._retry_band = None
        self._macro_left = 0
        self._macro_previous_band = -1
        if self._model is not None and self._model.physical_contract:
            if self._retune_table is None or self._retune_table.shape != (num_bands, num_bands):
                raise ValueError("physical MPC requires the receiver retune schedule")
        if self._model is not None and len(self._model.dwell_steps) > 1:
            if self._model.step_dim not in (STEP_FEATURE_DIM, STEP_FEATURE_DIM + 1):
                raise ValueError("unsupported macro observation width")
        self._macro_encoder = (
            MacroObservationEncoder(
                num_bands, include_elapsed=self._model.step_dim == STEP_FEATURE_DIM + 1
            )
            if self._model is not None and len(self._model.dwell_steps) > 1
            else None
        )

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
        if self._macro_encoder is not None:
            if self._macro_left:
                if not self._model.physical_contract:
                    self._macro_left -= 1
                return self._macro_band
            remaining = (
                max(1, self._episode_horizon - time_step)
                if self._episode_horizon is not None
                else max(self._model.dwell_steps) * 5
            )
            state = self._hidden.squeeze(0)
            forced = coverage_probe_action(
                self._macro_encoder.encoder.last_visit_step,
                time_step,
                self._macro_previous_band,
                self._retune_table,
                self._model.dwell_steps,
                remaining,
                self._coverage_probe_limit,
            )
            if forced is not None:
                action = forced
            elif self._num_simulations == 0:
                with torch.inference_mode():
                    logits, _ = self._model.predict(state)
                scores = logits[0].clone()
                for action in range(self._model.num_actions):
                    band, dwell = divmod(action, len(self._model.dwell_steps))
                    delay = (
                        int(self._retune_table[self._macro_previous_band, band])
                        if self._model.physical_contract
                        and self._macro_previous_band >= 0
                        and band < self._num_bands
                        else 0
                    )
                    if (
                        band >= self._num_bands
                        or self._model.dwell_steps[dwell] + delay > remaining
                    ):
                        scores[action] = -torch.inf
                action = int(scores.argmax().item())
            else:
                cfg = Config(
                    simulations=self._num_simulations,
                    gamma=self._gamma,
                    device=self._device.type,
                    dwell_steps=self._model.dwell_steps,
                    depth=self._depth,
                    normalize_search=self._normalize_search,
                    physical_contract=self._model.physical_contract,
                    search_method=self._search_method,
                    gumbel_candidates=self._gumbel_candidates,
                    gumbel_q_scale=self._gumbel_q_scale,
                )
                visits, _, chosen = search_batch(
                    self._model,
                    state,
                    [remaining],
                    cfg,
                    self._rng,
                    num_bands=self._num_bands,
                    current_bands=[self._macro_previous_band],
                    retune_tables=[self._retune_table],
                    return_actions=True,
                )
                action = int(chosen[0])
            band, dwell = divmod(action, len(self._model.dwell_steps))
            self._macro_band = band
            self._macro_previous_band = band
            self._macro_left = (
                self._model.dwell_steps[dwell]
                if self._model.physical_contract
                else self._model.dwell_steps[dwell] - 1
            )
            return band
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

        if self._macro_encoder is not None:
            self._macro_encoder.observe(observation)
            if self._model.physical_contract and observation.listening:
                self._macro_left -= 1
            if self._macro_left == 0:
                features = self._macro_encoder.finish()
                with torch.inference_mode():
                    x = torch.as_tensor(features, device=self._device).view(1, 1, -1)
                    _, self._hidden = self._model.representation(x, self._hidden)
            return

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
