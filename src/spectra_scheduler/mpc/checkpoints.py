"""Atomic model/training checkpoint storage and implementation provenance."""

from __future__ import annotations

import hashlib
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from ..experiments.storage import save_torch, write_json
from .config import GRU_HIDDEN, MAX_BANDS, MODEL_VERSION
from .model import NeuralMPCModel


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
            "step_dim": model.step_dim,
            "hidden_size": GRU_HIDDEN,
            "max_bands": MAX_BANDS,
            "observation_head": model._observation is not None,
            "dwell_steps": model.dwell_steps,
            "physical_contract": model.physical_contract,
        },
    }
    if metadata:
        checkpoint["metadata"] = metadata

    save_torch(path, checkpoint)

    # Write sidecar JSON
    sidecar = path.with_suffix(".json")
    info = {
        "version": MODEL_VERSION,
        "fingerprint": fingerprint,
        "config": checkpoint["config"],
    }
    if metadata:
        info["metadata"] = metadata
    write_json(sidecar, info)

    return path


def load_model(path: str | Path, device: torch.device | None = None) -> NeuralMPCModel:
    """Load a Neural-MPC model from a ``.pt`` checkpoint."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    version = checkpoint.get("version", 0)
    if version not in (2, 3, MODEL_VERSION):
        raise ValueError(f"Model version {version} != expected {MODEL_VERSION}")

    cfg = checkpoint["config"]
    model = NeuralMPCModel(
        step_dim=cfg["step_dim"],
        hidden_size=cfg["hidden_size"],
        max_bands=cfg["max_bands"],
        observation_head=cfg.get("observation_head", False),
        dwell_steps=tuple(cfg.get("dwell_steps", (1,))),
        physical_contract=cfg.get("physical_contract", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def atomic_json(path, value):
    write_json(path, value)


def save_training(path, payload):
    save_torch(path, payload)


def implementation_hashes():
    """Hash every MPC implementation module plus its entry points and world generator."""
    package = Path(__file__).parent
    root = package.parent
    paths = sorted(package.glob("*.py")) + [
        root / name
        for name in (
            "neural_mpc.py",
            "mpc_training.py",
            "rl_scenarios.py",
            "simulation.py",
            "receiver.py",
            "action_contract.py",
        )
    ]
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths}
