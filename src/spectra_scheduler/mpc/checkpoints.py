"""Atomic model/training checkpoint storage and implementation provenance."""

from __future__ import annotations

import hashlib
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from .config import GRU_HIDDEN, MAX_BANDS, MODEL_VERSION, STEP_FEATURE_DIM
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


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_training(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def implementation_hashes():
    """Hash every MPC implementation module plus its entry points and world generator."""
    package = Path(__file__).parent
    root = package.parent
    paths = sorted(package.glob("*.py")) + [
        root / name for name in ("neural_mpc.py", "mpc_training.py", "rl_scenarios.py")
    ]
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths}
