"""Expanded same-origin action head used by specialist RL and MOPD."""

import copy
import math

import torch
from torch import nn

from .experiments.storage import fingerprint, load_torch, save_torch
from .grouped_policy import ActionResidual
from .timing_belief import BeliefPolicyConfig, load_belief


class ExpandedActionResidual(ActionResidual):
    """Preserve the original policy at initialization; allow larger corrections."""

    def __init__(self, original):
        super().__init__(original.features, original.hidden)
        self.layers = copy.deepcopy(original.layers)
        self.prior_log_scale = nn.Parameter(torch.zeros(()))
        self.correction = nn.Linear(self.features, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, features, prior, legal):
        old_residual = 4 * (self.layers(features.float())[..., 0] / 4).tanh()
        scale = self.prior_log_scale.clamp(math.log(0.05), math.log(2.0)).exp()
        correction = self.correction(features.float())[..., 0]
        return (scale * prior + old_residual + correction).masked_fill(~legal, -1e9)


def save_mopd(path, actor, config, metadata):
    from dataclasses import asdict

    save_torch(
        path,
        {
            "version": 2,
            "architecture": "expanded-action-residual",
            "features": actor.features,
            "hidden": actor.hidden,
            "actor": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
            "policy": asdict(config),
            "metadata": metadata,
        },
    )


def load_mopd(path):
    if not torch.cuda.is_available():
        raise RuntimeError("MOPD policy neural inference requires CUDA")
    payload = load_torch(path)
    if payload.get("version") != 2 or payload.get("architecture") != "expanded-action-residual":
        raise ValueError("unsupported expanded MOPD policy checkpoint")
    forecaster = path.parent.parent / "forecaster.pt"
    if fingerprint(forecaster) != payload["metadata"]["semantic"]["initial_forecaster_sha256"]:
        raise ValueError("MOPD frozen forecaster changed")
    model, _ = load_belief(forecaster)
    actor = ExpandedActionResidual(ActionResidual(payload["features"], payload["hidden"]))
    actor.load_state_dict(payload["actor"], strict=True)
    actor.cuda().eval()
    config = dict(payload["policy"])
    config["dwells"] = tuple(config["dwells"])
    return model, actor, BeliefPolicyConfig(**config), payload["metadata"]
