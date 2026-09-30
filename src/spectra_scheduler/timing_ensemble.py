"""Frozen convex ensembles of causal timing-count forecasters."""

from dataclasses import asdict

import torch
from torch import nn

from .experiments.storage import load_torch, save_torch
from .timing_belief import BeliefConfig, TimingBeliefNetwork, load_belief


class TimingCountEnsemble(nn.Module):
    def __init__(self, models, weights):
        super().__init__()
        weights = tuple(float(w) for w in weights)
        if (not models or len(models) != len(weights)
                or any(not torch.isfinite(torch.tensor(w)) or w <= 0 for w in weights)
                or abs(sum(weights) - 1) > 1e-6):
            raise ValueError("positive finite weights summing to one are required")
        if any(model.config != models[0].config for model in models):
            raise ValueError("timing ensemble input and forecast schemas differ")
        self.config, self.weights = models[0].config, weights
        self.models = nn.ModuleList(models)
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, history):
        prediction = self.models[0](history) * self.weights[0]
        for model, weight in zip(self.models[1:], self.weights[1:], strict=True):
            prediction.add_(model(history), alpha=weight)
        return prediction


def save_ensemble(path, model, metadata):
    save_torch(path, {"kind": "timing_count_ensemble", "version": 1,
                      "config": asdict(model.config), "weights": list(model.weights),
                      "models": [m.state_dict() for m in model.models], "metadata": metadata})


def load_predictor(path):
    payload = load_torch(path)
    if payload.get("kind") != "timing_count_ensemble":
        return load_belief(path)
    if payload["version"] != 1:
        raise ValueError("unsupported timing ensemble version")
    models = []
    for state in payload["models"]:
        model = TimingBeliefNetwork(BeliefConfig(**payload["config"]))
        model.load_state_dict(state)
        models.append(model.cuda().eval())
    return TimingCountEnsemble(models, payload["weights"]), payload["metadata"]
