"""Probability-averaged independent predictors, sharing the scheduling contract.

Motivation: https://arxiv.org/abs/1612.01474. This implements predictive averaging,
not that paper's adversarial training or a claim of calibrated uncertainty.
"""

from functools import partial

import torch
from torch import nn

from .forecast_model import PredictorPolicy, load_predictor
from .policy_benchmark import PolicySpec, fingerprint
from .replay_env import validate_specification


class ForecastEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        if not models or any(m.config != models[0].config for m in models):
            raise ValueError("ensemble requires models with identical architecture contracts")
        self.members = nn.ModuleList(models)
        self.config = models[0].config

    def forward(self, history):
        probabilities, ratios = [], []
        for model in self.members:
            timing, ratio = model(history)
            probabilities.append(timing.softmax(-1))
            ratios.append(ratio)
        probability = torch.stack(probabilities).mean(0)
        # The standard policy's softmax recovers the averaged categorical distribution.
        return probability.clamp_min(torch.finfo(probability.dtype).tiny).log(), torch.stack(
            ratios
        ).mean(0)


class EnsemblePolicy(PredictorPolicy):
    def __init__(self, paths, digests, **kwargs):
        if not paths or len(paths) != len(digests) or len(set(digests)) != len(digests):
            raise ValueError("ensemble requires distinct registered checkpoints")
        super().__init__(paths[0], expected_sha256=digests[0], **kwargs)
        models, hashes = [self.model], set(self.training_hashes)
        for path, digest in zip(paths[1:], digests[1:], strict=True):
            if fingerprint(path) != digest:
                raise ValueError("ensemble member changed")
            model, specification, trained_on = load_predictor(path, self.device)
            validate_specification(self.saved_spec, specification)
            if fingerprint(path) != digest:
                raise ValueError("ensemble member changed during load")
            models.append(model)
            hashes.update(trained_on)
        self.model = ForecastEnsemble(models).eval()
        self.training_hashes = tuple(sorted(hashes))


def ensemble_spec(paths, name="ensemble", **kwargs):
    paths = tuple(str(path) for path in paths)
    digests = tuple(fingerprint(path) for path in paths)
    hashes = set()
    for path, digest in zip(paths, digests, strict=True):
        _, _, trained_on = load_predictor(path)
        hashes.update(trained_on)
        if fingerprint(path) != digest:
            raise ValueError("ensemble member changed during registration")
    return PolicySpec(
        name,
        partial(EnsemblePolicy, paths, digests, **kwargs),
        f"probability-mean:{digests};policy:{sorted(kwargs.items())}",
        tuple(sorted(hashes)),
    )
