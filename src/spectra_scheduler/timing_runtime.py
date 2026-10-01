"""Bounded CUDA graph inference for frozen timing models with fixed band shapes."""

import torch
from torch import nn


class CapturedTimingPredictor(nn.Module):
    """Reuse GPU allocations and launches; episode histories remain independent."""

    @torch.inference_mode()
    def __init__(self, model, batch=1, bands=8):
        super().__init__()
        if min(batch, bands) < 1 or next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA capture requires positive shapes and a CUDA model")
        if model.training:
            raise ValueError("CUDA timing capture requires a frozen evaluation model")
        self.model, self.config = model, model.config
        self.capacity, self.bands = batch, bands
        self.input = torch.zeros((batch, bands, 3, model.config.history), device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                model(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model(self.input)

    @torch.inference_mode()
    def forward(self, history):
        if (history.device.type != "cuda" or not 0 < len(history) <= self.capacity
                or tuple(history.shape[1:]) != tuple(self.input.shape[1:])):
            raise ValueError("timing inference differs from the captured CUDA shape")
        self.input[:len(history)].copy_(history)
        self.graph.replay()
        return self.output[:len(history)]
