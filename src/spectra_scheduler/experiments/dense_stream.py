"""Bounded, recording-shuffled CUDA batches from dense counterfactual caches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class DenseBlockStream:
    """Keep large replay caches on disk while overlapping reads and training.

    The loader shuffles contiguous blocks, then shuffles examples on CUDA within
    each block. At most two CPU blocks and one CUDA block are resident. This
    preserves full-epoch coverage without random page faults across a cache
    larger than GPU memory.
    """

    def __init__(self, caches: list[dict], block_size: int = 4096):
        if not caches or type(block_size) is not int or block_size < 1:
            raise ValueError("nonempty caches and positive block size required")
        names = ("history", "captured", "elapsed_us")
        shapes = None
        self.caches = []
        self.blocks = []
        for cache in caches:
            if any(name not in cache for name in names):
                raise ValueError("dense cache lacks a required array")
            count = len(cache["history"])
            if count < 1 or any(len(cache[name]) != count for name in names):
                raise ValueError("dense cache arrays have inconsistent lengths")
            current = tuple(cache[name].shape[1:] for name in names)
            if shapes is None:
                shapes = current
            elif current != shapes:
                raise ValueError("dense caches have incompatible shapes")
            index = len(self.caches)
            self.caches.append(cache)
            self.blocks.extend(
                (index, start, min(start + block_size, count))
                for start in range(0, count, block_size)
            )
        self.examples = sum(len(cache["history"]) for cache in caches)

    def __len__(self):
        return self.examples

    def batches(self, batch_size: int, seed: int, epoch: int):
        if batch_size < 1 or seed < 0 or epoch < 0:
            raise ValueError("invalid dense stream batch or seed")
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(
            len(self.blocks)
        )
        generator = torch.Generator(device="cuda").manual_seed(seed * 10_000 + epoch)

        def read_block(block_index):
            source, start, stop = self.blocks[int(block_index)]
            cache = self.caches[source]
            return {
                name: np.array(cache[name][start:stop], copy=True)
                for name in ("history", "captured", "elapsed_us")
            }

        with ThreadPoolExecutor(max_workers=1) as reader:
            pending = reader.submit(read_block, order[0])
            for position in range(len(order)):
                host = pending.result()
                if position + 1 < len(order):
                    pending = reader.submit(read_block, order[position + 1])
                block = {
                    name: torch.from_numpy(value).to("cuda")
                    for name, value in host.items()
                }
                permutation = torch.randperm(
                    len(block["history"]), device="cuda", generator=generator
                )
                for start in range(0, len(permutation), batch_size):
                    take = permutation[start : start + batch_size]
                    yield {name: value.index_select(0, take) for name, value in block.items()}
