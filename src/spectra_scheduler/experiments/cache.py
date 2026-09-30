"""Content-addressed causal replay cache, parallel collection and bounded GPU batches."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..policy_benchmark import _run_jobs, validate_plan
from ..replay_training import BatchConfig, UniformActionPolicy, training_batches
from .storage import fingerprint, run_lock, write_json

FIELDS = ("history", "action", "time_class", "ratio", "ratio_valid")


def _collect(job):
    index, root, plan, receiver, interface, config, directory = job
    # One recording per worker; write batches immediately, not growing Python lists.
    paths = []
    for number, batch in enumerate(
        training_batches(root, plan, UniformActionPolicy, receiver, interface, config)
    ):
        path = Path(directory) / f"{index}-{number}.npz"
        np.savez(path, **{name: batch[name] for name in FIELDS})
        paths.append((str(path), len(batch["action"])))
    return paths


def cache_configuration(plan, receiver, interface, history_steps, time_bins):
    package = Path(__file__).resolve().parent.parent
    return {
        "version": 1,
        "plan": plan,
        "receiver": asdict(receiver),
        "interface": asdict(interface),
        "history_steps": history_steps,
        "time_bins": time_bins,
        "behavior": "uniform-all-actions",
        "implementation": {
            name: fingerprint(package / name)
            for name in (
                "replay_training.py",
                "replay_env.py",
                "replay_features.py",
                "pulse_replay.py",
                "dataset_io.py",
                "experiments/cache.py",
            )
        },
    }


def build_cache(
    directory, root, plan, receiver, interface, *, history_steps=16, time_bins=16, workers=6
):
    """Train split only. Atomic manifest commits an immutable, hashed collection."""
    if plan.get("split") != "train":
        raise ValueError("cache collection requires training recordings")
    validate_plan(root, plan)
    expected = cache_configuration(plan, receiver, interface, history_steps, time_bins)
    directory = Path(directory).resolve()
    with run_lock(directory):
        manifest = directory / "manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["configuration"] != json.loads(json.dumps(expected)):
                raise ValueError("cache configuration changed; use a fresh cache directory")
            CachedBatches(directory)  # Verify bytes before reuse.
            return saved
        if any(p.name != "run.lock" for p in directory.iterdir()):
            raise ValueError("cache requires an empty directory or a committed manifest")
        with tempfile.TemporaryDirectory(prefix="spectra-cache-") as temporary:
            jobs = (
                (
                    i,
                    root,
                    {**plan, "recordings": [record]},
                    receiver,
                    interface,
                    BatchConfig(1024, history_steps, time_bins),
                    temporary,
                )
                for i, record in enumerate(plan["recordings"])
            )
            shards = [shard for group in _run_jobs(_collect, jobs, workers) for shard in group]
            rows = sum(count for _, count in shards)
            if not rows:
                raise ValueError("empty training cache")
            with np.load(shards[0][0]) as first:
                arrays = {
                    name: np.lib.format.open_memmap(
                        directory / f"{name}.npy",
                        mode="w+",
                        dtype=first[name].dtype,
                        shape=(rows, *first[name].shape[1:]),
                    )
                    for name in FIELDS
                }
            offset = 0
            for path, count in shards:
                with np.load(path) as shard:
                    for name, array in arrays.items():
                        array[offset : offset + count] = shard[name]
                offset += count
            for array in arrays.values():
                array.flush()
            del arrays
        validate_plan(root, plan)
        saved = {
            "configuration": expected,
            "samples": rows,
            "files": {name: fingerprint(directory / f"{name}.npy") for name in FIELDS},
        }
        write_json(manifest, saved)
        return saved


class CachedBatches:
    """Verified arrays; reuse resident GPU data or two bounded pinned staging slots."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.arrays = {}
        for name in FIELDS:
            path = self.directory / f"{name}.npy"
            if fingerprint(path) != self.manifest["files"][name]:
                raise ValueError(f"cache integrity failure: {name}")
            self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        rows = self.manifest["samples"]
        cfg = self.manifest["configuration"]
        actions = cfg["interface"]["bands"] * len(cfg["interface"]["dwell_us"])
        features = cfg["interface"]["bands"] * 9 + 2
        if self.arrays["history"].shape != (rows, cfg["history_steps"], features):
            raise ValueError("cache history dimensions differ")
        for name in FIELDS[1:]:
            if self.arrays[name].shape != (rows,):
                raise ValueError("cache target dimensions differ")
        # Chunked validation once, not synchronized checks on every GPU batch.
        for start in range(0, rows, 4096):
            block = {k: v[start : start + 4096] for k, v in self.arrays.items()}
            if (
                not np.isfinite(block["history"]).all()
                or not np.isfinite(block["ratio"]).all()
                or ((block["action"] < 0) | (block["action"] >= actions)).any()
                or ((block["time_class"] < 0) | (block["time_class"] > cfg["time_bins"])).any()
                or ((block["ratio"] < 0) | (block["ratio"] > 1)).any()
            ):
                raise ValueError("invalid cached targets or features")
        self.resident = None

    def batches(self, batch_size, seed, device, max_batches=None):
        import torch

        rows = self.manifest["samples"]
        order = np.random.default_rng(seed).permutation(rows)
        size = sum(a.nbytes for a in self.arrays.values())
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(2 * 1024**3, free // 4)
        if self.resident is None and size <= budget:
            self.resident = {k: torch.tensor(v, device=device) for k, v in self.arrays.items()}
        if self.resident is not None:
            indices = torch.tensor(order, device=device)
            for batch, start in enumerate(range(0, rows, batch_size)):
                if max_batches and batch >= max_batches:
                    break
                selection = indices[start : start + batch_size]
                yield {
                    **{k: v.index_select(0, selection) for k, v in self.resident.items()},
                    "time_bins": self.manifest["configuration"]["time_bins"],
                }
            return
        slots = [
            {
                k: torch.empty(
                    (batch_size, *v.shape[1:]),
                    dtype=torch.from_numpy(np.empty((), v.dtype)).dtype,
                    pin_memory=True,
                )
                for k, v in self.arrays.items()
            }
            for _ in range(2)
        ]
        events = [torch.cuda.Event(), torch.cuda.Event()]
        used = [False, False]
        for batch, start in enumerate(range(0, rows, batch_size)):
            if max_batches and batch >= max_batches:
                break
            slot = batch % 2
            if used[slot]:
                events[slot].synchronize()  # DMA must finish before host memory is reused.
            selected = order[start : start + batch_size]
            values = {}
            for name, host in slots[slot].items():
                np.take(self.arrays[name], selected, axis=0, out=host[: len(selected)].numpy())
                values[name] = host[: len(selected)].to(device, non_blocking=True)
            events[slot].record()
            used[slot] = True
            yield {**values, "time_bins": self.manifest["configuration"]["time_bins"]}
