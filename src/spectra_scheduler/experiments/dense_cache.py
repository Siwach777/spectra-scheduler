"""Bounded parallel collection of dense train-only counterfactual examples."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from ..policy_benchmark import _run_jobs, validate_plan
from ..pulse_replay import ReplayConfig
from ..replay_env import InterfaceConfig
from .dense_replay import dense_training_batches
from .storage import atomic_file, fingerprint, write_json

FIELDS = ("history", "captured", "elapsed_us")


def _collect_recording(job):
    root, record, template, receiver, interface, output, cache_key = job
    metadata = output.with_suffix(".json")
    if output.is_file() and metadata.is_file():
        previous = json.loads(metadata.read_text())
        if (
            previous.get("cache_key") == cache_key
            and previous.get("record_sha256") == record["sha256"]
            and previous.get("sha256") == fingerprint(output)
        ):
            return previous
    plan = {**template, "recordings": [record]}
    data = {key: [] for key in FIELDS}
    behaviors = {}
    for batch in dense_training_batches(root, plan, receiver, interface):
        count = len(batch["history"])
        behaviors[batch["behavior"]] = behaviors.get(batch["behavior"], 0) + count
        for key in FIELDS:
            data[key].append(batch[key].copy())
    merged = {key: np.concatenate(values) for key, values in data.items()}
    with atomic_file(output) as stream:
        np.savez(stream, **merged)
    shard = {
        "path": str(output),
        "sha256": fingerprint(output),
        "examples": len(merged["history"]),
        "behaviors": behaviors,
        "cache_key": cache_key,
        "record_sha256": record["sha256"],
    }
    write_json(metadata, shard)
    return shard


def cache_configuration(root, plan, receiver, interface):
    package = Path(__file__).resolve().parent.parent
    sources = (
        "experiments/dense_replay.py",
        "experiments/dense_cache.py",
        "pulse_replay.py",
        "replay_features.py",
        "replay_env.py",
        "replay_training.py",
        "replay_baselines.py",
        "dataset_io.py",
    )
    return {
        "version": 1,
        "root": str(Path(root).resolve()),
        "plan": plan,
        "receiver": asdict(receiver),
        "interface": asdict(interface),
        "history_steps": 16,
        "behaviors": ["UniformActionPolicy", "RateProbePolicy"],
        "source_sha256": {name: fingerprint(package / name) for name in sources},
    }


def build_dense_cache(root, plan, directory, *, workers=12, receiver=None, interface=None):
    """Commit file hashes only after all shards and merged arrays are flushed."""
    if plan.get("split") != "train":
        raise ValueError("dense cache requires training recordings")
    validate_plan(root, plan)
    receiver, interface = receiver or ReplayConfig(), interface or InterfaceConfig()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = cache_configuration(root, plan, receiver, interface)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration"] != json.loads(json.dumps(configuration)):
            raise ValueError("dense cache configuration or source code changed")
        for name, digest in manifest["arrays"].items():
            if fingerprint(directory / name) != digest:
                raise ValueError(f"dense cache array changed: {name}")
        return manifest
    template = {key: value for key, value in plan.items() if key != "recordings"}
    cache_key = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    jobs = [
        (
            root,
            record,
            template,
            receiver,
            interface,
            directory / f"recording-{index:04d}.npz",
            cache_key,
        )
        for index, record in enumerate(plan["recordings"])
    ]
    shards = _run_jobs(_collect_recording, jobs, workers)
    total = sum(shard["examples"] for shard in shards)
    if total < 1:
        raise ValueError("dense collection produced no examples")
    specs = {
        "history": ((total, 16, interface.bands * 9 + 2), np.float32),
        "captured": ((total, interface.bands * len(interface.dwell_us)), np.int32),
        "elapsed_us": ((total, interface.bands * len(interface.dwell_us)), np.float32),
    }
    arrays = {
        name: np.lib.format.open_memmap(directory / f"{name}.npy", "w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in specs.items()
    }
    cursor = 0
    for shard in shards:
        with np.load(shard["path"], allow_pickle=False) as chunk:
            stop = cursor + shard["examples"]
            for name in FIELDS:
                arrays[name][cursor:stop] = chunk[name]
            cursor = stop
    for array in arrays.values():
        array.flush()
    del arrays
    manifest = {
        "configuration": configuration,
        "examples": total,
        "shards": shards,
        "arrays": {f"{name}.npy": fingerprint(directory / f"{name}.npy") for name in FIELDS},
    }
    write_json(manifest_path, manifest)
    validate_plan(root, plan)
    return manifest


def open_dense_cache(directory, expected_configuration):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["configuration"] != json.loads(json.dumps(expected_configuration)):
        raise ValueError("dense cache configuration mismatch")
    for name, digest in manifest["arrays"].items():
        if fingerprint(directory / name) != digest:
            raise ValueError(f"dense cache array changed: {name}")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in FIELDS}
    total = manifest["examples"]
    if any(len(array) != total for array in arrays.values()):
        raise ValueError("dense cache array length mismatch")
    return manifest, arrays
