"""Atomic artifacts, content verification and single-writer run directories."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def atomic_file(path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_torch(path, value):
    import torch

    with atomic_file(path) as stream:
        torch.save(value, stream)


def load_torch(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


@contextmanager
def run_lock(directory):
    """POSIX advisory lock; process exit releases it, so stale files do not block."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another writer owns this run directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_artifacts(root, entry):
    root = Path(root).resolve()
    for relative, expected in entry["artifacts"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or fingerprint(path) != expected:
            raise ValueError(f"checkpoint integrity check failed: {relative}")


def checkpoint_path(root, entry, filename):
    relative = str(Path(entry["directory"]) / filename)
    if relative not in entry["artifacts"]:
        raise ValueError("checkpoint artifact not registered")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("checkpoint path outside run directory")
    return path
