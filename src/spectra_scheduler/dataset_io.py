"""Read completed TSRD pulse files without materialising the full dataset."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

FEATURES = ("toa_us", "frequency_mhz", "pulse_width_us", "aoa_deg", "amplitude_db")
_ALIASES = {
    "toa": "toa_us",
    "timeofarrival": "toa_us",
    "frequency": "frequency_mhz",
    "centrefrequency": "frequency_mhz",
    "centerfrequency": "frequency_mhz",
    "cf": "frequency_mhz",
    "pulsewidth": "pulse_width_us",
    "pw": "pulse_width_us",
    "aoa": "aoa_deg",
    "angleofarrival": "aoa_deg",
    "amplitude": "amplitude_db",
}


@dataclass(frozen=True)
class PulseFileInfo:
    rows: int
    source_features: tuple[str, ...]
    column_order: tuple[int, ...]
    dtype: str
    has_labels: bool


@dataclass(frozen=True)
class PulseBatch:
    start: int
    features: np.ndarray
    labels: np.ndarray | None


def discover_files(root: Path, mode: str = "scan", split: str = "train") -> list[Path]:
    """Never traverse archives, caches or the other train/validation/test splits."""
    if mode not in ("scan", "stare") or split not in ("train", "val", "test"):
        raise ValueError("mode must be scan/stare and split must be train/val/test")
    directory = Path(root) / mode / f"{split}_{mode}"
    if not directory.is_dir():
        raise ValueError(f"dataset split directory not found: {directory}")
    # Hugging Face downloads to .incomplete paths before final placement.
    return sorted(p for p in directory.glob("*.h5") if p.is_file() and not p.is_symlink())


def _header(handle: h5py.File) -> PulseFileInfo:
    if "data" not in handle or not isinstance(handle["data"], h5py.Dataset):
        raise ValueError("missing data dataset")
    data = handle["data"]
    if data.ndim != 2 or data.shape[1] != len(FEATURES):
        raise ValueError("data must have shape (N, 5)")
    if data.dtype.kind not in "fi":
        raise ValueError("data must have a real numeric dtype")
    if "metadata/feature_names" not in handle:
        raise ValueError("missing metadata/feature_names; column order cannot be inferred")
    names_dataset = handle["metadata/feature_names"]
    if not isinstance(names_dataset, h5py.Dataset) or names_dataset.shape != (5,):
        raise ValueError("feature_names must contain exactly five names")
    try:
        names = tuple(str(n) for n in names_dataset.asstr()[:])
    except (TypeError, UnicodeError) as error:
        raise ValueError("feature_names must contain UTF-8 strings") from error
    normalised = ["".join(c for c in name.lower() if c.isalnum()) for name in names]
    try:
        canonical = [_ALIASES[name] for name in normalised]
    except KeyError as error:
        raise ValueError(f"unsupported feature name: {error.args[0]}") from error
    if set(canonical) != set(FEATURES):
        raise ValueError("duplicate or missing pulse features")
    has_labels = "labels" in handle
    if has_labels:
        labels = handle["labels"]
        if not isinstance(labels, h5py.Dataset):
            raise ValueError("labels must be a dataset")
        if labels.shape not in ((data.shape[0],), (data.shape[0], 1)):
            raise ValueError("labels must have shape (N,) or (N, 1), aligned with data")
        if labels.dtype.kind not in "iu":
            raise ValueError("labels must be integers")
    return PulseFileInfo(
        data.shape[0],
        names,
        tuple(canonical.index(n) for n in FEATURES),
        str(data.dtype),
        has_labels,
    )


def inspect_header(path: Path) -> PulseFileInfo:
    with h5py.File(path, "r") as handle:
        return _header(handle)


def iter_pulses(path: Path, batch_rows: int = 65536) -> Iterator[PulseBatch]:
    """Yield canonical columns; labels and transmitter metadata are never features."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    with h5py.File(path, "r") as handle:
        info = _header(handle)
        previous_toa = None
        for start in range(0, info.rows, batch_rows):
            stop = min(start + batch_rows, info.rows)
            features = handle["data"][start:stop, :][:, info.column_order]
            if not np.isfinite(features).all():
                raise ValueError(f"non-finite pulse feature in rows {start}:{stop}")
            toa = features[:, 0]
            if np.any(toa[1:] < toa[:-1]) or (previous_toa is not None and toa[0] < previous_toa):
                raise ValueError(f"arrival times are not ordered at rows {start}:{stop}")
            previous_toa = toa[-1]
            labels = handle["labels"][start:stop].reshape(-1) if info.has_labels else None
            yield PulseBatch(start, features, labels)


def scan_file(
    path: Path,
    batch_rows: int = 65536,
    sample_rows: int = 0,
    seed: int = 0,
) -> tuple[dict, np.ndarray, np.ndarray | None, np.ndarray]:
    """Validate all rows and collect an optional uniform sample in one bounded pass.

    Sampling uses row indices only, never truth labels. Statistics describe the full
    file; clustering later describes only the explicitly reported sampled population.
    """
    if sample_rows < 0 or seed < 0 or batch_rows < 1:
        raise ValueError("sample_rows/seed must be nonnegative; batch_rows must be positive")
    before = path.stat()
    info = inspect_header(path)
    count = min(info.rows, sample_rows)
    indices = (
        np.sort(np.random.default_rng(seed).choice(info.rows, count, replace=False))
        if count
        else np.empty(0, dtype=np.int64)
    )
    sample = np.empty((count, 5), dtype=np.float64)
    truth = np.empty(count, dtype=np.int64) if info.has_labels else None
    minimum = np.full(5, np.inf)
    maximum = np.full(5, -np.inf)
    mean = np.zeros(5)
    m2 = np.zeros(5)
    observed = 0
    label_counts: Counter[int] = Counter()
    for batch in iter_pulses(path, batch_rows):
        values = batch.features.astype(np.float64, copy=False)
        n = len(values)
        batch_mean = values.mean(axis=0)
        delta = batch_mean - mean
        total = observed + n
        m2 += np.square(values - batch_mean).sum(axis=0) + delta**2 * observed * n / total
        mean += delta * n / total
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
        observed = total
        if batch.labels is not None:
            labels, counts = np.unique(batch.labels, return_counts=True)
            label_counts.update({int(k): int(v) for k, v in zip(labels, counts, strict=True)})
        lo, hi = np.searchsorted(indices, [batch.start, batch.start + n])
        if hi > lo:
            offsets = indices[lo:hi] - batch.start
            sample[lo:hi] = values[offsets]
            if truth is not None:
                truth[lo:hi] = batch.labels[offsets]
    if observed != info.rows:
        raise ValueError("file row count changed during inspection")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("file changed during inspection; retry after download completes")
    stats = {
        **asdict(info),
        "features": {
            name: {
                "min": float(minimum[i]) if observed else None,
                "max": float(maximum[i]) if observed else None,
                "mean": float(mean[i]) if observed else None,
                "std": float(np.sqrt(m2[i] / observed)) if observed else None,
            }
            for i, name in enumerate(FEATURES)
        },
        "emitter_count": len(label_counts) if info.has_labels else None,
        "largest_emitter_fraction": max(label_counts.values()) / observed
        if observed and info.has_labels
        else None,
        "sample_rows": count,
    }
    return stats, sample, truth, indices
