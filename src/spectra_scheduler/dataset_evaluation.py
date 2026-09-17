"""Bounded, file-local clustering benchmarks; not receiver-policy replay."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from spectra_scheduler.dataset_io import discover_files, scan_file


@dataclass(frozen=True)
class DatasetConfig:
    mode: str = "scan"
    split: str = "train"
    max_files: int = 10
    batch_rows: int = 65536
    sample_rows: int = 20000
    seed: int = 0
    features: str = "signature"
    min_cluster_size: int = 20
    min_samples: int = 10

    def __post_init__(self):
        if self.mode not in ("scan", "stare") or self.split not in ("train", "val", "test"):
            raise ValueError("invalid mode or split")
        if self.features not in ("raw", "signature"):
            raise ValueError("features must be raw or signature")
        if self.max_files < 0 or self.seed < 0:
            raise ValueError("max_files and seed must be nonnegative")
        if self.batch_rows < 1 or self.sample_rows < 2:
            raise ValueError("batch_rows must be positive and sample_rows at least two")
        if self.min_cluster_size < 2 or self.min_samples < 1:
            raise ValueError("min_cluster_size must be at least two; min_samples positive")


def transform_features(pulses: np.ndarray, mode: str) -> np.ndarray:
    from sklearn.preprocessing import RobustScaler

    if pulses.ndim != 2 or pulses.shape[1] != 5 or not np.isfinite(pulses).all():
        raise ValueError("expected finite pulse features with shape (N, 5)")
    if mode == "raw":
        return pulses.astype(np.float64, copy=True)
    if mode != "signature":
        raise ValueError("unknown feature transform")
    if np.any(pulses[:, 2] < 0):
        raise ValueError("signature transform requires nonnegative pulse widths")
    # Arrival time is deliberately excluded from this simple identity baseline.
    # Circular AoA avoids treating -180 and +180 degrees as opposite directions.
    continuous = np.column_stack((pulses[:, 1], np.log1p(pulses[:, 2]), pulses[:, 4]))
    continuous = RobustScaler().fit_transform(continuous)
    angle = np.deg2rad(pulses[:, 3])
    return np.column_stack((continuous, np.sin(angle), np.cos(angle)))


def predict_clusters(pulses: np.ndarray, config: DatasetConfig) -> np.ndarray:
    """Receives observations only; truth cannot influence fit or feature selection."""
    from sklearn.cluster import HDBSCAN
    from threadpoolctl import threadpool_limits

    if len(pulses) < max(config.min_cluster_size, config.min_samples, 2):
        return np.full(len(pulses), -1, dtype=np.int64)
    features = transform_features(pulses, config.features)
    # Parallelism is across files, not nested BLAS/OpenMP pools inside each worker.
    with threadpool_limits(limits=1):
        return HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            algorithm="kd_tree",
            n_jobs=1,
            copy=True,
        ).fit_predict(features)


def score_clusters(truth: np.ndarray, predicted: np.ndarray) -> dict:
    from sklearn.metrics import homogeneity_completeness_v_measure

    if truth.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError("nonempty aligned 1-D labels are required")
    # Combinatorial counts avoid constructing an N-by-N pair matrix.
    _, joint = np.unique(np.column_stack((truth, predicted)), axis=0, return_counts=True)
    _, actual = np.unique(truth, return_counts=True)
    _, joined = np.unique(predicted, return_counts=True)

    def pairs(counts):
        return sum(int(n) * (int(n) - 1) // 2 for n in counts)

    tp, predicted_pairs, true_pairs = pairs(joint), pairs(joined), pairs(actual)
    precision = tp / predicted_pairs if predicted_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth, predicted)
    return {
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _peak_rss_mib() -> float | None:
    import sys

    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _process_file(task: tuple) -> dict:
    root, relative, config, evaluate, profile = task
    start = perf_counter()
    path = root / relative
    seed = config.seed ^ int.from_bytes(sha256(relative.encode()).digest()[:8], "little")
    try:
        stats, sample, truth, indices = scan_file(
            path, config.batch_rows, config.sample_rows if evaluate else 0, seed
        )
        read_seconds = perf_counter() - start
        result = {"file": relative, "status": "ok", "statistics": stats}
        if not stats["rows"]:
            result["status"] = "empty"
        elif not stats["has_labels"]:
            result["status"] = "unlabelled"
        elif evaluate:
            prediction = predict_clusters(sample, config)
            noise = prediction < 0
            singleton = prediction.copy()
            singleton[noise] = -np.arange(1, noise.sum() + 1)
            result.update(
                {
                    "sample_seed": seed,
                    "sample_indices_sha256": sha256(indices.astype("<i8").tobytes()).hexdigest(),
                    "sample_features_sha256": sha256(sample.astype("<f8").tobytes()).hexdigest(),
                    "sample_predictions_sha256": sha256(
                        prediction.astype("<i8").tobytes()
                    ).hexdigest(),
                    "sample_emitters": int(len(np.unique(truth))),
                    "predicted_clusters": int(len(np.unique(prediction[~noise]))),
                    "noise_fraction": float(noise.mean()),
                    "metrics": score_clusters(truth, prediction),
                    "noise_singleton_metrics": score_clusters(truth, singleton),
                }
            )
        if profile:
            elapsed = perf_counter() - start
            result["profile"] = {
                "read_seconds": read_seconds,
                "total_seconds": elapsed,
                "read_pulses_per_second": stats["rows"] / read_seconds if read_seconds else None,
                "process_peak_rss_mib": _peak_rss_mib(),
            }
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"file": relative, "status": "error", "error": str(error)}


def run_dataset(
    root: Path,
    config: DatasetConfig,
    *,
    evaluate: bool = False,
    workers: int = 1,
    profile: bool = False,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root)
    paths = discover_files(root, config.mode, config.split)
    if not paths:
        raise ValueError("no completed .h5 files found in the selected split")
    selected = paths[: config.max_files] if config.max_files else paths
    tasks = [(root, p.relative_to(root).as_posix(), config, evaluate, profile) for p in selected]
    if workers == 1:
        results = [_process_file(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            results = list(pool.map(_process_file, tasks))
    scored = [r for r in results if "metrics" in r]
    summary = {
        "available_files": len(paths),
        "selected_files": len(selected),
        "valid_files": sum(r["status"] != "error" for r in results),
        "error_files": sum(r["status"] == "error" for r in results),
        "empty_files": sum(r["status"] == "empty" for r in results),
        "unlabelled_files": sum(r["status"] == "unlabelled" for r in results),
        "scored_files": len(scored),
        "validated_pulses": sum(r.get("statistics", {}).get("rows", 0) for r in results),
        "scored_pulses": sum(r["statistics"]["sample_rows"] for r in scored),
    }
    for field in ("metrics", "noise_singleton_metrics"):
        summary[f"macro_{field}"] = (
            {key: sum(r[field][key] for r in scored) / len(scored) for key in scored[0][field]}
            if scored
            else None
        )
    return {
        "schema_version": 1,
        "task": "pulse_association" if evaluate else "dataset_inspection",
        "config": asdict(config),
        "selection": "lexicographic completed filenames; uniform seeded pulse sample per file",
        "noise_policy": (
            "metrics: one noise cluster; noise_singleton_metrics: one cluster per noise pulse"
        ),
        "aggregation": "unweighted file macro means; emitter IDs never pooled across files",
        "versions": {name: version(name) for name in ("numpy", "h5py", "scikit-learn")},
        "summary": summary,
        "files": results,
    }
