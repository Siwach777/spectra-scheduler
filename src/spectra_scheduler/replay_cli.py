"""Run a sweep or CUDA timing scheduler against a completed TSRD stare recording."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

from .dataset_io import discover_files
from .pulse_replay import DwellAction, PulseReplay, ReplayConfig


def _timing_report(path, checkpoint, config, bands, adapter, compare):
    # Keep the sweep CLI usable without the optional PyTorch dependency.
    import torch

    from .experiments.storage import fingerprint
    from .replay_baselines import RateProbePolicy
    from .replay_env import InterfaceConfig
    from .replay_evaluation import ReferencePolicy, evaluate_policy
    from .timing_replay import TimingReplayPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("timing replay requires CUDA; CPU fallback is disabled")
    torch.set_num_threads(1)
    digest = fingerprint(checkpoint)
    policy = TimingReplayPolicy.from_checkpoint(checkpoint, maximum_batch=1, adapter=adapter)
    interface = InterfaceConfig(bands=bands, dwell_us=tuple(d * 1000 for d in policy.config.dwells))
    results = {"timing": evaluate_policy(path, policy, config, interface)}
    if compare:
        results["sweep-50"] = evaluate_policy(
            path, ReferencePolicy("sweep", interface.dwell_us.index(50000)), config, interface
        )
        results["rate-probe"] = evaluate_policy(path, RateProbePolicy(), config, interface)
        if len({r["truth_pulses"] for r in results.values()}) != 1 or len({
            r["simulated_us"] for r in results.values()
        }) != 1:
            raise ValueError("comparison policies received different truth or time budgets")
    if fingerprint(checkpoint) != digest:
        raise ValueError("checkpoint changed during replay")
    return dict(
        schema_version=1, policy="timing", checkpoint=str(checkpoint),
        checkpoint_sha256=digest, adapter=policy.adapter, results=results,
        scope="single external synthetic recording; not hardware or an aggregate benchmark",
        forecast_status="band/dwell decisions; external intercept-time calibration not established",
    )


def _write_report(path, rendered):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--dwell-us", type=float, default=10000, help="fixed-sweep listening dwell")
    parser.add_argument("--timing-model", type=Path, help="frozen timing checkpoint; requires CUDA")
    parser.add_argument("--timing-adapter", choices=("missing-power", "legacy"),
                        default="missing-power", help="legacy reproduces the old replay input path")
    parser.add_argument("--compare-controls", action="store_true",
                        help="compare timing with 50-ms sweep and RateProbe on the same recording")
    parser.add_argument("--output", type=Path)
    for name, field in ReplayConfig.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            default=field.default,
            type=int if name in ("batch_rows", "max_observation_pulses", "seed") else float,
        )
    args = parser.parse_args(arguments)
    try:
        config = ReplayConfig(
            **{name: getattr(args, name) for name in ReplayConfig.__dataclass_fields__}
        )
        span = config.max_frequency_mhz - config.min_frequency_mhz
        if args.bands < 1 or args.dwell_us <= 0:
            raise ValueError("bands and dwell must be positive")
        if args.bands * config.bandwidth_mhz != span:
            raise ValueError("sweep requires bands * bandwidth_mhz = frequency span")
        if args.compare_controls and args.timing_model is None:
            raise ValueError("compare-controls requires timing-model")
        if args.timing_model:
            if any(v % 1000 for v in (config.start_us, config.stop_us, config.retune_us)):
                raise ValueError("timing replay requires whole 1-ms ticks; use --retune-us 2000")
            if config.slew_mhz_per_us is not None:
                raise ValueError("timing replay does not support fractional slew timing")
        if args.output:
            output = args.output.resolve()
            if output.is_relative_to(args.root.resolve()) or (
                args.timing_model and output == args.timing_model.resolve()
            ):
                raise ValueError("output must not overwrite the dataset or timing checkpoint")
        files = discover_files(args.root, "stare", args.split)
        if not 0 <= args.file_index < len(files):
            raise ValueError(f"file-index out of range; found {len(files)} completed stare files")
        begin = perf_counter()
        if args.timing_model:
            report = _timing_report(files[args.file_index], args.timing_model, config,
                                    args.bands, args.timing_adapter, args.compare_controls)
            report.update(split=args.split, bands=args.bands, elapsed_seconds=perf_counter() - begin)
            rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
            if args.output:
                _write_report(args.output, rendered)
            print(rendered, end="")
            return 0
        with PulseReplay(files[args.file_index], config, source_mode="stare") as replay:
            step = 0
            while not replay.done:
                center = config.min_frequency_mhz + (step % args.bands + 0.5) * config.bandwidth_mhz
                replay.step(DwellAction(center, args.dwell_us))
                step += 1
            report = replay.report()
        report.update(
            split=args.split,
            policy="fixed-sweep",
            bands=args.bands,
            dwell_us=args.dwell_us,
            elapsed_seconds=perf_counter() - begin,
        )
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            _write_report(args.output, rendered)
        print(rendered, end="")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"replay failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
