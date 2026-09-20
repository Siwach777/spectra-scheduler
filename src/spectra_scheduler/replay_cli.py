"""Run a physical-time sweep against a completed TSRD stare recording."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from .dataset_io import discover_files
from .pulse_replay import DwellAction, PulseReplay, ReplayConfig


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--dwell-us", type=float, default=10000)
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
        files = discover_files(args.root, "stare", args.split)
        if not 0 <= args.file_index < len(files):
            raise ValueError(f"file-index out of range; found {len(files)} completed stare files")
        begin = perf_counter()
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
    except (ValueError, OSError) as error:
        parser.exit(1, f"replay failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
