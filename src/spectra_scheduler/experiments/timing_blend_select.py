"""Select fixed forecast blends using development selection worlds only."""

import argparse
import json
from pathlib import Path

import torch

from ..timing_belief import BeliefPolicyConfig, load_belief
from ..timing_ensemble import TimingCountEnsemble, save_ensemble
from .storage import fingerprint, run_lock, write_json
from .timing_refine import assess


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    semantic = json.loads((args.run_dir / "config.json").read_text())
    seeds = args.seeds or semantic["arguments"]["seeds"]
    if (len(seeds) not in (1, 2) or len(set(seeds)) != len(seeds)
            or not set(seeds) <= set(semantic["arguments"]["seeds"]) or args.batch_size < 1):
        parser.error("one or two completed training seeds and a positive batch are required")
    policy = dict(semantic["policy"])
    policy["dwells"] = tuple(policy["dwells"])
    config = BeliefPolicyConfig(**policy)
    paths = [args.run_dir / "initial.pt",
             *(args.run_dir / f"seed-{seed}" / "best.pt" for seed in seeds)]
    if fingerprint(paths[0]) != semantic["initial_sha256"]:
        raise ValueError("initial forecaster changed")
    for path in paths[1:]:
        frozen = json.loads((path.parent / "frozen.json").read_text())
        if frozen["checkpoint_sha256"] != fingerprint(path):
            raise ValueError("selected checkpoint changed")
        progress = json.loads((path.parent / "progress.json").read_text())
        if progress[-1]["epoch"] != semantic["arguments"]["epochs"]:
            raise ValueError("training budget incomplete")
    models = [load_belief(path)[0] for path in paths]
    # Freeze this small grid before inspecting its selection results.
    grid = [tuple(float(i == j) for i in range(len(models))) for j in range(len(models))]
    for weight in (0.25, 0.5, 0.75):
        for index in range(1, len(models)):
            item = [0.] * len(models)
            item[0], item[index] = weight, 1 - weight
            grid.append(tuple(item))
    if len(seeds) == 2:
        grid.extend(((0., 0.5, 0.5), (0.5, 0.25, 0.25)))
    with run_lock(args.output_dir):
        if (args.output_dir / "config.json").exists():
            raise ValueError("use a fresh blend-selection directory")
        frozen = {"checkpoints": {str(p): fingerprint(p) for p in paths},
                  "grid": grid, "selection_seeds": semantic["selection_seeds"],
                  "policy": semantic["policy"], "source_sha256": fingerprint(__file__),
                  "ensemble_source_sha256": fingerprint(
                      Path(__file__).parent.parent / "timing_ensemble.py"),
                  "objective": semantic["objective"], "training_seeds": seeds}
        write_json(args.output_dir / "config.json", frozen)
        rows, best = [], None
        for index, weights in enumerate(grid):
            active = [(model, weight) for model, weight in zip(models, weights, strict=True)
                      if weight > 0]
            predictor = TimingCountEnsemble([m for m, _ in active], [w for _, w in active])
            score, means, profile = assess(
                predictor, config, semantic["selection_seeds"], args.batch_size)
            row = {"index": index, "weights": weights, "score": score,
                   "selection": means, "profile": profile}
            rows.append(row)
            if best is None or score > best["score"]:
                best = row
                metadata = {"semantic": frozen, **row}
                save_ensemble(args.output_dir / "best.pt", predictor, metadata)
            write_json(args.output_dir / "progress.json", rows)
            print(json.dumps(row), flush=True)
        write_json(args.output_dir / "selection.json", {
            "configuration": frozen, "selected": best,
            "checkpoint_sha256": fingerprint(args.output_dir / "best.pt"), "candidates": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
