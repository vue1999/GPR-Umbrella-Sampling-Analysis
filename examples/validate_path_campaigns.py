"""Reproducible DAIS acceptance/sensitivity suite; never submits MD."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from gpr_umbrella.cli_path import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--target-normal-kappa", type=float, default=0.)
    args = parser.parse_args()
    cases = ["s110_MLp04", "s110_MLp08", "s100_ML", "s100_MLp02", "s100_MLp04"]
    case = cases[args.index]
    root = Path(args.data_root) / case
    # Vary one numerical/statistical setting at a time, without cherry-picking
    # a configuration because its reported barrier error happens to be small.
    variants = [("base", 30, 1000, 20), ("long_blocks", 30, 4000, 20),
                ("fine_bins", 45, 1000, 20), ("fine_stride", 30, 1000, 10)]
    if args.pilot:
        variants = variants[:1]
    for name, bins, block, stride in variants:
        output = Path(args.output_root) / case / name
        if (output / "summary.json").is_file():
            print(f"Already complete: {output}", flush=True)
            continue
        parameters = SimpleNamespace(colvar_dir=str(root / "COLVAR"),
            kappa_dir=str(root / "window_kappa"), path_reference=str(root / "neb_HH_RELZ_30img.dat"),
            output_dir=str(output), temperature=300., bins=bins, stride=stride,
            block_size=block, bootstraps=128, seed=2026, grid_points=201,
            reactant_interval=None, transition_interval=None, max_barrier_std=.05,
            target_normal_kappa=args.target_normal_kappa)
        try:
            run(parameters)
        except ValueError as error:
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(json.dumps({"status": "FAILED", "reason": str(error)}, indent=2))
            print(f"FAILED {case}/{name}: {error}", flush=True)


if __name__ == "__main__":
    main()
