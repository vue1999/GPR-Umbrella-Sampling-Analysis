"""Command-line entry point for 2D GPR umbrella integration."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .integration_2d import reconstruct_pmf_2d
from .pathways import find_lowest_barrier_path, save_lowest_barrier_path
from .plotting_2d import plot_lowest_barrier_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2D GPR umbrella integration for PLUMED 2-CV umbrella data."
    )
    p.add_argument("--colvar-dir", required=True,
                   help="Folder with COLVAR_window_*.dat (cols: time, cv0, cv1)")
    p.add_argument("--kappa-dir", default=None,
                   help="Folder with window_centers_kappa_*.txt "
                        "(c0, c1, kappa0, kappa1). Defaults to --colvar-dir.")
    p.add_argument("--cv-cols", type=int, nargs=2, default=(1, 2),
                   help="0-based CV column indices in COLVAR (default: 1 2)")
    p.add_argument("--kappa-kj", action="store_true",
                   help="Kappa values are kJ/mol/CV^2 and should be converted "
                        "to --energy-unit/CV^2")
    p.add_argument("--cv-names", nargs=2, default=("hh", "relz"))
    p.add_argument("--cv-units", nargs=2, default=("A", "A"))
    p.add_argument("--energy-unit", default="eV",
                   help="Numerical/output energy unit (default: eV)")
    p.add_argument("--grid-n", type=int, nargs=2, default=(60, 60))
    p.add_argument("--prediction-batch-size", type=int, default=10_000,
                   help="Maximum grid points predicted at once")
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--lengthscales", type=float, nargs=2, default=None,
                   metavar=("ELL0", "ELL1"),
                   help="Fix one GP lengthscale per CV")
    p.add_argument("--sigma-f", type=float, default=None)
    p.add_argument("--covariance-block-size", type=int, default=None, metavar="FRAMES",
                   help="Explicit covariance block length in saved frames; requires at least four blocks/window")
    p.add_argument("--fit-extra-noise", action="store_true",
                   help="Fit independent window-gradient discrepancy in each CV")
    p.add_argument("--extra-noise-scale", type=float, nargs=2, default=None,
                   help="Half-normal scales in energy/CV units; default is data-derived RMS forces")
    p.add_argument("--sigma-f-max", type=float, default=None,
                   help="Optional GP amplitude cap in energy units; unset by default")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--diagonal-noise", action="store_true",
                   help="Ignore cross-CV covariance within each window")
    p.add_argument(
        "--restrict-to-sampled-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict plots and path analysis to the union of kernel-scaled "
             "neighborhoods around sampled window means (default: enabled)",
    )
    p.add_argument(
        "--support-radius", type=float, default=0.5,
        help="Radius of each sampled neighborhood in GP lengthscales "
             "(default: 0.5)",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the PMF + uncertainty figure")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Skip the 8-panel sampling/fit diagnostics figure")
    p.add_argument("--find-lowest-barrier-path", action="store_true",
                   help="Find the exact minimum-PMF-range grid path")
    p.add_argument("--path-endpoints", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Required search-mode endpoint coordinates; snap to "
                        "the nearest grid cells and reject them if invalid")
    p.add_argument(
        "--path-mode", choices=("search", "corridor", "fixed"), default="search",
        help="Free grid search, corridor search, or direct reference-path evaluation",
    )
    p.add_argument(
        "--path-reference", default=None, metavar="FILE",
        help="Text file whose first two columns define the reference trajectory",
    )
    p.add_argument(
        "--path-corridor-radius", type=float, default=None, metavar="RADIUS",
        help="Corridor radius in dimensionless path-metric units",
    )
    p.add_argument("--path-metric-scales", type=float, nargs=2, default=None,
                   metavar=("SCALE0", "SCALE1"),
                   help="Positive scale for each CV when defining path length "
                        "(default: fitted GP lengthscales)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path_requested = args.find_lowest_barrier_path
    if path_requested and args.path_mode == "search" and args.path_endpoints is None:
        parser.error("--path-mode search requires --path-endpoints")
    if args.path_mode != "search" and args.path_reference is None:
        parser.error("--path-mode corridor/fixed requires --path-reference FILE")
    if args.path_mode == "corridor" and args.path_corridor_radius is None:
        parser.error("--path-mode corridor requires --path-corridor-radius")
    if args.path_mode == "search" and args.path_reference is not None:
        parser.error("--path-reference requires --path-mode corridor or fixed")
    if args.path_mode != "corridor" and args.path_corridor_radius is not None:
        parser.error("--path-corridor-radius requires --path-mode corridor")
    if args.path_mode != "search" and args.path_endpoints is not None:
        parser.error("corridor/fixed modes use the reference trajectory endpoints")
    reference_path = None
    if args.path_reference is not None:
        try:
            reference_path = np.loadtxt(
                args.path_reference, comments="#", usecols=(0, 1), ndmin=2,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read --path-reference: {exc}")
    results = reconstruct_pmf_2d(
        colvar_dir=args.colvar_dir,
        kappa_dir=args.kappa_dir,
        cv_cols=tuple(args.cv_cols),
        kappa_in_kj_per_mol=args.kappa_kj,
        cv_names=tuple(args.cv_names),
        cv_units=tuple(args.cv_units),
        energy_unit=args.energy_unit,
        grid_n=tuple(args.grid_n),
        prediction_batch_size=args.prediction_batch_size,
        optimize_hyperparams=not args.no_optimize,
        fixed_lengthscale=(tuple(args.lengthscales) if args.lengthscales else None),
        fixed_sigma_f=args.sigma_f,
        covariance_block_size=args.covariance_block_size,
        fit_extra_noise=args.fit_extra_noise,
        extra_noise_scale=args.extra_noise_scale,
        sigma_f_max=args.sigma_f_max,
        include_cross_component_covariance=not args.diagonal_noise,
        calibrate_uncertainty=not args.no_calibrate,
        restrict_to_sampled_support=args.restrict_to_sampled_support,
        support_radius=args.support_radius,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        plot=not args.no_plot,
        plot_diagnostics=not args.no_diagnostics,
        verbose=not args.quiet,
    )
    if path_requested:
        path_result = find_lowest_barrier_path(
            results,
            endpoints=(args.path_endpoints[:2], args.path_endpoints[2:])
            if args.path_endpoints else None,
            metric_scale=args.path_metric_scales,
            reference_path=reference_path,
            path_mode=args.path_mode,
            corridor_radius=args.path_corridor_radius,
        )
        pmf_file = Path(results["pmf_path"])
        prefix = pmf_file.name.removesuffix("_pmf_2d.dat")
        path_file = pmf_file.with_name(f"{prefix}_lowest_barrier_path.dat")
        save_lowest_barrier_path(path_result, str(path_file))
        if not args.no_plot:
            plot_lowest_barrier_path(
                results, path_result,
                output_path=str(path_file.with_suffix(".png")),
                output_prefix=prefix,
            )
        if not args.quiet:
            print(f"Endpoint-to-maximum rise: {path_result['endpoint_to_max']:.3f} "
                  f"+/- {path_result['endpoint_to_max_err']:.3f} {args.energy_unit}")
            print(f"Path energy range: {path_result['energy_range']:.3f} "
                  f"+/- {path_result['energy_range_err']:.3f} {args.energy_unit}")
            print(f"wrote {path_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
