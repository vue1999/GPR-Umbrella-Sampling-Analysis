"""Command-line entry point for 2D GPR umbrella integration."""
from __future__ import annotations

import argparse

from .integration_2d import reconstruct_pmf_2d


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
                   help="Find the minimum-bottleneck path between two states")
    p.add_argument("--path-endpoints", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Physical (cv0,cv1) coords of the two states to connect "
                        "(default: the two deepest minima)")
    p.add_argument(
        "--path-endpoint-radius", type=float, default=None,
        help="Search radius for relocating each requested endpoint to a "
             "path-valid minimum, in GP lengthscales "
             "(default: support radius)",
    )
    p.add_argument(
        "--adjust-path-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Relocate requested endpoints to nearby path-valid minima; "
             "disable to use the nearest restraint-window centres "
             "(default: enabled)",
    )
    p.add_argument("--path-metric-scales", type=float, nargs=2, default=None,
                   metavar=("SCALE0", "SCALE1"),
                   help="Positive scale for each CV when defining path length and "
                        "perpendicular directions (default: fitted GP lengthscales)")
    p.add_argument("--path-aligned-marginal", action="store_true",
                   help="Compute A(s) by Boltzmann-integrating the PMF along "
                        "directions perpendicular to the lowest-barrier path")
    p.add_argument("--thermal-energy", type=float, default=None, metavar="KBT",
                   help="kBT in --energy-unit; required with "
                        "--path-aligned-marginal")
    p.add_argument("--perpendicular-points", type=int, default=201,
                   help="Quadrature points on each perpendicular line (default: 201)")
    p.add_argument("--perpendicular-width", type=float, default=None,
                   help="Optional half-width in dimensionless path-metric coordinates")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.path_aligned_marginal and args.thermal_energy is None:
        parser.error("--path-aligned-marginal requires --thermal-energy KBT")
    reconstruct_pmf_2d(
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
        include_cross_component_covariance=not args.diagonal_noise,
        calibrate_uncertainty=not args.no_calibrate,
        restrict_to_sampled_support=args.restrict_to_sampled_support,
        support_radius=args.support_radius,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        plot=not args.no_plot,
        plot_diagnostics=not args.no_diagnostics,
        find_lowest_barrier=args.find_lowest_barrier_path,
        path_endpoints=((tuple(args.path_endpoints[:2]), tuple(args.path_endpoints[2:]))
                        if args.path_endpoints else None),
        path_endpoint_radius=args.path_endpoint_radius,
        adjust_path_endpoints=args.adjust_path_endpoints,
        path_metric_scale=(tuple(args.path_metric_scales)
                           if args.path_metric_scales else None),
        path_aligned_marginal=args.path_aligned_marginal,
        thermal_energy=args.thermal_energy,
        perpendicular_points=args.perpendicular_points,
        perpendicular_width=args.perpendicular_width,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
