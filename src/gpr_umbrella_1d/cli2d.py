"""Command-line entry point for 2D GPR umbrella integration."""
from __future__ import annotations

import argparse

from .gpr2d import gpr_umbrella_integration_2d


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
                   help="Kappa values are kJ/mol/CV^2 instead of eV/CV^2")
    p.add_argument("--cv-names", nargs=2, default=("hh", "relz"))
    p.add_argument("--cv-units", nargs=2, default=("A", "A"))
    p.add_argument("--energy-unit", default="eV")
    p.add_argument("--grid-n", type=int, nargs=2, default=(60, 60))
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--lengthscale", type=float, default=None,
                   help="Fix an isotropic lengthscale for both CVs")
    p.add_argument("--sigma-f", type=float, default=None)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the PMF + uncertainty figure")
    p.add_argument("--no-diagnostics", action="store_true",
                   help="Skip the 8-panel sampling/fit diagnostics figure")
    p.add_argument("--find-mep", action="store_true",
                   help="Locate minima, connect them with a minimum-energy path, "
                        "and write {prefix}_mep.dat + {prefix}_mep.png")
    p.add_argument("--mep-endpoints", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Physical (cv0,cv1) coords of the two states to connect "
                        "(default: the two deepest minima)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    gpr_umbrella_integration_2d(
        colvar_dir=args.colvar_dir,
        kappa_dir=args.kappa_dir,
        cv_cols=tuple(args.cv_cols),
        kappa_in_kj_per_mol=args.kappa_kj,
        cv_names=tuple(args.cv_names),
        cv_units=tuple(args.cv_units),
        energy_unit=args.energy_unit,
        grid_n=tuple(args.grid_n),
        optimize_hyperparams=not args.no_optimize,
        fixed_lengthscale=args.lengthscale,
        fixed_sigma_f=args.sigma_f,
        calibrate_uncertainty=not args.no_calibrate,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        plot=not args.no_plot,
        plot_diagnostics=not args.no_diagnostics,
        find_mep=args.find_mep,
        mep_endpoints=((tuple(args.mep_endpoints[:2]), tuple(args.mep_endpoints[2:]))
                       if args.mep_endpoints else None),
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
