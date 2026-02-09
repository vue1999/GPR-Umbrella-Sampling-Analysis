from __future__ import annotations

import argparse
import sys

from .gpr import gpr_umbrella_integration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPR-based umbrella integration for 1D PLUMED umbrella sampling data."
    )
    # ---- data source (mutually exclusive) ----
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-folder", help="Path to folder with window_*.ui_dat files")
    src.add_argument("--colvar-dir", help="Path to folder with COLVAR_window_*.dat files")

    # ---- force constant / centres ----
    parser.add_argument("--kappa", type=float, default=None,
                        help="Single force constant for all windows (eV/CV^2 by default)")
    parser.add_argument("--kappa-dir", default=None,
                        help="Directory with per-window window_centers_kappa_*.txt files")
    parser.add_argument("--centers", default=None,
                        help="File with one window centre per line (required with --kappa)")
    parser.add_argument("--kappa-kj", action="store_true",
                        help="Kappa values are in kJ/mol/CV^2 instead of eV/CV^2")
    parser.add_argument("--cv-col", type=int, default=1,
                        help="0-based column index for the CV in COLVAR files (default: 1)")

    # ---- units ----
    parser.add_argument("--cv-unit", default="nm",
                        help="Label for the collective-variable axis (default: nm)")
    parser.add_argument("--energy-unit", default="eV",
                        help="Label for the energy axis (default: eV)")

    # ---- outputs ----
    parser.add_argument("--output-dir", default=None, help="Directory for outputs (default: parent of data folder)")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output files")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimisation")
    parser.add_argument("--max-lag", type=int, default=1000, help="Max lag for autocorrelation")
    parser.add_argument("--acf-threshold", type=float, default=0.05, help="ACF cutoff for tau_int integration")
    parser.add_argument("--n-star", type=int, default=200, help="Number of prediction points")
    parser.add_argument("--no-plot", action="store_true", help="Skip diagnostic plot")
    parser.add_argument("--no-save", action="store_true", help="Do not save PMF/derivative outputs")
    parser.add_argument("--show", action="store_true", help="Show figure window")
    parser.add_argument("--fig-dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--fig-path", default=None, help="Explicit path to save figure")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        results = gpr_umbrella_integration(
            data_folder=args.data_folder,
            colvar_dir=args.colvar_dir,
            kappa=args.kappa,
            kappa_dir=args.kappa_dir,
            centers=args.centers,
            kappa_in_kj_per_mol=args.kappa_kj,
            cv_col=args.cv_col,
            cv_unit=args.cv_unit,
            energy_unit=args.energy_unit,
            output_prefix=args.output_prefix,
            output_dir=args.output_dir,
            optimize_hyperparams=not args.no_optimize,
            max_lag=args.max_lag,
            acf_threshold=args.acf_threshold,
            n_star=args.n_star,
            plot=not args.no_plot,
            save_fig=not args.no_plot,
            show=args.show,
            figure_dpi=args.fig_dpi,
            figure_path=args.fig_path,
            save_outputs=not args.no_save,
            verbose=not args.quiet,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cv_unit = results.get("cv_unit", "nm")
    energy_unit = results.get("energy_unit", "eV")

    if not args.quiet:
        print("\nSummary")
        print("-" * 40)
        print(f"sigma_f: {results['sigma_f']:.4f} {energy_unit}")
        print(f"lengthscale: {results['lengthscale']:.4f} {cv_unit}")
        print(f"LOO z-score std: {results['loo_z'].std():.2f}")
        if results.get("pmf_path"):
            print(f"PMF data: {results['pmf_path']}")
        if results.get("deriv_path"):
            print(f"Derivative data: {results['deriv_path']}")
        if results.get("figure_path"):
            print(f"Figure: {results['figure_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
