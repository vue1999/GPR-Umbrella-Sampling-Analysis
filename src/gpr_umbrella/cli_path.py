"""Original-bias projection and robust SE GPR, with the usual diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .integration_1d import _extract_window_index, compute_tau_int
from .plotting_1d import PALETTE, apply_plot_style, plot_diagnostics
from .projected_inputs import prepare_projected_observations
from .robust_1d import barrier_posterior, fit_linear_observations, profile_landmarks


def load_inputs(colvar_dir, kappa_dir, reference, cv_cols=(1, 2)):
    def indexed(folder, pattern):
        files = list(Path(folder).glob(pattern))
        result = {_extract_window_index(str(p)): p for p in files}
        if len(result) != len(files) or not files:
            raise ValueError("Missing files or duplicate numeric window IDs")
        return result

    colvars = indexed(colvar_dir, "COLVAR_window_*.dat")
    kappafiles = indexed(kappa_dir, "window_centers_kappa_*.txt")
    if colvars.keys() != kappafiles.keys():
        raise ValueError("COLVAR and kappa window IDs differ")
    trajectories, restraints, provenance, dt = [], [], [], []
    for i in sorted(colvars):
        raw = np.loadtxt(colvars[i], comments="#", ndmin=2)
        times = raw[:, 0]
        differences = np.diff(times)
        if (
            len(times) < 4
            or np.any(differences <= 0)
            or not np.allclose(differences, np.median(differences), rtol=1e-3)
        ):
            raise ValueError(
                f"Window {i}: time must be increasing and uniformly spaced"
            )
        trajectories.append(raw[:, cv_cols])
        dt.append(float(np.median(differences)))
        restraint = np.loadtxt(kappafiles[i], comments="#", ndmin=1)
        if restraint.shape != (4,):
            raise ValueError("2D restraint rows must be: center1 center2 kappa1 kappa2")
        restraints.append(restraint)
        provenance.append(
            {
                "window_id": i,
                "colvar": str(colvars[i].resolve()),
                "colvar_sha256": hashlib.sha256(colvars[i].read_bytes()).hexdigest(),
                "restraint_sha256": hashlib.sha256(
                    kappafiles[i].read_bytes()
                ).hexdigest(),
                "samples": len(raw),
                "time_start": times[0],
                "time_end": times[-1],
            }
        )
    if not np.allclose(dt, dt[0]):
        raise ValueError("Stored sample interval differs between windows")
    restraints = np.array(restraints)
    vertices = np.loadtxt(reference, comments="#", ndmin=2)
    return (
        trajectories,
        restraints[:, :2],
        restraints[:, 2:],
        vertices,
        provenance,
        dt[0],
    )


def _quality_plot(prepared, result, output):
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    table = result["hyperparameter_table"]
    scatter = axes[0, 0].scatter(
        table[:, 0], table[:, 1], c=table[:, -1], cmap="viridis", s=20
    )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="SE lengthscale (Å)",
        ylabel="SE amplitude (eV)",
        title="Hyperparameter posterior weights",
    )
    fig.colorbar(scatter, ax=axes[0, 0], label="Discrete posterior mass")
    axes[0, 1].imshow(prepared["overlap_matrix"], origin="lower", cmap="viridis")
    axes[0, 1].set(
        xlabel="Window",
        ylabel="Window",
        title=f"Original 2D-bias overlap: {prepared['overlap_components']} components",
    )
    axes[1, 0].plot(prepared["nodes"], prepared["block_ess"], color=PALETTE["sampling"])
    axes[1, 0].plot(
        prepared["nodes"],
        prepared["unweighted_block_ess"],
        color=PALETTE["guide"],
        ls="--",
        label="Unweighted blocks",
    )
    axes[1, 0].axhline(4, ls="--", color=PALETTE["warn"])
    axes[1, 0].legend()
    axes[1, 0].set(
        xlabel="Arclength (Å)",
        ylabel="Effective contributing time blocks",
        title="Reweighted support per bin",
    )
    for half, profile in enumerate(prepared["half_profiles"]):
        axes[1, 1].plot(prepared["nodes"], profile, label=f"Time half {half + 1}")
    axes[1, 1].plot(
        prepared["nodes"],
        prepared["histogram_pmf"],
        color=PALETTE["pmf"],
        label="Full trajectory",
        lw=2,
    )
    axes[1, 1].set(
        xlabel="Arclength (Å)",
        ylabel="ΔF (eV)",
        title="Time-split convergence (same reference bin)",
    )
    axes[1, 1].legend()
    fig.savefig(output / "selection_diagnostics.png", dpi=160)
    plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    code_provenance = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    trajectories, centers, kappas, vertices, provenance, dt = load_inputs(
        args.colvar_dir, args.kappa_dir, args.path_reference
    )
    campaign_root = Path(args.colvar_dir).parent
    manifest, path_xyz = (
        campaign_root / "source_campaign.json",
        campaign_root / "neb_path_30_images.xyz",
    )
    if manifest.is_file() and path_xyz.is_file():
        from .campaign_audit import audit_campaign

        campaign_audit = audit_campaign(manifest, path_xyz)
        if campaign_audit["n_windows"] != len(centers):
            raise ValueError("Campaign and supplied window counts differ")
        if not np.isclose(
            campaign_audit["temperature_K"], args.temperature, rtol=0, atol=1e-8
        ):
            raise ValueError("Requested temperature differs from the campaign metadata")
    else:
        campaign_audit = {"quality_issues": ["common_hamiltonian_not_verified"]}
    # Units are deliberately explicit: this frontend accepts Å, eV and fs.
    kbt = 8.617333262145e-5 * args.temperature
    prepared = prepare_projected_observations(
        trajectories,
        centers,
        kappas,
        vertices,
        kbt=kbt,
        bins=args.bins,
        stride=args.stride,
        block_size=args.block_size,
        bootstraps=args.bootstraps,
        seed=args.seed,
        target_normal_kappa=getattr(args, "target_normal_kappa", 0.0),
        profile_range=getattr(args, "s_range", None),
        bin_edges=(
            np.loadtxt(args.bin_edges, ndmin=1)
            if getattr(args, "bin_edges", None)
            else None
        ),
        projection_method=getattr(args, "projection_method", "soft"),
        smoothing_width=getattr(args, "smoothing_width", None),
        reweight_cache=getattr(args, "reweight_cache", None),
        progress=lambda s: print(s, flush=True),
    )
    nodes = prepared["nodes"]
    grid = np.linspace(nodes[0], nodes[-1], args.grid_points)
    center_s = np.sort(prepared["center_s"])
    resolution = max(np.median(np.diff(nodes)), np.median(np.diff(center_s)))
    result = fit_linear_observations(
        nodes,
        prepared["operator"],
        prepared["values"],
        prepared["noise_covariance"],
        grid,
        resolution=resolution,
    )
    issues = (
        prepared["quality_issues"]
        + result["quality_issues"]
        + campaign_audit["quality_issues"]
    )
    # A single run cannot certify binning, block-length, prior-bound invariance.
    issues.append("sensitivity_suite_required")
    status = "PROVISIONAL" if issues == ["sensitivity_suite_required"] else "UNRELIABLE"
    barrier = None
    if args.reactant_interval is not None or args.transition_interval is not None:
        if args.reactant_interval is None or args.transition_interval is None:
            raise ValueError("Both physical basin intervals must be specified")
        barrier = barrier_posterior(
            result, args.reactant_interval, args.transition_interval, seed=args.seed
        )
        if barrier["std"] > args.max_barrier_std:
            issues.append("barrier_uncertainty_exceeds_target")
            status = "UNRELIABLE"
        if (
            max(
                barrier["reactant_extremum_boundary_probability"],
                barrier["transition_extremum_boundary_probability"],
            )
            > 0.5
        ):
            issues.append("barrier_extremum_at_interval_boundary")
            status = "UNRELIABLE"
    else:
        issues.append("physical_barrier_basins_not_specified")
    # Reuse the package's original diagnostic figure, not a parallel plot style.
    positions = prepared["all_positions"]
    tau = np.array([compute_tau_int(p) for p in positions])
    for i, p in enumerate(positions):
        if p.var(ddof=1) > 0:
            for blocks in (8, 16, 32, 64):
                means = np.array([a.mean() for a in np.array_split(p, blocks)])
                tau[i] = max(
                    tau[i], len(p) * means.var(ddof=1) / (2 * blocks * p.var(ddof=1))
                )
    result.update(
        {
            "x_centers": prepared["center_s"],
            "x_means": np.array([p.mean() for p in positions]),
            "x_vars": np.array([p.var(ddof=1) for p in positions]),
            "n_samples": np.array([len(p) for p in positions]),
            "tau_ints": tau,
            "n_eff": np.array([len(p) for p in positions]) / (2 * tau),
            "observation_x": (nodes[:-1] + nodes[1:]) / 2,
            "observation_label": "Reweighted interval slopes ±2σ",
            "observation_index_label": "Arclength interval index",
            "loo_title": "Model-averaged LOO",
            "derivatives": prepared["values"],
            "derivative_errors": np.sqrt(np.diag(prepared["noise_covariance"])),
            "training_std_residuals": result["training_residuals"]
            / np.sqrt(np.diag(prepared["noise_covariance"])),
            "all_positions": [p[:: max(1, len(p) // 2000)] for p in positions],
            "kappa": kappas,
            "cv_unit": "Å",
            "energy_unit": "eV",
            "deriv_unit": "eV/Å",
            "uncertainty_calibration_factor": None,
            "quality_status": status,
        }
    )
    np.savetxt(
        output / "pmf_1d.dat",
        np.c_[grid, result["pmf_mean"], result["pmf_std"]],
        header="s_A F_eV sigma_eV; endpoint-bin referenced; conditional + hyperparameter mixture",
    )
    np.savetxt(
        output / "hyperparameters.tsv",
        result["hyperparameter_table"],
        header="\t".join(result["hyperparameter_columns"]),
        delimiter="\t",
    )
    np.savez_compressed(
        output / "fit_arrays.npz",
        nodes=nodes,
        bin_edges=prepared["bin_edges"],
        operator=prepared["operator"],
        values=prepared["values"],
        noise_covariance=prepared["noise_covariance"],
        grid=grid,
        pmf=result["pmf_mean"],
        covariance=result["pmf_covariance"],
        bootstrap_pmf=prepared["bootstrap_pmf"],
        overlap=prepared["overlap_matrix"],
        block_ess=prepared["block_ess"],
        unweighted_block_ess=prepared["unweighted_block_ess"],
        relative_block_ess=prepared["relative_block_ess"],
        half_profiles=prepared["half_profiles"],
        loo_predictive_quantile_z=result["loo_z"],
        map_loo_z=result["map_loo_z"],
        loo_means=result["loo_means"],
        loo_stds=result["loo_stds"],
    )
    summary = {
        "status": status,
        "quality_issues": issues,
        "barrier": barrier,
        "profile_landmarks": profile_landmarks(result),
        "method": "original-2D-bias MBAR preparation + correlated-increment SE GPR",
        "target": (
            "path-normal-restrained arclength marginal"
            if getattr(args, "target_normal_kappa", 0.0)
            else "unrestrained arclength marginal"
        )
        + "; common physical walls retained; not F along a 2D NEB slice",
        "target_normal_kappa_eV_A2": getattr(args, "target_normal_kappa", 0.0),
        "projection_method": prepared["projection_method"],
        "smoothing_width_A": prepared["smoothing_width"],
        "reweight_cache_hits": prepared["reweight_cache_hits"],
        "lengthscale_A": result["lengthscale"],
        "sigma_f_eV": result["sigma_f"],
        "loo_rms": float(np.sqrt(np.mean(result["loo_z"] ** 2))),
        "loo_max_abs": float(np.max(np.abs(result["loo_z"]))),
        "loo_method": result["loo_method"],
        "map_loo_rms": float(np.sqrt(np.mean(result["map_loo_z"] ** 2))),
        "map_loo_max_abs": float(np.max(np.abs(result["map_loo_z"]))),
        "map_blocked_cv_rms": result["map_blocked_cv_rms"],
        "blocked_cv_rms": result["blocked_cv_rms"],
        "overlap_components": prepared["overlap_components"],
        "overlap_scalar": prepared["overlap_scalar"],
        "minimum_effective_blocks": float(prepared["block_ess"].min()),
        "minimum_relative_effective_blocks": float(
            prepared["relative_block_ess"].min()
        ),
        "half_profile_difference_range_eV": prepared["half_profile_difference_range"],
        "settings": vars(args),
        "campaign_audit": campaign_audit,
        "stored_sample_interval_fs": dt,
        "code_sha256": code_provenance,
        "python": sys.version,
        "dependencies": {
            name: version(name) for name in ("numpy", "scipy", "pymbar", "matplotlib")
        },
        "block_duration_ps": args.block_size * dt / 1000,
        "covariance_shrinkage": prepared["covariance_shrinkage"],
        "successful_bootstraps": len(prepared["bootstrap_pmf"]),
        "failed_bootstraps": prepared["failed_bootstraps"],
        "observed_support_A": result["observed_support"],
        "projection": prepared["projection_summary"],
        "inputs": provenance,
        "reference_sha256": hashlib.sha256(
            Path(args.path_reference).read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure = plot_diagnostics(
        result, output_prefix=f"{prepared['projection_method']} path progress SE"
    )
    figure.savefig(output / "diagnostics_1d.png", dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    _quality_plot(prepared, result, output)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "quality_issues",
                    "lengthscale_A",
                    "loo_rms",
                    "barrier",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colvar-dir", required=True)
    parser.add_argument("--kappa-dir", required=True)
    parser.add_argument(
        "--path-reference", required=True, help="Ordered two-column HH/RELZ path in Å"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument(
        "--target-normal-kappa",
        type=float,
        default=0.0,
        help="Explicit alternative target: common harmonic distance-to-path restraint in eV/Å²; default 0 unrestrained",
    )
    parser.add_argument("--bins", type=int)
    parser.add_argument(
        "--bin-edges",
        help="Optional explicit nonuniform arclength bin-edge file; enables local endpoint refinement",
    )
    parser.add_argument(
        "--projection-method", choices=("soft", "polyline"), default="soft"
    )
    parser.add_argument(
        "--smoothing-width",
        type=float,
        help="Soft path-coordinate width in Å; default half the median reference spacing",
    )
    parser.add_argument(
        "--reweight-cache",
        help="Optional SHA-keyed cache of verified block-bootstrap window normalisations",
    )
    parser.add_argument(
        "--s-range",
        nargs=2,
        type=float,
        help="Optional sampled arclength range, including negative/extended endpoint tails; empty bins still fail",
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--block-size",
        type=int,
        default=1000,
        help="Original stored samples, not thinned samples",
    )
    parser.add_argument("--bootstraps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument("--reactant-interval", nargs=2, type=float)
    parser.add_argument("--transition-interval", nargs=2, type=float)
    parser.add_argument(
        "--max-barrier-std",
        type=float,
        default=0.05,
        help="Reporting target in eV; never used to shrink errors",
    )
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, np.linalg.LinAlgError) as error:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps(
                {"status": "FAILED", "reason": str(error), "settings": vars(args)},
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
