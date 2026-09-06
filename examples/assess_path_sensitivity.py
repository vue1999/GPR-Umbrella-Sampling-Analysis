"""Assess completed suites, refitting cached observations under wider GP priors."""
import argparse
import json
from pathlib import Path

import numpy as np

from gpr_umbrella.acceptance import compare_profiles, diagnostic_status
from gpr_umbrella.campaign_audit import audit_campaign
from gpr_umbrella.robust_1d import fit_linear_observations, profile_landmarks
from gpr_umbrella.cli_path import load_inputs
from gpr_umbrella.arclength import project_arclength
from gpr_umbrella.projected_inputs import unweighted_block_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    all_cases = []
    for root in sorted(output.glob("s*")):
        if not root.is_dir():
            continue
        profiles = []; summaries = []; issues = []; coordinate_profiles = []
        source = Path(args.data_root) / root.name
        trajectories, _, _, vertices, _, _ = load_inputs(source/"COLVAR", source/"window_kappa", source/"neb_HH_RELZ_30img.dat")
        position_cache = {}
        base_summary_path = root/"base/summary.json"
        if not base_summary_path.is_file():
            print(f"Missing baseline for {root.name}", flush=True)
            continue
        smooth = json.loads(base_summary_path.read_text()).get("projection_method") == "soft"
        variants = ["base", "long_blocks", "fine_bins", "fine_stride"]
        if smooth:
            variants += ["projection_halfwidth", "projection_doublewidth"]
        for variant in variants:
            if not (root / variant / "summary.json").is_file():
                issues.append(f"missing_or_failed_variant_{variant}")
                continue
            summary = json.loads((root / variant / "summary.json").read_text())
            data = np.load(root / variant / "fit_arrays.npz")
            profile = {"x_star": data["grid"], "pmf_mean": data["pmf"],
                       "pmf_std": np.sqrt(np.maximum(np.diag(data["covariance"]), 0))}
            (coordinate_profiles if variant.startswith("projection_") else profiles).append(profile)
            summary["variant"] = variant
            summaries.append(summary)
            issues.extend(i for i in summary["quality_issues"] if i not in ("sensitivity_suite_required", "low_effective_block_support"))
            key = (summary.get("projection_method", "polyline"), summary.get("smoothing_width_A"))
            if key not in position_cache:
                position_cache[key] = [project_arclength(q, vertices, method=key[0], smoothing_width=key[1])["s"] for q in trajectories]
            positions = position_cache[key]
            edges = data["bin_edges"] if "bin_edges" in data else np.linspace(0, np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum(), len(data["nodes"])+1)
            baseline = unweighted_block_support(positions, edges, block_size=summary["settings"]["block_size"], stride=summary["settings"]["stride"])
            ratio = data["block_ess"] / baseline
            summary["minimum_relative_effective_blocks"] = float(ratio.min())
            if data["block_ess"].min() < 4 or ratio.min() < .25:
                issues.append("low_effective_block_support")
        if not profiles:
            continue
        base = np.load(root / "base/fit_arrays.npz")
        table = np.loadtxt(root / "base/hyperparameters.tsv")
        lower = table[:, 0].min()
        base_refit = fit_linear_observations(base["nodes"], base["operator"], base["values"],
                                            base["noise_covariance"], base["grid"], resolution=lower)
        hypers = []
        for factor in (1.5, 2.):
            fit = fit_linear_observations(base["nodes"], base["operator"], base["values"],
                     base["noise_covariance"], base["grid"], resolution=lower,
                     lengthscales=np.geomspace(lower * factor, table[:, 0].max() * 2, 35),
                     amplitudes=np.geomspace(table[:, 1].min()/3, table[:, 1].max()*3, 25))
            profiles.append(fit)
            hypers.append({"lengthscale_lower_factor": factor, "quality_issues": fit["quality_issues"],
                           "lengthscale": fit["lengthscale"], "sigma_f": fit["sigma_f"]})
            issues.extend(fit["quality_issues"])
        comparison = compare_profiles(profiles)
        issues.extend(comparison["quality_issues"])
        coordinate_comparison = compare_profiles([profiles[0], *coordinate_profiles]) if coordinate_profiles else None
        if coordinate_comparison and coordinate_comparison["maximum_profile_spread"] > .05:
            issues.append("coordinate_definition_sensitivity")
        source = Path(args.data_root) / root.name
        audit = audit_campaign(source / "source_campaign.json", source / "neb_path_30_images.xyz")
        issues.extend(audit["quality_issues"])
        issues = sorted(set(issues))
        report = {"campaign": root.name, "status": diagnostic_status(issues),
                  "quality_issues": issues, "sensitivity": comparison,
                  "coordinate_sensitivity": coordinate_comparison,
                  "prior_sensitivity": hypers, "campaign_audit": audit,
                  "profile_landmarks": profile_landmarks(base_refit),
                  "support_rule": "at least 4 effective blocks and 25% of unweighted block ESS in every bin",
                  "variants": [{k: s[k] for k in ("variant", "lengthscale_A", "loo_rms", "loo_max_abs", "minimum_effective_blocks", "minimum_relative_effective_blocks", "half_profile_difference_range_eV")} for s in summaries]}
        (root / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
        all_cases.append(report)
        print(json.dumps(report, indent=2), flush=True)
    (output / "acceptance.json").write_text(json.dumps(all_cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
