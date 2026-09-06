"""Check finite-bootstrap covariance regularisation without changing any fit.

This is an audit, not a search for whichever shrinkage makes LOO look good.
The production value remains 0.02 and all alternatives are reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gpr_umbrella.acceptance import compare_profiles
from gpr_umbrella.robust_1d import fit_linear_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    reports = []
    for root in sorted(output.glob("s*")):
        filename = root / "base/fit_arrays.npz"
        if not filename.is_file():
            continue
        data = np.load(filename)
        table = np.loadtxt(root / "base/hyperparameters.tsv")
        covariance = np.cov(data["bootstrap_pmf"] @ data["operator"].T, rowvar=False)
        profiles = []
        variants = []
        for shrinkage in (0.0, 0.02, 0.05, 0.1):
            noise = (1 - shrinkage) * covariance + shrinkage * np.diag(
                np.diag(covariance)
            )
            fit = fit_linear_observations(
                data["nodes"],
                data["operator"],
                data["values"],
                noise,
                data["grid"],
                resolution=table[:, 0].min(),
                lengthscales=np.unique(table[:, 0]),
                amplitudes=np.unique(table[:, 1]),
            )
            profiles.append(fit)
            variants.append(
                {
                    "covariance_shrinkage": shrinkage,
                    "loo_rms": float(np.sqrt(np.mean(fit["loo_z"] ** 2))),
                    "loo_max_abs": float(np.max(np.abs(fit["loo_z"]))),
                    "quality_issues": fit["quality_issues"],
                }
            )
        report = {
            "campaign": root.name,
            "production_shrinkage": 0.02,
            "sensitivity": compare_profiles(profiles),
            "variants": variants,
            "note": "Audit only; no alternative selected to improve diagnostics",
        }
        (root / "covariance_sensitivity.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    (output / "covariance_sensitivity.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
