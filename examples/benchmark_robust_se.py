"""Repeated-noise benchmark of the GP stage against a known 1 eV barrier.

This tests conditional GP coverage and hyperparameter stability, not MD
equilibration or MBAR reweighting (covered separately by trajectory tests).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gpr_umbrella.robust_1d import barrier_posterior, fit_linear_observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    nodes = np.linspace(-0.4, 3.4, 35)
    grid = np.linspace(nodes[0], nodes[-1], 201)
    operator = np.diff(np.eye(len(nodes)), axis=0) / np.diff(nodes)[:, None]
    true_f = np.sin(np.pi * nodes / 3) ** 2
    covariance_f = 0.015**2 * np.exp(-np.abs(nodes[:, None] - nodes) / 0.2)
    noise = operator @ covariance_f @ operator.T
    rng = np.random.default_rng(827)
    records = []
    for repeat in range(args.repeats):
        observations = operator @ (
            true_f + rng.multivariate_normal(np.zeros(len(nodes)), covariance_f)
        )
        fit = fit_linear_observations(nodes, operator, observations, noise, grid)
        barrier = barrier_posterior(
            fit, (-0.3, 0.3), (1.2, 1.8), draws=1000, seed=100 + repeat
        )
        table = fit["hyperparameter_table"]
        broad = fit_linear_observations(
            nodes,
            operator,
            observations,
            noise,
            grid,
            lengthscales=np.geomspace(
                table[:, 0].min() * 1.5, table[:, 0].max() * 2, 35
            ),
            amplitudes=np.geomspace(table[:, 1].min() / 3, table[:, 1].max() * 3, 25),
        )
        records.append(
            {
                "repeat": repeat,
                "estimated_barrier": barrier["median"],
                "sigma": barrier["std"],
                "ci95": barrier["ci95"],
                "truth_inside_ci95": barrier["ci95"][0] <= 1 <= barrier["ci95"][1],
                "max_prior_sensitivity": float(
                    np.max(np.abs(fit["pmf_mean"] - broad["pmf_mean"]))
                ),
                "loo_rms": float(np.sqrt(np.mean(fit["loo_z"] ** 2))),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    report = {
        "true_barrier_eV": 1.0,
        "repeats": args.repeats,
        "median_absolute_error_eV": float(
            np.median([abs(r["estimated_barrier"] - 1) for r in records])
        ),
        "maximum_absolute_error_eV": float(
            max(abs(r["estimated_barrier"] - 1) for r in records)
        ),
        "empirical_ci95_coverage": float(
            np.mean([r["truth_inside_ci95"] for r in records])
        ),
        "maximum_prior_sensitivity_eV": float(
            max(r["max_prior_sensitivity"] for r in records)
        ),
        "records": records,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
