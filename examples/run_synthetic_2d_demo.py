"""Synthetic 2D GPR umbrella-integration demo.

Reconstructs a known double-Gaussian PMF from biased windows and writes a
PMF + uncertainty figure. Mirrors the H-H x rel-z desorption setup but uses an
analytic surface so it runs without any simulation data.

    python examples/run_synthetic_2d_demo.py
"""
import numpy as np

from gpr_umbrella import reconstruct_pmf_2d

# Reuse the synthetic builder from the test suite.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_gpr_2d_synthetic import build_synthetic_data, true_pmf  # noqa: E402


def main():
    data = build_synthetic_data(n_per_side=6, n_samples=4000)
    res = reconstruct_pmf_2d(
        data=data,
        cv_names=("x", "y"),
        cv_units=("u", "u"),
        output_dir="outputs_2d_demo",
        output_prefix="synthetic2d",
        verbose=True,
    )

    truth = true_pmf(res["GX"], res["GY"])
    truth -= truth.min()
    pred = res["pmf"] - res["pmf"].min()
    rmse = np.sqrt(np.mean((pred[5:-5, 5:-5] - truth[5:-5, 5:-5]) ** 2))
    print(f"\ninterior PMF RMSE vs analytic truth: {rmse:.4f}")
    print("Figure + PMF table written under outputs_2d_demo/")


if __name__ == "__main__":
    main()
