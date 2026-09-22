"""Synthetic 2D GPR umbrella-integration demo.

Reconstructs a known double-Gaussian PMF from biased windows and writes a
PMF, uncertainty and diagnostics figures. Uses an analytic surface so it
runs without simulation data or imports from the test suite.

    python examples/run_synthetic_2d_demo.py
"""
import numpy as np

from gpr_umbrella import reconstruct_pmf_2d

def true_pmf(x, y):
    """Two basins separated by a barrier; smooth and well-behaved."""
    return (-1.5 * np.exp(-((x + 1.0) ** 2 + (y + 0.5) ** 2) / 0.8)
            - 1.2 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) / 0.8))


def true_grad(x, y):
    g1 = -1.5 * np.exp(-((x + 1.0) ** 2 + (y + 0.5) ** 2) / 0.8)
    g2 = -1.2 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) / 0.8)
    dx = g1 * (-2 * (x + 1.0) / 0.8) + g2 * (-2 * (x - 1.0) / 0.8)
    dy = g1 * (-2 * (y + 0.5) / 0.8) + g2 * (-2 * (y - 0.6) / 0.8)
    return np.array([dx, dy])


def build_synthetic_data(n_per_side=6, n_samples=4000, kappa=(20.0, 20.0),
                         seed=0):
    rng = np.random.default_rng(seed)
    cx = np.linspace(-2.2, 2.2, n_per_side)
    cy = np.linspace(-2.0, 2.0, n_per_side)
    kappa = np.asarray(kappa, dtype=float)

    centers, kappas, positions = [], [], []
    for x0 in cx:
        for y0 in cy:
            c = np.array([x0, y0])
            # Sample around the biased minimum: with a stiff harmonic bias the
            # window distribution is ~ Gaussian centred near c, shifted by the
            # local PMF gradient (linear-response): mean ~= c - grad/kappa.
            shift = true_grad(x0, y0) / kappa
            mean = c - shift
            cov = np.diag(1.0 / kappa)        # width set by the restraint
            samp = rng.multivariate_normal(mean, cov, size=n_samples)
            centers.append(c)
            kappas.append(kappa.copy())
            positions.append(samp)

    return {
        "window_files": [f"w{i}" for i in range(len(centers))],
        "centers": np.array(centers),
        "kappa": np.array(kappas),
        "all_positions": positions,
    }



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
