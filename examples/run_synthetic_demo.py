"""Run the GPR pipeline on a synthetic harmonic PMF.

Generates the same data used by ``tests/test_gpr_synthetic.py`` but writes
a full diagnostics figure so the plotting layout can be inspected on a
dataset where the answer is known.

The true PMF is F(x) = 1/2 a x^2 with a = 2 eV/nm^2, sampled with
umbrella windows of kappa = 50 eV/nm^2 centred uniformly in [-2, 2] nm.
"""
from pathlib import Path

import numpy as np

from gpr_umbrella import reconstruct_pmf_1d


HERE = Path(__file__).resolve().parent
OUT = HERE / "synthetic_demo"
COLVAR_DIR = OUT / "COLVAR"
OUT.mkdir(exist_ok=True)
COLVAR_DIR.mkdir(exist_ok=True)


def make_data(n_windows: int = 15, n_samples: int = 5000, seed: int = 12345):
    a = 2.0
    kappa = 50.0
    kappa_eff = kappa + a

    centers = np.linspace(-2.0, 2.0, n_windows)
    np.savetxt(OUT / "window_centers.txt", centers, fmt="%.6f")

    rng = np.random.default_rng(seed)
    for i, c in enumerate(centers):
        x_mean = kappa * c / kappa_eff
        x_std = np.sqrt(1.0 / kappa_eff)
        positions = rng.normal(x_mean, x_std, n_samples)
        time = np.arange(n_samples, dtype=float) * 0.001
        np.savetxt(
            COLVAR_DIR / f"COLVAR_window_{i}.dat",
            np.column_stack([time, positions]),
            header="#! FIELDS time x", comments="", fmt="%.6f",
        )

    return kappa, OUT / "window_centers.txt"


if __name__ == "__main__":
    kappa, centers_file = make_data()
    results = reconstruct_pmf_1d(
        colvar_dir=str(COLVAR_DIR),
        kappa=kappa,
        centers=str(centers_file),
        cv_unit="nm",
        energy_unit="eV",
        output_dir=str(OUT),
        output_prefix="synthetic_harmonic",
        show=False,
    )
    print("\nFigure:", results["figure_path"])
