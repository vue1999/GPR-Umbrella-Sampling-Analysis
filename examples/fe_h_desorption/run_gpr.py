"""Example: H desorption from an Fe surface (PMF along the relative-z CV).

Umbrella sampling with 36 windows and per-window force constants.  This uses
the COLVAR-based loader together with a ``window_kappa`` directory, where each
``window_centers_kappa_*.txt`` file holds one ``center, kappa`` line.

The COLVAR trajectories shipped here are subsampled (every 20th frame) to keep
the repository light; the resulting PMF is essentially identical to the
full-resolution run.
"""
from pathlib import Path
from gpr_umbrella_1d import gpr_umbrella_integration

HERE = Path(__file__).resolve().parent

results = gpr_umbrella_integration(
    colvar_dir=str(HERE / "COLVAR"),
    kappa_dir=str(HERE / "window_kappa"),
    cv_col=1,
    cv_unit="nm",
    energy_unit="eV",
    output_dir=str(HERE / "outputs"),
    output_prefix="fe_h_desorption",
    show=False,
)

print("\n--- Results ---")
print(f"PMF file:        {results['pmf_path']}")
print(f"Derivative file: {results['deriv_path']}")
print(f"Figure:          {results['figure_path']}")
