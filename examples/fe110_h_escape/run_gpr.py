#!/usr/bin/env python
"""Example: Fe(110) H escape PMF from PLUMED umbrella sampling COLVAR files.

This script uses the COLVAR-based loader with a single kappa value
and a window-centres file.  It is equivalent to:

    gpr-umbrella --colvar-dir COLVAR --kappa 24.305 \
                 --centers window_centers.txt --cv-unit nm
"""
from pathlib import Path
from gpr_umbrella_1d import gpr_umbrella_integration

HERE = Path(__file__).resolve().parent

results = gpr_umbrella_integration(
    colvar_dir=str(HERE / "COLVAR"),
    kappa=24.305,                           # eV/nm^2
    centers=str(HERE / "window_centers.txt"),
    cv_unit="nm",
    energy_unit="eV",
    output_dir=str(HERE / "outputs"),
    output_prefix="fe110_h_escape",
    show=False,
)

print("\n--- Results ---")
print(f"PMF file:        {results['pmf_path']}")
print(f"Derivative file: {results['deriv_path']}")
print(f"Figure:          {results['figure_path']}")
