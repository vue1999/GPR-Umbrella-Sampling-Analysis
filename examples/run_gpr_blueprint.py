"""Template example for GPR umbrella integration.

Copy this file and edit the paths / parameters to match your data.
Neither option below is runnable as-is -- update the paths first.
"""
from gpr_umbrella_1d import gpr_umbrella_integration

# --- Option A: preprocessed window_*.ui_dat files ---
# DATA_FOLDER = "/path/to/processed_data"
# results = gpr_umbrella_integration(
#     data_folder=DATA_FOLDER,
#     output_dir="outputs",
#     output_prefix="example_run",
#     show=False,
# )

# --- Option B: raw PLUMED COLVAR files + single kappa ---
# COLVAR_DIR = "/path/to/COLVAR"
# KAPPA = 24.305          # eV/nm^2 (set kappa_in_kj_per_mol=True if in kJ/mol)
# CENTERS_FILE = "/path/to/window_centers.txt"  # one centre per line
#
# results = gpr_umbrella_integration(
#     colvar_dir=COLVAR_DIR,
#     kappa=KAPPA,
#     centers=CENTERS_FILE,
#     cv_unit="nm",
#     energy_unit="eV",
#     output_dir="outputs",
#     output_prefix="example_run",
#     show=False,
# )
#
# print("PMF file:", results["pmf_path"])
# print("Derivative file:", results["deriv_path"])
# print("Figure:", results["figure_path"])
