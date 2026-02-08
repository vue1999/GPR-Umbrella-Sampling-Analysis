# GPR Umbrella Integration (1D)

Gaussian process regression (GPR) based umbrella integration for 1D PLUMED umbrella sampling outputs. This package implements the method in
"Free-energy surface reconstruction from umbrella samples using Gaussian process regression" and is tailored to PLUMED `window_*.ui_dat` files.

## Features
- Computes mean force and uncertainty from umbrella windows
- Estimates autocorrelation time for effective sample size
- Optimizes GP hyperparameters by marginal likelihood
- Produces PMF and derivative predictions with uncertainties
- Generates a multi-panel diagnostics figure
- **Reads raw PLUMED COLVAR files directly** (no preprocessing required)
- Configurable units (energy, collective-variable axis)

## Installation

```bash
pip install -e .
```

## Quick Start

### From COLVAR files (recommended)

Supply a directory of `COLVAR_window_*.dat` files together with force-constant
information.  For a single kappa shared across all windows, provide a centres
file:

```bash
gpr-umbrella --colvar-dir COLVAR --kappa 24.305 \
             --centers window_centers.txt --cv-unit nm
```

Or from Python:

```python
from gpr_umbrella_1d import gpr_umbrella_integration

results = gpr_umbrella_integration(
    colvar_dir="COLVAR",
    kappa=24.305,                    # eV/nm^2
    centers="window_centers.txt",    # one centre per line
    cv_unit="nm",
    energy_unit="eV",
    output_dir="outputs",
    output_prefix="my_system",
    show=False,
)
```

If each window has its own kappa, point to a directory of per-window files
instead:

```bash
gpr-umbrella --colvar-dir COLVAR --kappa-dir window_kappa/
```

### From preprocessed window files

```bash
gpr-umbrella --data-folder /path/to/processed_data
```

```python
results = gpr_umbrella_integration(
    data_folder="/path/to/processed_data",
    output_dir="outputs",
    output_prefix="my_system",
    show=False,
)
```

## Input Data Formats

### COLVAR files (preferred)
Standard PLUMED `COLVAR_window_*.dat` files with columns for `time` and the
collective variable.  The CV column index can be set with `--cv-col` (default 1).

Force constant and window centres can be provided in two ways:

1. **Single kappa** (`--kappa`) + a centres file (`--centers`) listing one
   centre per line.
2. **Per-window kappa directory** (`--kappa-dir`) containing
   `window_centers_kappa_*.txt` files with `centre, kappa` on each data line.

By default, kappa is expected in eV/CV_unit².  Pass `--kappa-kj` if values are
in kJ/mol/CV_unit² (PLUMED convention).

### window_*.ui_dat files (legacy)
Each file must contain at least three numeric columns:
1. reaction coordinate samples
2. window centre (constant per file)
3. force constant kappa in kJ/mol/CV_unit² (constant per file)

## Outputs
- `*_pmf_gpr.dat`: reaction coordinate, PMF mean, PMF uncertainty
- `*_deriv_gpr.dat`: reaction coordinate, mean force, mean force uncertainty
- `*_gpr_analysis.png`: diagnostics figure

## Example

See `examples/fe110_h_escape/` for a complete example using COLVAR data from
an Fe(110) H-escape umbrella sampling simulation:

```bash
cd examples/fe110_h_escape
python run_gpr.py
```

## Citation
If you use this tool in published work, please cite the original GPR umbrella integration paper and acknowledge this implementation.

## License
MIT
