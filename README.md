# GPR Umbrella Integration (1D)

Gaussian process regression (GPR) based umbrella integration for 1D PLUMED umbrella sampling outputs. This package implements the method described in

> T. Stecher, N. Bernstein, and G. Csányi, "Free Energy Surface Reconstruction from Umbrella Samples Using Gaussian Process Regression," *J. Chem. Theory Comput.* **2014**, *10* (9), 4079–4097. [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v)

and is tailored to PLUMED `window_*.ui_dat` files. Specifically, it implements the
gradient-based reconstruction variant referred to as **GPR(d)** in that paper:
mean forces are estimated per umbrella window (Sec. 2.3, eq 15), their statistical
noise is propagated into the likelihood (Sec. 4, eq 37), and the free-energy profile
is reconstructed by GPR on the derivative observations using a (periodic) squared-exponential
kernel (Sec. 4–4.1).

## Features

- Computes mean force and uncertainty from umbrella windows
- Estimates autocorrelation time for effective sample size
- Optimizes GP hyperparameters by marginal likelihood
- Produces PMF and derivative predictions with uncertainties
- Generates a multi-panel diagnostics figure
- Reads raw PLUMED COLVAR files directly (no preprocessing required)
- Configurable units (energy, collective-variable axis)

## Installation

```bash
pip install -e .
```

## Quick Start

### From COLVAR files

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

### COLVAR files

Standard PLUMED `COLVAR_window_*.dat` files with columns for `time` and the
collective variable.  The CV column index can be set with `--cv-col` (default 1).

Force constant and window centres can be provided in two ways:

1. **Single kappa** (`--kappa`) + a centres file (`--centers`) listing one
   centre per line.
2. **Per-window kappa directory** (`--kappa-dir`) containing
   `window_centers_kappa_*.txt` files with `centre, kappa` on each data line.

By default, kappa is expected in eV/CV_unit².  Pass `--kappa-kj` if values are
in kJ/mol/CV_unit² (PLUMED convention).

### window_*.ui_dat files

Each file must contain at least three numeric columns:

1. reaction coordinate samples
2. window centre (constant per file)
3. force constant kappa in kJ/mol/CV_unit² (constant per file)

## Outputs

- `*_pmf_gpr.dat`: reaction coordinate, PMF mean, PMF uncertainty
- `*_deriv_gpr.dat`: reaction coordinate, mean force, mean force uncertainty
- `*_gpr_analysis.png`: diagnostics figure

## Example

See `examples/fe_h_desorption/` for a complete example using COLVAR data from
an Fe-surface H-desorption umbrella sampling simulation (36 windows with
per-window force constants):

```bash
cd examples/fe_h_desorption
python run_gpr.py
```

## 2D umbrella integration (`multiD` branch)

`gpr_umbrella_1d.gpr2d` extends the same scheme to a separable-bias 2-CV setup
(e.g. H–H distance × relative-z desorption umbrella sampling). Each window
applies one harmonic restraint per CV, so it yields a 2-vector mean-force
estimate `kappa_d * (center_d - <x_d>)`. A Gaussian process with a separable
squared-exponential kernel is conditioned on this **gradient field** (a
derivative-observation GP, using the same kernel derivatives as the 1D code) to
reconstruct the scalar 2D PMF up to an additive constant, with LOO-calibrated
uncertainty.

Expected per-window inputs (written by `desorption_2dUS/ui_md_umbrella_2d.py`):

- `COLVAR_window_<i>.dat` with columns `time, cv0, cv1` (CV columns set by
  `--cv-cols`, default `1 2`).
- `window_centers_kappa_<i>.txt` with one data line `c0, c1, kappa0, kappa1`.

```bash
# CLI
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
                --cv-names hh relz --cv-units A A

# or from Python
from gpr_umbrella_1d import gpr_umbrella_integration_2d
res = gpr_umbrella_integration_2d(colvar_dir="COLVAR", kappa_dir="COLVAR")
```

Outputs: `*_pmf2d_gpr.dat` (cv0, cv1, PMF, sigma on a grid),
`*_pmf2d_gpr.png` (PMF contour + uncertainty, window centres overlaid) and
`*_diagnostics2d.png` — an 8-panel sampling/fit diagnostics figure (PMF and
calibrated uncertainty; window drift centre→mean, mean-force field, and
autocorrelation time; window-overlap ellipses, per-observation LOO z-scores,
and the LOO calibration histogram). Pass `plot_diagnostics=False`
(CLI: `--no-diagnostics`) to skip it.

Run the self-contained synthetic check (no simulation data needed):

```bash
python examples/run_synthetic_2d_demo.py   # reconstructs a known 2D PMF
```

## Citation

This implementation is based on the method introduced in:

> T. Stecher, N. Bernstein, and G. Csányi, "Free Energy Surface Reconstruction from Umbrella Samples Using Gaussian Process Regression," *J. Chem. Theory Comput.* **2014**, *10* (9), 4079–4097. [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v)

If you use this tool in published work, please cite the paper above and acknowledge
this implementation.

A closely related follow-up paper extends the same GPR-based reconstruction to combined
exploration + sampling (metadynamics biasing with an instantaneous-collective-force
gradient estimator, reconstructed with GPR):

> L. Mones, N. Bernstein, and G. Csányi, "Exploration, Sampling, And Reconstruction of Free Energy Surfaces with Gaussian Process Regression," *J. Chem. Theory Comput.* **2016**, *12* (10), 5100–5110. [doi:10.1021/acs.jctc.6b00553](https://doi.org/10.1021/acs.jctc.6b00553)
