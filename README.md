# GPR Umbrella Integration

Gaussian-process regression (GPR) umbrella integration for one- and
two-dimensional PLUMED umbrella-sampling outputs. This package implements the method described in

> T. Stecher, N. Bernstein, and G. Csányi, "Free Energy Surface Reconstruction from Umbrella Samples Using Gaussian Process Regression," *J. Chem. Theory Comput.* **2014**, *10* (9), 4079–4097. [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v)

and is tailored to PLUMED `window_*.ui_dat` files. Specifically, it implements the
gradient-based reconstruction variant referred to as **GPR(d)** in that paper:
mean forces are estimated per umbrella window (Sec. 2.3, eq 15), their statistical
noise is propagated into the likelihood (Sec. 4, eq 37), and the free-energy profile
is reconstructed by GPR on the derivative observations using a squared-exponential
kernel (Sec. 4–4.1).

## Features

- Computes mean force and uncertainty from umbrella windows
- Estimates autocorrelation time for effective sample size
- Optimizes GP hyperparameters by marginal likelihood
- Produces PMF and derivative predictions with uncertainties
- Retains cross-component sampling covariance in two-CV windows
- Propagates GP covariance into relative barriers and path marginals
- Finds lowest-barrier grid paths and transverse Boltzmann marginals
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
from gpr_umbrella import reconstruct_pmf_1d

results = reconstruct_pmf_1d(
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
results = reconstruct_pmf_1d(
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

By default, kappa is expected in `energy_unit/CV_unit²` (eV/CV_unit² with the
defaults). Pass `--kappa-kj` if values are in kJ/mol/CV_unit² (PLUMED
convention); they are then converted to the selected numerical `energy_unit`.

### window_*.ui_dat files

Each file must contain at least three numeric columns:

1. reaction coordinate samples
2. window centre (constant per file)
3. force constant kappa in kJ/mol/CV_unit² (constant per file)

## Outputs

- `*_pmf_1d.dat`: reaction coordinate, PMF mean, PMF uncertainty
- `*_mean_force_1d.dat`: reaction coordinate, mean force, mean force uncertainty
- `*_diagnostics_1d.png`: diagnostics figure

## Example

See `examples/fe_h_desorption/` for a complete example using COLVAR data from
an Fe-surface H-desorption umbrella sampling simulation (36 windows with
per-window force constants):

```bash
cd examples/fe_h_desorption
python run_gpr.py
```

## 2D umbrella integration

`gpr_umbrella.integration_2d` extends the same scheme to a separable-bias 2-CV setup
(e.g. H–H distance × relative-z desorption umbrella sampling). Each window
applies one harmonic restraint per CV, so it yields a 2-vector mean-force
estimate `kappa_d * (center_d - <x_d>)`. A Gaussian process with a separable
squared-exponential kernel is conditioned on this **gradient field** (a
derivative-observation GP, using the same kernel derivatives as the 1D code) to
reconstruct the scalar 2D PMF up to an additive constant, with LOO-calibrated
uncertainty.

The two CV components from one window are treated as a correlated vector
observation. Multivariate batch means estimate their full covariance, which is
propagated through the force constants and retained as a 2x2 likelihood block.
Raw GP uncertainty and the LOO-scaled uncertainty are both reported.

Expected per-window inputs (written by `desorption_2dUS/ui_md_umbrella_2d.py`):

- `COLVAR_window_<i>.dat` with columns `time, cv0, cv1` (CV columns set by
  `--cv-cols`, default `1 2`).
- `window_centers_kappa_<i>.txt` with one data line `c0, c1, kappa0, kappa1`.

```bash
# CLI
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
                --cv-names hh relz --cv-units A A

# or from Python
from gpr_umbrella import reconstruct_pmf_2d
res = reconstruct_pmf_2d(colvar_dir="COLVAR", kappa_dir="COLVAR")
```

Outputs: `*_pmf_2d.dat` (cv0, cv1, PMF, raw sigma, calibrated sigma,
sampling-support flag, and path-valid flag on a grid),
`*_pmf_2d.png` (PMF contour + uncertainty, window centres overlaid) and
`*_diagnostics_2d.png` — an 8-panel sampling/fit diagnostics figure (PMF and
calibrated uncertainty; window drift centre→mean, mean-force field, and
autocorrelation time; window-overlap ellipses, per-observation LOO z-scores,
and the LOO calibration histogram). Pass `plot_diagnostics=False`
(CLI: `--no-diagnostics`) to skip it.

The 2D colour scale and path-valid region use one shared, observation-anchored
policy:

- Geometric support is the union of circles or axis-aligned ellipses centred at
  sampled window means. Their semiaxes are `support_radius * lengthscale`.
- `support_radius=0.5` is the default. It is deliberately local: it does not
  fill a convex hull or bridge disconnected groups of windows.
- The normal PMF colour interval contains the complete PMF range evaluated at
  window means plus a fixed 25% margin.
- Cells outside geometric support are blank. Supported PMF values outside the
  normal interval are red and excluded from path analysis.
- The path-valid mask is finite PMF ∩ geometric support ∩ normal PMF interval.
  No additional uncertainty cutoff is applied.

Use `--support-radius R` for data with different window spacing. The existing
`--no-restrict-to-sampled-support` escape hatch restores rectangular geometric
support, while the non-red PMF-range check remains active.

Run the self-contained synthetic check with:

```bash
python examples/run_synthetic_2d_demo.py
```

### Path analysis and path-aligned marginal PMFs

#### At a glance

- **Free search:** `--path-mode search --path-endpoints X0 Y0 X1 Y1`
  - Both endpoints are required.
  - Each coordinate is snapped only to its nearest grid cell.
  - Invalid, coincident, or disconnected endpoint cells are rejected.
- **Reference corridor:** `--path-mode corridor --path-reference PATH
  --path-corridor-radius R`
  - The first and last reference points define the endpoints.
  - Search is limited to the path-valid cells within `R` of the reference.
- **Fixed trajectory:** `--path-mode fixed --path-reference PATH`
  - No graph search is performed.
  - The densified trajectory is evaluated directly and rejected if any part
    leaves the path-valid region.
- **Path metric:** fitted GP lengthscales by default; override with
  `--path-metric-scales SCALE0 SCALE1`.
- **Optional 1D marginal:** add `--path-aligned-marginal --thermal-energy KBT`.
- **Reported barrier:** `max(path PMF) - min(path PMF)`, with uncertainty from
  the full GP posterior covariance between those extrema.

#### Free search

```bash
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-mode search \
    --path-endpoints X0 Y0 X1 Y1
```

Search and corridor modes find the narrowest PMF interval `[F_low, F_high]`
that contains both endpoints and connects them through path-valid cells. This
exactly minimizes `F_high - F_low` on the finite 8-neighbour grid. It is not a
guarantee for the continuous GPR surface and is not a string or NEB refinement.

Within that exact interval, one deterministic representative path minimizes
lengthscale-scaled length with a fixed gradient-alignment penalty. Motion
perpendicular to a reliable local GP gradient is penalized; the penalty fades
where the predicted gradient is weak. There is no user-facing gradient weight
or uncertainty/UCB path-selection parameter.

Python usage follows the same contract:

```python
from gpr_umbrella import find_lowest_barrier_path

path = find_lowest_barrier_path(
    result,
    endpoints=((x_start, y_start), (x_end, y_end)),
)
```

#### Reference trajectories

The reference file may contain `#` comments; its first two columns are the two
CV coordinates.

```bash
# Search only within a dimensionless path-metric radius of the reference.
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-mode corridor \
    --path-reference neb_xy.dat --path-corridor-radius 0.5

# Evaluate the reference exactly, with no graph search.
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-mode fixed \
    --path-reference neb_xy.dat
```

The corridor radius is dimensionless in the path metric. With isotropic metric
scale 0.5 A, for example, radius 0.5 corresponds to 0.25 A. Disconnected
corridors and invalid fixed trajectories fail explicitly instead of crossing,
clipping, or rerouting through untrusted cells.

#### Path-aligned 1D marginal

```bash
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-mode search \
    --path-endpoints X0 Y0 X1 Y1 \
    --path-aligned-marginal --thermal-energy 0.02585
```

`--thermal-energy` is `kBT` in `energy_unit`. The path coordinate and transverse
coordinate are dimensionless metric arclengths. `--perpendicular-points` and
`--perpendicular-width` control the transverse quadrature. Outputs are written
to `*_lowest_barrier_path.dat`, `*_lowest_barrier_path.png`, and
`*_path_aligned_pmf_1d.dat`.

### Path implementation structure

- `support.py` owns sampled-support geometry and the shared display/path-valid
  policy.
- `path_graph.py` owns exact finite-grid minimum-range search and the fixed
  gradient-aware tie-break.
- `trajectory.py` validates, densifies, and measures reference trajectories.
- `path_profile.py` evaluates path PMFs and covariance-aware differences.
- `pathways.py` orchestrates modes and optional transverse marginalization.

## Citation

This implementation is based on the method introduced in:

> T. Stecher, N. Bernstein, and G. Csányi, "Free Energy Surface Reconstruction from Umbrella Samples Using Gaussian Process Regression," *J. Chem. Theory Comput.* **2014**, *10* (9), 4079–4097. [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v)

If you use this tool in published work, please cite the paper above and acknowledge
this implementation.

A closely related follow-up paper extends the same GPR-based reconstruction to combined
exploration + sampling (metadynamics biasing with an instantaneous-collective-force
gradient estimator, reconstructed with GPR):

> L. Mones, N. Bernstein, and G. Csányi, "Exploration, Sampling, And Reconstruction of Free Energy Surfaces with Gaussian Process Regression," *J. Chem. Theory Comput.* **2016**, *12* (10), 5100–5110. [doi:10.1021/acs.jctc.6b00553](https://doi.org/10.1021/acs.jctc.6b00553)
