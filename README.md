# GPR umbrella integration

Reconstruct one- or two-dimensional free-energy surfaces from harmonic umbrella
sampling using Gaussian processes conditioned on window mean forces. The method
is based on [Stecher, Bernstein and Csányi (2014)](https://doi.org/10.1021/ct500438v).
The fitted free energy is defined up to an additive constant.

## Installation and examples

```bash
pip install -e '.[test]'
python examples/run_synthetic_demo.py
python examples/run_synthetic_2d_demo.py
pytest
```

The examples generate known analytic surfaces without simulation input. The
`examples/fe_h_desorption/` directory also contains a complete 1D COLVAR example.
The package requires NumPy, SciPy and Matplotlib; pytest is a test dependency.

## 1D reconstruction

```python
from gpr_umbrella import reconstruct_pmf_1d

result = reconstruct_pmf_1d(
    colvar_dir="COLVAR", kappa=24.305, centers="window_centers.txt",
    cv_unit="nm", energy_unit="eV", output_dir="outputs", show=False,
)
```

Use `kappa_dir="window_kappa"` instead of `kappa`/`centers` for per-window
restraints, or `data_folder="processed"` for `window_*.ui_dat` files.

```bash
gpr-umbrella --colvar-dir COLVAR --kappa 24.305 \
    --centers window_centers.txt --cv-unit nm
```

COLVAR files are named `COLVAR_window_<i>.dat`; the CV column defaults to index
1 after time. Per-window `window_centers_kappa_<i>.txt` files contain a centre
and force constant. Processed `.ui_dat` files contain sample, centre and kappa;
their kappa is interpreted in kJ/mol/CV². For raw COLVAR inputs, force constants
use the requested energy unit unless `kappa_in_kj_per_mol=True` / `--kappa-kj`.

Outputs are `*_pmf_gpr.dat`, `*_deriv_gpr.dat` and `*_gpr_analysis.png`.
The original documented imports remain available:

```python
from gpr_umbrella_1d import (
    gpr_umbrella_integration, load_plumed_colvar_data, load_window_data,
)
```

`gpr_umbrella_integration` forwards to `reconstruct_pmf_1d`; the existing
`gpr-umbrella` command remains unchanged.

## 2D reconstruction

Each window applies one separable harmonic restraint per CV. Its free-energy
gradient estimate is `kappa * (centre - sampled_mean)`, evaluated at the sampled
mean. A squared-exponential derivative GP reconstructs a scalar surface from
these vector observations. Each coordinate has its own lengthscale and unit.

Input files:

- `COLVAR_window_<i>.dat`: time, CV0, CV1 (column indices configurable).
- `window_centers_kappa_<i>.txt`: one data row `centre0 centre1 kappa0 kappa1`.

```python
from gpr_umbrella import reconstruct_pmf_2d

surface = reconstruct_pmf_2d(
    colvar_dir="COLVAR", kappa_dir="COLVAR",
    cv_names=("distance", "height"), cv_units=("A", "A"), energy_unit="eV",
    fit_extra_noise=True, output_dir="outputs",
)
```

```bash
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --cv-names distance height --cv-units A A --fit-extra-noise
```

For in-memory input, provide `data` instead of `colvar_dir`:

```python
surface = reconstruct_pmf_2d(data={
    "centers": centers,          # shape (n_windows, 2)
    "kappa": force_constants,    # shape (n_windows, 2)
    "all_positions": trajectories,  # list of (n_frames, 2) arrays
})
```

Window means, variances and sample counts are derived from trajectories. An
optional `window_files` list supplies labels. Input arrays are not modified.
Force constants are in `energy_unit / cv_units[d]²`; gradients are in
`energy_unit / cv_units[d]`. Labels alone do not convert coordinate values.

Outputs include the PMF grid, raw and calibrated uncertainty, support masks,
fit metadata, a PMF/uncertainty figure, and the full diagnostic figure. The
latter keeps its three-row layout: PMF and uncertainty; target-to-mean drift,
mean-force field and autocorrelation; sampling ellipses, component LOO scores
and their histogram. The header records lengthscales, amplitude, covariance
blocks, additional noise and calibration. Use `--no-diagnostics` to skip it.

### Sampling covariance and additional force noise

The likelihood separates sampling covariance, optional unresolved force scatter,
and numerical regularization:

- Multivariate batch means retain the full within-window 2×2 covariance.
  `covariance_block_size=None` uses autocorrelation-adaptive blocks. An explicit
  size is in saved frames and requires at least four complete blocks/window.
  A trailing incomplete block is omitted from covariance estimation; every
  frame still contributes to the mean force.
- `fit_extra_noise=True` fits a separate additional gradient standard deviation
  for each CV. Half-normal prior scales default to the RMS observed gradients,
  including sampling variance. `extra_noise_scale=(s0, s1)` overrides them in
  energy/CV units. The default remains `fit_extra_noise=False`.
- Numerical jitter is fixed from the data, independent of fitted amplitude or
  additional noise. The objective, posterior and LOO use the same covariance.
- Optimization uses analytic derivatives and six deterministic, dimensionless
  starts. Only successful starts are eligible; complete failure raises an error.
- `fixed_lengthscale=(ell0, ell1)` and `fixed_sigma_f=value` fix parameters.
  `sigma_f_max=None` adds no amplitude cap. An explicit cap is validated and
  recorded; broad data-scaled numerical bounds remain in either case.

For example, `covariance_block_size=4000` means 10 ps if saved frames are 2.5 fs
apart. The equivalent CLI flags are `--covariance-block-size 4000`,
`--extra-noise-scale S0 S1`, `--lengthscales L0 L1` and `--sigma-f-max VALUE`.
No block time, physical lengthscale or expected barrier is hard-coded.

Results expose `gradient_noise_cov`, `extra_noise`, `observation_noise_cov`,
`numerical_jitter` and `optimization`. `*_fit_metadata.json` records all starts,
prior scales, bounds and bound hits. Whole-window LOO calibration is a post-fit
diagnostic at fixed hyperparameters. Raw and calibrated errors remain available.
Additional noise does not correct biased sampling or establish equilibration;
compare block sizes and independent samples to assess those limitations.

### Support and display

The existing shared support/display policy is preserved:

- Geometric support is the union of ellipses at sampled means, with semiaxes
  `support_radius * lengthscale`; `support_radius=0.5` by default.
- The ordinary colour interval spans PMFs at window means plus a 25% margin.
  Unsupported regions are blank; supported values outside that interval are red.
- Paths require finite PMF, geometric support and the ordinary colour interval.
  There is no additional uncertainty threshold and no bridging of support gaps.

`--no-restrict-to-sampled-support` restores rectangular geometric support but
retains the colour-interval validity check. `--support-radius R` changes the
neighborhood size. Support depends on fitted lengthscales and window spacing.

## Path analysis

Reconstruct first, then call the standalone path function:

```python
from gpr_umbrella import find_lowest_barrier_path
from gpr_umbrella.pathways import save_lowest_barrier_path
from gpr_umbrella.plotting_2d import plot_lowest_barrier_path

path = find_lowest_barrier_path(
    surface, path_mode="fixed", reference_path=reference_xy,
)
save_lowest_barrier_path(path, "path.dat")
plot_lowest_barrier_path(surface, path, output_path="path.png")
```

Three modes share profile evaluation and covariance-aware uncertainty:

| Mode | Required input | Behavior |
|---|---|---|
| `fixed` | `reference_path` | Evaluate the supplied curve; preserve vertices and endpoints; reject unsupported segments |
| `search` | `endpoints=((x0,y0),(x1,y1))` | Find the minimum-energy-range supported grid path |
| `corridor` | `reference_path`, `corridor_radius` | Run the same search within a corridor around the curve |

Search endpoints snap to the nearest grid points, which must be valid; they do
not move to local minima or umbrella centres. Requested and snapped endpoints
are recorded. Corridor mode uses the reference endpoints. The search minimizes
`max(F) - min(F)` exactly on the supported eight-neighbor graph; a deterministic
gradient-aligned tie-break selects a representative within the optimal interval.
It does not minimize only the highest energy or an uncertainty-penalized score.

The default path metric scales each CV by its fitted GP lengthscale. Override
with `metric_scale=(scale0, scale1)`; arc length and corridor radius are
dimensionless in that metric. A reference curve need not pass through windows,
but it must lie inside the supported region. No smoothing changes that curve.

Both quantities are reported explicitly in energy units:

- `energy_range = max(F) - min(F)` along the path.
- `endpoint_to_max = max(F) - F(start)` for the forward endpoint rise.

Each has `_err_raw`, `_err_calibrated` and configured `_err` values, using full
GP covariance for free-energy differences. These uncertainties condition on the
chosen path, fitted parameters and its selected extrema; they do not marginalize
over alternative paths or transition-state locations.

```bash
gpr-umbrella-2d --colvar-dir COLVAR --fit-extra-noise \
    --find-lowest-barrier-path --path-mode fixed --path-reference curve.dat
```

For search use `--path-mode search --path-endpoints X0 Y0 X1 Y1`; for a corridor
use `--path-mode corridor --path-reference curve.dat --path-corridor-radius R`.
Path tables include compact JSON metadata in their comment header.

## Migration and citation

See [MIGRATION.md](MIGRATION.md) for changes from the full experimental branch.
Projected-arclength MBAR/GPR and transverse marginalization remain preserved on
`codex/noise-aware-2d` at `2d03bc3`, outside this package's current scope.

Please cite T. Stecher, N. Bernstein and G. Csányi, *J. Chem. Theory Comput.*
**2014**, 10, 4079–4097, [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v).
A related extension is L. Mones, N. Bernstein and G. Csányi, *J. Chem. Theory
Comput.* **2016**, 12, 5100–5110,
[doi:10.1021/acs.jctc.6b00553](https://doi.org/10.1021/acs.jctc.6b00553).
