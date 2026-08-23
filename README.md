# GPR Free-Energy Analysis

Free-energy reconstruction and convergence diagnostics for PLUMED umbrella,
OPES, and metadynamics simulations. The original umbrella interfaces remain
backward compatible, while adaptive-bias estimators use a shared named-field
PLUMED parser and a bias-independent derivative-observation Gaussian process.

The umbrella implementation follows

> T. Stecher, N. Bernstein, and G. Csányi, "Free Energy Surface Reconstruction from Umbrella Samples Using Gaussian Process Regression," *J. Chem. Theory Comput.* **2014**, *10* (9), 4079–4097. [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v)

and implements the gradient-based reconstruction variant referred to as
**GPR(d)** in that paper:
mean forces are estimated per umbrella window (Sec. 2.3, eq 15), their statistical
noise is propagated into the likelihood (Sec. 4, eq 37), and the free-energy profile
is reconstructed by GPR on the derivative observations using a squared-exponential
kernel (Sec. 4–4.1).

The ICF route follows Mones, Bernstein, and Csányi (2016): an adaptive bias is
used for exploration, the unbiased instantaneous collective force (ICF) is the
local observable, and GPR reconstructs a scalar free-energy surface from those
gradient observations. Direct OPES/MTD density reweighting and PLUMED
`sum_hills` are deliberately kept as separate estimators.

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
- Parses named PLUMED fields across repeated restart headers
- Reweights OPES with the instantaneous total bias while excluding `opes.rct`
- Reweights metadynamics from an explicit log weight or normalized `*.rbias`
- Reports global/local importance ESS and maximum single-frame leverage
- Compares cumulative and disjoint time-block free-energy estimates
- Detects hysteretic basin transitions and completed round trips
- Reconstructs 1D or 2D PMFs from explicit paper-convention ICF observations
- Records Equation 8, quasi-equilibrium, physical-force, and metric assumptions
- Delegates HILLS reconstruction to `plumed sum_hills` through a shell-free API

## Estimator architecture

The package does not treat every biased trajectory as the same statistical
problem:

| Route | Observable supplied to reconstruction | Main validity requirement |
|---|---|---|
| Umbrella GPR(d) | Window mean force | Equilibrated stationary harmonic windows |
| OPES density | `exp(beta * total instantaneous bias)` weights | Every applied bias is included; usable local weight support |
| MTD density | Explicit log weight or normalized `bias - c(t)` (`*.rbias`) | Raw time-dependent bias is not silently accepted |
| MTD HILLS | Bias history evaluated by PLUMED | Correct PLUMED action and HILLS semantics |
| OPES/MTD ICF-GPR | `grad A = -conditional_mean(ICF)` | Valid physical ICF and equilibrium/quasi-equilibrium conditional sampling |

GPR uncertainty describes reconstruction from the supplied observations. It is
not, by itself, evidence that an adaptive simulation has equilibrated or that
hidden slow coordinates were sampled.

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

Outputs: `*_pmf_2d.dat` (cv0, cv1, PMF, raw sigma, calibrated sigma, and a
sampling-support flag on a grid),
`*_pmf_2d.png` (PMF contour + uncertainty, window centres overlaid) and
`*_diagnostics_2d.png` — an 8-panel sampling/fit diagnostics figure (PMF and
calibrated uncertainty; window drift centre→mean, mean-force field, and
autocorrelation time; window-overlap ellipses, per-observation LOO z-scores,
and the LOO calibration histogram). Pass `plot_diagnostics=False`
(CLI: `--no-diagnostics`) to skip it.

The default reproduces and plots the GP over the complete rectangular grid.
Sampling-support masking is optional: pass
`restrict_to_sampled_support=True` (CLI:
`--restrict-to-sampled-support`) to limit the reference, plots, path search,
and transverse integration to the convex hull of sampled window means. Use
`support_radius` (CLI: `--support-radius`) to additionally limit that hull by
distance in fitted GP lengthscales.

Run the self-contained synthetic check (no simulation data needed):

```bash
python examples/run_synthetic_2d_demo.py   # reconstructs a known 2D PMF
```

### Lowest-barrier paths and path-aligned marginal PMFs

#### At a glance

- **Enable path analysis:** `--find-lowest-barrier-path`.
- **Choose the allowed path region:**
  - `--path-mode search` (default): search anywhere in the connected valid
    sampled region; do not supply `--path-reference`.
  - `--path-mode corridor --path-reference neb_xy.dat
    --path-corridor-radius R`: search only in the valid region within `R` of
    the supplied trajectory.
  - `--path-mode fixed --path-reference neb_xy.dat`: perform no graph search;
    evaluate the supplied trajectory directly.
- **Choose the endpoints for `search` or `corridor`:**
  - Omit `--path-endpoints`: choose suitable minima automatically.
  - `--path-endpoints X0 Y0 X1 Y1` (default endpoint behavior): move each
    requested point to the deepest nearby valid grid-local minimum.
  - `--path-endpoint-radius R`: set the endpoint-minimum search radius in GP
    lengthscales; the default is `--support-radius`.
  - `--no-adjust-path-endpoints`: use the restraint-window centre nearest each
    requested endpoint instead of moving to a minimum.
  - `fixed` mode always uses the first and last reference-trajectory points and
    does not accept `--path-endpoints`.
- **Choose the primary path objective:**
  - `--path-uncertainty-weight 0` (default): minimize the mean-PMF bottleneck.
  - `--path-uncertainty-weight BETA`, with `BETA > 0`: minimize the
    upper-confidence bottleneck `mean dF + BETA * sigma(dF)`; `BETA=1` gives a
    one-sigma risk-aware search.
  - Uncertainty weighting works with `search` and `corridor`, but not with a
    fixed trajectory because there is no path to select in `fixed` mode.
- **Choose among paths with the same optimal bottleneck:**
  - `--path-gradient-weight 1` (default): prefer gradient-aligned, MEP-like
    paths.
  - `--path-gradient-weight 0`: prefer the geometrically shortest path.
  - Other non-negative values tune the gradient-alignment penalty.
- **Choose the path metric:**
  - By default, each CV is scaled by its fitted or fixed GP lengthscale.
  - `--path-metric-scales SCALE0 SCALE1` overrides those physical scales.
  - `--path-corridor-radius` is dimensionless in this path metric; for isotropic
    metric scale `ell`, physical radius `r` corresponds to `R = r / ell`.
  - `--path-endpoint-radius` is always measured in the GP-lengthscale metric,
    independently of `--path-metric-scales`.
- **Optional path-aligned 1D PMF:**
  - `--path-aligned-marginal --thermal-energy KBT` computes the transverse
    Boltzmann marginal `A(s)`.
  - `--perpendicular-points N` and `--perpendicular-width W` control its
    transverse quadrature.
- **Validity rule:** endpoint relocation, free/corridor searches, fixed-path
  evaluation, and transverse integration all stay inside the sampled,
  non-red `path_valid` region. Invalid or disconnected requests fail explicitly.
- **Reported quantities:** the barrier is `max(path PMF) - min(path PMF)`;
  reaction `dF` is end minus start; uncertainties use the full GP posterior
  covariance between the relevant points.

The following sections describe these choices and their numerical meaning in
more detail.

By default, `find_lowest_barrier_path` finds the grid path whose highest PMF is
as low as possible. It computes the exact minimax threshold, then minimizes an
additive lengthscale-scaled path cost inside that exact sublevel set. The
default `gradient_alignment_weight=1.0` (CLI: `--path-gradient-weight 1.0`)
adds a weak-gradient-aware penalty for motion perpendicular to the predicted
free-energy gradient. Set it to zero for the geometrically shortest minimax
path.

Uncertainty-aware selection is opt-in. A positive `uncertainty_weight=beta`
(CLI: `--path-uncertainty-weight BETA`) replaces the node score by the
one-sided upper-confidence quantity
`mean dF from start + beta * sigma(dF from start)`. The graph algorithm exactly
minimizes the largest such score. `beta=1` is a one-sigma risk-aware path;
`beta=0` preserves the mean-only result. This takes uncertainty into account
when choosing the path, but does not integrate over uncertainty in which path
is selected.

The reported barrier is always `max(path PMF) - min(path PMF)`, with uncertainty
from the full posterior covariance between those two points. Reaction ΔF
remains end minus start. The searched bottleneck and secondary cost are exact
on the finite 8-neighbor graph; they are not guarantees for the continuous GPR
surface and do not constitute NEB/string refinement:

```python
from gpr_umbrella import find_lowest_barrier_path

path = find_lowest_barrier_path(
    res,
    endpoints=((x_start, y_start), (x_end, y_end)),
)
```

Requested endpoints are relocated to the deepest grid-local minima within an
elliptical neighborhood by default. The endpoint search uses the same
lengthscale metric as sampled support and defaults to `support_radius`; set
`endpoint_search_radius` (CLI: `--path-endpoint-radius`) independently when
needed. Set `adjust_endpoints=False` (CLI:
`--no-adjust-path-endpoints`) to skip the minimum search and instead use the
restraint-window centre nearest each requested endpoint, snapped to the path
grid. A selected window centre must snap into the path-valid region; otherwise
the code stops with an actionable error rather than silently moving it. The
window-centre mode requires explicit endpoints so the corresponding endpoint
windows can be identified. Both selected endpoints must belong to the same
connected path-valid component (sampled support intersected with the non-red,
window-anchored PMF range). The complete minimum-bottleneck path may move
anywhere in that component; disconnected endpoint neighborhoods produce an
actionable error instead of silently crossing an invalid region.

A supplied two-column trajectory can either be enforced exactly or used as a
soft corridor constraint:

```bash
# Evaluate the supplied trajectory; no graph search is performed.
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-mode fixed \
    --path-reference neb_xy.dat

# Search only near that trajectory.
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-mode corridor \
    --path-reference neb_xy.dat --path-corridor-radius 0.5
```

The reference file may contain comments beginning with `#`; its first two
columns are the two CV coordinates. A fixed trajectory is densified for stable
barrier evaluation and must remain inside the same trustworthy `path_valid`
region used by the free search. Corridor mode intersects that valid region with
a tube around the reference polyline. The radius is dimensionless in the path
metric: with isotropic 0.5 A metric scales, radius 0.5 corresponds to 0.25 A.
Disconnected corridors and invalid fixed trajectories fail explicitly.

The 2D CLI can find that path and optionally compute the path-aligned marginal
PMF, `A(s)`, by Boltzmann-integrating the transverse coordinate `u` at each
position along the path:

```bash
gpr-umbrella-2d --colvar-dir COLVAR --kappa-dir COLVAR \
    --find-lowest-barrier-path --path-aligned-marginal \
    --thermal-energy 0.02585
```

The value passed to `--thermal-energy` is `kBT` in the selected `energy_unit`.
Path length and perpendicular directions are computed after scaling each CV by
its own fitted GP lengthscale. Override those two physical scales with
`--path-metric-scales SCALE0 SCALE1`; `SCALE0` is expressed in the first CV's
unit and `SCALE1` in the second CV's unit. The resulting path coordinate `s`
and transverse coordinate `u` are dimensionless metric arclengths.
The marginal profile is written to `*_path_aligned_pmf_1d.dat`; the
lowest-barrier path itself is written to `*_lowest_barrier_path.dat` and
`*_lowest_barrier_path.png` when the corresponding outputs are enabled.

### Path implementation structure

The pathway contribution is split by responsibility to keep reviews local:

- `path_graph.py` contains only the generic two-pass minimax/Dijkstra solver.
- `trajectory.py` validates, densifies, and measures distances to reference paths.
- `path_profile.py` evaluates path PMFs and covariance-aware differences.
- `pathways.py` orchestrates endpoints, validity masks, modes, and optional
  transverse marginalization.

The existing `find_lowest_barrier_path` entry point and default mean-only free
search are preserved for compatibility.

## OPES analysis

OPES direct reweighting uses the instantaneous OPES bias plus every other
applied restraint or wall. `opes.rct` is retained as a diagnostic and is never
used in the weights. By default, the loader raises if the COLVAR contains an
unlisted `*.bias` field, preventing an incomplete total bias from being used
silently.

```python
from gpr_umbrella import analyze_opes_1d

result = analyze_opes_1d(
    "T300/COLVAR",
    cv_field="path.s",
    temperature=300.0,
    energy_unit="eV",
    bias_energy_unit="eV",  # numerical unit printed in the bias columns
    bias_field="opes.bias",
    other_bias_fields=(
        "wall_path.bias",
        "wall_hh_lo.bias",
        "wall_hh_hi.bias",
        "wall_relz_lo.bias",
        "wall_relz_hi.bias",
        "wall_corner.bias",
        "wall_dz.bias",
    ),
    bins=160,
    cumulative_cutoffs=(250_000, 500_000, 750_000, 1_000_000, 1_500_000),
    n_blocks=5,
    regions={"desorbed": (0.9, 2.5), "transition": (4.0, 10.0)},
    basin_a=(10.0, 16.0),
    basin_b=(0.9, 2.5),
    transition_region=(4.0, 10.0),
)

profile = result["pmf"]
print(profile["importance_weight_ess"])
print(profile["local_ess"])
print(profile["local_maximum_weight_fraction"])
print(result["opes_diagnostics"])
```

The equivalent CLI writes a 1D table and a concise JSON summary when
`--output-dir` is supplied:

```bash
gpr-biased opes T300/COLVAR --cv-field path.s --temperature 300 \
  --energy-unit eV --bias-energy-unit eV \
  --other-bias-field wall_path.bias \
  --other-bias-field wall_hh_lo.bias --other-bias-field wall_hh_hi.bias \
  --other-bias-field wall_relz_lo.bias --other-bias-field wall_relz_hi.bias \
  --other-bias-field wall_corner.bias --other-bias-field wall_dz.bias \
  --bins 160 --n-blocks 5 --basin-a 10 16 --basin-b 0.9 2.5 \
  --transition-region 4 10 --output-dir analysis/T300
```

Cutoffs use the units printed in the COLVAR `time` field. The returned Kish ESS
is an importance-weight diagnostic, not an autocorrelation-adjusted number of
independent configurations. A high global ESS can coexist with a transition
region dominated by one frame, so inspect local ESS and local maximum-weight
fractions. Empty bins carry a false support flag and `NaN` leverage rather than
looking artificially well behaved.

`start_time`, `stop_time`, and `stride` (CLI: `--start-time`, `--stop-time`,
`--stride`) apply an inclusive analysis cutoff after restart deduplication.
The selected range and record counts are retained in `opes_data["selection"]`.
Use several physically motivated cutoffs: dropping an early adaptive segment
can reveal that an apparently stable cumulative PMF has no later support in a
required basin.

When both basin ranges are supplied, the analyzer also reports hysteretic
basin-to-basin events, completed round trips, and the last qualifying time in
each basin. With disjoint blocks it reports how many blocks support each basin
and a basin-population free energy. Block RMS values are computed only on their
common supported bins; a low RMS is not reassuring if later blocks have lost a
requested basin.

The CLI basin ranges act on the analyzed one-dimensional CV. For a chemical
state definition involving several observables, construct the two Boolean
masks explicitly and use the public detector:

```python
from gpr_umbrella import detect_hysteretic_transitions

events = detect_hysteretic_transitions(
    (HH >= 2.0) & (RELZ >= -1.7),
    (HH <= 1.0) & (RELZ <= -3.0),
    times=time,
    state_names=("adsorbed", "desorbed"),
)
print(events["events"], events["completed_round_trips"])
print(events["basin_evidence"]["desorbed"]["last_qualifying_time"])
```

`free_energy_from_opes_bias` converts an OPES bias grid evaluated by the
matching PLUMED version using `F = -V/(1 - 1/gamma)`. The package reads saved
state metadata, validates an `OPES_METAD_state` action when that metadata is
present, but does not reimplement compressed OPES kernel semantics.

## Metadynamics analysis

Time-dependent MTD reweighting requires a normalized bias. Prefer an explicit
dimensionless log-weight field or PLUMED's `*.rbias = bias - c(t)` output:

```python
from gpr_umbrella import analyze_metadynamics

result = analyze_metadynamics(
    "COLVAR",
    cv_fields=("path.s",),
    rbias_field="metad.rbias",
    extra_bias_fields=("wall.bias",),
    thermal_energy=0.025852,  # kBT in the same energy unit as the biases
    bias_energy_factor=1.0,   # explicit input-bias to kBT-unit conversion
    start_time=100.0,         # optional, in the COLVAR time-field unit
    stop_time=1000.0,         # inclusive bounds after restart deduplication
    stride=10,                # applied after the time bounds
    bins=160,
    n_blocks=5,
    hills_files="HILLS",
)
```

```bash
gpr-biased metad COLVAR --cv-field path.s --rbias-field metad.rbias \
  --extra-bias-field wall.bias --temperature 300 --energy-unit eV \
  --bias-energy-factor 1.0 --start-time 100 --stop-time 1000 --stride 10 \
  --hills HILLS --n-blocks 5 --output-dir analysis
```

Raw `metad.bias` is rejected by default because it is not interchangeable with
the normalized bias. The raw-bias route requires both the explicit
`allow_quasistatic_bias=True` opt-in and a finite, nonzero `start_time` to
`stop_time` range:

```python
raw_segment = analyze_metadynamics(
    "COLVAR",
    cv_fields=("path.s",),
    raw_bias_field="metad.bias",
    extra_bias_fields=("wall.bias",),
    allow_quasistatic_bias=True,
    start_time=800.0,
    stop_time=1000.0,
    thermal_energy=0.025852,
)
```

```bash
gpr-biased metad COLVAR --cv-field path.s --raw-bias-field metad.bias \
  --extra-bias-field wall.bias --allow-quasistatic-bias \
  --start-time 800 --stop-time 1000 --temperature 300 --energy-unit eV
```

Time bounds use the units of the named COLVAR time field and are inclusive.
Restart overlap is deduplicated first, the time bounds are applied second, and
the stride is applied last. At least two records must remain. The returned
`selection` metadata records the requested and realized range and all record
counts. Explicit-logweight and normalized-`rbias` routes support the same
optional selection controls. Bounding a raw-bias segment records the scope of
the quasi-static assumption; it does not demonstrate stationarity.

Restart-overlap records are deduplicated by named time. As for OPES, every
printed wall/restraint `*.bias` must be accounted for. With `action.rbias`, the
matching `action.bias` is recognized as the raw bias that was replaced and is
not added twice. A precomputed log weight cannot be audited from its values, so
printed bias columns require an explicit `allow_unlisted_bias_fields=True`
acknowledgement after confirming they were included when that log weight was
created.

`load_hills_diagnostics` summarizes deposition intervals and hill-height
evolution. `run_sum_hills` invokes PLUMED with an argument list (never a shell)
for conventional HILLS reconstruction, avoiding a partial Python
reimplementation of well-tempered, flexible-hill, or multiple-walker rules.
`analyze_metadynamics` applies the same explicit `bias_energy_factor` to HILLS
heights and COLVAR bias energies so their numerical units remain consistent.
COLVAR time selection does not crop a separately supplied HILLS file; HILLS
diagnostics continue to summarize every record in that file.

## ICF/GPR for OPES or metadynamics

The paper-style route is bias-agnostic once valid instantaneous collective
forces are available. With the paper's convention, `ICF` is a thermodynamic
force and the package always applies

> L. Mones, N. Bernstein, and G. Csányi, "Exploration, Sampling, And
> Reconstruction of Free Energy Surfaces with Gaussian Process Regression,"
> *J. Chem. Theory Comput.* **2016**, *12* (10), 5100–5110.
> [doi:10.1021/acs.jctc.6b00553](https://doi.org/10.1021/acs.jctc.6b00553)
> ([local PDF](../ct6b00553.pdf))

```text
grad A(xi) = - conditional_mean(ICF | xi)
```

```python
from gpr_umbrella import reconstruct_pmf_icf

result = reconstruct_pmf_icf(
    "COLVAR_ICF",
    cv_fields=("path.s",),
    icf_fields=("path_s.icf",),
    cv_factors=1.0,       # input-to-output coordinate conversion
    energy_factor=1.0,   # input-to-output energy conversion
    aggregation_bins=80,
    time_block_size=20_000,
    bias_depends_only_on_modeled_cvs=True,
    quasi_equilibrium=True,
    physical_force_excludes_bias=True,
    metric_correction_included=True,
    grid_n=300,
)
```

```bash
gpr-biased icf COLVAR_ICF --cv-field path.s --icf-field path_s.icf \
  --aggregation-bins 80 --time-block-size 20000 \
  --bias-depends-only-on-modeled-cvs --quasi-equilibrium \
  --physical-force-excludes-bias --metric-correction-included \
  --require-paper-assumptions --output-dir analysis
```

The ICF must be computed from the unbiased physical potential, with all bias
forces removed, and must include the CV metric/Jacobian divergence term when
required. `COLVAR`, HILLS, an OPES state, and bias values alone do not provide
this observable. For an adaptive bias, Equation 8 also requires all applied
biases to depend only on the complete reconstructed CV vector and requires
equilibrium or justified quasi-equilibrium conditional sampling. A wall on an
omitted CV breaks that invariance.

All four paper-validity conditions are required by default. The Python API's
`require_paper_assumptions=False` and CLI's
`--allow-unverified-paper-assumptions` are deliberately named unsafe opt-outs;
the returned validity record remains explicit. Pointwise LOO calibration is
also prevented from shrinking raw GP uncertainty by default because coherent
time-correlated force errors can look artificially predictable. Only enable
`allow_uncertainty_downscaling` when validation groups are demonstrably
independent.

Coordinate and ICF conversions are coupled: when `cv_factors` converts input
CV coordinates, the derived thermodynamic-force factor is
`energy_factor / cv_factor`. An explicit `icf_factors` override is available
for already converted force columns and is recorded in the result.

Exact derivative GPR scales cubically and is guarded at 1,000 scalar gradient
components (`N_points * N_CVs`) by default. Larger trajectories must use
explicit local/time-block aggregation or a future sparse-GP implementation.
Aggregation reduces size, but its covariance is only decorrelation-aware when
blocks are longer than the relevant correlation time.

### Generic derivative-observation GP

The sampler-independent core is also public. `fit_gradient_gp(points,
gradients, gradient_noise_cov)` expects `points` and `gradients` with shape
`(N, D)` and a full `(N*D, N*D)` observation covariance in point-major order.
`fit_icf_gp` accepts the same covariance but requires an explicit
`force_convention`: use `"thermodynamic_force"` for the paper convention
`ICF = -grad A`, or `"free_energy_gradient"` when the supplied values already
are `grad A`. `predict_gradient_gp` and
`posterior_covariance_gradient_gp` then return scalar-free-energy predictions
and covariance, optionally relative to an explicit reference point.

The shared GP currently uses a nonperiodic squared-exponential kernel. Unwrap
or otherwise transform torsional CVs; a periodic derivative kernel is not yet
implemented.

## Reading convergence evidence

The adaptive-bias APIs intentionally do not return a universal `converged`
Boolean. Judge each reported observable using several independent checks:

- cumulative and disjoint-block PMF, barrier, and basin-population stability;
- global and local importance ESS plus maximum single-frame leverage;
- repeated transitions and completed round trips between the relevant basins;
- OPES `rct`, `zed`, `neff`, and kernel-count evolution, or MTD hill evolution;
- agreement between reweighted density, saved-bias/HILLS, and ICF estimators;
- sensitivity to equilibration cutoff, binning, and basin definitions;
- agreement between independent replicas or walkers;
- stability of hidden/orthogonal observables at fixed modeled CV.

Agreement of two estimators built from one trajectory is internal consistency.
It cannot replace basin recurrence, independent runs, or evidence that slow
orthogonal coordinates equilibrated.

## Citation

This implementation is based on the method introduced in:

> T. Stecher, N. Bernstein, and G. Csányi, "Free Energy Surface Reconstruction from Umbrella Samples Using Gaussian Process Regression," *J. Chem. Theory Comput.* **2014**, *10* (9), 4079–4097. [doi:10.1021/ct500438v](https://doi.org/10.1021/ct500438v)

If you use this tool in published work, please cite the paper above and acknowledge
this implementation.

The adaptive-bias ICF/GPR route follows:

> L. Mones, N. Bernstein, and G. Csányi, "Exploration, Sampling, And Reconstruction of Free Energy Surfaces with Gaussian Process Regression," *J. Chem. Theory Comput.* **2016**, *12* (10), 5100–5110. [doi:10.1021/acs.jctc.6b00553](https://doi.org/10.1021/acs.jctc.6b00553)

The supplied manuscript is available locally as
[ct6b00553.pdf](../ct6b00553.pdf).
