# Robust arclength SE GPR

This extends the existing package and its squared-exponential kernels and
diagnostic plotting functions. It is not a replacement plotting/GP framework.

## What is being estimated?

For samples q=(HH, RELZ), the reference is an **ordered 2D polyline**. Its scalar
coordinate is geometric arclength, not sorted HH. Both coordinates must be in
the same explicitly chosen metric (the DAIS frontend uses Å). The endpoint
segments extend as rays, so samples beyond the first/last image are not piled
onto one endpoint. The profile itself is reported only on sampled histogram
support within the finite NEB range.

Nonmonotonic HH is valid. Retracing and self-intersections are rejected; nearby
arclength-remote branches are flagged. Neither window ID nor trajectory history
is used to change the projection. At a true intersection, two CV values cannot
identify which physical branch a structure belongs to. Add a structural CV or
analyse separate pathways; do not manufacture a window-dependent coordinate.

The old external `run_arclength_1d.py` replaced the original 2D restraints with a
scalar kappa after projection. That is not exact at bends or endpoints. The new
`gpr-umbrella-path` entry point instead evaluates **both original harmonic biases
on every sample**, uses PyMBAR to determine window normalisations, then projects
the correctly reweighted samples. Constant common physical energies cancel;
common walls remain part of the target. Different temperatures, walls, fixed
backgrounds, models or atom sets cannot be mixed using bias energies alone.

Two explicitly different thermodynamic targets are supported:

* `--target-normal-kappa 0` (default): unrestrained arclength marginal, within
  the original common physical setup. Narrow 2D ribbons may have inadequate
  transverse support: rare large-weight frames are a failure, not certainty.
* `--target-normal-kappa K`: arclength marginal retaining a **common** potential
  K d(q,path)^2/2. This is a path-normal-restrained PMF, not the unrestrained
  marginal or a pointwise slice through F(HH,RELZ). The actual 2D bias is still
  treated exactly. The target choice is recorded and must accompany any barrier.
  It cannot be selected just because it produces a desired barrier/error.

Neither profile is automatically a global reaction free energy or a kinetic
reaction coordinate. Both require equilibrated sampling and common Hamiltonians.

## Statistical fitting

1. MBAR uses the full original 2D reduced bias matrix. Thinning is a computational
   control, **not** an assertion of independence. Convergence is checked explicitly.
2. Contiguous time blocks are resampled independently within each window; every
   replicate re-solves the window normalisations. PyMBAR's IID covariance is not
   applied to correlated MD. Empty bins are not filled with pseudocounts.
3. Reweighted bin free energies supply finite-interval slopes and their **full
   correlated covariance**. The GP observes differences of its endpoint values,
   not fictitious exact point derivatives. Finite-bin averaging is an approximation
   checked by changing bin resolution. Bootstrap covariance uses a reported 2%
   diagonal shrinkage and requires at least twice as many replicates as bins.
4. The existing SE function and derivative covariances supply all predictions.
   Explicit log-uniform lengthscale/amplitude grids replace trust in one local
   optimiser. Sub-spacing lengthscales are disallowed. The grid bounds and weights
   are saved. Conditional uncertainty and between-hyperparameter covariance are
   combined. No fitted white-noise inflation or outlier deletion hides failures.
5. Raw conditional LOO and contiguous-interval held-out diagnostics remain visible.
   Hyperparameters are selected using the full data, so these are diagnostic CV
   scores, not an independent external validation claim.
6. Barriers require explicit ordered reactant and transition intervals. Full
   correlated posterior curves are sampled, including hyperparameter and extrema
   selection uncertainty. Extrema repeatedly hitting interval boundaries are
   flagged. Arbitrary first-1.5-Å basins and quadrature of unrelated point errors
   are not used.

The existing scalar umbrella fitter also now uses a resolution lower bound,
multiscale batch-mean sampling errors by default, raw quality flags, and an
uncertainty scale that never shrinks errors. Its scalar-bias assumption still
applies: use the new path entry point for 2D-biased MD.

## Usage

Install `pip install -e '.[path,test]'` in an isolated environment. Input COLVARs
must already have their documented burn-in removed; time is in fs, CVs in Å,
restraints in eV/Å². Each indexed restraint file contains four whitespace-separated
values: HH centre, RELZ centre, kappa HH, kappa RELZ. Numeric window IDs must match.

```bash
MPLBACKEND=Agg gpr-umbrella-path \
  --colvar-dir /absolute/path/COLVAR \
  --kappa-dir /absolute/path/window_kappa \
  --path-reference /absolute/path/ordered_HH_RELZ.dat \
  --temperature 300 --bins 30 --stride 20 \
  --block-size 1000 --bootstraps 128 \
  --output-dir /absolute/path/new_analysis
```

For barrier sampling also supply `--reactant-interval LOW HIGH` and
`--transition-interval LOW HIGH` in Å, chosen using the physical states. The
reporting target `--max-barrier-std` never changes the fit to reduce an error.

On DAIS, `/u/vueszter/work/projects/Desorption/GPR/run_arclength_1d.py` now delegates
to this entry point. Use the isolated `.venv-robust-gpr/bin/python` beside it.
Its previous approximate implementation is retained as
`run_arclength_1d.legacy-20260906.py`. The wrapper is versioned under `examples/`.

`source_campaign.json` and `neb_path_30_images.xyz` beside COLVAR inputs trigger
an audit of atom order, cell and frozen atoms. Unverified provenance is flagged;
differences require cross-Hamiltonian energies or consistently prepared new MD.

Outputs retain `diagnostics_1d.png` with the existing layout/palette, and add
`selection_diagnostics.png`, `pmf_1d.dat`, `hyperparameters.tsv`, `fit_arrays.npz`
and machine-readable `summary.json`. Code/dependency/input hashes are recorded.
Numerical results are saved before plotting. Existing output folders are not
overwritten. Statistical/model failures are visible, not silently discarded.

## Acceptance, not cosmetic success

Run `examples/validate_path_campaigns.py` for the completed DAIS datasets. It
varies binning, block length and thinning independently. Then run
`examples/assess_path_sensitivity.py`, which also broadens GP priors and raises
the minimum lengthscale before comparing gauge-aligned profiles.

Current explicit reporting thresholds (not universal physical laws): raw LOO
RMS <=2 and max |z| <=4; blocked-CV RMS <=3; at least 4 effective contributing
time blocks per bin and at least 25% of the unweighted block ESS in that bin;
overlap graph connected at 0.001; maximum profile sensitivity
and uncertainty <=0.05 eV. Prior-bound mass, time splits, projection ambiguity and
Hamiltonian mismatches are separate flags. Record changes to these thresholds;
do not loosen them until a problematic dataset turns green.

`UNRELIABLE` means a fit exists but failed checks. `PROVISIONAL` means the
sensitivity suite is outstanding. `FIT_STABLE_BARRIER_UNDEFINED` still requires
physical basin definitions. Passing specified checks is not proof of equilibrium,
complete transverse sampling, or publication readiness.

## References

* Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008),
  https://doi.org/10.1063/1.2978177 (MBAR).
* https://pymbar.readthedocs.io/en/stable/mbar.html (original bias matrices,
  overlap and independent-sample assumptions).

Block resampling here is explicit because these MD trajectories are correlated.
