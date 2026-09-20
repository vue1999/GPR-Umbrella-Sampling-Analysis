# Migration from the experimental noise-aware branch

This revision simplifies `codex/noise-aware-2d` at `2d03bc3`. The original code
and history remain on that branch and the `archive/noise-aware-2d-2d03bc3` ref.

## Reconstruction and paths

Remove `find_lowest_barrier`, `path_endpoints`, `path_metric_scale`,
`path_reference`, `path_mode` and `path_corridor_radius` from
`reconstruct_pmf_2d(...)`. Reconstruct the surface, then call
`find_lowest_barrier_path(surface, ...)`, using `endpoints`, `metric_scale`,
`reference_path`, `path_mode` and `corridor_radius` respectively.
The CLI continues to orchestrate both steps with its existing path flags.
Plot and save standalone path results explicitly using `plot_lowest_barrier_path`
and `save_lowest_barrier_path`.

`max_minima` and reporting-only minima/alignment metadata are removed. No path
selection depends on minima enumeration. The graph solver and support rules
retain their previous numerical behavior.

## Result fields

Path `barrier` becomes `energy_range`, with the same max-minus-min definition.
Its `barrier_err*` fields become `energy_range_err*`. New `endpoint_to_max` and
matching error fields report the rise from the starting endpoint. Path profile
plots use the start as their reference and label both quantities explicitly.
Path tables now use a compact JSON metadata comment instead of repeated prose.

For in-memory 2D reconstruction, supply `centers`, `kappa` and `all_positions`.
`window_files` is optional. Means, variances and counts are derived internally;
legacy redundant summary fields are ignored. Return values still include those
statistics for diagnostics.

## Deferred features and compatibility

The projected-arclength MBAR/GPR command, its campaign-specific examples and its
ASE/PyMBAR dependencies are excluded. Transverse GP marginalization and its
options (`path_aligned_marginal`, `thermal_energy`, `perpendicular_points`,
`perpendicular_width`) are deferred too. Both remain in the archived branch.

Existing ordinary 1D behavior is unchanged from `2d03bc3`. Main's documented
`gpr_umbrella_1d` imports and `gpr-umbrella` command remain available. The 2D noise
model, optimization, uncertainty calibration and diagnostic panels are retained;
no expected barrier or material-specific numerical constraint has been added.
