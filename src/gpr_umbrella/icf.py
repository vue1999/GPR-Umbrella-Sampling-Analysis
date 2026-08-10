"""Instantaneous-collective-force adapters for derivative-observation GPR.

This module implements the ICF/GPR conventions described by Mones,
Bernstein, and Csanyi, *J. Chem. Theory Comput.* 2016, 12, 5100--5110,
doi:10.1021/acs.jctc.6b00553.

For the convention used by Mones, Bernstein, and Csanyi, the instantaneous
collective force ``f`` is a thermodynamic force and

``grad A(xi) = - E[f | xi]``.

This module preserves that sign explicitly: input columns are always ICFs and
are negated before fitting the shared gradient GP.  It does not attempt to
derive an ICF from Cartesian forces.  Such a derivation must remove bias forces
and include the CV metric/Jacobian (divergence) correction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .plumed_io import read_plumed_table


__all__ = [
    "aggregate_icf_observations",
    "assess_icf_validity",
    "load_icf_data",
    "reconstruct_pmf_icf",
]


def _optional_bool(value: bool | None, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True, False, or None")
    return bool(value)


def assess_icf_validity(
    *,
    bias_depends_only_on_modeled_cvs: bool | None = None,
    quasi_equilibrium: bool | None = None,
    physical_force_excludes_bias: bool | None = None,
    metric_correction_included: bool | None = None,
) -> dict:
    """Record, but do not infer, the assumptions behind ICF reconstruction.

    Equation 8 of Mones et al. makes a conditional ICF average invariant to a
    bias only if that bias depends solely on the complete modeled CV vector and
    the biased dynamics is at equilibrium (or a justified quasi-equilibrium).
    The other two flags concern whether the supplied quantity is an ICF at all.
    ``None`` means that the condition has not been established.
    """
    conditions = {
        "bias_depends_only_on_modeled_cvs": _optional_bool(
            bias_depends_only_on_modeled_cvs,
            "bias_depends_only_on_modeled_cvs",
        ),
        "quasi_equilibrium": _optional_bool(
            quasi_equilibrium, "quasi_equilibrium"
        ),
        "physical_force_excludes_bias": _optional_bool(
            physical_force_excludes_bias, "physical_force_excludes_bias"
        ),
        "metric_correction_included": _optional_bool(
            metric_correction_included, "metric_correction_included"
        ),
    }

    eq8_values = (
        conditions["bias_depends_only_on_modeled_cvs"],
        conditions["quasi_equilibrium"],
    )
    definition_values = (
        conditions["physical_force_excludes_bias"],
        conditions["metric_correction_included"],
    )

    def tri_state(values: tuple[bool | None, ...]) -> bool | None:
        if any(value is False for value in values):
            return False
        if all(value is True for value in values):
            return True
        return None

    eq8 = tri_state(eq8_values)
    definition = tri_state(definition_values)
    complete = tri_state(tuple(conditions.values()))
    caveats: list[str] = []
    if conditions["bias_depends_only_on_modeled_cvs"] is not True:
        caveats.append(
            "All applied biases must depend only on the complete modeled CV "
            "vector; a bias on an omitted CV changes the conditional ensemble."
        )
    if conditions["quasi_equilibrium"] is not True:
        caveats.append(
            "An adaptive bias must evolve slowly relative to the underlying "
            "conditional dynamics for the quasi-equilibrium argument to hold."
        )
    if conditions["physical_force_excludes_bias"] is not True:
        caveats.append(
            "The physical collective force must exclude every applied bias "
            "force."
        )
    if conditions["metric_correction_included"] is not True:
        caveats.append(
            "Nonlinear CVs require the metric/Jacobian divergence correction."
        )
    return {
        "paper_equation_8_applicable": eq8,
        "icf_definition_complete": definition,
        "paper_assumptions_satisfied": complete,
        "conditions": conditions,
        "caveats": tuple(caveats),
        "convergence_assessed": False,
        "note": (
            "Validity assumptions and convergence evidence are distinct; no "
            "universal convergence Boolean is inferred."
        ),
    }


def _broadcast_factors(values, dimensions: int, name: str) -> np.ndarray:
    try:
        factors = np.broadcast_to(
            np.asarray(values, dtype=float), (dimensions,)
        ).copy()
    except ValueError as exc:
        raise ValueError(
            f"{name} must be scalar or have shape ({dimensions},)"
        ) from exc
    if np.any(~np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError(f"{name} values must be finite and positive")
    return factors


def _positive_scalar_factor(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar")
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    factor = float(array)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return factor


def _resolve_icf_scales(
    cv_factors,
    dimensions: int,
    energy_factor,
    icf_factors,
) -> tuple[np.ndarray, float, np.ndarray, str]:
    position_scale = _broadcast_factors(
        cv_factors, dimensions, "cv_factors"
    )
    energy_scale = _positive_scalar_factor(energy_factor, "energy_factor")
    if icf_factors is None:
        force_scale = energy_scale / position_scale
        source = "derived_energy_factor_over_cv_factors"
    else:
        if energy_scale != 1.0:
            raise ValueError(
                "A non-default energy_factor cannot be combined with explicit "
                "icf_factors; omit icf_factors to derive force scaling as "
                "energy_factor / cv_factor"
            )
        force_scale = _broadcast_factors(
            icf_factors, dimensions, "icf_factors"
        )
        source = "explicit_icf_factors"
    return position_scale, energy_scale, force_scale, source


def load_icf_data(
    colvar_file: str | Path,
    *,
    cv_fields: Sequence[str],
    icf_fields: Sequence[str],
    time_field: str = "time",
    cv_factors=1.0,
    energy_factor: float = 1.0,
    icf_factors=None,
    start_time: float | None = None,
    stop_time: float | None = None,
    stride: int = 1,
    bias_depends_only_on_modeled_cvs: bool | None = None,
    quasi_equilibrium: bool | None = None,
    physical_force_excludes_bias: bool | None = None,
    metric_correction_included: bool | None = None,
    strict: bool = False,
) -> dict:
    """Load named CV and paper-convention ICF columns from a PLUMED table.

    Restart-overlap records are deduplicated by *time_field*, keeping the last
    copy.  The complete shared-parser result is returned as ``table`` so that
    malformed-row counts, duplicate counts, settings, and file provenance are
    retained for the scientific audit trail.
    """
    cv_fields = tuple(cv_fields)
    icf_fields = tuple(icf_fields)
    if not cv_fields or len(cv_fields) != len(icf_fields):
        raise ValueError(
            "cv_fields and icf_fields must be non-empty and have equal length"
        )
    if len(set(cv_fields)) != len(cv_fields):
        raise ValueError("cv_fields must be unique")
    if len(set(icf_fields)) != len(icf_fields):
        raise ValueError("icf_fields must be unique")
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise ValueError("stride must be a positive integer")
    if stride < 1:
        raise ValueError("stride must be a positive integer")

    path = Path(colvar_file)
    table = read_plumed_table(
        path,
        required_fields=(time_field, *cv_fields, *icf_fields),
        deduplicate_field=time_field,
        duplicate_policy="last",
        strict=strict,
    )
    if table["n_records"] == 0:
        raise ValueError("ICF COLVAR contains no valid numeric records")
    columns = table["columns"]
    time = columns[time_field]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(
            "ICF time must be strictly increasing after restart "
            "deduplication"
        )
    selected = np.ones(len(time), dtype=bool)
    if start_time is not None:
        if not np.isfinite(start_time):
            raise ValueError("start_time must be finite")
        selected &= time >= start_time
    if stop_time is not None:
        if not np.isfinite(stop_time):
            raise ValueError("stop_time must be finite")
        selected &= time <= stop_time
    indices = np.flatnonzero(selected)[::int(stride)]
    if len(indices) < 2:
        raise ValueError("At least two ICF frames are required after selection")

    dimensions = len(cv_fields)
    position_scale, energy_scale, force_scale, force_scale_source = (
        _resolve_icf_scales(
            cv_factors, dimensions, energy_factor, icf_factors
        )
    )
    positions = np.column_stack([
        columns[name][indices] for name in cv_fields
    ]) * position_scale
    collective_forces = np.column_stack([
        columns[name][indices] for name in icf_fields
    ]) * force_scale
    gradients = -collective_forces
    validity = assess_icf_validity(
        bias_depends_only_on_modeled_cvs=bias_depends_only_on_modeled_cvs,
        quasi_equilibrium=quasi_equilibrium,
        physical_force_excludes_bias=physical_force_excludes_bias,
        metric_correction_included=metric_correction_included,
    )
    return {
        "source_file": str(path),
        "table": table,
        "time": time[indices].copy(),
        "positions": positions,
        "collective_forces": collective_forces,
        "icf": collective_forces,
        "gradients": gradients,
        "cv_names": cv_fields,
        "icf_fields": icf_fields,
        "cv_factors": position_scale,
        "energy_factor": energy_scale,
        "icf_factors": force_scale,
        "unit_provenance": {
            "cv_factors": position_scale.copy(),
            "energy_factor": energy_scale,
            "icf_factors": force_scale.copy(),
            "icf_factors_source": force_scale_source,
            "force_scaling_relation": (
                "icf_factor = energy_factor / cv_factor"
                if icf_factors is None
                else "explicit icf_factors"
            ),
        },
        "force_convention": "thermodynamic_force",
        "gradient_relation": "grad_A = -conditional_mean(ICF)",
        "validity": validity,
    }


def _normalize_icf_data(data: Mapping) -> dict:
    required = ("positions",)
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Missing ICF data key(s): {', '.join(missing)}")
    if "collective_forces" in data:
        forces = data["collective_forces"]
    elif "icf" in data:
        forces = data["icf"]
    else:
        raise ValueError(
            "ICF data must contain 'collective_forces' or 'icf'; a generic "
            "'gradients' key is intentionally not accepted because its sign "
            "is ambiguous"
        )
    positions = np.asarray(data["positions"], dtype=float)
    forces = np.asarray(forces, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]
    if forces.ndim == 1:
        forces = forces[:, None]
    if positions.ndim != 2 or forces.shape != positions.shape:
        raise ValueError(
            "positions and collective_forces must have the same (N, D) shape"
        )
    if len(positions) < 2:
        raise ValueError("At least two ICF observations are required")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(forces)):
        raise ValueError("ICF positions and forces must be finite")
    time = np.asarray(data.get("time", np.arange(len(positions))), dtype=float)
    if time.shape != (len(positions),) or not np.all(np.isfinite(time)):
        raise ValueError("time must be a finite array with one value per frame")
    if np.any(np.diff(time) < 0.0):
        raise ValueError("ICF time must be nondecreasing")
    dimensions = positions.shape[1]
    cv_names = tuple(data.get(
        "cv_names", tuple(f"cv{index}" for index in range(dimensions))
    ))
    if len(cv_names) != dimensions:
        raise ValueError("cv_names must contain one name per CV")
    validity = data.get("validity")
    if validity is None:
        validity = assess_icf_validity()
    normalized = {
        "time": time.copy(),
        "positions": positions.copy(),
        "collective_forces": forces.copy(),
        "icf": forces.copy(),
        "gradients": -forces,
        "cv_names": cv_names,
        "force_convention": "thermodynamic_force",
        "gradient_relation": "grad_A = -conditional_mean(ICF)",
        "validity": validity,
        **({"source_file": data["source_file"]} if "source_file" in data else {}),
        **({"table": data["table"]} if "table" in data else {}),
    }
    for name in ("cv_factors", "energy_factor", "icf_factors"):
        if name in data:
            value = data[name]
            normalized[name] = (
                np.asarray(value, dtype=float).copy()
                if name != "energy_factor"
                else float(value)
            )
    if "unit_provenance" in data:
        normalized["unit_provenance"] = dict(data["unit_provenance"])
    return normalized


def _as_bin_edges(
    positions: np.ndarray,
    bins,
    ranges,
) -> tuple[np.ndarray, ...]:
    histogram, edges = np.histogramdd(positions, bins=bins, range=ranges)
    if histogram.size == 0:
        raise ValueError("Aggregation grid is empty")
    return tuple(edges)


def _point_bin_indices(
    positions: np.ndarray,
    edges: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    valid = np.ones(len(positions), dtype=bool)
    for dimension, edge in enumerate(edges):
        index = np.searchsorted(edge, positions[:, dimension], side="right") - 1
        index[positions[:, dimension] == edge[-1]] = len(edge) - 2
        valid &= (index >= 0) & (index < len(edge) - 1)
        indices.append(index)
    flat = np.full(len(positions), -1, dtype=int)
    if np.any(valid):
        shape = tuple(len(edge) - 1 for edge in edges)
        flat[valid] = np.ravel_multi_index(
            tuple(index[valid] for index in indices), shape
        )
    return flat, valid


def _block_diagonal(blocks: Sequence[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=float)
    start = 0
    for block in blocks:
        stop = start + block.shape[0]
        result[start:stop, start:stop] = block
        start = stop
    return result


def aggregate_icf_observations(
    data: Mapping,
    *,
    bins=20,
    ranges=None,
    time_block_size: int | None = None,
    min_samples: int = 2,
) -> dict:
    """Average ICFs within local spatial bins and optional time blocks.

    Each occupied ``(time block, spatial bin)`` supplies one vector
    observation.  Its force-noise covariance is the within-group sample
    covariance divided by the group count.  Time blocking can reduce leakage
    between validation blocks, but this simple estimate does not itself prove
    decorrelation of the underlying trajectory.
    """
    normalized = _normalize_icf_data(data)
    if isinstance(min_samples, bool) or not isinstance(
        min_samples, (int, np.integer)
    ) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if time_block_size is not None and (
        isinstance(time_block_size, bool)
        or not isinstance(time_block_size, (int, np.integer))
        or time_block_size < 1
    ):
        raise ValueError("time_block_size must be a positive integer or None")

    positions = normalized["positions"]
    forces = normalized["collective_forces"]
    dimensions = positions.shape[1]
    edges = _as_bin_edges(positions, bins, ranges)
    spatial_bin, valid = _point_bin_indices(positions, edges)
    if time_block_size is None:
        time_block = np.zeros(len(positions), dtype=int)
    else:
        time_block = np.arange(len(positions)) // int(time_block_size)

    groups: dict[tuple[int, int], list[int]] = {}
    for index in np.flatnonzero(valid):
        key = (int(time_block[index]), int(spatial_bin[index]))
        groups.setdefault(key, []).append(int(index))

    mean_positions: list[np.ndarray] = []
    mean_forces: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    counts: list[int] = []
    group_keys: list[tuple[int, int]] = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=int)
        if len(indices) < min_samples:
            continue
        group_forces = forces[indices]
        mean_positions.append(np.mean(positions[indices], axis=0))
        mean_forces.append(np.mean(group_forces, axis=0))
        if len(indices) == 1:
            covariance = np.zeros((dimensions, dimensions), dtype=float)
        else:
            covariance = np.atleast_2d(
                np.cov(group_forces, rowvar=False, ddof=1)
            ) / len(indices)
            if covariance.shape != (dimensions, dimensions):
                covariance = covariance.reshape(dimensions, dimensions)
        covariance = 0.5 * (covariance + covariance.T)
        covariances.append(covariance)
        counts.append(len(indices))
        group_keys.append(key)

    if len(mean_positions) < 2:
        raise ValueError(
            "ICF aggregation produced fewer than two observations; reduce "
            "min_samples or use fewer bins"
        )
    result_positions = np.asarray(mean_positions, dtype=float)
    result_forces = np.asarray(mean_forces, dtype=float)
    result = {
        "positions": result_positions,
        "collective_forces": result_forces,
        "icf": result_forces,
        "gradients": -result_forces,
        "force_noise_cov": _block_diagonal(covariances),
        "force_covariances": np.asarray(covariances),
        "counts": np.asarray(counts, dtype=int),
        "group_keys": tuple(group_keys),
        "bin_edges": edges,
        "time_block_size": time_block_size,
        "cv_names": normalized["cv_names"],
        "force_convention": "thermodynamic_force",
        "gradient_relation": "grad_A = -conditional_mean(ICF)",
        "validity": normalized["validity"],
        "aggregation": {
            "kind": "local_spatial_bins",
            "uses_time_blocks": time_block_size is not None,
            "within_group_independence_assumed": True,
            "note": (
                "Block means reduce data size; autocorrelation-aware "
                "uncertainty still requires blocks longer than the relevant "
                "correlation time."
            ),
        },
    }
    for name in (
        "cv_factors", "energy_factor", "icf_factors", "unit_provenance"
    ):
        if name in normalized:
            result[name] = normalized[name]
    return result


def _force_covariance(
    n_points: int,
    dimensions: int,
    *,
    force_noise_cov=None,
    force_errors=None,
) -> np.ndarray:
    if force_noise_cov is not None and force_errors is not None:
        raise ValueError("Specify force_noise_cov or force_errors, not both")
    size = n_points * dimensions
    if force_noise_cov is not None:
        covariance = np.asarray(force_noise_cov, dtype=float)
        if covariance.shape == (dimensions, dimensions):
            covariance = _block_diagonal([covariance] * n_points)
        if covariance.shape != (size, size):
            raise ValueError(
                f"force_noise_cov must have shape ({size}, {size}) or "
                f"({dimensions}, {dimensions})"
            )
        return covariance
    if force_errors is None:
        raise ValueError(
            "Exact ICF fitting requires force_errors or force_noise_cov; the "
            "instantaneous-force noise cannot be inferred safely from COLVAR"
        )
    errors = np.asarray(force_errors, dtype=float)
    try:
        errors = np.broadcast_to(errors, (n_points, dimensions)).copy()
    except ValueError as exc:
        raise ValueError(
            f"force_errors must be scalar or broadcast to "
            f"({n_points}, {dimensions})"
        ) from exc
    if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("force_errors must be finite and nonnegative")
    return np.diag(errors.ravel() ** 2)


def _prediction_grid(
    positions: np.ndarray,
    grid_n,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...]]:
    dimensions = positions.shape[1]
    if dimensions == 1:
        values = np.asarray(grid_n)
        if values.ndim == 0:
            count = int(values)
        elif values.shape == (1,):
            count = int(values[0])
        else:
            raise ValueError("grid_n must be an integer for one CV")
        if count < 2:
            raise ValueError("grid_n must be at least 2")
        axis = np.linspace(np.min(positions[:, 0]), np.max(positions[:, 0]), count)
        return axis[:, None], (axis,), (count,)
    if dimensions == 2:
        values = np.asarray(grid_n)
        if values.ndim == 0:
            counts = (int(values), int(values))
        elif values.shape == (2,):
            counts = tuple(int(value) for value in values)
        else:
            raise ValueError("grid_n must be an integer or a pair for two CVs")
        if min(counts) < 2:
            raise ValueError("grid_n values must be at least 2")
        axes = tuple(
            np.linspace(np.min(positions[:, dimension]),
                        np.max(positions[:, dimension]), counts[dimension])
            for dimension in range(2)
        )
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([component.ravel() for component in mesh])
        return points, axes, counts
    raise ValueError(
        "prediction_points are required for ICF reconstructions above two CVs"
    )


def reconstruct_pmf_icf(
    colvar_file: str | Path | None = None,
    *,
    data: Mapping | None = None,
    cv_fields: Sequence[str] | None = None,
    icf_fields: Sequence[str] | None = None,
    time_field: str = "time",
    cv_factors=1.0,
    energy_factor: float = 1.0,
    icf_factors=None,
    start_time: float | None = None,
    stop_time: float | None = None,
    stride: int = 1,
    bias_depends_only_on_modeled_cvs: bool | None = None,
    quasi_equilibrium: bool | None = None,
    physical_force_excludes_bias: bool | None = None,
    metric_correction_included: bool | None = None,
    require_paper_assumptions: bool = True,
    strict: bool = False,
    aggregation_bins=None,
    aggregation_ranges=None,
    time_block_size: int | None = None,
    min_samples_per_observation: int = 2,
    force_noise_cov=None,
    force_errors=None,
    max_exact_observations: int = 1000,
    prediction_points=None,
    grid_n=200,
    reference=None,
    optimize_hyperparams: bool = True,
    fixed_lengthscale=None,
    fixed_sigma_f: float | None = None,
    calibrate_uncertainty: bool = True,
    allow_uncertainty_downscaling: bool = False,
    prediction_batch_size: int = 10_000,
) -> dict:
    """Fit and predict a PMF from explicit paper-convention ICF samples.

    Provide exactly one of *colvar_file* or an in-memory *data* mapping.  Raw
    exact fitting is guarded by *max_exact_observations*.  For compatibility,
    that public name is retained, but the limit applies to the scalar gradient
    components ``N * D`` that determine the dense covariance size.  Larger
    problems must opt into local aggregation or use a genuine sparse-GP method.

    Equation-8 and ICF-definition assumptions are required by default.
    Passing ``require_paper_assumptions=False`` is an explicit scientific
    opt-out and leaves the failed or unknown conditions in ``icf_validity``.
    LOO calibration is not allowed to shrink raw GP uncertainty unless
    ``allow_uncertainty_downscaling=True`` is explicitly requested; pointwise
    LOO is otherwise unsafe for time-correlated force errors.
    """
    if (colvar_file is None) == (data is None):
        raise ValueError("Provide exactly one of colvar_file or data")
    if isinstance(max_exact_observations, bool) or not isinstance(
        max_exact_observations, (int, np.integer)
    ) or max_exact_observations < 2:
        raise ValueError("max_exact_observations must be an integer >= 2")
    if not isinstance(require_paper_assumptions, (bool, np.bool_)):
        raise ValueError("require_paper_assumptions must be Boolean")
    if not isinstance(allow_uncertainty_downscaling, (bool, np.bool_)):
        raise ValueError("allow_uncertainty_downscaling must be Boolean")
    resolved_energy_factor = _positive_scalar_factor(
        energy_factor, "energy_factor"
    )
    if icf_factors is not None and resolved_energy_factor != 1.0:
        raise ValueError(
            "A non-default energy_factor cannot be combined with explicit "
            "icf_factors; omit icf_factors to derive force scaling as "
            "energy_factor / cv_factor"
        )

    validity_options = {
        "bias_depends_only_on_modeled_cvs": bias_depends_only_on_modeled_cvs,
        "quasi_equilibrium": quasi_equilibrium,
        "physical_force_excludes_bias": physical_force_excludes_bias,
        "metric_correction_included": metric_correction_included,
    }
    if colvar_file is not None:
        if cv_fields is None or icf_fields is None:
            raise ValueError(
                "cv_fields and icf_fields are required with colvar_file"
            )
        loaded = load_icf_data(
            colvar_file,
            cv_fields=cv_fields,
            icf_fields=icf_fields,
            time_field=time_field,
            cv_factors=cv_factors,
            energy_factor=resolved_energy_factor,
            icf_factors=icf_factors,
            start_time=start_time,
            stop_time=stop_time,
            stride=stride,
            strict=strict,
            **validity_options,
        )
    else:
        loaded = _normalize_icf_data(data)  # type: ignore[arg-type]
        if any(value is not None for value in validity_options.values()):
            loaded["validity"] = assess_icf_validity(**validity_options)

    validity = loaded["validity"]
    if require_paper_assumptions and (
        validity.get("paper_assumptions_satisfied") is not True
    ):
        raise ValueError(
            "Paper assumptions are false or unestablished; inspect "
            "result['icf_validity'] or explicitly establish all validity flags"
        )

    if aggregation_bins is not None:
        observations = aggregate_icf_observations(
            loaded,
            bins=aggregation_bins,
            ranges=aggregation_ranges,
            time_block_size=time_block_size,
            min_samples=min_samples_per_observation,
        )
        base_covariance = observations["force_noise_cov"]
        if force_noise_cov is not None or force_errors is not None:
            base_covariance = base_covariance + _force_covariance(
                len(observations["positions"]),
                observations["positions"].shape[1],
                force_noise_cov=force_noise_cov,
                force_errors=force_errors,
            )
        observations["force_noise_cov"] = base_covariance
    else:
        n_points, dimensions = loaded["positions"].shape
        n_components = n_points * dimensions
        if n_components > max_exact_observations:
            raise ValueError(
                f"Exact ICF GP received {n_points} observation points with "
                f"{n_components} scalar gradient components, above "
                f"max_exact_observations={max_exact_observations}; use "
                "aggregation_bins or a sparse-GP implementation"
            )
        observations = dict(loaded)
        observations["aggregation"] = {
            "kind": "none",
            "paper_raw_icf_observations": True,
            "trajectory_correlation_modeled": False,
        }
        observations["force_noise_cov"] = _force_covariance(
            len(observations["positions"]),
            observations["positions"].shape[1],
            force_noise_cov=(
                force_noise_cov
                if force_noise_cov is not None
                else data.get("force_noise_cov")
                if data is not None and "force_noise_cov" in data
                else None
            ),
            force_errors=force_errors,
        )

    from .gradient_gpr import fit_icf_gp, predict_gradient_gp

    model = fit_icf_gp(
        observations["positions"],
        observations["collective_forces"],
        observations["force_noise_cov"],
        force_convention="thermodynamic_force",
        optimize_hyperparams=optimize_hyperparams,
        fixed_lengthscale=fixed_lengthscale,
        fixed_sigma_f=fixed_sigma_f,
        calibrate_uncertainty=calibrate_uncertainty,
        allow_uncertainty_downscaling=allow_uncertainty_downscaling,
    )

    if prediction_points is None:
        query, axes, grid_shape = _prediction_grid(
            observations["positions"], grid_n
        )
    else:
        query = np.asarray(prediction_points, dtype=float)
        if query.ndim == 1 and observations["positions"].shape[1] == 1:
            query = query[:, None]
        axes = ()
        grid_shape = (len(query),)

    if reference is None:
        support_prediction = predict_gradient_gp(
            model,
            observations["positions"],
            calibrated=calibrate_uncertainty,
            prediction_batch_size=prediction_batch_size,
        )
        reference = observations["positions"][
            int(np.argmin(support_prediction["mean"]))
        ]
        reference_source = "lowest_prediction_on_observation_support"
    else:
        reference_source = "explicit"
    prediction = predict_gradient_gp(
        model,
        query,
        reference=reference,
        calibrated=calibrate_uncertainty,
        prediction_batch_size=prediction_batch_size,
    )

    result = dict(model)
    result.update({
        "method": "icf_gradient_gpr",
        "collective_forces": observations["collective_forces"],
        "force_convention": "thermodynamic_force",
        "gradient_relation": "grad_A = -conditional_mean(ICF)",
        "icf_validity": validity,
        "aggregation": observations["aggregation"],
        "prediction": prediction,
        "prediction_points": query,
        "grid_axes": axes,
        "grid_shape": grid_shape,
        "pmf": prediction["mean"].reshape(grid_shape),
        "pmf_std": prediction["std"].reshape(grid_shape),
        "pmf_std_raw": prediction["std_raw"].reshape(grid_shape),
        "pmf_std_calibrated": prediction["std_calibrated"].reshape(grid_shape),
        "reference_source": reference_source,
        "cv_names": observations["cv_names"],
        "icf_fields": loaded.get("icf_fields"),
        "source_file": loaded.get("source_file"),
        "input_table": loaded.get("table"),
        "convergence_evidence": {
            "assessed": False,
            "note": (
                "This fit alone does not assess time-block stability, repeated "
                "round trips, or hidden-coordinate equilibration."
            ),
        },
    })
    for name in (
        "cv_factors", "energy_factor", "icf_factors", "unit_provenance"
    ):
        if name in observations:
            result[name] = observations[name]
    if axes and len(axes) == 1:
        result["x_star"] = axes[0]
        result["pmf_mean"] = result["pmf"]
    elif axes and len(axes) == 2:
        result["gx"], result["gy"] = axes
    return result
