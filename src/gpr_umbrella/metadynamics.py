"""Metadynamics input, reweighting, and convergence-evidence helpers.

This module deliberately keeps two metadynamics estimators separate:

* a density estimate obtained from normalized per-frame log weights; and
* a conventional bias estimate reconstructed from ``HILLS`` by PLUMED.

Neither estimator is an instantaneous-collective-force (ICF) reconstruction.
The latter lives in :mod:`gpr_umbrella.icf`.  Keeping these routes explicit is
important because a raw, time-dependent metadynamics bias is not a valid
replacement for the normalized ``bias - c(t)`` used for reweighting.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess

import numpy as np

from .plumed_io import read_plumed_table


__all__ = [
    "analyze_metadynamics",
    "load_hills_diagnostics",
    "load_metadynamics_data",
    "run_sum_hills",
]


def _fallback_read_plumed_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a named PLUMED table when the shared parser is unavailable."""
    path = Path(path)
    fields: tuple[str, ...] | None = None
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#! FIELDS"):
                current = tuple(stripped.split()[2:])
                if not current or len(set(current)) != len(current):
                    raise ValueError(
                        f"Invalid or duplicate PLUMED fields in {path}"
                    )
                if fields is not None and current != fields:
                    raise ValueError(
                        f"PLUMED fields change within {path} at line "
                        f"{line_number}"
                    )
                fields = current
                continue
            if stripped.startswith("#"):
                continue
            if fields is None:
                raise ValueError(f"Missing '#! FIELDS' header in {path}")
            parts = stripped.split()
            if len(parts) != len(fields):
                raise ValueError(
                    f"Expected {len(fields)} columns in {path} at line "
                    f"{line_number}, found {len(parts)}"
                )
            try:
                rows.append([float(value) for value in parts])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric PLUMED row in {path} at line {line_number}"
                ) from exc
    if fields is None:
        raise ValueError(f"Missing '#! FIELDS' header in {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"PLUMED data in {path} must be finite")
    return {name: values[:, index] for index, name in enumerate(fields)}


def _normalize_plumed_table(table, path: str | Path) -> dict[str, np.ndarray]:
    """Normalize shared-parser return variants to a field-to-array mapping."""
    if isinstance(table, Mapping):
        columns = table.get("columns")
        if isinstance(columns, Mapping):
            candidate = columns
        elif "fields" in table and ("data" in table or "values" in table):
            fields = tuple(table["fields"])
            values = np.asarray(table.get("data", table.get("values")), dtype=float)
            if values.ndim != 2 or values.shape[1] != len(fields):
                raise ValueError(f"Invalid PLUMED table returned for {path}")
            candidate = {
                name: values[:, index] for index, name in enumerate(fields)
            }
        else:
            candidate = {
                name: value
                for name, value in table.items()
                if isinstance(name, str)
                and np.asarray(value).ndim == 1
                and np.issubdtype(np.asarray(value).dtype, np.number)
            }
    else:
        fields = getattr(table, "fields", None)
        columns = getattr(table, "columns", None)
        if isinstance(columns, Mapping):
            candidate = columns
        else:
            values = getattr(table, "data", getattr(table, "values", None))
            if fields is None or values is None:
                raise TypeError(
                    "read_plumed_table must return named numeric columns"
                )
            values = np.asarray(values, dtype=float)
            candidate = {
                name: values[:, index]
                for index, name in enumerate(tuple(fields))
            }

    result = {
        str(name): np.asarray(value, dtype=float)
        for name, value in candidate.items()
    }
    if not result:
        raise ValueError(f"No named numeric columns found in {path}")
    lengths = {array.shape for array in result.values()}
    if len(lengths) != 1 or next(iter(lengths))[0] == 0:
        raise ValueError(f"PLUMED columns in {path} have inconsistent lengths")
    if any(array.ndim != 1 for array in result.values()):
        raise ValueError(f"PLUMED columns in {path} must be one-dimensional")
    if any(not np.all(np.isfinite(array)) for array in result.values()):
        raise ValueError(f"PLUMED data in {path} must be finite")
    return result


def _read_plumed_mapping(path: str | Path) -> dict[str, np.ndarray]:
    """Read a PLUMED table through the shared parser, with a local fallback."""
    try:
        from .plumed_io import read_plumed_table
    except (ImportError, AttributeError):
        return _fallback_read_plumed_table(path)
    return _normalize_plumed_table(read_plumed_table(path), path)


def _require_fields(
    columns: Mapping[str, np.ndarray],
    names: Sequence[str],
    path: str | Path,
) -> None:
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(
            f"Missing PLUMED field(s) {', '.join(missing)} in {path}"
        )


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _optional_finite_float(value: float | None, name: str) -> float | None:
    """Return a finite optional float while rejecting Boolean sentinels."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite or None")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite or None") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite or None")
    return result


def _printed_bias_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Return PLUMED fields that conventionally contain applied biases."""
    return tuple(
        field for field in fields
        if field == "bias" or field.endswith(".bias")
    )


def _raw_field_replaced_by_rbias(rbias_field: str) -> str | None:
    """Map ``action.rbias`` to its printed raw ``action.bias`` counterpart."""
    if rbias_field == "rbias":
        return "bias"
    if rbias_field.endswith(".rbias"):
        return rbias_field[:-len(".rbias")] + ".bias"
    return None


def load_metadynamics_data(
    colvar_file: str | Path,
    *,
    cv_fields: Sequence[str],
    time_field: str = "time",
    logweight_field: str | None = None,
    rbias_field: str | None = None,
    extra_bias_fields: Sequence[str] = (),
    raw_bias_field: str | None = None,
    allow_quasistatic_bias: bool = False,
    allow_unlisted_bias_fields: bool = False,
    thermal_energy: float | None = None,
    bias_energy_factor: float = 1.0,
    strict: bool = False,
    start_time: float | None = None,
    stop_time: float | None = None,
    stride: int = 1,
) -> dict:
    """Load named metadynamics data and construct dimensionless log weights.

    Exactly one weight source is required.  A precomputed *logweight_field* is
    already dimensionless.  An *rbias_field* is interpreted as the normalized
    metadynamics bias ``bias - c(t)`` and is combined with any static
    *extra_bias_fields* before division by ``thermal_energy``.

    Every printed ``*.bias`` field must be accounted for.  When
    *rbias_field* is ``action.rbias``, the corresponding ``action.bias`` is
    treated as replaced by that normalized source, not added a second time.
    Other walls and restraints must be named in *extra_bias_fields*.  The
    audit escape hatch *allow_unlisted_bias_fields* records, but does not fix,
    omitted fields.

    *bias_energy_factor* explicitly converts every supplied bias-energy field
    into the same energy unit as *thermal_energy* before division.  A value of
    one is therefore an explicit assertion that those units already match; no
    unit is inferred from the file.

    A raw time-dependent bias is rejected unless
    ``allow_quasistatic_bias=True`` explicitly records the user's assumption
    that the selected trajectory segment sees a static or quasi-static bias.
    That route additionally requires finite *start_time* and *stop_time* so it
    can never silently treat the entire adaptive run as quasi-static.  Bounds
    are inclusive, and *stride* is applied after time selection and restart
    deduplication.  This explicit bounding still does not verify stationarity.
    """
    cv_fields = tuple(cv_fields)
    extra_bias_fields = tuple(extra_bias_fields)
    if not cv_fields or len(set(cv_fields)) != len(cv_fields):
        raise ValueError("cv_fields must contain unique field names")
    if len(set(extra_bias_fields)) != len(extra_bias_fields):
        raise ValueError("extra_bias_fields must be unique")
    if not isinstance(allow_unlisted_bias_fields, (bool, np.bool_)):
        raise ValueError("allow_unlisted_bias_fields must be Boolean")
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise ValueError("stride must be a positive integer")
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    start_time = _optional_finite_float(start_time, "start_time")
    stop_time = _optional_finite_float(stop_time, "stop_time")
    if (
        start_time is not None
        and stop_time is not None
        and stop_time <= start_time
    ):
        raise ValueError("stop_time must be greater than start_time")
    sources = [
        logweight_field is not None,
        rbias_field is not None,
        raw_bias_field is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Specify exactly one of logweight_field, rbias_field, or "
            "raw_bias_field"
        )
    if logweight_field is not None and extra_bias_fields:
        raise ValueError(
            "extra_bias_fields cannot be combined with a precomputed log "
            "weight; include all biases when producing that field"
        )
    if raw_bias_field is not None and not allow_quasistatic_bias:
        raise ValueError(
            "Raw time-dependent metadynamics bias is not a normalized "
            "reweighting factor. Supply logweight_field or rbias_field, or "
            "set allow_quasistatic_bias=True for an explicitly selected "
            "quasi-static segment."
        )
    if raw_bias_field is not None and (
        start_time is None or stop_time is None
    ):
        raise ValueError(
            "Quasi-static raw-bias analysis requires explicit finite "
            "start_time and stop_time bounds"
        )

    source_field = logweight_field or rbias_field or raw_bias_field
    if source_field in extra_bias_fields:
        raise ValueError(
            "The primary weight source cannot also appear in "
            "extra_bias_fields"
        )
    replaced_raw_bias = (
        _raw_field_replaced_by_rbias(rbias_field)
        if rbias_field is not None else None
    )
    if replaced_raw_bias is not None and replaced_raw_bias in extra_bias_fields:
        raise ValueError(
            f"{replaced_raw_bias!r} is replaced by normalized field "
            f"{rbias_field!r} and must not be added again"
        )

    path = Path(colvar_file)
    requested = [time_field, *cv_fields, *extra_bias_fields]
    requested.append(source_field)  # type: ignore[arg-type]
    table = read_plumed_table(
        path,
        required_fields=requested,
        deduplicate_field=time_field,
        duplicate_policy="last",
        strict=strict,
    )
    if table["n_records"] == 0:
        raise ValueError("Metadynamics COLVAR contains no valid numeric records")
    columns = table["columns"]

    available_bias_fields = _printed_bias_fields(table["fields"])
    accounted_bias_fields = set(extra_bias_fields)
    if raw_bias_field is not None:
        accounted_bias_fields.add(raw_bias_field)
    elif rbias_field is not None and replaced_raw_bias is not None:
        # ``rbias`` is bias-c(t); its matching raw bias is diagnostic here and
        # must not be added to the normalized reweighting source.
        accounted_bias_fields.add(replaced_raw_bias)
    unlisted_bias_fields = tuple(
        field for field in available_bias_fields
        if field not in accounted_bias_fields
    )
    if unlisted_bias_fields and not allow_unlisted_bias_fields:
        raise ValueError(
            "COLVAR contains unlisted bias fields that would be omitted from "
            "metadynamics reweighting: " + ", ".join(unlisted_bias_fields)
            + ". List every additional wall/restraint in extra_bias_fields, "
            "or explicitly set allow_unlisted_bias_fields=True after "
            "auditing the precomputed log weights."
        )

    all_time = columns[time_field]
    if np.any(np.diff(all_time) <= 0.0):
        raise ValueError(
            "COLVAR time must be strictly increasing after restart "
            "deduplication"
        )
    within_bounds = np.ones(len(all_time), dtype=bool)
    if start_time is not None:
        within_bounds &= all_time >= start_time
    if stop_time is not None:
        within_bounds &= all_time <= stop_time
    bounded_indices = np.flatnonzero(within_bounds)
    indices = bounded_indices[::int(stride)]
    if len(indices) < 2:
        raise ValueError(
            "At least two metadynamics records are required after time and "
            "stride selection"
        )
    time = all_time[indices].copy()
    positions = np.column_stack([
        columns[name][indices] for name in cv_fields
    ])
    factor = _positive_float(bias_energy_factor, "bias_energy_factor")

    if logweight_field is not None:
        log_weights = columns[logweight_field][indices].copy()
        weight_source = "explicit_logweight"
        normalized_time_dependent_bias = True
        quasistatic_assumption = False
    else:
        if thermal_energy is None:
            raise ValueError(
                "thermal_energy is required when constructing weights from "
                "bias energies"
            )
        kbt = _positive_float(thermal_energy, "thermal_energy")
        bias_name = rbias_field if rbias_field is not None else raw_bias_field
        total_bias = columns[bias_name][indices].copy()  # type: ignore[index]
        for name in extra_bias_fields:
            total_bias += columns[name][indices]
        log_weights = factor * total_bias / kbt
        weight_source = (
            "normalized_rbias" if rbias_field is not None
            else "quasistatic_raw_bias"
        )
        normalized_time_dependent_bias = rbias_field is not None
        quasistatic_assumption = raw_bias_field is not None

    if not np.all(np.isfinite(log_weights)):
        raise ValueError("Constructed metadynamics log weights must be finite")

    return {
        "source_file": str(path),
        "table": table,
        "time": time,
        "positions": positions,
        "cv_names": cv_fields,
        "log_weights": log_weights,
        "weight_source": weight_source,
        "thermal_energy": thermal_energy,
        "bias_energy_factor": factor,
        "bias_energy_conversion": {
            "factor_into_thermal_energy_unit": factor,
            "applied": logweight_field is None,
            "input_unit_inferred": False,
            "unit_contract": (
                "Each bias-energy column is multiplied by "
                "bias_energy_factor to reach the unit of thermal_energy; "
                "factor=1 explicitly asserts the units already match."
            ),
        },
        "extra_bias_fields": extra_bias_fields,
        "available_bias_fields": available_bias_fields,
        "accounted_bias_fields": tuple(
            field for field in available_bias_fields
            if field in accounted_bias_fields
        ),
        "unlisted_bias_fields": unlisted_bias_fields,
        "allow_unlisted_bias_fields": bool(allow_unlisted_bias_fields),
        "rbias_replaces_raw_bias_field": replaced_raw_bias,
        "selection": {
            "requested_start_time": start_time,
            "requested_stop_time": stop_time,
            "stride": int(stride),
            "n_records_before_selection": int(len(all_time)),
            "n_records_within_time_bounds": int(len(bounded_indices)),
            "n_records_selected": int(len(indices)),
            "selected_start_time": float(time[0]),
            "selected_stop_time": float(time[-1]),
            "explicit_finite_time_range": (
                start_time is not None and stop_time is not None
            ),
            "bounds_inclusive": True,
            "selection_order": (
                "restart_deduplication", "time_bounds", "stride"
            ),
        },
        "validity": {
            "estimator": "metadynamics_reweighted_density",
            "normalized_time_dependent_bias": normalized_time_dependent_bias,
            "quasistatic_bias_assumed": quasistatic_assumption,
            "quasistatic_segment_explicitly_bounded": bool(
                quasistatic_assumption
                and start_time is not None
                and stop_time is not None
            ),
            "assumption_verified_by_code": False,
            "note": (
                "These fields document estimator validity; they are not a "
                "universal convergence decision."
            ),
        },
    }


def _normalized_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Return stable normalized weights and Kish effective sample size."""
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 1 or len(log_weights) == 0:
        raise ValueError("log_weights must be a non-empty one-dimensional array")
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Metadynamics weights have zero or invalid total")
    square_total = float(np.dot(weights, weights))
    ess = float(total**2 / square_total)
    weights /= total
    return weights, ess


def _histogram_analysis(
    positions: np.ndarray,
    log_weights: np.ndarray,
    *,
    bins,
    ranges,
    thermal_energy: float,
) -> dict:
    """Build a weighted histogram/FES and local weight diagnostics."""
    positions = np.asarray(positions, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    if positions.ndim != 2 or positions.shape[1] == 0:
        raise ValueError("positions must have shape (n_frames, n_CVs)")
    if log_weights.ndim != 1 or len(log_weights) != len(positions):
        raise ValueError("log_weights must match the number of position frames")
    if len(log_weights) == 0:
        raise ValueError("positions and log_weights must not be empty")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(log_weights)):
        raise ValueError("positions and log_weights must contain only finite values")
    thermal_energy = _positive_float(thermal_energy, "thermal_energy")

    weights, ess = _normalized_weights(log_weights)
    # Let NumPy resolve integer/explicit edge specifications, then assign every
    # frame ourselves.  Weighted histogramdd uses cumulative sums internally;
    # subtracting them can erase a tiny bin that follows a dominant one.
    _, edges = np.histogramdd(positions, bins=bins, range=ranges)
    shape = tuple(len(edge) - 1 for edge in edges)
    n_flat_bins = int(np.prod(shape))
    indices = []
    valid = np.ones(len(positions), dtype=bool)
    for dimension, edge in enumerate(edges):
        index = np.searchsorted(
            edge, positions[:, dimension], side="right"
        ) - 1
        index[positions[:, dimension] == edge[-1]] = len(edge) - 2
        valid &= (index >= 0) & (index < len(edge) - 1)
        indices.append(index)
    if not np.any(valid):
        raise ValueError("No metadynamics samples fall inside histogram range")
    flat_indices = np.ravel_multi_index(
        tuple(index[valid] for index in indices), shape
    )
    valid_logs = log_weights[valid]
    counts_flat = np.bincount(flat_indices, minlength=n_flat_bins)

    # One maximum-log shift per flat N-D bin retains both tiny probability
    # masses and local ESS/leverage across arbitrarily separated global scales.
    bin_maximum = np.full(n_flat_bins, -np.inf, dtype=float)
    np.maximum.at(bin_maximum, flat_indices, valid_logs)
    relative_logs = valid_logs - bin_maximum[flat_indices]
    scaled = np.exp(relative_logs)
    scaled_sums = np.zeros(n_flat_bins, dtype=float)
    scaled_square_sums = np.zeros(n_flat_bins, dtype=float)
    np.add.at(scaled_sums, flat_indices, scaled)
    np.add.at(scaled_square_sums, flat_indices, scaled**2)

    nonempty_flat = counts_flat > 0
    global_maximum = float(np.max(log_weights))
    global_scaled_sum = float(np.sum(np.exp(log_weights - global_maximum)))
    log_probability_flat = np.full(n_flat_bins, -np.inf, dtype=float)
    log_probability_flat[nonempty_flat] = (
        bin_maximum[nonempty_flat]
        - global_maximum
        + np.log(scaled_sums[nonempty_flat])
        - np.log(global_scaled_sum)
    )
    probability_mass = np.exp(log_probability_flat).reshape(shape)
    log_probability_mass = log_probability_flat.reshape(shape)
    counts = counts_flat.reshape(shape)

    local_ess_flat = np.zeros(n_flat_bins, dtype=float)
    local_ess_flat[nonempty_flat] = (
        scaled_sums[nonempty_flat] ** 2
        / scaled_square_sums[nonempty_flat]
    )
    local_ess = local_ess_flat.reshape(shape)
    local_leverage_flat = np.full(n_flat_bins, np.nan, dtype=float)
    local_leverage_flat[nonempty_flat] = (
        1.0 / scaled_sums[nonempty_flat]
    )
    local_leverage = local_leverage_flat.reshape(shape)

    # A density, unlike a probability mass, accounts for unequal bin widths.
    cell_volume = np.ones(shape, dtype=float)
    for dimension, edge in enumerate(edges):
        shape = [1] * positions.shape[1]
        shape[dimension] = len(edge) - 1
        cell_volume *= np.diff(edge).reshape(shape)
    log_density = log_probability_mass - np.log(cell_volume)
    density = np.exp(log_density)
    support = counts > 0
    free_energy = np.full_like(density, np.nan)
    free_energy[support] = -thermal_energy * log_density[support]
    free_energy[support] -= np.min(free_energy[support])

    return {
        "density": density,
        "log_density": log_density,
        "probability_mass": probability_mass,
        "log_probability_mass": log_probability_mass,
        "free_energy": free_energy,
        "support_mask": support,
        "bin_edges": tuple(edges),
        "bin_centers": tuple(
            0.5 * (edge[:-1] + edge[1:]) for edge in edges
        ),
        "counts": counts.astype(int),
        "local_ess": local_ess,
        "local_weight_leverage": local_leverage,
        "weights": weights,
        "global_ess": ess,
        "global_ess_fraction": ess / len(weights),
        "maximum_weight_fraction": 1.0 / global_scaled_sum,
    }


def _time_block_evidence(
    data: dict,
    *,
    edges: tuple[np.ndarray, ...],
    thermal_energy: float,
    n_blocks: int,
) -> tuple[list[dict], list[float]]:
    """Calculate aligned block FESs and common-support RMS differences."""
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, (int, np.integer)):
        raise ValueError("n_blocks must be an integer")
    if n_blocks < 1 or n_blocks > len(data["time"]):
        raise ValueError("n_blocks must lie between 1 and the number of frames")
    block_results: list[dict] = []
    for indices in np.array_split(np.arange(len(data["time"])), n_blocks):
        result = _histogram_analysis(
            data["positions"][indices],
            data["log_weights"][indices],
            bins=edges,
            ranges=None,
            thermal_energy=thermal_energy,
        )
        block_results.append({
            "start_index": int(indices[0]),
            "stop_index": int(indices[-1] + 1),
            "start_time": float(data["time"][indices[0]]),
            "stop_time": float(data["time"][indices[-1]]),
            "free_energy": result["free_energy"],
            "support_mask": result["support_mask"],
            "global_ess": result["global_ess"],
            "maximum_weight_fraction": result["maximum_weight_fraction"],
        })

    successive_rms: list[float] = []
    for previous, current in zip(block_results[:-1], block_results[1:]):
        common = previous["support_mask"] & current["support_mask"]
        if not np.any(common):
            successive_rms.append(float("nan"))
            continue
        first = previous["free_energy"][common]
        second = current["free_energy"][common]
        # Each profile was shifted to its own minimum. Remove the remaining
        # least-squares constant on the common support before comparing shape.
        difference = second - first
        difference -= np.mean(difference)
        successive_rms.append(float(np.sqrt(np.mean(difference**2))))
    return block_results, successive_rms


def analyze_metadynamics(
    colvar_file: str | Path,
    *,
    cv_fields: Sequence[str],
    thermal_energy: float,
    bins=100,
    ranges=None,
    n_blocks: int = 4,
    hills_files: str | Path | Sequence[str | Path] | None = None,
    **load_options,
) -> dict:
    """Analyze a metadynamics trajectory without issuing a binary verdict.

    The returned block-to-block differences, ESS values, and HILLS summaries
    are convergence *evidence*.  Interpreting them requires the scientific
    observable and state definitions, so no universal ``converged`` Boolean is
    produced.
    """
    kbt = _positive_float(thermal_energy, "thermal_energy")
    data = load_metadynamics_data(
        colvar_file,
        cv_fields=cv_fields,
        thermal_energy=kbt,
        **load_options,
    )
    histogram = _histogram_analysis(
        data["positions"],
        data["log_weights"],
        bins=bins,
        ranges=ranges,
        thermal_energy=kbt,
    )
    blocks, rms = _time_block_evidence(
        data,
        edges=histogram["bin_edges"],
        thermal_energy=kbt,
        n_blocks=n_blocks,
    )
    result = dict(data)
    result.update(histogram)
    result["time_blocks"] = blocks
    result["convergence_evidence"] = {
        "n_time_blocks": n_blocks,
        "successive_block_shape_rms": np.asarray(rms, dtype=float),
        "energy_unit": "same unit as thermal_energy",
        "interpretation": (
            "Evidence only; barrier, basin-population, and global sampling "
            "convergence must be assessed for the requested observable."
        ),
    }
    if hills_files is not None:
        result["hills_diagnostics"] = load_hills_diagnostics(
            hills_files,
            bias_energy_factor=load_options.get("bias_energy_factor", 1.0),
        )
    return result


def _as_paths(
    paths: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        result = (Path(paths),)
    else:
        result = tuple(Path(path) for path in paths)
    if not result:
        raise ValueError("At least one HILLS file is required")
    return result


def load_hills_diagnostics(
    hills_files: str | Path | Sequence[str | Path],
    *,
    time_field: str = "time",
    height_field: str = "height",
    cv_fields: Sequence[str] | None = None,
    bias_energy_factor: float = 1.0,
) -> dict:
    """Read HILLS deposition diagnostics without reconstructing an MTD FES."""
    paths = _as_paths(hills_files)
    factor = _positive_float(bias_energy_factor, "bias_energy_factor")
    walkers: list[dict] = []
    excluded = {time_field, height_field, "biasf", "clock"}
    for path in paths:
        columns = _read_plumed_mapping(path)
        _require_fields(columns, (time_field, height_field), path)
        inferred = tuple(
            name for name in columns
            if name not in excluded and not name.startswith("sigma_")
        )
        names = tuple(cv_fields) if cv_fields is not None else inferred
        if not names:
            raise ValueError(f"Could not infer biased CV fields from {path}")
        _require_fields(columns, names, path)
        time = columns[time_field].copy()
        heights = factor * columns[height_field]
        positive_intervals = np.diff(time)
        positive_intervals = positive_intervals[positive_intervals > 0.0]
        walkers.append({
            "source_file": str(path),
            "time": time,
            "centers": np.column_stack([columns[name] for name in names]),
            "cv_names": names,
            "heights": heights,
            "bias_factor": columns.get("biasf"),
            "n_hills": len(time),
            "time_monotonic": bool(np.all(np.diff(time) >= 0.0)),
            "median_deposition_interval": (
                float(np.median(positive_intervals))
                if len(positive_intervals) else float("nan")
            ),
            "initial_height": float(heights[0]),
            "final_height": float(heights[-1]),
            "height_ratio_final_to_initial": (
                float(heights[-1] / heights[0])
                if heights[0] != 0.0 else float("nan")
            ),
        })
    return {
        "walkers": walkers,
        "n_files": len(walkers),
        "n_hills": int(sum(walker["n_hills"] for walker in walkers)),
        "bias_energy_factor": factor,
        "interpretation": (
            "Hill decay and deposition history are diagnostics, not a "
            "standalone proof of free-energy convergence."
        ),
    }


def _comma_values(values: Sequence, name: str) -> str:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return ",".join(str(value) for value in values)


def run_sum_hills(
    hills_files: str | Path | Sequence[str | Path],
    *,
    output_file: str | Path,
    plumed_executable: str | Path = "plumed",
    bins: Sequence[int] | None = None,
    minimum: Sequence[float | str] | None = None,
    maximum: Sequence[float | str] | None = None,
    stride: int | None = None,
    thermal_energy: float | None = None,
    integrate_variables: Sequence[str] | None = None,
    mintozero: bool = True,
    negative_bias: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``plumed sum_hills`` through a shell-free argument boundary."""
    paths = _as_paths(hills_files)
    if any("," in str(path) for path in paths):
        raise ValueError("HILLS paths cannot contain commas")
    command = [
        str(plumed_executable),
        "sum_hills",
        "--hills",
        ",".join(str(path) for path in paths),
        "--outfile",
        str(output_file),
    ]
    if bins is not None:
        if any(isinstance(value, bool) or int(value) < 1 for value in bins):
            raise ValueError("bins values must be positive integers")
        command.extend(["--bin", _comma_values(tuple(int(v) for v in bins), "bins")])
    if minimum is not None:
        command.extend(["--min", _comma_values(minimum, "minimum")])
    if maximum is not None:
        command.extend(["--max", _comma_values(maximum, "maximum")])
    if stride is not None:
        if isinstance(stride, bool) or int(stride) < 1:
            raise ValueError("stride must be a positive integer")
        command.extend(["--stride", str(int(stride))])
    if thermal_energy is not None:
        command.extend([
            "--kt", str(_positive_float(thermal_energy, "thermal_energy"))
        ])
    if integrate_variables is not None:
        command.extend([
            "--idw", _comma_values(integrate_variables, "integrate_variables")
        ])
    if mintozero:
        command.append("--mintozero")
    if negative_bias:
        command.append("--negbias")
    try:
        return subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"PLUMED executable not found: {plumed_executable}"
        ) from exc
