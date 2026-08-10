"""OPES-specific loading, reweighting, and convergence diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .biased_sampling import (
    block_reweighted_pmf_1d,
    cumulative_reweighted_pmf_1d,
    detect_hysteretic_transitions,
    log_reweighting_weights,
    pmf_landmarks_1d,
    region_weight_diagnostics,
    reweighted_pmf_1d,
)
from .plumed_io import read_plumed_table


def _field_tuple(fields: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(fields, str):
        return (fields,)
    return tuple(str(field) for field in fields)


def _optional_finite_float(value: float | None, name: str) -> float | None:
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


def _is_rct_field(field: str) -> bool:
    return field.lower().rsplit(".", 1)[-1] == "rct"


def _diagnostic_fields(bias_field: str) -> tuple[str, ...]:
    if bias_field.endswith(".bias"):
        prefix = bias_field[:-4]
    elif bias_field == "bias":
        prefix = ""
    else:
        prefix = bias_field.rsplit(".", 1)[0] + "."
    return tuple(prefix + suffix for suffix in ("rct", "zed", "neff", "nker"))


def load_opes_colvar(
    paths: str | Path | Iterable[str | Path],
    *,
    cv_fields: str | Iterable[str],
    bias_field: str = "opes.bias",
    other_bias_fields: str | Iterable[str] = (),
    time_field: str = "time",
    start_time: float | None = None,
    stop_time: float | None = None,
    stride: int = 1,
    allow_unlisted_bias_fields: bool = False,
    strict: bool = False,
) -> dict:
    """Load restart-aware OPES COLVAR data by named fields.

    The total applied bias is the instantaneous OPES bias plus every field
    explicitly listed in *other_bias_fields*.  ``opes.rct`` is a convergence
    diagnostic, not a reweighting offset, and is rejected if supplied as a
    bias field.  Time bounds are inclusive, and *stride* is applied after
    restart deduplication and time selection.
    """
    cvs = _field_tuple(cv_fields)
    other_biases = _field_tuple(other_bias_fields)
    if not cvs:
        raise ValueError("At least one cv_field is required")
    if len(set(cvs)) != len(cvs):
        raise ValueError("cv_fields must be unique")
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
    bias_fields = (str(bias_field),) + other_biases
    if len(set(bias_fields)) != len(bias_fields):
        raise ValueError("bias fields must be unique")
    invalid_rct = [field for field in bias_fields if _is_rct_field(field)]
    if invalid_rct:
        raise ValueError(
            "OPES rct is diagnostic only and must not be used for "
            "reweighting: " + ", ".join(invalid_rct)
        )

    required = (time_field,) + cvs + bias_fields
    table = read_plumed_table(
        paths,
        required_fields=required,
        deduplicate_field=time_field,
        duplicate_policy="last",
        strict=strict,
    )
    if table["n_records"] == 0:
        raise ValueError("OPES COLVAR contains no valid numeric records")

    columns = table["columns"]
    available_bias_fields = tuple(
        field
        for field in table["fields"]
        if field == "bias" or field.endswith(".bias")
    )
    unlisted_bias_fields = tuple(
        field for field in available_bias_fields if field not in bias_fields
    )
    if unlisted_bias_fields and not allow_unlisted_bias_fields:
        raise ValueError(
            "COLVAR contains unlisted bias fields that would be omitted from "
            "reweighting: " + ", ".join(unlisted_bias_fields) + ". List every "
            "applied bias in other_bias_fields, or explicitly set "
            "allow_unlisted_bias_fields=True."
        )
    all_time = np.asarray(columns[time_field], dtype=float)
    if np.any(np.diff(all_time) <= 0.0):
        raise ValueError(
            "OPES time must be strictly increasing after restart "
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
            "At least two OPES records are required after time and stride "
            "selection"
        )
    time = all_time[indices].copy()
    cv_values = np.column_stack([
        columns[field][indices] for field in cvs
    ])
    bias_components = {
        field: np.asarray(columns[field][indices], dtype=float)
        for field in bias_fields
    }
    total_bias = np.sum(
        np.column_stack(list(bias_components.values())), axis=1
    )
    diagnostics = {
        field: np.asarray(columns[field][indices], dtype=float)
        for field in _diagnostic_fields(bias_field)
        if field in columns
    }
    return {
        "table": table,
        "time": time,
        "time_field": time_field,
        "cv_fields": cvs,
        "cv_values": cv_values,
        "bias_fields": bias_fields,
        "bias_components": bias_components,
        "total_bias": total_bias,
        "diagnostics": diagnostics,
        "available_bias_fields": available_bias_fields,
        "unlisted_bias_fields": unlisted_bias_fields,
        "allow_unlisted_bias_fields": bool(allow_unlisted_bias_fields),
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
    }


def opes_convergence_diagnostics(
    opes_data: dict,
    *,
    tail_fraction: float = 0.2,
) -> dict:
    """Summarize OPES state series without declaring global convergence.

    ``rct`` and ``zed`` should flatten, while ``neff`` should grow.  The
    returned tail changes and monotonicity counts are evidence for judging
    those conditions; they are deliberately not collapsed into a universal
    converged/not-converged flag.
    """
    tail_fraction = float(tail_fraction)
    if not np.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1]")
    diagnostics = opes_data.get("diagnostics", {})
    n_records = len(opes_data.get("time", ()))
    tail_records = max(1, int(np.ceil(tail_fraction * n_records)))
    summaries = {}
    for field, values in diagnostics.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) != n_records:
            raise ValueError(f"Diagnostic field {field!r} has invalid shape")
        if np.any(~np.isfinite(values)):
            raise ValueError(f"Diagnostic field {field!r} contains nonfinite values")
        tail = values[-tail_records:]
        summary = {
            "initial": float(values[0]),
            "final": float(values[-1]),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "tail_initial": float(tail[0]),
            "tail_final": float(tail[-1]),
            "tail_drift": float(tail[-1] - tail[0]),
            "tail_range": float(np.ptp(tail)),
        }
        component = field.lower().rsplit(".", 1)[-1]
        if component == "neff":
            summary["decrease_count"] = int(np.count_nonzero(np.diff(values) < 0))
        if component == "nker":
            summary["change_count"] = int(np.count_nonzero(np.diff(values) != 0))
        summaries[field] = summary

    expected = set(_diagnostic_fields(opes_data["bias_fields"][0]))
    return {
        "n_records": n_records,
        "tail_fraction": tail_fraction,
        "tail_records": tail_records,
        "series": summaries,
        "missing_fields": tuple(sorted(expected - set(diagnostics))),
        "rct_used_for_reweighting": False,
        "interpretation": (
            "Tail stability and OPES effective-sample diagnostics are necessary "
            "checks, not proof of global free-energy convergence"
        ),
    }


def free_energy_from_opes_bias(
    bias,
    biasfactor: float,
    *,
    zero_minimum: bool = True,
) -> np.ndarray:
    r"""Convert an evaluated converged OPES bias to a free-energy estimate.

    For the well-tempered OPES target,
    ``F = -V / (1 - 1 / gamma)`` up to an additive constant.  This function
    converts an already evaluated bias grid; it does not approximate or
    reimplement PLUMED's compressed-kernel state evaluator.
    """
    bias_values = np.asarray(bias, dtype=float)
    if bias_values.size == 0 or np.any(~np.isfinite(bias_values)):
        raise ValueError("bias must contain finite values")
    gamma = float(biasfactor)
    if not (np.isinf(gamma) or (np.isfinite(gamma) and gamma > 1)):
        raise ValueError("biasfactor must be greater than one or infinity")
    scale = 1.0 if np.isinf(gamma) else 1.0 - 1.0 / gamma
    free_energy = -bias_values / scale
    if zero_minimum:
        free_energy = free_energy - np.min(free_energy)
    return free_energy


def read_opes_state(path: str | Path) -> dict:
    """Read OPES state kernels and their ``#! SET`` metadata.

    Kernel rows are returned for inspection and provenance.  Exact bias-grid
    evaluation remains the responsibility of the matching PLUMED version.
    """
    table = read_plumed_table(path, deduplicate_field=None)
    if table["n_records"] == 0:
        raise ValueError("OPES state contains no kernels")
    action = table["settings"].get("action")
    if action is not None and action != "OPES_METAD_state":
        raise ValueError(
            "OPES state action must be 'OPES_METAD_state' when present"
        )
    if "biasfactor" not in table["settings"]:
        raise ValueError("OPES state is missing '#! SET biasfactor'")
    try:
        biasfactor = float(table["settings"]["biasfactor"])
    except ValueError as exc:
        raise ValueError("OPES state biasfactor is not numeric") from exc
    if not (np.isinf(biasfactor) or (np.isfinite(biasfactor) and biasfactor > 1)):
        raise ValueError("OPES state biasfactor must be greater than one")
    biasfactor_history = table["setting_history"].get("biasfactor", ())
    try:
        historical_biasfactors = np.asarray(biasfactor_history, dtype=float)
    except ValueError as exc:
        raise ValueError("OPES state biasfactor history is not numeric") from exc
    if (
        len(historical_biasfactors) > 1
        and not np.allclose(
            historical_biasfactors,
            historical_biasfactors[0],
            rtol=1e-12,
            atol=0.0,
        )
    ):
        raise ValueError("OPES state contains conflicting biasfactor settings")
    adaptive_counter = None
    if "adaptive_counter" in table["settings"]:
        try:
            counter_value = float(table["settings"]["adaptive_counter"])
        except ValueError as exc:
            raise ValueError("OPES state adaptive_counter is not numeric") from exc
        if (
            not np.isfinite(counter_value)
            or counter_value < 0
            or not counter_value.is_integer()
        ):
            raise ValueError(
                "OPES state adaptive_counter must be a non-negative integer"
            )
        adaptive_counter = int(counter_value)
    return {
        "table": table,
        "action": action,
        "biasfactor": biasfactor,
        "adaptive_counter": adaptive_counter,
        "n_kernels": table["n_records"],
    }


def analyze_opes_1d(
    paths: str | Path | Iterable[str | Path],
    *,
    cv_field: str,
    temperature: float,
    bias_field: str = "opes.bias",
    other_bias_fields: str | Iterable[str] = (),
    time_field: str = "time",
    start_time: float | None = None,
    stop_time: float | None = None,
    stride: int = 1,
    energy_unit: str = "eV",
    bias_energy_unit: str | None = None,
    bins=100,
    value_range: tuple[float, float] | None = None,
    min_count: int = 1,
    cumulative_cutoffs: Iterable[float] | None = None,
    n_blocks: int | None = None,
    regions: Mapping[str, tuple[float, float]] | None = None,
    basin_a: tuple[float, float] | None = None,
    basin_b: tuple[float, float] | None = None,
    transition_region: tuple[float, float] | None = None,
    diagnostic_tail_fraction: float = 0.2,
    allow_unlisted_bias_fields: bool = False,
    strict: bool = False,
) -> dict:
    """Run direct 1D OPES reweighting and numerical convergence summaries.

    ``bias_energy_unit`` describes the numerical unit of the printed bias
    columns.  It defaults to the requested PMF ``energy_unit`` only when
    omitted; the two units otherwise remain independent.
    """
    resolved_bias_energy_unit = (
        energy_unit if bias_energy_unit is None else bias_energy_unit
    )
    bias_energy_unit_source = (
        "output_energy_unit_default"
        if bias_energy_unit is None
        else "explicit_bias_energy_unit"
    )
    data = load_opes_colvar(
        paths,
        cv_fields=(cv_field,),
        bias_field=bias_field,
        other_bias_fields=other_bias_fields,
        time_field=time_field,
        start_time=start_time,
        stop_time=stop_time,
        stride=stride,
        allow_unlisted_bias_fields=allow_unlisted_bias_fields,
        strict=strict,
    )
    log_weights = log_reweighting_weights(
        data["total_bias"],
        temperature=temperature,
        energy_unit=resolved_bias_energy_unit,
    )
    pmf = reweighted_pmf_1d(
        data["cv_values"][:, 0],
        log_weights,
        bins=bins,
        value_range=value_range,
        temperature=temperature,
        energy_unit=energy_unit,
        min_count=min_count,
    )
    cumulative = None
    if cumulative_cutoffs is not None:
        cumulative = cumulative_reweighted_pmf_1d(
            data["cv_values"][:, 0],
            data["time"],
            log_weights,
            cumulative_cutoffs,
            bins=pmf["bin_edges"],
            temperature=temperature,
            energy_unit=energy_unit,
            min_count=min_count,
        )
    blocks = None
    if n_blocks is not None:
        blocks = block_reweighted_pmf_1d(
            data["cv_values"][:, 0],
            log_weights,
            n_blocks=n_blocks,
            bins=pmf["bin_edges"],
            temperature=temperature,
            energy_unit=energy_unit,
            min_count=min_count,
        )
    regional_evidence = None
    if regions is not None:
        regional_evidence = region_weight_diagnostics(
            data["cv_values"][:, 0],
            pmf["normalized_weights"],
            regions,
        )
    if (basin_a is None) != (basin_b is None):
        raise ValueError("basin_a and basin_b must be supplied together")
    if transition_region is not None and basin_a is None:
        raise ValueError(
            "transition_region requires both basin_a and basin_b"
        )
    landmarks = None
    transitions = None
    if basin_a is not None and basin_b is not None:
        landmarks = pmf_landmarks_1d(
            pmf,
            basin_a,
            basin_b,
            transition_region=transition_region,
        )
        bounds_a = landmarks["basin_a"]["bounds"]
        bounds_b = landmarks["basin_b"]["bounds"]
        cv_values = data["cv_values"][:, 0]
        transitions = detect_hysteretic_transitions(
            (cv_values >= bounds_a[0]) & (cv_values <= bounds_a[1]),
            (cv_values >= bounds_b[0]) & (cv_values <= bounds_b[1]),
            times=data["time"],
            state_names=("basin_a", "basin_b"),
        )
        if cumulative is not None:
            for snapshot in cumulative["snapshots"]:
                if snapshot.get("status") == "ok":
                    snapshot["landmarks"] = pmf_landmarks_1d(
                        snapshot,
                        basin_a,
                        basin_b,
                        transition_region=transition_region,
                    )
        if blocks is not None:
            block_landmark_support = []
            for block in blocks["blocks"]:
                if block.get("status") == "ok":
                    block["landmarks"] = pmf_landmarks_1d(
                        block,
                        basin_a,
                        basin_b,
                        transition_region=transition_region,
                    )
                    block_landmarks = block["landmarks"]
                    block_landmark_support.append({
                        "block_index": int(block["block_index"]),
                        "profile_status": "ok",
                        "supported_bins": int(np.count_nonzero(
                            block["support_mask"]
                        )),
                        "landmark_status": block_landmarks["status"],
                        "basin_a_status": block_landmarks["basin_a"]["status"],
                        "basin_a_supported_bins": int(
                            block_landmarks["basin_a"]["supported_bins"]
                        ),
                        "basin_b_status": block_landmarks["basin_b"]["status"],
                        "basin_b_supported_bins": int(
                            block_landmarks["basin_b"]["supported_bins"]
                        ),
                        "population_delta_f_status": block_landmarks[
                            "population_delta_f_status"
                        ],
                    })
                else:
                    block_landmark_support.append({
                        "block_index": int(block["block_index"]),
                        "profile_status": block.get("status"),
                        "supported_bins": 0,
                        "landmark_status": "unavailable",
                        "basin_a_status": "unavailable",
                        "basin_a_supported_bins": 0,
                        "basin_b_status": "unavailable",
                        "basin_b_supported_bins": 0,
                        "population_delta_f_status": "unavailable",
                    })
            blocks["landmark_support"] = block_landmark_support
    return {
        "method": "direct_total_bias_reweighting",
        "opes_data": data,
        "log_weights": log_weights,
        "pmf": pmf,
        "cumulative": cumulative,
        "blocks": blocks,
        "regions": regional_evidence,
        "landmarks": landmarks,
        "transitions": transitions,
        "opes_diagnostics": opes_convergence_diagnostics(
            data, tail_fraction=diagnostic_tail_fraction
        ),
        "convergence_evidence": {
            "cumulative_profiles": cumulative is not None,
            "disjoint_block_profiles": blocks is not None,
            "regional_weight_diagnostics": regional_evidence is not None,
            "observable_landmarks": landmarks is not None,
            "hysteretic_transition_events": transitions is not None,
            "completed_round_trips": (
                None
                if transitions is None
                else int(transitions["completed_round_trips"])
            ),
            "interpretation": (
                "Observable-specific numerical evidence only; recurrence, "
                "hidden-coordinate equilibration, and independent replicas "
                "remain separate requirements."
            ),
        },
        "rct_used_for_reweighting": False,
        "energy_unit": energy_unit,
        "bias_energy_unit": resolved_bias_energy_unit,
        "unit_provenance": {
            "pmf_energy_unit": energy_unit,
            "bias_energy_unit": resolved_bias_energy_unit,
            "bias_energy_unit_source": bias_energy_unit_source,
            "loader_applied_unit_conversion": False,
        },
    }
