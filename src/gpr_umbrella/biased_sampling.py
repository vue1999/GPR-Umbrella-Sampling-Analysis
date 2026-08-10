"""Shared numerical utilities for time-dependent biased simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


__all__ = [
    "block_reweighted_pmf_1d",
    "cumulative_reweighted_pmf_1d",
    "detect_hysteretic_transitions",
    "importance_weight_diagnostics",
    "log_reweighting_weights",
    "normalize_log_weights",
    "pmf_landmarks_1d",
    "region_weight_diagnostics",
    "reweighted_pmf_1d",
    "thermal_energy",
]


def thermal_energy(temperature: float, energy_unit: str = "eV") -> float:
    """Return ``k_B T`` in a supported numerical energy unit."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    normalized = energy_unit.strip().lower().replace(" ", "")
    boltzmann_constants = {
        "ev": 8.617333262145e-5,
        "kj/mol": 8.31446261815324e-3,
        "kjmol-1": 8.31446261815324e-3,
        "kcal/mol": 1.98720425864083e-3,
        "kcalmol-1": 1.98720425864083e-3,
        "j/mol": 8.31446261815324,
        "jmol-1": 8.31446261815324,
        "hartree": 3.1668115634556e-6,
        "ha": 3.1668115634556e-6,
        "eh": 3.1668115634556e-6,
    }
    try:
        return temperature * boltzmann_constants[normalized]
    except KeyError as exc:
        supported = "eV, kJ/mol, kcal/mol, J/mol, or Hartree"
        raise ValueError(
            f"Unsupported energy unit {energy_unit!r}; expected {supported}"
        ) from exc


def _finite_vector(values, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def log_reweighting_weights(
    total_bias,
    *,
    temperature: float,
    energy_unit: str = "eV",
) -> np.ndarray:
    """Return log importance weights ``beta * total_bias``.

    The returned logarithms are intentionally not exponentiated here.  Passing
    them to :func:`normalize_log_weights` avoids overflow even for large bias
    offsets.
    """
    bias = _finite_vector(total_bias, "total_bias")
    return bias / thermal_energy(temperature, energy_unit)


def normalize_log_weights(log_weights) -> np.ndarray:
    """Normalize log weights using a stable log-sum-exp shift."""
    log_weights = _finite_vector(log_weights, "log_weights")
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("log_weights could not be normalized")
    return weights / total


def _logsumexp(values: np.ndarray) -> float:
    """Return log(sum(exp(values))) without losing separated small terms."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    maximum = float(np.max(values))
    if maximum == -np.inf:
        return -np.inf
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _log_weight_diagnostics(log_weights: np.ndarray) -> dict:
    """Compute global weight diagnostics after a maximum-log shift."""
    log_weights = _finite_vector(log_weights, "log_weights")
    maximum = float(np.max(log_weights))
    scaled = np.exp(log_weights - maximum)
    scaled_sum = float(np.sum(scaled))
    scaled_square_sum = float(np.dot(scaled, scaled))
    return {
        "importance_weight_ess": scaled_sum**2 / scaled_square_sum,
        "maximum_weight_fraction": 1.0 / scaled_sum,
        "importance_weight_ess_note": (
            "Kish ESS of importance weights; not an autocorrelation-adjusted "
            "count of independent dynamical samples"
        ),
    }


def importance_weight_diagnostics(normalized_weights) -> dict:
    """Return Kish importance-weight ESS and maximum weight leverage."""
    weights = _finite_vector(normalized_weights, "normalized_weights")
    if np.any(weights < 0):
        raise ValueError("normalized_weights must be non-negative")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("normalized_weights must have positive sum")
    positive = weights > 0.0
    return _log_weight_diagnostics(np.log(weights[positive]))


def _region_bounds(region, name: str) -> tuple[float, float]:
    """Validate and return one closed numerical region."""
    try:
        lower, upper = region
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly two bounds") from exc
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} bounds must be finite and strictly increasing")
    return lower, upper


def region_weight_diagnostics(
    values,
    normalized_weights,
    regions: Mapping[str, tuple[float, float]],
) -> dict:
    """Report importance-weight support within named coordinate regions.

    Region bounds are closed.  Input weights are normalized again defensively,
    so ``weight_mass`` is always a fraction of the complete input trajectory.
    ``maximum_local_leverage`` is the largest frame weight divided by the total
    weight in that region.  Kish ESS and leverage are importance-weight
    diagnostics only and are not adjusted for trajectory autocorrelation.
    Regions may overlap; ``covered_weight_mass`` counts their union once.
    """
    values = _finite_vector(values, "values")
    weights = _finite_vector(normalized_weights, "normalized_weights")
    if len(values) != len(weights):
        raise ValueError("values and normalized_weights must have the same length")
    if np.any(weights < 0.0):
        raise ValueError("normalized_weights must be non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("normalized_weights must have positive sum")
    weights = weights / total
    if not isinstance(regions, Mapping) or len(regions) == 0:
        raise ValueError("regions must be a non-empty mapping of names to bounds")

    diagnostics = {}
    covered = np.zeros(len(values), dtype=bool)
    for name, region in regions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("region names must be non-empty strings")
        lower, upper = _region_bounds(region, f"region {name!r}")
        mask = (values >= lower) & (values <= upper)
        covered |= mask
        local = weights[mask]
        mass = float(np.sum(local))
        if local.size == 0:
            status = "no_records"
            kish_ess = 0.0
            maximum_leverage = None
        elif mass <= 0.0 or not np.any(local > 0.0):
            status = "zero_weight"
            kish_ess = 0.0
            maximum_leverage = None
        else:
            status = "ok"
            local_diagnostics = _log_weight_diagnostics(
                np.log(local[local > 0.0])
            )
            kish_ess = local_diagnostics["importance_weight_ess"]
            maximum_leverage = local_diagnostics["maximum_weight_fraction"]
        diagnostics[name] = {
            "bounds": (lower, upper),
            "status": status,
            "n_records": int(np.count_nonzero(mask)),
            "weight_mass": mass,
            "kish_ess": float(kish_ess),
            "maximum_local_leverage": maximum_leverage,
        }

    return {
        "regions": diagnostics,
        "n_records": len(values),
        "covered_records": int(np.count_nonzero(covered)),
        "covered_weight_mass": float(np.sum(weights[covered])),
        "diagnostic_note": (
            "Regional Kish ESS is not an autocorrelation-adjusted count of "
            "independent dynamical samples"
        ),
    }


def _resolve_bin_edges(values: np.ndarray, bins, value_range) -> np.ndarray:
    if np.isscalar(bins):
        if isinstance(bins, (bool, np.bool_)):
            raise ValueError("bins must be an integer or one-dimensional edges")
        bins_float = float(bins)
        if (
            not np.isfinite(bins_float)
            or bins_float != np.floor(bins_float)
            or bins_float < 1
        ):
            raise ValueError("bins must be a positive integer")
        return np.histogram_bin_edges(
            values, bins=int(bins_float), range=value_range
        )
    edges = _finite_vector(bins, "bins")
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin edges must be strictly increasing")
    if value_range is not None:
        raise ValueError("value_range cannot be combined with explicit bin edges")
    return edges


def reweighted_pmf_1d(
    values,
    log_weights,
    *,
    bins=100,
    value_range: tuple[float, float] | None = None,
    temperature: float,
    energy_unit: str = "eV",
    min_count: int = 1,
) -> dict:
    """Estimate a one-dimensional PMF by direct bias reweighting.

    Empty or undersampled bins remain unsupported and have ``NaN`` free
    energy.  ``local_ess`` is the Kish importance-weight ESS within each bin;
    like the global ESS, it does not correct for trajectory autocorrelation.
    """
    values = _finite_vector(values, "values")
    log_weights = _finite_vector(log_weights, "log_weights")
    if len(values) != len(log_weights):
        raise ValueError("values and log_weights must have the same length")
    if isinstance(min_count, (bool, np.bool_)) or int(min_count) != min_count:
        raise ValueError("min_count must be a positive integer")
    min_count = int(min_count)
    if min_count < 1:
        raise ValueError("min_count must be a positive integer")

    edges = _resolve_bin_edges(values, bins, value_range)
    weights = normalize_log_weights(log_weights)
    n_bins = len(edges) - 1
    bin_indices = np.searchsorted(edges, values, side="right") - 1
    bin_indices[values == edges[-1]] = n_bins - 1
    included = (bin_indices >= 0) & (bin_indices < n_bins)
    if not np.any(included):
        raise ValueError("No weighted samples fall inside the histogram range")

    included_indices = bin_indices[included]
    included_logs = log_weights[included]
    raw_counts = np.bincount(included_indices, minlength=n_bins)

    # Reduce each bin relative to its own maximum log weight.  Unlike
    # np.histogram's cumulative weighted summation, this cannot erase a tiny
    # bin merely because a dominant bin precedes it in the cumulative total.
    bin_maximum = np.full(n_bins, -np.inf, dtype=float)
    np.maximum.at(bin_maximum, included_indices, included_logs)
    relative_logs = included_logs - bin_maximum[included_indices]
    scaled = np.exp(relative_logs)
    scaled_sums = np.zeros(n_bins, dtype=float)
    scaled_square_sums = np.zeros(n_bins, dtype=float)
    np.add.at(scaled_sums, included_indices, scaled)
    np.add.at(scaled_square_sums, included_indices, scaled**2)

    nonempty = raw_counts > 0
    global_maximum = float(np.max(log_weights))
    global_scaled_sum = float(np.sum(np.exp(log_weights - global_maximum)))
    log_weighted_mass = np.full(n_bins, -np.inf, dtype=float)
    log_weighted_mass[nonempty] = (
        bin_maximum[nonempty]
        - global_maximum
        + np.log(scaled_sums[nonempty])
        - np.log(global_scaled_sum)
    )
    weighted_mass = np.exp(log_weighted_mass)

    included_maximum = float(np.max(included_logs))
    included_scaled_sum = float(
        np.sum(np.exp(included_logs - included_maximum))
    )
    widths = np.diff(edges)
    log_density = np.full(n_bins, -np.inf, dtype=float)
    log_density[nonempty] = (
        bin_maximum[nonempty]
        - included_maximum
        + np.log(scaled_sums[nonempty])
        - np.log(included_scaled_sum)
        - np.log(widths[nonempty])
    )
    density = np.exp(log_density)
    included_weight = float(np.exp(
        included_maximum
        - global_maximum
        + np.log(included_scaled_sum)
        - np.log(global_scaled_sum)
    ))
    support = (raw_counts >= min_count) & nonempty
    if not np.any(support):
        raise ValueError("No histogram bin satisfies min_count")

    pmf = np.full(n_bins, np.nan, dtype=float)
    kbt = thermal_energy(temperature, energy_unit)
    pmf[support] = -kbt * log_density[support]
    pmf[support] -= np.min(pmf[support])

    local_ess = np.zeros(n_bins, dtype=float)
    local_ess[nonempty] = (
        scaled_sums[nonempty] ** 2 / scaled_square_sums[nonempty]
    )
    # Highest normalized contribution within each bin.  A value near one
    # reveals a locally single-frame-dominated PMF even when global ESS looks
    # less alarming.
    local_maximum_weight_fraction = np.full(n_bins, np.nan, dtype=float)
    local_maximum_weight_fraction[nonempty] = 1.0 / scaled_sums[nonempty]
    diagnostics = _log_weight_diagnostics(log_weights)
    return {
        "bin_edges": edges,
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
        "pmf": pmf,
        "density": density,
        "log_density": log_density,
        "weighted_mass": weighted_mass,
        "log_weighted_mass": log_weighted_mass,
        "raw_counts": raw_counts,
        "support_mask": support,
        "local_ess": local_ess,
        "local_maximum_weight_fraction": local_maximum_weight_fraction,
        "normalized_weights": weights,
        "included_weight_fraction": included_weight,
        "temperature": float(temperature),
        "energy_unit": energy_unit,
        **diagnostics,
    }


def cumulative_reweighted_pmf_1d(
    values,
    times,
    log_weights,
    cutoffs: Iterable[float],
    *,
    bins=100,
    value_range: tuple[float, float] | None = None,
    temperature: float,
    energy_unit: str = "eV",
    min_count: int = 1,
) -> dict:
    """Evaluate direct-reweighting PMFs over cumulative time prefixes."""
    values = _finite_vector(values, "values")
    times = _finite_vector(times, "times")
    log_weights = _finite_vector(log_weights, "log_weights")
    if not (len(values) == len(times) == len(log_weights)):
        raise ValueError("values, times, and log_weights must have equal length")
    cutoff_array = _finite_vector(cutoffs, "cutoffs")
    if np.any(np.diff(cutoff_array) <= 0):
        raise ValueError("cutoffs must be strictly increasing")

    common_edges = _resolve_bin_edges(values, bins, value_range)
    snapshots = []
    for cutoff in cutoff_array:
        selected = times <= cutoff
        if not np.any(selected):
            snapshots.append({
                "cutoff": float(cutoff),
                "n_records": 0,
                "status": "no_samples",
            })
            continue
        try:
            result = reweighted_pmf_1d(
                values[selected],
                log_weights[selected],
                bins=common_edges,
                temperature=temperature,
                energy_unit=energy_unit,
                min_count=min_count,
            )
        except ValueError as exc:
            snapshots.append({
                "cutoff": float(cutoff),
                "n_records": int(np.count_nonzero(selected)),
                "status": "insufficient_support",
                "reason": str(exc),
            })
            continue
        result["cutoff"] = float(cutoff)
        result["n_records"] = int(np.count_nonzero(selected))
        result["status"] = "ok"
        snapshots.append(result)
    return {
        "cutoffs": cutoff_array,
        "bin_edges": common_edges,
        "snapshots": snapshots,
    }


def block_reweighted_pmf_1d(
    values,
    log_weights,
    *,
    n_blocks: int,
    bins=100,
    value_range: tuple[float, float] | None = None,
    temperature: float,
    energy_unit: str = "eV",
    min_count: int = 1,
    min_common_bins: int = 2,
) -> dict:
    """Estimate PMFs in contiguous, non-overlapping trajectory blocks.

    Every block uses the same histogram edges.  Each successive pair is
    compared only on bins supported by both blocks.  The newer PMF receives
    the constant offset that minimizes its unweighted squared difference from
    the preceding PMF on that common support; ``offset_aligned_rms`` is the
    resulting shape difference.  A comparison with fewer than
    ``min_common_bins`` is explicitly marked ``insufficient_support``.
    """
    values = _finite_vector(values, "values")
    log_weights = _finite_vector(log_weights, "log_weights")
    if len(values) != len(log_weights):
        raise ValueError("values and log_weights must have the same length")
    if (
        isinstance(n_blocks, (bool, np.bool_))
        or not isinstance(n_blocks, (int, np.integer))
        or n_blocks < 2
    ):
        raise ValueError("n_blocks must be an integer >= 2")
    if n_blocks > len(values):
        raise ValueError("n_blocks cannot exceed the number of records")
    if (
        isinstance(min_common_bins, (bool, np.bool_))
        or not isinstance(min_common_bins, (int, np.integer))
        or min_common_bins < 1
    ):
        raise ValueError("min_common_bins must be a positive integer")

    common_edges = _resolve_bin_edges(values, bins, value_range)
    index_blocks = np.array_split(np.arange(len(values)), n_blocks)
    block_results = []
    for block_index, indices in enumerate(index_blocks):
        metadata = {
            "block_index": block_index,
            "start_index": int(indices[0]),
            "stop_index_exclusive": int(indices[-1] + 1),
            "n_records": len(indices),
        }
        try:
            result = reweighted_pmf_1d(
                values[indices],
                log_weights[indices],
                bins=common_edges,
                temperature=temperature,
                energy_unit=energy_unit,
                min_count=min_count,
            )
        except ValueError as exc:
            block_results.append({
                **metadata,
                "status": "insufficient_support",
                "reason": str(exc),
            })
            continue
        result.update(metadata)
        result["status"] = "ok"
        block_results.append(result)

    comparisons = []
    for previous_index in range(n_blocks - 1):
        previous = block_results[previous_index]
        current = block_results[previous_index + 1]
        comparison = {
            "previous_block": previous_index,
            "current_block": previous_index + 1,
        }
        if previous["status"] != "ok" or current["status"] != "ok":
            comparison.update({
                "status": "insufficient_support",
                "reason": "one or both block PMFs are unavailable",
                "common_support_bins": 0,
                "offset_to_previous": None,
                "offset_aligned_rms": None,
            })
            comparisons.append(comparison)
            continue
        common = (
            previous["support_mask"]
            & current["support_mask"]
            & np.isfinite(previous["pmf"])
            & np.isfinite(current["pmf"])
        )
        common_count = int(np.count_nonzero(common))
        if common_count < min_common_bins:
            comparison.update({
                "status": "insufficient_support",
                "reason": (
                    f"only {common_count} common supported bins; "
                    f"at least {min_common_bins} required"
                ),
                "common_support_bins": common_count,
                "offset_to_previous": None,
                "offset_aligned_rms": None,
            })
            comparisons.append(comparison)
            continue
        offset = float(np.mean(previous["pmf"][common] - current["pmf"][common]))
        residual = current["pmf"][common] + offset - previous["pmf"][common]
        comparison.update({
            "status": "ok",
            "common_support_bins": common_count,
            "common_support_mask": common,
            "offset_to_previous": offset,
            "offset_aligned_rms": float(np.sqrt(np.mean(residual**2))),
        })
        comparisons.append(comparison)

    return {
        "n_blocks": int(n_blocks),
        "bin_edges": common_edges,
        "blocks": block_results,
        "successive_comparisons": comparisons,
    }


def pmf_landmarks_1d(
    profile: Mapping,
    basin_a: tuple[float, float],
    basin_b: tuple[float, float],
    transition_region: tuple[float, float] | None = None,
) -> dict:
    """Extract basin minima, barriers, and basin-population free energy.

    ``profile`` follows :func:`reweighted_pmf_1d` and must at least provide
    ``bin_centers`` and ``pmf``.  The full interbasin peak is reported only
    when every bin between the two located minima is supported.  An optional
    transition region produces a separately labelled, prespecified peak.  No
    convergence decision is inferred from these observables.
    """
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be a mapping")
    if "bin_centers" not in profile or "pmf" not in profile:
        raise ValueError("profile must contain bin_centers and pmf")
    centers = _finite_vector(profile["bin_centers"], "profile bin_centers")
    pmf = np.asarray(profile["pmf"], dtype=float)
    if pmf.ndim != 1 or len(pmf) != len(centers):
        raise ValueError("profile pmf must match the one-dimensional bin centers")
    if np.any(np.isinf(pmf)):
        raise ValueError("profile pmf must not contain infinite values")
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("profile bin_centers must be strictly increasing")
    if "support_mask" in profile:
        support = np.asarray(profile["support_mask"], dtype=bool)
        if support.shape != pmf.shape:
            raise ValueError("profile support_mask must match pmf")
        if np.any(support & ~np.isfinite(pmf)):
            raise ValueError("supported PMF bins must contain finite values")
        support = support & np.isfinite(pmf)
    else:
        support = np.isfinite(pmf)

    bounds_a = _region_bounds(basin_a, "basin_a")
    bounds_b = _region_bounds(basin_b, "basin_b")
    if not (bounds_a[1] < bounds_b[0] or bounds_b[1] < bounds_a[0]):
        raise ValueError(
            "closed basin_a and basin_b regions must be strictly separated"
        )

    masks = {
        "basin_a": (centers >= bounds_a[0]) & (centers <= bounds_a[1]),
        "basin_b": (centers >= bounds_b[0]) & (centers <= bounds_b[1]),
    }
    basin_results = {}
    minimum_indices = {}
    reasons = []
    for name, bounds in (("basin_a", bounds_a), ("basin_b", bounds_b)):
        available = masks[name] & support
        count = int(np.count_nonzero(available))
        if count == 0:
            basin_results[name] = {
                "bounds": bounds,
                "status": "insufficient_support",
                "supported_bins": 0,
                "minimum_position": None,
                "minimum_free_energy": None,
            }
            reasons.append(f"{name} has no supported PMF bin")
            continue
        indices = np.flatnonzero(available)
        minimum_index = int(indices[np.argmin(pmf[indices])])
        minimum_indices[name] = minimum_index
        basin_results[name] = {
            "bounds": bounds,
            "status": "ok",
            "supported_bins": count,
            "minimum_bin_index": minimum_index,
            "minimum_position": float(centers[minimum_index]),
            "minimum_free_energy": float(pmf[minimum_index]),
        }

    interbasin = {
        "status": "insufficient_support",
        "peak_position": None,
        "peak_free_energy": None,
        "barrier_from_a": None,
        "barrier_from_b": None,
    }
    interval_mask = np.zeros(len(centers), dtype=bool)
    if len(minimum_indices) == 2:
        left, right = sorted(minimum_indices.values())
        interval_mask[left:right + 1] = True
        unsupported = interval_mask & ~support
        if np.any(unsupported):
            missing = int(np.count_nonzero(unsupported))
            interbasin["reason"] = f"{missing} unsupported bins between minima"
            reasons.append(interbasin["reason"])
        else:
            indices = np.flatnonzero(interval_mask)
            peak_index = int(indices[np.argmax(pmf[indices])])
            peak = float(pmf[peak_index])
            interbasin = {
                "status": "ok",
                "supported_bins": len(indices),
                "peak_bin_index": peak_index,
                "peak_position": float(centers[peak_index]),
                "peak_free_energy": peak,
                "barrier_from_a": (
                    peak - basin_results["basin_a"]["minimum_free_energy"]
                ),
                "barrier_from_b": (
                    peak - basin_results["basin_b"]["minimum_free_energy"]
                ),
            }

    transition = None
    if transition_region is not None:
        transition_bounds = _region_bounds(transition_region, "transition_region")
        expected = (
            (centers >= transition_bounds[0])
            & (centers <= transition_bounds[1])
            & interval_mask
        )
        transition = {
            "bounds": transition_bounds,
            "status": "insufficient_support",
            "peak_position": None,
            "peak_free_energy": None,
            "barrier_from_a": None,
            "barrier_from_b": None,
        }
        if not np.any(expected):
            transition["reason"] = "transition region has no bin between basin minima"
            reasons.append(transition["reason"])
        elif np.any(expected & ~support):
            missing = int(np.count_nonzero(expected & ~support))
            transition["reason"] = (
                f"transition region contains {missing} unsupported bins"
            )
            reasons.append(transition["reason"])
        else:
            indices = np.flatnonzero(expected)
            peak_index = int(indices[np.argmax(pmf[indices])])
            peak = float(pmf[peak_index])
            transition = {
                "bounds": transition_bounds,
                "status": "ok",
                "supported_bins": len(indices),
                "peak_bin_index": peak_index,
                "peak_position": float(centers[peak_index]),
                "peak_free_energy": peak,
                "barrier_from_a": (
                    peak - basin_results["basin_a"]["minimum_free_energy"]
                ),
                "barrier_from_b": (
                    peak - basin_results["basin_b"]["minimum_free_energy"]
                ),
            }

    population_masses = None
    population_log_masses = None
    population_source = None
    if "log_weighted_mass" in profile:
        log_mass = np.asarray(profile["log_weighted_mass"], dtype=float)
        if log_mass.shape != pmf.shape:
            raise ValueError("profile log_weighted_mass must match pmf")
        if np.any(np.isnan(log_mass)) or np.any(log_mass == np.inf):
            raise ValueError(
                "profile log_weighted_mass must contain finite values or -inf"
            )
        population_log_masses = (
            _logsumexp(log_mass[masks["basin_a"]]),
            _logsumexp(log_mass[masks["basin_b"]]),
        )
        population_masses = tuple(
            float(np.exp(value)) for value in population_log_masses
        )
        population_source = "log_weighted_mass"
    elif "weighted_mass" in profile:
        weighted_mass = np.asarray(profile["weighted_mass"], dtype=float)
        if weighted_mass.shape != pmf.shape:
            raise ValueError("profile weighted_mass must match pmf")
        if np.any(~np.isfinite(weighted_mass)) or np.any(weighted_mass < 0.0):
            raise ValueError("profile weighted_mass must be finite and non-negative")
        population_masses = (
            float(np.sum(weighted_mass[masks["basin_a"]])),
            float(np.sum(weighted_mass[masks["basin_b"]])),
        )
        population_log_masses = tuple(
            -np.inf if value <= 0.0 else float(np.log(value))
            for value in population_masses
        )
        population_source = "weighted_mass"
    elif "density" in profile and "bin_edges" in profile:
        density = np.asarray(profile["density"], dtype=float)
        edges = _finite_vector(profile["bin_edges"], "profile bin_edges")
        if density.shape != pmf.shape or len(edges) != len(pmf) + 1:
            raise ValueError("profile density/bin_edges must match pmf")
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("profile density must be finite and non-negative")
        masses = density * np.diff(edges)
        population_masses = (
            float(np.sum(masses[masks["basin_a"]])),
            float(np.sum(masses[masks["basin_b"]])),
        )
        population_log_masses = tuple(
            -np.inf if value <= 0.0 else float(np.log(value))
            for value in population_masses
        )
        population_source = "density_integral"

    population_delta_f = None
    population_status = "unavailable"
    population_reason = "profile lacks weighted mass or density information"
    if population_masses is not None and population_log_masses is not None:
        mass_a, mass_b = population_masses
        log_mass_a, log_mass_b = population_log_masses
        if not np.isfinite(log_mass_a) or not np.isfinite(log_mass_b):
            population_reason = "one or both basin populations have zero weight"
        elif "temperature" not in profile:
            population_reason = "profile lacks temperature for population delta F"
        else:
            energy_unit = profile.get("energy_unit", "eV")
            kbt = thermal_energy(profile["temperature"], energy_unit)
            population_delta_f = float(-kbt * (log_mass_b - log_mass_a))
            population_status = "ok"
            population_reason = None
            basin_results["basin_a"]["population_mass"] = mass_a
            basin_results["basin_b"]["population_mass"] = mass_b
            basin_results["basin_a"]["log_population_mass"] = log_mass_a
            basin_results["basin_b"]["log_population_mass"] = log_mass_b

    status = "ok" if not reasons else "insufficient_support"
    return {
        "status": status,
        "reasons": reasons,
        "basin_a": basin_results["basin_a"],
        "basin_b": basin_results["basin_b"],
        "interbasin": interbasin,
        "transition_region": transition,
        "population_delta_f": population_delta_f,
        "population_delta_f_status": population_status,
        "population_delta_f_reason": population_reason,
        "population_delta_f_definition": "F_b - F_a = -kBT ln(P_b / P_a)",
        "population_source": population_source,
    }


def detect_hysteretic_transitions(
    state_a_mask,
    state_b_mask,
    *,
    times=None,
    state_names: tuple[str, str] = ("state_a", "state_b"),
) -> dict:
    """Detect transitions between disjoint basin masks with hysteresis.

    Frames in neither basin retain the most recently established state.  This
    prevents a transition region from producing rapid false recrossings.
    """
    state_a = np.asarray(state_a_mask, dtype=bool)
    state_b = np.asarray(state_b_mask, dtype=bool)
    if state_a.ndim != 1 or state_b.ndim != 1 or len(state_a) != len(state_b):
        raise ValueError("state masks must be one-dimensional and equally sized")
    if np.any(state_a & state_b):
        raise ValueError("state masks must not overlap")
    if len(state_names) != 2 or state_names[0] == state_names[1]:
        raise ValueError("state_names must contain two distinct names")
    if times is None:
        time_values = np.arange(len(state_a), dtype=float)
    else:
        time_values = _finite_vector(times, "times", allow_empty=True)
        if len(time_values) != len(state_a):
            raise ValueError("times and state masks must have the same length")
    if np.any(np.diff(time_values) < 0.0):
        raise ValueError("times must be nondecreasing")

    labels = np.full(len(state_a), "unclassified", dtype=object)
    current: int | None = None
    initial: int | None = None
    events = []
    for index in range(len(state_a)):
        observed = 0 if state_a[index] else (1 if state_b[index] else None)
        if current is None:
            if observed is None:
                continue
            current = observed
            initial = observed
            labels[index] = state_names[current]
            continue
        if observed is not None and observed != current:
            old = current
            current = observed
            events.append({
                "index": index,
                "time": float(time_values[index]),
                "from_state": state_names[old],
                "to_state": state_names[current],
                "event": f"{state_names[old]}_to_{state_names[current]}",
            })
        labels[index] = state_names[current]

    forward = sum(
        event["from_state"] == state_names[0]
        and event["to_state"] == state_names[1]
        for event in events
    )
    reverse = sum(
        event["from_state"] == state_names[1]
        and event["to_state"] == state_names[0]
        for event in events
    )
    completed_round_trips = 0
    if initial is not None:
        completed_round_trips = sum(
            event["to_state"] == state_names[initial] for event in events
        )

    basin_evidence = {}
    for name, mask in zip(state_names, (state_a, state_b)):
        indices = np.flatnonzero(mask)
        basin_evidence[name] = {
            "n_qualifying_records": int(len(indices)),
            "first_qualifying_index": (
                None if len(indices) == 0 else int(indices[0])
            ),
            "last_qualifying_index": (
                None if len(indices) == 0 else int(indices[-1])
            ),
            "first_qualifying_time": (
                None if len(indices) == 0 else float(time_values[indices[0]])
            ),
            "last_qualifying_time": (
                None if len(indices) == 0 else float(time_values[indices[-1]])
            ),
        }
    return {
        "states": labels,
        "events": events,
        "initial_state": None if initial is None else state_names[initial],
        "final_state": None if current is None else state_names[current],
        "forward_events": forward,
        "reverse_events": reverse,
        "completed_round_trips": completed_round_trips,
        "last_event_time": None if not events else events[-1]["time"],
        "basin_evidence": basin_evidence,
    }
