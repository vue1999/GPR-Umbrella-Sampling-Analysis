"""Supported fixed reference curves and exact minimum-range grid paths.

The graph objective minimizes the energy range, not the forward activation
barrier. Profiles report both quantities explicitly with correlated errors.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .path_graph import minimum_range_path
from .path_profile import evaluate_path_profile
from .trajectory import distance_to_polyline, resample_polyline, validate_reference_path


def _snap(xy, gx, gy) -> tuple[int, int]:
    return int(np.argmin(np.abs(gx - xy[0]))), int(np.argmin(np.abs(gy - xy[1])))


def _support_components(support_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label support islands with the same 8-connectivity used by paths."""
    structure = np.ones((3, 3), dtype=np.uint8)
    return label(np.asarray(support_mask, dtype=bool), structure=structure)


def _nearest_grid_support(
    points: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Classify arbitrary points by the nearest grid cell's support flag."""
    if len(gx) < 2 or len(gy) < 2:
        raise ValueError("Reference-path support needs at least a 2x2 grid")
    ix = np.searchsorted(gx, points[:, 0])
    iy = np.searchsorted(gy, points[:, 1])
    ix = np.clip(ix, 1, len(gx) - 1)
    iy = np.clip(iy, 1, len(gy) - 1)
    ix -= np.abs(points[:, 0] - gx[ix - 1]) <= np.abs(points[:, 0] - gx[ix])
    iy -= np.abs(points[:, 1] - gy[iy - 1]) <= np.abs(points[:, 1] - gy[iy])
    return support_mask[ix, iy]


def _posterior_mean_gradient_2d(
    results: dict,
    points: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Evaluate the derivative-observation GP posterior mean gradient."""
    from .integration_2d import _k_grad_grad

    state = results.get("_gp_state")
    if not isinstance(state, dict):
        raise ValueError("GP state is required for gradient-aligned path search")
    points = np.atleast_2d(np.asarray(points, dtype=float))
    training = np.asarray(state["X"], dtype=float)
    alpha = np.asarray(state["alpha"], dtype=float)
    sigma_f = float(state["sigma_f"])
    lengthscale = np.asarray(state["lengthscale"], dtype=float)
    gradient = np.empty((len(points), 2), dtype=float)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        covariance = _k_grad_grad(
            points[start:stop], training, sigma_f, lengthscale
        )
        gradient[start:stop] = (covariance @ alpha).reshape(-1, 2)
    return gradient


def find_lowest_barrier_path(
    results: dict,
    endpoints=None,
    metric_scale: tuple[float, float] | np.ndarray | None = None,
    *,
    reference_path: np.ndarray | None = None,
    path_mode: str = "search",
    corridor_radius: float | None = None,
) -> dict:
    """Evaluate a fixed curve or search a supported grid between explicit states.

    Search and corridor modes minimize ``max(F) - min(F)`` exactly on the
    8-neighbour grid, with a fixed gradient-alignment tie-break. Fixed mode
    evaluates the supplied reference curve without moving vertices. Search
    endpoints snap only to the nearest grid cells; reference modes use the
    first and last vertices. No mode chooses a physical basin automatically.
    """
    gx, gy = (np.asarray(results[key], dtype=float) for key in ("gx", "gy"))
    pmf = np.asarray(results["pmf"], dtype=float)
    geometric_support = np.asarray(
        results.get("support_mask", np.ones_like(pmf)), dtype=bool
    )
    support = np.asarray(
        results.get("path_valid_mask", geometric_support), dtype=bool
    )
    if any(mask.shape != (len(gx), len(gy)) for mask in (pmf, geometric_support, support)):
        raise ValueError("PMF and validity masks must match the gx/gy grid")
    support = support & geometric_support & np.isfinite(pmf)
    if not np.any(support):
        raise ValueError("The path-valid region contains no finite PMF point")

    try:
        lengthscale = np.broadcast_to(np.asarray(results["lengthscale"], dtype=float), (2,))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("results lengthscale must be scalar or contain two values") from exc
    if not np.all(np.isfinite(lengthscale)) or np.any(lengthscale <= 0):
        raise ValueError("results lengthscale must contain two finite positive values")
    metric_description = "dimensionless; CVs scaled by " + (
        "GP lengthscales" if metric_scale is None else "user metric_scale"
    )
    metric_scale = np.asarray(lengthscale if metric_scale is None else metric_scale, dtype=float)
    if (metric_scale.shape != (2,) or not np.all(np.isfinite(metric_scale))
            or np.any(metric_scale <= 0)):
        raise ValueError("metric_scale must contain two finite positive CV scales")

    if path_mode not in {"search", "corridor", "fixed"}:
        raise ValueError("path_mode must be one of: search, corridor, fixed")
    reference = None if reference_path is None else validate_reference_path(reference_path)
    if path_mode == "search":
        if endpoints is None:
            raise ValueError("search path mode requires two explicit endpoints")
        if reference is not None:
            raise ValueError("reference_path requires corridor or fixed path mode")
        endpoint_values = np.asarray(endpoints, dtype=float)
    else:
        if reference is None:
            raise ValueError(f"path_mode={path_mode!r} requires reference_path")
        if endpoints is not None:
            raise ValueError(f"path_mode={path_mode!r} uses the reference trajectory endpoints")
        endpoint_values = reference[[0, -1]]
    if endpoint_values.shape != (2, 2) or not np.all(np.isfinite(endpoint_values)):
        raise ValueError("endpoints must contain two finite (x, y) coordinates")

    reference_distance = None
    if path_mode == "corridor":
        if isinstance(corridor_radius, (bool, np.bool_)):
            raise ValueError("corridor_radius must be finite and positive")
        try:
            corridor_radius = float(corridor_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError("corridor_radius must be finite and positive") from exc
        if not np.isfinite(corridor_radius) or corridor_radius <= 0:
            raise ValueError("corridor_radius must be finite and positive")
        grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        reference_distance = distance_to_polyline(
            grid_points, reference, metric_scale
        ).reshape(pmf.shape)
        support = support & (reference_distance <= corridor_radius + 1.0e-12)
        if not np.any(support):
            raise ValueError("The reference corridor does not overlap the path-valid region")
    elif corridor_radius is not None:
        raise ValueError("corridor_radius is only valid with corridor path mode")

    components, component_count = _support_components(support)
    start_ij, end_ij = (_snap(point, gx, gy) for point in endpoint_values)
    profile_inputs = {}
    if path_mode == "fixed":
        max_step = 0.5 * min((gx[1] - gx[0]) / metric_scale[0],
                             (gy[1] - gy[0]) / metric_scale[1])
        points = resample_polyline(reference, metric_scale, max_step)
        inside = np.all((points >= [gx[0], gy[0]]) & (points <= [gx[-1], gy[-1]]), axis=1)
        valid = inside & _nearest_grid_support(points, gx, gy, support)
        if not np.all(valid):
            first = points[int(np.flatnonzero(~valid)[0])]
            raise ValueError("The fixed reference trajectory leaves the path-valid region "
                             f"near ({first[0]:.6g}, {first[1]:.6g})")
        objective = "fixed_reference_trajectory"
        selection = "fixed_reference_vertex"
        max_reference_distance = 0.0
    else:
        if start_ij == end_ij:
            raise ValueError("The two requested endpoints collapse to one grid point")
        if not support[start_ij]:
            raise ValueError("The start endpoint snaps outside the path-valid region")
        if not support[end_ij]:
            raise ValueError("The end endpoint snaps outside the path-valid region")
        if components[start_ij] != components[end_ij]:
            raise ValueError("Requested endpoints lie in disconnected path-valid regions")
        dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
        dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
        grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        gradient = _posterior_mean_gradient_2d(results, grid_points).reshape(pmf.shape + (2,))
        path, lower, upper = minimum_range_path(
            pmf, start_ij, end_ij, dx / metric_scale[0], dy / metric_scale[1],
            support, gradient_scaled=gradient * metric_scale,
        )
        pi, pj = np.asarray(path).T
        points = np.column_stack([gx[pi], gy[pj]])
        profile_inputs = {"energy": pmf[pi, pj],
                          "latent_variance": results["latent_variance_raw"][pi, pj]}
        objective = "minimum_pmf_range_then_fixed_gradient_aligned_cost"
        selection = "nearest_grid_cell"
        max_reference_distance = (float(np.max(reference_distance[pi, pj]))
                                  if reference_distance is not None else None)

    profile = evaluate_path_profile(results, points, metric_scale, **profile_inputs)
    if path_mode != "fixed":
        expected_range = upper - lower
        if abs(profile["energy_range"] - expected_range) > 1e-10 * max(1.0, abs(expected_range)):
            raise RuntimeError("Selected path does not realize the exact minimum energy interval")
    else:
        lower, upper = profile["path_min_pmf"], profile["bottleneck_energy"]
    endpoint_metadata = {
        name + "_endpoint": {
            "ij": ij, "xy": profile[name + "_xy"], "requested_xy": tuple(requested),
            "component": int(components[ij]), "selection": selection,
        }
        for name, ij, requested in zip(("start", "end"), (start_ij, end_ij), endpoint_values)
    }
    return {
        **profile, **endpoint_metadata,
        "nominal_start_xy": tuple(endpoint_values[0]),
        "nominal_end_xy": tuple(endpoint_values[1]),
        "support_component": int(components[start_ij]),
        "support_component_count": int(component_count),
        "path_graph_connectivity": None if path_mode == "fixed" else 8,
        "path_objective": objective,
        "optimal_energy_lower": float(lower), "optimal_energy_upper": float(upper),
        "path_mode": path_mode, "reference_path": reference,
        "corridor_radius": corridor_radius, "max_reference_distance": max_reference_distance,
        "validity_kind": results.get("path_valid_kind", "path_valid_mask"),
        "support_radius": results.get("support_radius"),
        "metric_scale": metric_scale, "path_coordinate_description": metric_description,
        "cv_names": results.get("cv_names", ("cv0", "cv1")),
        "cv_units": results.get("cv_units", ("", "")),
        "energy_unit": results.get("energy_unit", "eV"),
        "default_uncertainty": results["default_uncertainty"],
    }


def save_lowest_barrier_path(path_result: dict, path: str) -> None:
    """Write the profile with explicit energy-range and endpoint-rise labels."""
    p = path_result
    names, units, energy_unit = p["cv_names"], p["cv_units"], p["energy_unit"]
    header = [
        "Path on a two-dimensional GPR PMF",
        f"path mode = {p['path_mode']}; path objective = {p['path_objective']}",
        "path metric scales = " + ", ".join(
            f"{scale:.6g} {unit}" for scale, unit in zip(p["metric_scale"], units)),
        f"requested/reference start = {p['nominal_start_xy']}; end = {p['nominal_end_xy']}",
        f"evaluated start = {p['start_xy']}; end = {p['end_xy']}",
        f"path minimum = {p['path_min_xy']}; maximum = {p['ts_xy']}",
        f"selected energy interval = [{p['optimal_energy_lower']:.6g}, {p['optimal_energy_upper']:.6g}] {energy_unit}",
    ]
    if p["corridor_radius"] is not None:
        header.append(f"reference corridor radius = {p['corridor_radius']:.6g} path-metric units")
    for key, definition in (("energy_range", "max(path PMF) - min(path PMF)"),
                            ("endpoint_to_max", "max(path PMF) - PMF(start)"),
                            ("delta_f", "PMF(end) - PMF(start)")):
        header.append(f"{key} = {definition} = {p[key]:.6g} {energy_unit}; "
                      f"sigma_raw = {p[key + '_err_raw']:.6g}; "
                      f"sigma_calibrated = {p[key + '_err_calibrated']:.6g} {energy_unit}")
    header.append(
        f"s(metric_arclength) {names[0]}({units[0]}) {names[1]}({units[1]}) "
        f"PMF({energy_unit}) PMF_rel_start({energy_unit}) sigma_from_start_raw({energy_unit}) "
        f"sigma_from_start_calibrated({energy_unit}) PMF_rel_path_min({energy_unit}) "
        f"sigma_from_path_min_raw({energy_unit}) sigma_from_path_min_calibrated({energy_unit})"
    )
    columns = ("s", "x", "y", "pmf", "pmf_rel", "sigma_raw", "sigma_calibrated",
               "pmf_rel_path_min", "sigma_from_path_min_raw", "sigma_from_path_min_calibrated")
    np.savetxt(path, np.column_stack([p[key] for key in columns]),
               header="\n".join(header), fmt="%.6f")
