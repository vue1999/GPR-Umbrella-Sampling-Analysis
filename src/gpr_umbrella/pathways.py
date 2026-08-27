"""Lowest-barrier paths on reconstructed two-dimensional PMFs.

The grid algorithm exactly minimizes the PMF range visited between two states.
A fixed gradient-alignment cost chooses one representative path inside that
optimal interval; this is not a continuous minimum-energy-path refinement.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_solve
from scipy.ndimage import label, minimum_filter
from scipy.special import logsumexp

from .path_graph import minimum_range_path as _minimum_range_path
from .path_profile import evaluate_path_profile
from .trajectory import (
    distance_to_polyline,
    resample_polyline,
    validate_reference_path,
)


def find_minima(
    pmf: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    max_minima: int = 8,
    support_mask: np.ndarray | None = None,
) -> list[dict]:
    """Return de-duplicated local minima inside the supported grid region."""
    if support_mask is None:
        support_mask = np.ones_like(pmf, dtype=bool)
    valid = np.asarray(support_mask, dtype=bool) & np.isfinite(pmf)
    work = np.where(valid, pmf, np.inf)
    local = minimum_filter(work, size=3, mode="constant", cval=np.inf)
    ii, jj = np.where(valid & (work == local))
    order = np.argsort(work[ii, jj])

    kept: list[dict] = []
    for i, j in zip(ii[order], jj[order]):
        if any(abs(i - m["ij"][0]) <= 2 and abs(j - m["ij"][1]) <= 2
               for m in kept):
            continue
        kept.append({
            "ij": (int(i), int(j)),
            "xy": (float(gx[i]), float(gy[j])),
            "energy": float(pmf[i, j]),
        })
        if len(kept) >= max_minima:
            break
    return kept


def _snap(xy, gx, gy) -> tuple[int, int]:
    return int(np.argmin(np.abs(gx - xy[0]))), int(np.argmin(np.abs(gy - xy[1])))


def _support_components(support_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label support islands with the same 8-connectivity used by paths."""
    structure = np.ones((3, 3), dtype=np.uint8)
    return label(np.asarray(support_mask, dtype=bool), structure=structure)


def _trapezoid_weights(coordinate: np.ndarray) -> np.ndarray:
    """Return positive trapezoidal quadrature weights for a 1D grid."""
    coordinate = np.asarray(coordinate, dtype=float)
    if coordinate.ndim != 1 or len(coordinate) < 2:
        raise ValueError("At least two perpendicular quadrature points are required")
    differences = np.diff(coordinate)
    if np.any(differences <= 0):
        raise ValueError("Perpendicular quadrature coordinates must increase")
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    if len(coordinate) > 2:
        weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


def _perpendicular_interval(
    point: np.ndarray,
    normal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    half_width: float | None,
) -> tuple[float, float]:
    """Intersect a normal line with rectangular bounds in metric space."""
    low, high = -np.inf, np.inf
    for component in range(2):
        if abs(normal[component]) < 1e-14:
            continue
        first = (lower[component] - point[component]) / normal[component]
        second = (upper[component] - point[component]) / normal[component]
        low = max(low, min(first, second))
        high = min(high, max(first, second))
    if half_width is not None:
        low, high = max(low, -half_width), min(high, half_width)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("A path normal does not cross a finite PMF-grid interval")
    return float(low), float(high)


def _nearest_grid_support(
    points: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Classify arbitrary points by the nearest grid cell's support flag."""
    if len(gx) < 2 or len(gy) < 2:
        raise ValueError("Path-aligned marginalization needs at least a 2x2 grid")
    ix = np.searchsorted(gx, points[:, 0])
    iy = np.searchsorted(gy, points[:, 1])
    ix = np.clip(ix, 1, len(gx) - 1)
    iy = np.clip(iy, 1, len(gy) - 1)
    ix -= np.abs(points[:, 0] - gx[ix - 1]) <= np.abs(points[:, 0] - gx[ix])
    iy -= np.abs(points[:, 1] - gy[iy - 1]) <= np.abs(points[:, 1] - gy[iy])
    return support_mask[ix, iy]


def _supported_segment(
    coordinate: np.ndarray,
    points: np.ndarray,
    supported: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the supported normal-line segment connected to the path centre."""
    centre = int(np.argmin(np.abs(coordinate)))
    # Rounding at a convex-hull boundary can classify the exact centre as just
    # outside. The centre itself lies on an already supported grid path.
    supported = np.asarray(supported, dtype=bool).copy()
    supported[centre] = True
    left = centre
    while left > 0 and supported[left - 1]:
        left -= 1
    right = centre
    while right + 1 < len(coordinate) and supported[right + 1]:
        right += 1
    if right - left + 1 < 2:
        raise ValueError("No finite supported perpendicular interval")
    return coordinate[left:right + 1], points[left:right + 1]


def _supported_boundary_from_centre(
    centre: np.ndarray,
    normal: np.ndarray,
    bound: float,
    metric_scale: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    support_mask: np.ndarray,
) -> float | None:
    """Locate the connected nearest-grid support boundary on one half-line."""
    if bound == 0.0:
        return None
    # Search from very near the supported path centre outwards, so a detached
    # supported island can never be mistaken for transverse support connected
    # to the path. The geometric scan makes coarse quadrature robust to a
    # one-cell-wide supported cross-section.
    trial_u = bound * np.exp2(np.arange(-40, 1, dtype=float))
    trial_points = (
        centre[None, :] + trial_u[:, None] * normal[None, :]
    ) * metric_scale
    trial_supported = _nearest_grid_support(
        trial_points, gx, gy, support_mask
    )
    if not trial_supported[0]:
        return None
    inside = float(trial_u[0])
    for candidate, supported in zip(trial_u[1:], trial_supported[1:]):
        if supported:
            inside = float(candidate)
            continue
        outside = float(candidate)
        for _ in range(32):
            midpoint = 0.5 * (inside + outside)
            point = (centre + midpoint * normal) * metric_scale
            if _nearest_grid_support(
                point[None, :], gx, gy, support_mask
            )[0]:
                inside = midpoint
            else:
                outside = midpoint
        return inside
    return float(trial_u[-1])


def _posterior_mean_at_points(results: dict, points: np.ndarray) -> np.ndarray:
    """Evaluate the fitted latent PMF mean at arbitrary 2D points."""
    state = results.get("_gp_state")
    if state is not None and "alpha" in state:
        from .integration_2d import _k_f_grad

        cross = _k_f_grad(
            points, state["X"], state["sigma_f"], state["lengthscale"]
        )
        # A derivative-observation GP has an arbitrary additive constant. Keep
        # this shift-free latent mean; relative path marginals do not need the
        # grid PMF's display reference and are more stable without it.
        return cross @ state["alpha"]

    # This fallback keeps the geometry helper useful for externally supplied
    # surfaces. Covariance-aware marginal errors still require fitted GP state.
    interpolator = RegularGridInterpolator(
        (results["gx"], results["gy"]),
        results["pmf"],
        bounds_error=True,
    )
    return np.asarray(interpolator(points), dtype=float)


def _path_aligned_marginal_pmf(
    results: dict,
    path_result: dict,
    *,
    thermal_energy: float,
    perpendicular_points: int,
    perpendicular_width: float | None,
) -> dict:
    """Boltzmann-integrate the PMF along normals to a grid pathway.

    Geometry is evaluated in the dimensionless metric used by the path. The
    reported uncertainty is a first-order (delta-method) propagation of the
    *joint* GP posterior covariance, including covariance both within each
    transverse integral and between that integral and the marginal reference.
    """
    if not np.isfinite(thermal_energy) or thermal_energy <= 0:
        raise ValueError("thermal_energy must be a positive kBT in energy_unit")
    try:
        perpendicular_points_float = float(perpendicular_points)
    except (TypeError, ValueError):
        perpendicular_points_float = np.nan
    if (
        isinstance(perpendicular_points, (bool, np.bool_))
        or not np.isscalar(perpendicular_points)
        or not np.isfinite(perpendicular_points_float)
        or not perpendicular_points_float.is_integer()
    ):
        raise ValueError("perpendicular_points must be an integer >= 3")
    perpendicular_points = int(perpendicular_points_float)
    if perpendicular_points < 3:
        raise ValueError("perpendicular_points must be an integer >= 3")
    if perpendicular_width is not None:
        if not np.isfinite(perpendicular_width) or perpendicular_width <= 0:
            raise ValueError("perpendicular_width must be positive when provided")

    state = results.get("_gp_state")
    if state is None:
        raise ValueError(
            "Path-aligned marginal uncertainty requires fitted GP state in results"
        )

    metric_scale = np.asarray(path_result["metric_scale"], dtype=float)
    path_s = np.asarray(path_result["s"], dtype=float)
    if len(path_s) < 2 or path_s[-1] <= path_s[0]:
        raise ValueError("The lowest-barrier path must contain at least two points")

    # Uniform resampling reduces staircase-normal artefacts while preserving
    # the original path's metric arclength and endpoints.
    s = np.linspace(path_s[0], path_s[-1], len(path_s))
    x = np.interp(s, path_s, path_result["x"])
    y = np.interp(s, path_s, path_result["y"])
    metric_points = np.column_stack([x, y]) / metric_scale
    tangents = np.empty_like(metric_points)
    tangents[0] = metric_points[1] - metric_points[0]
    tangents[-1] = metric_points[-1] - metric_points[-2]
    if len(metric_points) > 2:
        tangents[1:-1] = metric_points[2:] - metric_points[:-2]
    tangent_norm = np.linalg.norm(tangents, axis=1)
    if np.any(tangent_norm <= np.finfo(float).eps):
        raise ValueError("The resampled path contains a zero-length tangent")
    tangents /= tangent_norm[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    gx = np.asarray(results["gx"], dtype=float)
    gy = np.asarray(results["gy"], dtype=float)
    support = np.asarray(
        results.get(
            "path_valid_mask",
            results.get("support_mask", np.ones_like(results["pmf"])),
        ), dtype=bool,
    )
    lower = np.array([gx.min(), gy.min()]) / metric_scale
    upper = np.array([gx.max(), gy.max()]) / metric_scale

    line_points: list[np.ndarray] = []
    line_u: list[np.ndarray] = []
    valid_stations: list[int] = []
    for station, (centre, normal) in enumerate(zip(metric_points, normals)):
        try:
            low, high = _perpendicular_interval(
                centre, normal, lower, upper, perpendicular_width
            )
        except ValueError:
            # A path endpoint at a rectangular corner can have a zero-measure
            # normal cross-section. A marginal density is undefined there, so
            # omit that station instead of inventing transverse support.
            continue
        u = np.linspace(low, high, perpendicular_points)
        if not np.any(np.isclose(u, 0.0, atol=1e-13, rtol=0.0)):
            u = np.sort(np.append(u, 0.0))
        points = (centre[None, :] + u[:, None] * normal[None, :]) * metric_scale
        supported = _nearest_grid_support(points, gx, gy, support)
        try:
            u, points = _supported_segment(u, points, supported)
        except ValueError:
            extra_u = []
            for bound in (low, high):
                boundary = _supported_boundary_from_centre(
                    centre, normal, bound, metric_scale, gx, gy, support
                )
                if boundary is not None:
                    extra_u.extend(boundary * np.array([0.25, 0.5, 0.75, 1.0]))
            if extra_u:
                extra_u = np.asarray(extra_u, dtype=float)
                extra_points = (
                    centre[None, :] + extra_u[:, None] * normal[None, :]
                ) * metric_scale
                extra_supported = _nearest_grid_support(
                    extra_points, gx, gy, support
                )
                order = np.argsort(np.concatenate([u, extra_u]))
                u = np.concatenate([u, extra_u])[order]
                points = np.concatenate([points, extra_points], axis=0)[order]
                supported = np.concatenate([supported, extra_supported])[order]
            try:
                u, points = _supported_segment(u, points, supported)
            except ValueError:
                continue
        valid_stations.append(station)
        line_u.append(u)
        line_points.append(points)

    if len(valid_stations) < 2:
        raise ValueError(
            "Fewer than two path stations have a finite supported "
            "perpendicular interval"
        )
    valid_stations_array = np.asarray(valid_stations, dtype=int)
    if np.any(np.diff(valid_stations_array) > 1):
        raise ValueError(
            "The path-aligned marginal has undefined internal stations; "
            "refusing to bridge an unsupported gap"
        )
    s = s[valid_stations_array]
    x = x[valid_stations_array]
    y = y[valid_stations_array]

    unreferenced = np.empty(len(s), dtype=float)
    boltzmann_weights: list[np.ndarray] = []
    line_energies = [
        _posterior_mean_at_points(results, points) for points in line_points
    ]
    energy_offset = min(float(np.min(energies)) for energies in line_energies)
    for station, (u, energies) in enumerate(zip(line_u, line_energies)):
        quadrature = _trapezoid_weights(u)
        log_terms = np.log(quadrature) - (energies - energy_offset) / thermal_energy
        log_integral = logsumexp(log_terms)
        unreferenced[station] = -thermal_energy * log_integral
        boltzmann_weights.append(np.exp(log_terms - log_integral))

    # Delta-method covariance. We need each marginal variance and its
    # covariance with the chosen reference, not an expensive all-station
    # covariance matrix. The full GP covariance within every normal integral
    # is nevertheless retained exactly at first order.
    from .integration_2d import _k_f_grad, _se

    aggregated_cross = []
    prior_variance = np.empty(len(s), dtype=float)
    for station, (points, weights) in enumerate(zip(line_points, boltzmann_weights)):
        cross = _k_f_grad(
            points, state["X"], state["sigma_f"], state["lengthscale"]
        )
        aggregated_cross.append(weights @ cross)
        prior_variance[station] = float(
            weights
            @ _se(points, points, state["sigma_f"], state["lengthscale"])
            @ weights
        )
    aggregated_cross = np.asarray(aggregated_cross)
    solved_cross = cho_solve(
        (state["cho_factor"], state["lower"]), aggregated_cross.T
    )
    posterior_variance = np.clip(
        prior_variance - np.einsum("ij,ji->i", aggregated_cross, solved_cross),
        0.0,
        np.inf,
    )

    reference = int(np.argmin(unreferenced))
    reference_points = line_points[reference]
    reference_weights = boltzmann_weights[reference]
    prior_covariance_to_reference = np.empty(len(s), dtype=float)
    for station, (points, weights) in enumerate(zip(line_points, boltzmann_weights)):
        prior_covariance_to_reference[station] = float(
            weights
            @ _se(
                points,
                reference_points,
                state["sigma_f"],
                state["lengthscale"],
            )
            @ reference_weights
        )
    posterior_covariance_to_reference = (
        prior_covariance_to_reference
        - aggregated_cross @ solved_cross[:, reference]
    )
    relative_variance = np.clip(
        posterior_variance
        + posterior_variance[reference]
        - 2.0 * posterior_covariance_to_reference,
        0.0,
        np.inf,
    )
    relative_variance[reference] = 0.0
    sigma_raw = np.sqrt(relative_variance)
    sigma_calibrated = sigma_raw * results.get("loo_calibration_factor", 1.0)
    use_calibrated = results.get("default_uncertainty", "raw") == "calibrated"
    sigma = sigma_calibrated if use_calibrated else sigma_raw
    pmf = unreferenced - unreferenced[reference]

    return {
        "s": s,
        "x": x,
        "y": y,
        "pmf": pmf,
        "pmf_rel": pmf,
        "pmf_unreferenced": unreferenced,
        "integration_energy_offset": energy_offset,
        "sigma_raw": sigma_raw,
        "sigma_calibrated": sigma_calibrated,
        "sigma": sigma,
        "reference_index": reference,
        "thermal_energy": float(thermal_energy),
        "u_min": np.array([u[0] for u in line_u]),
        "u_max": np.array([u[-1] for u in line_u]),
        "perpendicular_samples": np.array([len(u) for u in line_u]),
        "metric_scale": metric_scale.copy(),
        "cv_names": path_result["cv_names"],
        "cv_units": path_result["cv_units"],
        "energy_unit": path_result["energy_unit"],
        "default_uncertainty": path_result["default_uncertainty"],
    }


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


def _fixed_reference_path_result(
    results: dict,
    reference: np.ndarray,
    metric_scale: np.ndarray,
    metric_description: str,
    support: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    max_minima: int,
    path_aligned_marginal: bool,
    thermal_energy: float | None,
    perpendicular_points: int,
    perpendicular_width: float | None,
) -> dict:
    """Densify, validate, and evaluate an immutable reference trajectory."""
    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])
    max_step = 0.5 * min(dx / metric_scale[0], dy / metric_scale[1])
    points = resample_polyline(reference, metric_scale, max_step)
    lower = np.array([gx[0], gy[0]])
    upper = np.array([gx[-1], gy[-1]])
    inside_grid = np.all((points >= lower) & (points <= upper), axis=1)
    valid = inside_grid & _nearest_grid_support(points, gx, gy, support)
    if not np.all(valid):
        first = points[int(np.flatnonzero(~valid)[0])]
        raise ValueError(
            "The fixed reference trajectory leaves the path-valid region "
            f"near ({first[0]:.6g}, {first[1]:.6g})"
        )

    profile = evaluate_path_profile(results, points, metric_scale)
    components, component_count = _support_components(support)
    start_ij = _snap(points[0], gx, gy)
    end_ij = _snap(points[-1], gx, gy)
    minima = find_minima(
        results["pmf"], gx, gy, max_minima=max_minima, support_mask=support
    )
    path_result = {
        **profile,
        "minima": minima,
        "nominal_start_xy": tuple(reference[0]),
        "nominal_end_xy": tuple(reference[-1]),
        "start_endpoint": {
            "ij": start_ij,
            "xy": profile["start_xy"],
            "requested_xy": tuple(reference[0]),
            "component": int(components[start_ij]),
            "selection": "fixed_reference_vertex",
        },
        "end_endpoint": {
            "ij": end_ij,
            "xy": profile["end_xy"],
            "requested_xy": tuple(reference[-1]),
            "component": int(components[end_ij]),
            "selection": "fixed_reference_vertex",
        },
        "support_component": int(components[start_ij]),
        "support_component_count": int(component_count),
        "path_graph_connectivity": None,
        "path_objective": "fixed_reference_trajectory",
        "optimal_energy_lower": float(profile["path_min_pmf"]),
        "optimal_energy_upper": float(profile["bottleneck_energy"]),
        "path_mode": "fixed",
        "reference_path": reference,
        "corridor_radius": None,
        "max_reference_distance": 0.0,
        "path_median_abs_gradient_cosine": None,
        "path_length_weighted_gradient_misalignment": None,
        "validity_kind": results.get("path_valid_kind", "path_valid_mask"),
        "support_radius": results.get("support_radius"),
        "metric_scale": metric_scale,
        "path_coordinate_description": metric_description,
        "cv_names": results.get("cv_names", ("cv0", "cv1")),
        "cv_units": results.get("cv_units", ("", "")),
        "energy_unit": results.get("energy_unit", "eV"),
        "default_uncertainty": results["default_uncertainty"],
    }
    if path_aligned_marginal:
        if thermal_energy is None:
            raise ValueError(
                "thermal_energy is required when path_aligned_marginal=True"
            )
        path_result["path_aligned_marginal"] = _path_aligned_marginal_pmf(
            results, path_result, thermal_energy=thermal_energy,
            perpendicular_points=perpendicular_points,
            perpendicular_width=perpendicular_width,
        )
    return path_result


def find_lowest_barrier_path(
    results: dict,
    endpoints=None,
    max_minima: int = 8,
    metric_scale: tuple[float, float] | np.ndarray | None = None,
    *,
    reference_path: np.ndarray | None = None,
    path_mode: str = "search",
    corridor_radius: float | None = None,
    path_aligned_marginal: bool = False,
    thermal_energy: float | None = None,
    perpendicular_points: int = 201,
    perpendicular_width: float | None = None,
) -> dict:
    """Find an exact minimum-range path on the path-valid finite grid.

    Search and corridor modes minimize ``max(F) - min(F)`` exactly on the
    8-neighbour grid, then choose a deterministic gradient-aligned shortest
    path inside the optimal energy interval. Search mode requires two explicit
    physical endpoints. Corridor and fixed modes use the first and last points
    of ``reference_path``. No mode silently moves an endpoint to a minimum or
    restraint-window centre.
    """
    gx = np.asarray(results["gx"], dtype=float)
    gy = np.asarray(results["gy"], dtype=float)
    pmf = np.asarray(results["pmf"], dtype=float)
    geometric_support = np.asarray(
        results.get("support_mask", np.ones_like(pmf)), dtype=bool
    )
    support = np.asarray(
        results.get("path_valid_mask", geometric_support), dtype=bool
    )
    if (
        pmf.shape != (len(gx), len(gy))
        or geometric_support.shape != pmf.shape
        or support.shape != pmf.shape
    ):
        raise ValueError("PMF and validity masks must match the gx/gy grid")
    support = support & geometric_support & np.isfinite(pmf)
    if not np.any(support):
        raise ValueError("The path-valid region contains no finite PMF point")

    try:
        lengthscale = np.broadcast_to(
            np.asarray(results["lengthscale"], dtype=float), (2,)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "results lengthscale must be scalar or contain two values"
        ) from exc
    if not np.all(np.isfinite(lengthscale)) or np.any(lengthscale <= 0):
        raise ValueError("results lengthscale must contain two finite positive values")

    if metric_scale is None:
        metric_scale = lengthscale.copy()
        metric_description = "dimensionless; CVs scaled by GP lengthscales"
    else:
        metric_scale = np.asarray(metric_scale, dtype=float)
        metric_description = "dimensionless; CVs scaled by user metric_scale"
    if (
        metric_scale.shape != (2,)
        or not np.all(np.isfinite(metric_scale))
        or np.any(metric_scale <= 0)
    ):
        raise ValueError("metric_scale must contain two finite positive CV scales")

    if path_mode not in {"search", "corridor", "fixed"}:
        raise ValueError("path_mode must be one of: search, corridor, fixed")
    reference = (
        None if reference_path is None else validate_reference_path(reference_path)
    )
    reference_distance = None
    if path_mode == "search":
        if endpoints is None:
            raise ValueError("search path mode requires two explicit endpoints")
        if reference is not None:
            raise ValueError("reference_path requires corridor or fixed path mode")
        if corridor_radius is not None:
            raise ValueError("corridor_radius is only valid with corridor path mode")
        endpoint_values = np.asarray(endpoints, dtype=float)
    else:
        if reference is None:
            raise ValueError(f"path_mode={path_mode!r} requires reference_path")
        if endpoints is not None:
            raise ValueError(
                f"path_mode={path_mode!r} uses the reference trajectory endpoints"
            )
        endpoint_values = np.asarray((reference[0], reference[-1]), dtype=float)

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
            raise ValueError(
                "The reference corridor does not overlap the path-valid region"
            )
    elif corridor_radius is not None:
        raise ValueError("corridor_radius is only valid with corridor path mode")

    if path_mode == "fixed":
        return _fixed_reference_path_result(
            results, reference, metric_scale, metric_description, support,
            gx, gy, max_minima, path_aligned_marginal, thermal_energy,
            perpendicular_points, perpendicular_width,
        )

    if endpoint_values.shape != (2, 2) or not np.all(np.isfinite(endpoint_values)):
        raise ValueError("endpoints must contain two finite (x, y) coordinates")
    start_ij = _snap(endpoint_values[0], gx, gy)
    end_ij = _snap(endpoint_values[1], gx, gy)
    if start_ij == end_ij:
        raise ValueError("The two requested endpoints collapse to one grid point")
    if not support[start_ij]:
        raise ValueError("The start endpoint snaps outside the path-valid region")
    if not support[end_ij]:
        raise ValueError("The end endpoint snaps outside the path-valid region")

    components, component_count = _support_components(support)
    if components[start_ij] != components[end_ij]:
        raise ValueError("Requested endpoints lie in disconnected path-valid regions")
    minima = find_minima(
        pmf, gx, gy, max_minima=max_minima, support_mask=support
    )

    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    gradient_scaled = _posterior_mean_gradient_2d(
        results, grid_points
    ).reshape(pmf.shape + (2,))
    gradient_scaled = gradient_scaled * metric_scale

    path, interval_lower, interval_upper = _minimum_range_path(
        pmf, start_ij, end_ij, dx / metric_scale[0], dy / metric_scale[1],
        support, gradient_scaled=gradient_scaled,
    )
    pi = np.array([point[0] for point in path])
    pj = np.array([point[1] for point in path])
    px, py = gx[pi], gy[pj]
    ds = np.hypot(
        np.diff(px) / metric_scale[0], np.diff(py) / metric_scale[1]
    )
    energy = pmf[pi, pj]

    optimal_mask = support & (pmf >= interval_lower) & (pmf <= interval_upper)
    typical_gradient = float(np.median(
        np.linalg.norm(gradient_scaled[optimal_mask], axis=1)
    ))
    gradient_floor = 0.1 * typical_gradient if typical_gradient > 0.0 else 1.0
    edge_gradient = 0.5 * (
        gradient_scaled[pi[:-1], pj[:-1]]
        + gradient_scaled[pi[1:], pj[1:]]
    )
    edge_magnitude = np.linalg.norm(edge_gradient, axis=1)
    edge_direction = np.column_stack([
        np.diff(pi) * dx / metric_scale[0],
        np.diff(pj) * dy / metric_scale[1],
    ]) / ds[:, None]
    cosine = np.abs(np.einsum(
        "ij,ij->i", edge_gradient, edge_direction
    )) / np.maximum(edge_magnitude, np.finfo(float).tiny)
    cosine = np.clip(cosine, 0.0, 1.0)
    reliable = edge_magnitude >= gradient_floor
    median_abs_gradient_cosine = (
        float(np.median(cosine[reliable])) if np.any(reliable) else None
    )
    reliability = edge_magnitude**2 / (edge_magnitude**2 + gradient_floor**2)
    length_weighted_gradient_misalignment = float(np.average(
        reliability * (1.0 - cosine**2), weights=ds
    ))

    points = np.column_stack([px, py])
    profile = evaluate_path_profile(
        results, points, metric_scale,
        energy=energy,
        latent_variance=results["latent_variance_raw"][pi, pj],
    )
    expected_range = interval_upper - interval_lower
    tolerance = 1e-10 * max(1.0, abs(expected_range))
    if abs(profile["barrier"] - expected_range) > tolerance:
        raise RuntimeError(
            "Selected path does not realize the exact minimum energy interval"
        )
    max_reference_distance = (
        float(np.max(reference_distance[pi, pj]))
        if reference_distance is not None else None
    )
    start_xy = (float(gx[start_ij[0]]), float(gy[start_ij[1]]))
    end_xy = (float(gx[end_ij[0]]), float(gy[end_ij[1]]))

    path_result = {
        **profile,
        "minima": minima,
        "nominal_start_xy": tuple(endpoint_values[0]),
        "nominal_end_xy": tuple(endpoint_values[1]),
        "start_endpoint": {
            "ij": start_ij,
            "xy": start_xy,
            "requested_xy": tuple(endpoint_values[0]),
            "component": int(components[start_ij]),
            "selection": "nearest_grid_cell",
        },
        "end_endpoint": {
            "ij": end_ij,
            "xy": end_xy,
            "requested_xy": tuple(endpoint_values[1]),
            "component": int(components[end_ij]),
            "selection": "nearest_grid_cell",
        },
        "support_component": int(components[start_ij]),
        "support_component_count": int(component_count),
        "path_graph_connectivity": 8,
        "path_objective": "minimum_pmf_range_then_fixed_gradient_aligned_cost",
        "optimal_energy_lower": float(interval_lower),
        "optimal_energy_upper": float(interval_upper),
        "path_mode": path_mode,
        "reference_path": reference,
        "corridor_radius": corridor_radius,
        "max_reference_distance": max_reference_distance,
        "path_median_abs_gradient_cosine": median_abs_gradient_cosine,
        "path_length_weighted_gradient_misalignment": (
            length_weighted_gradient_misalignment
        ),
        "validity_kind": results.get("path_valid_kind", "path_valid_mask"),
        "support_radius": results.get("support_radius"),
        "metric_scale": metric_scale,
        "path_coordinate_description": metric_description,
        "cv_names": results.get("cv_names", ("cv0", "cv1")),
        "cv_units": results.get("cv_units", ("", "")),
        "energy_unit": results.get("energy_unit", "eV"),
        "default_uncertainty": results["default_uncertainty"],
    }
    if path_aligned_marginal:
        if thermal_energy is None:
            raise ValueError(
                "thermal_energy is required when path_aligned_marginal=True"
            )
        path_result["path_aligned_marginal"] = _path_aligned_marginal_pmf(
            results,
            path_result,
            thermal_energy=thermal_energy,
            perpendicular_points=perpendicular_points,
            perpendicular_width=perpendicular_width,
        )
    return path_result


def save_lowest_barrier_path(path_result: dict, path: str) -> None:
    """Write a selected path and its covariance-aware uncertainty profile."""
    cvn, cvu = path_result["cv_names"], path_result["cv_units"]
    eu = path_result["energy_unit"]
    sx, sy = path_result["start_xy"]
    ex, ey = path_result["end_xy"]
    nsx, nsy = path_result.get("nominal_start_xy", (sx, sy))
    nex, ney = path_result.get("nominal_end_xy", (ex, ey))
    tx, ty = path_result["ts_xy"]
    mx, my = path_result["path_min_xy"]
    metric = np.asarray(path_result["metric_scale"], dtype=float)
    mode = path_result.get("path_mode", "search")
    endpoint_note = (
        f"requested/reference start = ({nsx:.4f} {cvu[0]}, {nsy:.4f} {cvu[1]}); "
        f"requested/reference end = ({nex:.4f} {cvu[0]}, {ney:.4f} {cvu[1]})\n"
    )
    alignment_note = ""
    if path_result.get("path_median_abs_gradient_cosine") is not None:
        alignment_note = (
            "fixed gradient alignment: median |cos| = "
            f"{path_result['path_median_abs_gradient_cosine']:.4f}; "
            "length-weighted misalignment = "
            f"{path_result['path_length_weighted_gradient_misalignment']:.4f}\n"
        )
    mode_note = f"path mode = {mode}\n"
    if path_result.get("corridor_radius") is not None:
        mode_note += (
            "reference corridor radius = "
            f"{path_result['corridor_radius']:.6g} path-metric units; "
            "maximum selected-path distance = "
            f"{path_result['max_reference_distance']:.6g}\n"
        )
    interval_note = (
        "selected energy interval = "
        f"[{path_result['optimal_energy_lower']:.6g}, "
        f"{path_result['optimal_energy_upper']:.6g}] {eu}\n"
    )
    header = (
        "Path on a two-dimensional GPR PMF\n"
        f"{mode_note}"
        f"path metric scales = ({metric[0]:.6g} {cvu[0]}, "
        f"{metric[1]:.6g} {cvu[1]})\n"
        f"path objective = {path_result['path_objective']}\n"
        f"{interval_note}"
        f"{alignment_note}"
        f"{endpoint_note}"
        f"start = ({sx:.4f} {cvu[0]}, {sy:.4f} {cvu[1]})\n"
        f"end = ({ex:.4f} {cvu[0]}, {ey:.4f} {cvu[1]})\n"
        f"path minimum = ({mx:.4f} {cvu[0]}, {my:.4f} {cvu[1]}) "
        f"at dimensionless metric arclength s = {path_result['path_min_s']:.4f}\n"
        f"transition-state candidate = ({tx:.4f} {cvu[0]}, {ty:.4f} {cvu[1]}) "
        f"at dimensionless metric arclength s = {path_result['ts_s']:.4f}\n"
        f"barrier = max(path PMF) - min(path PMF) = "
        f"{path_result['barrier']:.4f} {eu}; "
        f"sigma_raw = {path_result['barrier_err_raw']:.4f} {eu}; "
        f"sigma_calibrated = {path_result['barrier_err_calibrated']:.4f} {eu}\n"
        f"reaction dF = {path_result['delta_f']:.4f} {eu}; "
        f"sigma_raw = {path_result['delta_f_err_raw']:.4f} {eu}; "
        f"sigma_calibrated = {path_result['delta_f_err_calibrated']:.4f} {eu}\n"
        f"s(metric_arclength) {cvn[0]}({cvu[0]}) {cvn[1]}({cvu[1]}) "
        f"PMF({eu}) PMF_rel_start({eu}) sigma_from_start_raw({eu}) "
        f"sigma_from_start_calibrated({eu}) PMF_rel_path_min({eu}) "
        f"sigma_from_path_min_raw({eu}) sigma_from_path_min_calibrated({eu})"
    )
    data = np.column_stack([
        path_result["s"], path_result["x"], path_result["y"],
        path_result["pmf"], path_result["pmf_rel"],
        path_result["sigma_raw"], path_result["sigma_calibrated"],
        path_result["pmf_rel_path_min"],
        path_result["sigma_from_path_min_raw"],
        path_result["sigma_from_path_min_calibrated"],
    ])
    np.savetxt(path, data, header=header, fmt="%.6f")



def save_path_aligned_marginal(marginal: dict, path: str) -> None:
    """Write a perpendicular-Boltzmann-integrated PMF A(s)."""
    cvn, cvu = marginal["cv_names"], marginal["cv_units"]
    eu = marginal["energy_unit"]
    metric = np.asarray(marginal["metric_scale"], dtype=float)
    header = (
        "Path-aligned marginal PMF: A(s) = -kBT log integral_u exp[-F(s,u)/kBT]\n"
        f"kBT = {marginal['thermal_energy']:.8g} {eu}; s and u use the "
        "dimensionless path metric\n"
        f"path metric scales = ({metric[0]:.6g} {cvu[0]}, "
        f"{metric[1]:.6g} {cvu[1]})\n"
        f"s(metric_arclength) {cvn[0]}({cvu[0]}) {cvn[1]}({cvu[1]}) "
        f"A_rel_min({eu}) sigma_raw({eu}) sigma_calibrated({eu}) "
        "u_min(metric) u_max(metric) perpendicular_samples"
    )
    data = np.column_stack([
        marginal["s"], marginal["x"], marginal["y"], marginal["pmf"],
        marginal["sigma_raw"], marginal["sigma_calibrated"],
        marginal["u_min"], marginal["u_max"],
        marginal["perpendicular_samples"],
    ])
    np.savetxt(path, data, header=header, fmt="%.6f")
