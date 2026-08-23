"""Lowest-barrier paths on reconstructed two-dimensional PMFs.

The grid algorithm minimizes the largest PMF value encountered between two
states.  It is deliberately called a *lowest-barrier path*, not a
minimum-energy path: a true MEP additionally satisfies a local force/path
condition and normally requires string- or NEB-like refinement.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_solve
from scipy.ndimage import label, minimum_filter
from scipy.special import logsumexp


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


def _endpoint_candidates(
    pmf: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    support_mask: np.ndarray,
    components: np.ndarray,
    nominal_xy,
    lengthscale: np.ndarray,
    radius: float,
    endpoint_name: str,
) -> list[dict]:
    """Return the best supported endpoint candidate in each support island."""
    nominal = np.asarray(nominal_xy, dtype=float)
    if nominal.shape != (2,) or not np.all(np.isfinite(nominal)):
        raise ValueError(f"{endpoint_name} endpoint must contain two finite values")
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    distance = np.sqrt(
        ((GX - nominal[0]) / lengthscale[0]) ** 2
        + ((GY - nominal[1]) / lengthscale[1]) ** 2
    )
    valid = np.asarray(support_mask, dtype=bool) & np.isfinite(pmf)
    neighborhood = valid & (distance <= radius + 1.0e-12)
    if not np.any(neighborhood):
        nearest = float(np.min(distance[valid])) if np.any(valid) else np.inf
        raise ValueError(
            f"The {endpoint_name} endpoint neighborhood contains no supported "
            f"grid point (radius={radius:g} GP lengthscales; nearest={nearest:.3g})"
        )

    work = np.where(valid, pmf, np.inf)
    local = valid & (
        work <= minimum_filter(work, size=3, mode="constant", cval=np.inf)
    )

    candidates = []
    for component in np.unique(components[neighborhood]):
        if component == 0:
            continue
        component_neighborhood = neighborhood & (components == component)
        local_pool = component_neighborhood & local
        used_fallback = not np.any(local_pool)
        pool = component_neighborhood if used_fallback else local_pool
        indices = np.argwhere(pool)
        selected = indices[int(np.argmin(pmf[pool]))]
        i, j = int(selected[0]), int(selected[1])
        candidates.append({
            "ij": (i, j),
            "xy": (float(gx[i]), float(gy[j])),
            "energy": float(pmf[i, j]),
            "component": int(component),
            "nominal_xy": (float(nominal[0]), float(nominal[1])),
            "shift_kernel_distance": float(distance[i, j]),
            "selection": (
                "supported_neighborhood_minimum"
                if used_fallback else "grid_local_minimum"
            ),
            "used_fallback": used_fallback,
            "local_minima_in_neighborhood": int(
                np.count_nonzero(component_neighborhood & local)
            ),
            "supported_points_in_neighborhood": int(
                np.count_nonzero(component_neighborhood)
            ),
        })
    return candidates


def _adjust_endpoint_pair(
    pmf: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    support_mask: np.ndarray,
    endpoints,
    lengthscale: np.ndarray,
    radius: float,
) -> tuple[dict, dict]:
    """Choose nearby endpoint minima that belong to one support component."""
    components, _ = _support_components(support_mask)
    start_candidates = _endpoint_candidates(
        pmf, gx, gy, support_mask, components, endpoints[0], lengthscale,
        radius, "start",
    )
    end_candidates = _endpoint_candidates(
        pmf, gx, gy, support_mask, components, endpoints[1], lengthscale,
        radius, "end",
    )
    pairs = [
        (start, end)
        for start in start_candidates
        for end in end_candidates
        if start["component"] == end["component"] and start["ij"] != end["ij"]
    ]
    if not pairs:
        start_components = sorted(item["component"] for item in start_candidates)
        end_components = sorted(item["component"] for item in end_candidates)
        raise ValueError(
            "Endpoint neighborhoods do not reach the same connected path-valid "
            "component; adjust the validity range or endpoint search radius. "
            f"start components={start_components}, end components={end_components}"
        )
    return min(
        pairs,
        key=lambda pair: (
            max(pair[0]["energy"], pair[1]["energy"]),
            pair[0]["energy"] + pair[1]["energy"],
            pair[0]["shift_kernel_distance"] + pair[1]["shift_kernel_distance"],
        ),
    )


def _window_center_endpoint(
    nominal_xy, centers, gx, gy, pmf, support_mask, components, name, lengthscale
) -> dict:
    """Select the restraint-window centre nearest a requested endpoint."""
    nominal = np.asarray(nominal_xy, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if (
        centers.ndim != 2
        or centers.shape[1] != 2
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError(
            "results centers must be a finite (n_windows, 2) array when "
            "adjust_endpoints=False"
        )
    distances = np.linalg.norm((centers - nominal) / lengthscale, axis=1)
    window_index = int(np.argmin(distances))
    center = centers[window_index]
    ij = _snap(center, gx, gy)
    if not support_mask[ij] or not np.isfinite(pmf[ij]):
        raise ValueError(
            f"The {name} endpoint window centre (window {window_index}) snaps "
            "outside the path-valid region; enable endpoint adjustment or "
            "change the sampled-support settings"
        )
    selected = np.array([gx[ij[0]], gy[ij[1]]], dtype=float)
    return {
        "ij": ij,
        "xy": (float(selected[0]), float(selected[1])),
        "energy": float(pmf[ij]),
        "component": int(components[ij]),
        "nominal_xy": (float(nominal[0]), float(nominal[1])),
        "window_index": window_index,
        "window_center_xy": (float(center[0]), float(center[1])),
        "shift_kernel_distance": float(distances[window_index]),
        "grid_snap_kernel_distance": float(
            np.linalg.norm((selected - center) / lengthscale)
        ),
        "selection": "nearest_window_center",
        "used_fallback": False,
        "local_minima_in_neighborhood": 0,
        "supported_points_in_neighborhood": 1,
    }


def _gradient_alignment_penalty(
    gradient: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    direction: np.ndarray,
    gradient_floor: float,
) -> float:
    """Return a weak-gradient-aware squared sine for one directed edge."""
    local_gradient = 0.5 * (gradient[start] + gradient[end])
    magnitude = float(np.linalg.norm(local_gradient))
    if magnitude == 0.0:
        return 0.0
    cosine = float(np.dot(local_gradient, direction) / magnitude)
    sine_squared = max(0.0, 1.0 - min(1.0, cosine * cosine))
    reliability = magnitude**2 / (magnitude**2 + gradient_floor**2)
    return reliability * sine_squared


def _minimax_path(
    pmf: np.ndarray,
    start_ij: tuple[int, int],
    end_ij: tuple[int, int],
    dx_scaled: float,
    dy_scaled: float,
    support_mask: np.ndarray,
    gradient_scaled: np.ndarray | None = None,
    gradient_alignment_weight: float = 0.0,
) -> list[tuple[int, int]]:
    """Find a gradient-aligned path at the globally minimal PMF bottleneck.

    The first Dijkstra pass computes the exact minimax threshold. The second
    pass minimizes an additive metric-length plus gradient-misalignment cost in
    the subgraph at or below that threshold. A zero alignment weight recovers
    the geometrically shortest path.
    """
    nx, ny = pmf.shape
    diagonal = float(np.hypot(dx_scaled, dy_scaled))
    # Axis 0 indexes gx and therefore uses dx; axis 1 uses dy.
    steps = [
        (-1, 0, dx_scaled), (1, 0, dx_scaled),
        (0, -1, dy_scaled), (0, 1, dy_scaled),
        (-1, -1, diagonal), (-1, 1, diagonal),
        (1, -1, diagonal), (1, 1, diagonal),
    ]

    best = np.full((nx, ny), np.inf)
    si, sj = start_ij
    ti, tj = end_ij
    if not support_mask[si, sj] or not support_mask[ti, tj]:
        raise ValueError("Both pathway endpoints must lie in the path-valid region")
    best[si, sj] = pmf[si, sj]
    queue = [(pmf[si, sj], si, sj)]

    while queue:
        bottleneck, i, j = heapq.heappop(queue)
        if bottleneck > best[i, j]:
            continue
        if (i, j) == (ti, tj):
            break
        for di, dj, _ in steps:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny and support_mask[ni, nj]):
                continue
            new_bottleneck = max(bottleneck, pmf[ni, nj])
            if new_bottleneck < best[ni, nj]:
                best[ni, nj] = new_bottleneck
                heapq.heappush(queue, (new_bottleneck, ni, nj))

    if not np.isfinite(best[ti, tj]):
        raise RuntimeError("No path-valid path found between the requested endpoints")

    threshold = best[ti, tj]
    allowed = support_mask & (pmf <= threshold)
    if isinstance(gradient_alignment_weight, (bool, np.bool_)):
        raise ValueError("gradient_alignment_weight must be finite and non-negative")
    gradient_alignment_weight = float(gradient_alignment_weight)
    if not np.isfinite(gradient_alignment_weight) or gradient_alignment_weight < 0:
        raise ValueError("gradient_alignment_weight must be finite and non-negative")
    gradient_floor = 1.0
    if gradient_scaled is not None:
        gradient_scaled = np.asarray(gradient_scaled, dtype=float)
        if gradient_scaled.shape != pmf.shape + (2,):
            raise ValueError("gradient_scaled must have shape pmf.shape + (2,)")
        if not np.all(np.isfinite(gradient_scaled[allowed])):
            raise ValueError("gradient_scaled must be finite in the minimax sublevel set")
        magnitudes = np.linalg.norm(gradient_scaled[allowed], axis=1)
        typical_gradient = float(np.median(magnitudes))
        if typical_gradient > 0.0:
            gradient_floor = 0.1 * typical_gradient
    elif gradient_alignment_weight > 0.0:
        raise ValueError("gradient_scaled is required when gradient alignment is enabled")

    best_cost = np.full((nx, ny), np.inf)
    best_cost[si, sj] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, si, sj)]
    while queue:
        cost, i, j = heapq.heappop(queue)
        if cost > best_cost[i, j]:
            continue
        if (i, j) == (ti, tj):
            break
        for di, dj, distance in steps:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny and allowed[ni, nj]):
                continue
            direction = np.array([di * dx_scaled, dj * dy_scaled]) / distance
            alignment = (
                _gradient_alignment_penalty(
                    gradient_scaled, (i, j), (ni, nj), direction, gradient_floor
                )
                if gradient_scaled is not None else 0.0
            )
            edge_cost = distance * (
                1.0 + gradient_alignment_weight * alignment
            )
            new_cost = cost + edge_cost
            if new_cost < best_cost[ni, nj]:
                best_cost[ni, nj] = new_cost
                previous[(ni, nj)] = (i, j)
                heapq.heappush(queue, (new_cost, ni, nj))

    if not np.isfinite(best_cost[ti, tj]):
        raise RuntimeError("Internal error finding a path at the minimax threshold")
    path = [(ti, tj)]
    while path[-1] != (si, sj):
        path.append(previous[path[-1]])
    return path[::-1]


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


def find_lowest_barrier_path(
    results: dict,
    endpoints=None,
    max_minima: int = 8,
    metric_scale: tuple[float, float] | np.ndarray | None = None,
    *,
    endpoint_search_radius: float | None = None,
    adjust_endpoints: bool = True,
    gradient_alignment_weight: float = 1.0,
    path_aligned_marginal: bool = False,
    thermal_energy: float | None = None,
    perpendicular_points: int = 201,
    perpendicular_width: float | None = None,
) -> dict:
    """Find a minimum-bottleneck path inside one path-valid component.

    The barrier is minimized first. Among paths at that exact barrier, an
    additive metric-length plus GP-gradient-misalignment cost is minimized.
    Explicit endpoints are moved to the lowest nearby path-valid minima by
    default. With ``adjust_endpoints=False``, each endpoint instead uses the
    nearest restraint-window centre (snapped to the path grid).
    ``endpoint_search_radius`` is measured in GP-lengthscale units;
    when omitted it uses the reconstruction's sampled-support radius. Endpoint
    selection and the complete path are restricted to one connected component
    of ``results['path_valid_mask']`` when present, otherwise
    ``results['support_mask']``. By default path length divides each CV by
    its fitted GP lengthscale; ``metric_scale`` may override those two scales.
    ``gradient_alignment_weight`` controls the secondary MEP-like preference;
    zero recovers the geometrically shortest minimax path.
    If ``path_aligned_marginal`` is true, ``thermal_energy`` supplies kBT in
    ``results['energy_unit']``.
    """
    from .integration_2d import posterior_covariance_2d
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
    if not isinstance(adjust_endpoints, (bool, np.bool_)):
        raise ValueError("adjust_endpoints must be Boolean")

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

    components, component_count = _support_components(support)
    minima = find_minima(
        pmf, gx, gy, max_minima=max_minima, support_mask=support
    )
    for minimum in minima:
        minimum["support_component"] = int(components[minimum["ij"]])

    resolved_endpoint_radius = None
    if endpoints is None and not adjust_endpoints:
        raise ValueError(
            "adjust_endpoints=False requires two explicit endpoints so their "
            "nearest restraint windows can be identified"
        )
    if endpoints is not None:
        endpoint_values = np.asarray(endpoints, dtype=float)
        if endpoint_values.shape != (2, 2) or not np.all(np.isfinite(endpoint_values)):
            raise ValueError("endpoints must contain two finite (x, y) coordinates")
        if adjust_endpoints:
            resolved_endpoint_radius = endpoint_search_radius
            if resolved_endpoint_radius is None:
                resolved_endpoint_radius = results.get("support_radius")
            if resolved_endpoint_radius is None:
                resolved_endpoint_radius = 1.0
            if isinstance(resolved_endpoint_radius, (bool, np.bool_)):
                raise ValueError(
                    "endpoint_search_radius must be finite and positive"
                )
            try:
                resolved_endpoint_radius = float(resolved_endpoint_radius)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "endpoint_search_radius must be finite and positive"
                ) from exc
            if (
                not np.isfinite(resolved_endpoint_radius)
                or resolved_endpoint_radius <= 0
            ):
                raise ValueError(
                    "endpoint_search_radius must be finite and positive"
                )
            start_endpoint, end_endpoint = _adjust_endpoint_pair(
                pmf, gx, gy, support, endpoint_values, lengthscale,
                float(resolved_endpoint_radius),
            )
        else:
            try:
                centers = results["centers"]
            except KeyError as exc:
                raise ValueError(
                    "results must contain restraint-window centers when "
                    "adjust_endpoints=False"
                ) from exc
            start_endpoint = _window_center_endpoint(
                endpoint_values[0], centers, gx, gy, pmf, support, components,
                "start", lengthscale,
            )
            end_endpoint = _window_center_endpoint(
                endpoint_values[1], centers, gx, gy, pmf, support, components,
                "end", lengthscale,
            )
            if start_endpoint["component"] != end_endpoint["component"]:
                raise ValueError(
                    "Requested endpoints lie in disconnected path-valid "
                    "components; increase support_radius or enable endpoint "
                    "adjustment"
                )
    else:
        pairs = [
            (first, second)
            for index, first in enumerate(minima)
            for second in minima[index + 1:]
            if first["support_component"] == second["support_component"]
        ]
        if not pairs:
            raise ValueError(
                "No connected path-valid component contains two detected "
                f"minima (components={component_count}, minima={len(minima)}); "
                "pass endpoints explicitly or increase support_radius"
            )
        first, second = min(
            pairs,
            key=lambda pair: (
                max(pair[0]["energy"], pair[1]["energy"]),
                pair[0]["energy"] + pair[1]["energy"],
            ),
        )
        start_endpoint = {
            **first,
            "component": first["support_component"],
            "nominal_xy": first["xy"],
            "shift_kernel_distance": 0.0,
            "selection": "global_supported_minimum",
            "used_fallback": False,
            "local_minima_in_neighborhood": 1,
            "supported_points_in_neighborhood": 1,
        }
        end_endpoint = {
            **second,
            "component": second["support_component"],
            "nominal_xy": second["xy"],
            "shift_kernel_distance": 0.0,
            "selection": "global_supported_minimum",
            "used_fallback": False,
            "local_minima_in_neighborhood": 1,
            "supported_points_in_neighborhood": 1,
        }

    start_ij = start_endpoint["ij"]
    end_ij = end_endpoint["ij"]
    if start_ij == end_ij:
        raise ValueError("The two selected endpoints collapse to one grid point")

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

    dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    if isinstance(gradient_alignment_weight, (bool, np.bool_)):
        raise ValueError(
            "gradient_alignment_weight must be finite and non-negative"
        )
    gradient_alignment_weight = float(gradient_alignment_weight)
    if not np.isfinite(gradient_alignment_weight) or gradient_alignment_weight < 0:
        raise ValueError(
            "gradient_alignment_weight must be finite and non-negative"
        )
    gradient_scaled = None
    if gradient_alignment_weight > 0.0:
        grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        gradient_scaled = _posterior_mean_gradient_2d(
            results, grid_points
        ).reshape(pmf.shape + (2,))
        gradient_scaled = gradient_scaled * metric_scale

    path = _minimax_path(
        pmf, start_ij, end_ij, dx / metric_scale[0], dy / metric_scale[1],
        support, gradient_scaled=gradient_scaled,
        gradient_alignment_weight=gradient_alignment_weight,
    )
    pi = np.array([p[0] for p in path])
    pj = np.array([p[1] for p in path])
    px, py = gx[pi], gy[pj]
    ds = np.hypot(
        np.diff(px) / metric_scale[0], np.diff(py) / metric_scale[1]
    )
    s = np.concatenate([[0.0], np.cumsum(ds)])
    energy = pmf[pi, pj]
    energy_relative = energy - energy[0]
    ts = int(np.argmax(energy_relative))

    median_abs_gradient_cosine = None
    length_weighted_gradient_misalignment = None
    if gradient_scaled is not None:
        threshold_mask = support & (pmf <= float(np.max(energy)))
        typical_gradient = float(np.median(
            np.linalg.norm(gradient_scaled[threshold_mask], axis=1)
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
        if np.any(reliable):
            median_abs_gradient_cosine = float(np.median(cosine[reliable]))
        reliability = edge_magnitude**2 / (
            edge_magnitude**2 + gradient_floor**2
        )
        length_weighted_gradient_misalignment = float(np.average(
            reliability * (1.0 - cosine**2), weights=ds
        ))

    points = np.column_stack([px, py])
    covariance_to_start = posterior_covariance_2d(
        results, points, points[[0]], calibrated=False
    ).ravel()
    latent_variance = results["latent_variance_raw"][pi, pj]
    relative_variance = np.clip(
        latent_variance + latent_variance[0] - 2.0 * covariance_to_start,
        0,
        np.inf,
    )
    relative_variance[0] = 0.0  # F(start) - F(start) is exactly known to be zero.
    sigma_raw = np.sqrt(relative_variance)
    sigma_calibrated = sigma_raw * results["loo_calibration_factor"]
    use_calibrated = results["default_uncertainty"] == "calibrated"
    sigma_default = sigma_calibrated if use_calibrated else sigma_raw

    path_result = {
        "s": s,
        "x": px,
        "y": py,
        "pmf": energy,
        "pmf_rel": energy_relative,
        "sigma_raw": sigma_raw,
        "sigma_calibrated": sigma_calibrated,
        "sigma": sigma_default,
        "minima": minima,
        "nominal_start_xy": tuple(start_endpoint["nominal_xy"]),
        "nominal_end_xy": tuple(end_endpoint["nominal_xy"]),
        "start_endpoint": start_endpoint,
        "end_endpoint": end_endpoint,
        "explicit_endpoints": endpoints is not None,
        "endpoints_adjusted": bool(endpoints is not None and adjust_endpoints),
        "endpoint_selection": (
            "nearby_local_minimum" if endpoints is not None and adjust_endpoints
            else "window_centre" if endpoints is not None
            else "automatic_minima"
        ),
        "endpoint_search_radius": (
            float(resolved_endpoint_radius)
            if resolved_endpoint_radius is not None else None
        ),
        "support_component": int(start_endpoint["component"]),
        "support_component_count": int(component_count),
        "start_xy": (float(px[0]), float(py[0])),
        "end_xy": (float(px[-1]), float(py[-1])),
        "ts_xy": (float(px[ts]), float(py[ts])),
        "ts_s": float(s[ts]),
        "bottleneck_energy": float(np.max(energy)),
        "path_metric_length": float(s[-1]),
        "path_graph_connectivity": 8,
        "path_objective": (
            "minimum_bottleneck_then_gradient_aligned_metric_cost"
            if gradient_alignment_weight > 0.0
            else "minimum_bottleneck_then_shortest_metric_length"
        ),
        "path_gradient_alignment_weight": gradient_alignment_weight,
        "path_median_abs_gradient_cosine": median_abs_gradient_cosine,
        "path_length_weighted_gradient_misalignment": (
            length_weighted_gradient_misalignment
        ),
        "barrier": float(energy_relative[ts]),
        "barrier_err_raw": float(sigma_raw[ts]),
        "barrier_err_calibrated": float(sigma_calibrated[ts]),
        "barrier_err": float(sigma_default[ts]),
        "delta_f": float(energy_relative[-1]),
        "delta_f_err_raw": float(sigma_raw[-1]),
        "delta_f_err_calibrated": float(sigma_calibrated[-1]),
        "delta_f_err": float(sigma_default[-1]),
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
    """Write the lowest-barrier path and both uncertainty estimates."""
    cvn, cvu = path_result["cv_names"], path_result["cv_units"]
    eu = path_result["energy_unit"]
    sx, sy = path_result["start_xy"]
    ex, ey = path_result["end_xy"]
    nsx, nsy = path_result.get("nominal_start_xy", (sx, sy))
    nex, ney = path_result.get("nominal_end_xy", (ex, ey))
    endpoint_note = ""
    if path_result.get("explicit_endpoints", False):
        endpoint_note = (
            f"requested start = ({nsx:.4f} {cvu[0]}, {nsy:.4f} {cvu[1]}); "
            f"requested end = ({nex:.4f} {cvu[0]}, {ney:.4f} {cvu[1]})\n"
        )
        if path_result["endpoint_selection"] == "nearby_local_minimum":
            radius = path_result["endpoint_search_radius"]
            endpoint_note += (
                "endpoint selection = nearby path-valid minima; "
                f"search radius = {radius:.6g} GP lengthscales\n"
            )
        else:
            start = path_result["start_endpoint"]
            end = path_result["end_endpoint"]
            endpoint_note += (
                "endpoint selection = nearest restraint-window centres; "
                f"window indices = ({start['window_index']}, "
                f"{end['window_index']})\n"
            )
    tx, ty = path_result["ts_xy"]
    metric = np.asarray(path_result["metric_scale"], dtype=float)
    alignment_note = ""
    if path_result.get("path_median_abs_gradient_cosine") is not None:
        alignment_note = (
            "gradient alignment: median |cos| = "
            f"{path_result['path_median_abs_gradient_cosine']:.4f}; "
            "length-weighted misalignment = "
            f"{path_result['path_length_weighted_gradient_misalignment']:.4f}\n"
        )
    header = (
        "Lowest-barrier (minimum-bottleneck) grid path\n"
        f"path metric scales = ({metric[0]:.6g} {cvu[0]}, "
        f"{metric[1]:.6g} {cvu[1]})\n"
        f"path objective = {path_result['path_objective']}; "
        f"gradient weight = "
        f"{path_result.get('path_gradient_alignment_weight', 0.0):.6g}\n"
        f"{alignment_note}"
        f"{endpoint_note}"
        f"start = ({sx:.4f} {cvu[0]}, {sy:.4f} {cvu[1]})\n"
        f"end = ({ex:.4f} {cvu[0]}, {ey:.4f} {cvu[1]})\n"
        f"transition-state candidate = ({tx:.4f} {cvu[0]}, {ty:.4f} {cvu[1]}) "
        f"at dimensionless metric arclength s = {path_result['ts_s']:.4f}\n"
        f"barrier = {path_result['barrier']:.4f} {eu}; "
        f"sigma_raw = {path_result['barrier_err_raw']:.4f} {eu}; "
        f"sigma_calibrated = {path_result['barrier_err_calibrated']:.4f} {eu}\n"
        f"reaction dF = {path_result['delta_f']:.4f} {eu}; "
        f"sigma_raw = {path_result['delta_f_err_raw']:.4f} {eu}; "
        f"sigma_calibrated = {path_result['delta_f_err_calibrated']:.4f} {eu}\n"
        f"s(metric_arclength) {cvn[0]}({cvu[0]}) {cvn[1]}({cvu[1]}) "
        f"PMF({eu}) PMF_rel_start({eu}) sigma_raw({eu}) sigma_calibrated({eu})"
    )
    data = np.column_stack([
        path_result["s"], path_result["x"], path_result["y"],
        path_result["pmf"], path_result["pmf_rel"],
        path_result["sigma_raw"], path_result["sigma_calibrated"],
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
