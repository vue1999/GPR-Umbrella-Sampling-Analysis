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
from scipy.ndimage import minimum_filter
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


def _minimax_path(
    pmf: np.ndarray,
    start_ij: tuple[int, int],
    end_ij: tuple[int, int],
    dx_scaled: float,
    dy_scaled: float,
    support_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Minimize path bottleneck, with normalized length as the tie-break."""
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
    best_len = np.full((nx, ny), np.inf)
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    si, sj = start_ij
    ti, tj = end_ij
    if not support_mask[si, sj] or not support_mask[ti, tj]:
        raise ValueError("Both pathway endpoints must lie in the sampled support region")
    best[si, sj] = pmf[si, sj]
    best_len[si, sj] = 0.0
    queue = [(pmf[si, sj], 0.0, si, sj)]

    while queue:
        bottleneck, length, i, j = heapq.heappop(queue)
        if (bottleneck, length) > (best[i, j], best_len[i, j]):
            continue
        if (i, j) == (ti, tj):
            break
        for di, dj, distance in steps:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny and support_mask[ni, nj]):
                continue
            new_bottleneck = max(bottleneck, pmf[ni, nj])
            new_length = length + distance
            if (new_bottleneck, new_length) < (best[ni, nj], best_len[ni, nj]):
                best[ni, nj] = new_bottleneck
                best_len[ni, nj] = new_length
                previous[(ni, nj)] = (i, j)
                heapq.heappush(queue, (new_bottleneck, new_length, ni, nj))

    if not np.isfinite(best[ti, tj]):
        raise RuntimeError("No supported path found between the requested endpoints")
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
        results.get("support_mask", np.ones_like(results["pmf"])), dtype=bool
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


def find_lowest_barrier_path(
    results: dict,
    endpoints=None,
    max_minima: int = 8,
    metric_scale: tuple[float, float] | np.ndarray | None = None,
    *,
    path_aligned_marginal: bool = False,
    thermal_energy: float | None = None,
    perpendicular_points: int = 201,
    perpendicular_width: float | None = None,
) -> dict:
    """Find a minimum-bottleneck path between two states on the PMF grid.

    Grid length is only a tie-break between paths with the same bottleneck.
    By default each CV is divided by its fitted GP lengthscale, making this
    tie-break dimensionless and invariant to consistent unit conversions.
    ``metric_scale`` may instead provide two positive, physically motivated
    CV scales. If ``path_aligned_marginal`` is true, ``thermal_energy`` must
    provide kBT in ``results['energy_unit']``; the returned path then includes
    ``A(s) = -kBT log integral exp[-F(s,u)/kBT] du`` under
    ``path_aligned_marginal``.
    """
    from .integration_2d import posterior_covariance_2d

    gx, gy = results["gx"], results["gy"]
    pmf = results["pmf"]
    support = np.asarray(results.get("support_mask", np.ones_like(pmf)), dtype=bool)
    minima = find_minima(pmf, gx, gy, max_minima=max_minima, support_mask=support)

    if endpoints is not None:
        start_ij = _snap(endpoints[0], gx, gy)
        end_ij = _snap(endpoints[1], gx, gy)
    else:
        if len(minima) < 2:
            raise ValueError(
                f"Need at least two supported minima, found {len(minima)}; "
                "pass endpoints=((x0, y0), (x1, y1)) explicitly"
            )
        start_ij, end_ij = minima[0]["ij"], minima[1]["ij"]

    if metric_scale is None:
        metric_scale = np.asarray(results["lengthscale"], dtype=float)
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
    path = _minimax_path(
        pmf, start_ij, end_ij, dx / metric_scale[0], dy / metric_scale[1], support
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
        "start_xy": (float(px[0]), float(py[0])),
        "end_xy": (float(px[-1]), float(py[-1])),
        "ts_xy": (float(px[ts]), float(py[ts])),
        "ts_s": float(s[ts]),
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
    tx, ty = path_result["ts_xy"]
    metric = np.asarray(path_result["metric_scale"], dtype=float)
    header = (
        "Lowest-barrier (minimum-bottleneck) grid path\n"
        f"path metric scales = ({metric[0]:.6g} {cvu[0]}, "
        f"{metric[1]:.6g} {cvu[1]})\n"
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
