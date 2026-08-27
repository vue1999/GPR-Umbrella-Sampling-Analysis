"""Exact minimum-range paths on a finite two-dimensional grid."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import label


_NEIGHBORHOOD = np.ones((3, 3), dtype=np.uint8)


def _connected(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    """Return whether two cells share one 8-connected component."""
    if not mask[start] or not mask[end]:
        return False
    components, _ = label(mask, structure=_NEIGHBORHOOD)
    return bool(components[start] == components[end])


def minimum_energy_interval(
    node_energy: np.ndarray,
    start_ij: tuple[int, int],
    end_ij: tuple[int, int],
    support_mask: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Find the narrowest energy interval connecting two supported cells.

    The lower threshold is advanced through sorted node energies while the
    smallest feasible upper threshold advances monotonically. The returned
    interval therefore exactly minimizes ``upper - lower`` on the finite
    8-neighbour graph. Equal-width intervals prefer the lower upper bound.
    """
    energy = np.asarray(node_energy, dtype=float)
    support = np.asarray(support_mask, dtype=bool) & np.isfinite(energy)
    if energy.ndim != 2 or support.shape != energy.shape:
        raise ValueError("node_energy and support_mask must be matching 2D arrays")
    if not support[start_ij] or not support[end_ij]:
        raise ValueError("Both pathway endpoints must lie in the path-valid region")
    if not _connected(support, start_ij, end_ij):
        raise ValueError("Pathway endpoints lie in disconnected path-valid regions")

    start_energy = float(energy[start_ij])
    end_energy = float(energy[end_ij])
    endpoint_low = min(start_energy, end_energy)
    endpoint_high = max(start_energy, end_energy)
    levels = np.unique(energy[support])
    lower_levels = levels[levels <= endpoint_low]
    upper_levels = levels[levels >= endpoint_high]

    best: tuple[float, float, float] | None = None
    upper_index = 0
    for lower in lower_levels:
        while upper_index < len(upper_levels):
            upper = upper_levels[upper_index]
            interval_mask = support & (energy >= lower) & (energy <= upper)
            if _connected(interval_mask, start_ij, end_ij):
                break
            upper_index += 1
        if upper_index == len(upper_levels):
            break
        upper = float(upper_levels[upper_index])
        lower = float(lower)
        width = upper - lower
        candidate = (width, upper, -lower)
        scale = max(1.0, abs(width), abs(upper), abs(lower))
        tolerance = 1e-12 * scale
        if best is None or candidate[0] < best[0] - tolerance or (
            abs(candidate[0] - best[0]) <= tolerance
            and candidate[1:] < best[1:]
        ):
            best = candidate

    if best is None:
        raise RuntimeError("No path-valid energy interval connects the endpoints")
    lower = -best[2]
    upper = best[1]
    allowed = support & (energy >= lower) & (energy <= upper)
    return float(lower), float(upper), allowed


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


def minimum_range_path(
    node_energy: np.ndarray,
    start_ij: tuple[int, int],
    end_ij: tuple[int, int],
    dx_scaled: float,
    dy_scaled: float,
    support_mask: np.ndarray,
    gradient_scaled: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """Return a representative path in the exact minimum energy interval.

    The interval width is the primary objective. Inside that interval Dijkstra
    minimizes length multiplied by ``1 + reliability * sin(angle)**2``. This
    fixed secondary rule prefers gradient-aligned paths without changing the
    exact minimum-range guarantee.
    """
    energy = np.asarray(node_energy, dtype=float)
    lower, upper, allowed = minimum_energy_interval(
        energy, start_ij, end_ij, support_mask
    )
    nx, ny = energy.shape
    diagonal = float(np.hypot(dx_scaled, dy_scaled))
    steps = [
        (-1, 0, dx_scaled), (1, 0, dx_scaled),
        (0, -1, dy_scaled), (0, 1, dy_scaled),
        (-1, -1, diagonal), (-1, 1, diagonal),
        (1, -1, diagonal), (1, 1, diagonal),
    ]

    gradient_floor = 1.0
    if gradient_scaled is not None:
        gradient_scaled = np.asarray(gradient_scaled, dtype=float)
        if gradient_scaled.shape != energy.shape + (2,):
            raise ValueError("gradient_scaled must have shape node_energy.shape + (2,)")
        if not np.all(np.isfinite(gradient_scaled[allowed])):
            raise ValueError("gradient_scaled must be finite in the optimal interval")
        typical = float(np.median(np.linalg.norm(gradient_scaled[allowed], axis=1)))
        if typical > 0.0:
            gradient_floor = 0.1 * typical

    best_cost = np.full((nx, ny), np.inf)
    best_cost[start_ij] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, *start_ij)]
    while queue:
        cost, i, j = heapq.heappop(queue)
        if cost > best_cost[i, j]:
            continue
        if (i, j) == end_ij:
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
            candidate = cost + distance * (1.0 + alignment)
            if candidate < best_cost[ni, nj]:
                best_cost[ni, nj] = candidate
                previous[(ni, nj)] = (i, j)
                heapq.heappush(queue, (candidate, ni, nj))

    if not np.isfinite(best_cost[end_ij]):
        raise RuntimeError("Internal error selecting a path in the optimal interval")
    path = [end_ij]
    while path[-1] != start_ij:
        path.append(previous[path[-1]])
    return path[::-1], lower, upper
