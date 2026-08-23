"""Finite-grid minimax path search with optional gradient alignment."""
from __future__ import annotations

import heapq

import numpy as np


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


def minimax_path(
    node_score: np.ndarray,
    start_ij: tuple[int, int],
    end_ij: tuple[int, int],
    dx_scaled: float,
    dy_scaled: float,
    support_mask: np.ndarray,
    gradient_scaled: np.ndarray | None = None,
    gradient_alignment_weight: float = 0.0,
) -> list[tuple[int, int]]:
    """Find a gradient-aligned path at the globally minimal node bottleneck.

    The first Dijkstra pass computes the exact minimax threshold. The second
    minimizes metric length plus gradient misalignment inside that sublevel
    graph. ``node_score`` may be a mean PMF or a risk-adjusted PMF score.
    """
    nx, ny = node_score.shape
    diagonal = float(np.hypot(dx_scaled, dy_scaled))
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
    best[si, sj] = node_score[si, sj]
    queue = [(node_score[si, sj], si, sj)]
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
            candidate = max(bottleneck, node_score[ni, nj])
            if candidate < best[ni, nj]:
                best[ni, nj] = candidate
                heapq.heappush(queue, (candidate, ni, nj))
    if not np.isfinite(best[ti, tj]):
        raise RuntimeError("No path-valid path found between the requested endpoints")

    threshold = best[ti, tj]
    allowed = support_mask & (node_score <= threshold)
    if isinstance(gradient_alignment_weight, (bool, np.bool_)):
        raise ValueError("gradient_alignment_weight must be finite and non-negative")
    gradient_alignment_weight = float(gradient_alignment_weight)
    if not np.isfinite(gradient_alignment_weight) or gradient_alignment_weight < 0:
        raise ValueError("gradient_alignment_weight must be finite and non-negative")
    gradient_floor = 1.0
    if gradient_scaled is not None:
        gradient_scaled = np.asarray(gradient_scaled, dtype=float)
        if gradient_scaled.shape != node_score.shape + (2,):
            raise ValueError("gradient_scaled must have shape node_score.shape + (2,)")
        if not np.all(np.isfinite(gradient_scaled[allowed])):
            raise ValueError("gradient_scaled must be finite in the minimax sublevel set")
        typical = float(np.median(np.linalg.norm(gradient_scaled[allowed], axis=1)))
        if typical > 0.0:
            gradient_floor = 0.1 * typical
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
            candidate = cost + distance * (
                1.0 + gradient_alignment_weight * alignment
            )
            if candidate < best_cost[ni, nj]:
                best_cost[ni, nj] = candidate
                previous[(ni, nj)] = (i, j)
                heapq.heappush(queue, (candidate, ni, nj))
    if not np.isfinite(best_cost[ti, tj]):
        raise RuntimeError("Internal error finding a path at the minimax threshold")
    path = [(ti, tj)]
    while path[-1] != (si, sj):
        path.append(previous[path[-1]])
    return path[::-1]
