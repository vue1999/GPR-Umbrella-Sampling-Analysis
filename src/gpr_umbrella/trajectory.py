"""Geometry helpers for reference-constrained two-dimensional paths."""
from __future__ import annotations

import numpy as np


def validate_reference_path(points: np.ndarray) -> np.ndarray:
    """Return a finite, duplicate-free ``(n, 2)`` reference polyline."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("reference_path must have shape (n_points >= 2, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("reference_path must contain only finite coordinates")
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("reference_path must contain two distinct coordinates")
    return points


def distance_to_polyline(
    points: np.ndarray,
    vertices: np.ndarray,
    metric_scale: np.ndarray,
) -> np.ndarray:
    """Return exact point-to-polyline distances in scaled path coordinates."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    vertices = validate_reference_path(vertices)
    metric_scale = np.asarray(metric_scale, dtype=float)
    if metric_scale.shape != (2,) or np.any(metric_scale <= 0):
        raise ValueError("metric_scale must contain two positive values")
    scaled_points = points / metric_scale
    scaled_vertices = vertices / metric_scale
    best = np.full(len(points), np.inf)
    for start, end in zip(scaled_vertices[:-1], scaled_vertices[1:]):
        segment = end - start
        fraction = np.clip(
            ((scaled_points - start) @ segment) / (segment @ segment),
            0.0,
            1.0,
        )
        projection = start + fraction[:, None] * segment
        best = np.minimum(best, np.linalg.norm(scaled_points - projection, axis=1))
    return best


def resample_polyline(
    vertices: np.ndarray,
    metric_scale: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Densify a polyline without moving its vertices or endpoints."""
    vertices = validate_reference_path(vertices)
    metric_scale = np.asarray(metric_scale, dtype=float)
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be finite and positive")
    pieces = []
    for index, (start, end) in enumerate(zip(vertices[:-1], vertices[1:])):
        distance = np.linalg.norm((end - start) / metric_scale)
        intervals = max(1, int(np.ceil(distance / max_step)))
        fraction = np.linspace(0.0, 1.0, intervals + 1)
        segment = start + fraction[:, None] * (end - start)
        pieces.append(segment if index == 0 else segment[1:])
    return np.vstack(pieces)
