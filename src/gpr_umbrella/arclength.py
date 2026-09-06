"""Ordered, window-independent projection in CV space (not CV1 sorting).

The first/last segments are extended as rays: endpoint samples are never
silently clipped. Self intersections cannot be disambiguated using two CVs
alone and are rejected. No window ID or trajectory history changes the CV.
"""
from __future__ import annotations

import numpy as np


def validate_path(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2:
        raise ValueError("Path must contain at least two ordered 2D vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Path contains non-finite coordinates")
    delta = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    scale = max(float(np.max(lengths)), np.finfo(float).tiny)
    if np.any(lengths < scale * 1e-10):
        raise ValueError("Path contains duplicate consecutive vertices")
    if len(delta) > 1:
        cosine = np.sum(delta[:-1] * delta[1:], axis=1) / (lengths[:-1] * lengths[1:])
        if np.any(cosine < -1 + 1e-10):
            raise ValueError("Self-intersecting path: adjacent segments retrace one another")
    # Non-adjacent intersecting segments include exact retracing/closed loops.
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    for i, a in enumerate(delta):
        for j in range(i + 2, len(delta)):
            b = delta[j]
            r = vertices[j] - vertices[i]
            determinant = cross(a, b)
            tol = lengths[i] * lengths[j] * 1e-10
            if abs(determinant) > tol:
                t, u = cross(r, b) / determinant, cross(r, a) / determinant
                intersects = -1e-10 <= t <= 1 + 1e-10 and -1e-10 <= u <= 1 + 1e-10
            else:
                aligned = abs(cross(r, a)) <= scale * lengths[i] * 1e-10
                t = np.dot(r, a) / lengths[i] ** 2
                u = t + np.dot(b, a) / lengths[i] ** 2
                intersects = aligned and max(min(t, u), 0) <= min(max(t, u), 1) + 1e-10
            if intersects:
                raise ValueError(f"Self-intersecting path at segments {i}, {j}; supply another CV or separate pathways")
    return vertices, delta, lengths


def project_arclength(points, vertices, *, ambiguity_distance=0.01,
                      branch_separation=None, chunk_size=2048):
    """Return s, transverse distances, segment IDs and remote-branch ambiguity.

    All coordinates must use the same, explicitly chosen metric/units.
    Nonmonotonic CV1 is supported. A nearby *arclength-remote* second solution
    is flagged even if it is not an exact intersection. Ties are deterministic
    but never treated as evidence of an unambiguous reaction coordinate.
    """
    vertices, delta, lengths = validate_path(vertices)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Samples must be a finite (N, 2) array")
    if ambiguity_distance < 0 or chunk_size < 1:
        raise ValueError("Invalid projection tolerance or chunk size")
    cumulative = np.r_[0., np.cumsum(lengths)]
    if branch_separation is None:
        branch_separation = min(3 * np.median(lengths), cumulative[-1] / 4)
    if branch_separation <= 0:
        raise ValueError("Branch separation must be positive")
    output = {key: [] for key in ("s", "distance", "segment", "ambiguous")}
    for start in range(0, len(points), chunk_size):
        q = points[start:start + chunk_size]
        t = np.einsum("nkd,kd->nk", q[:, None] - vertices[:-1], delta) / lengths**2
        clipped = np.clip(t, 0, 1)
        clipped[:, 0] = np.minimum(t[:, 0], 1)
        clipped[:, -1] = np.maximum(t[:, -1], 0)
        if len(delta) == 1:
            clipped = t
        closest = vertices[:-1] + clipped[:, :, None] * delta
        distance = np.linalg.norm(q[:, None] - closest, axis=2)
        candidate_s = cumulative[:-1] + clipped * lengths
        segment = np.argmin(distance, axis=1)
        row = np.arange(len(q))
        s, d = candidate_s[row, segment], distance[row, segment]
        remote = np.abs(candidate_s - s[:, None]) > branch_separation
        ambiguous = np.any(remote & (distance <= d[:, None] + ambiguity_distance), axis=1)
        for key, value in zip(output, (s, d, segment, ambiguous)):
            output[key].append(value)
    result = {key: np.concatenate(value) for key, value in output.items()}
    result["vertex_s"] = cumulative
    return result
