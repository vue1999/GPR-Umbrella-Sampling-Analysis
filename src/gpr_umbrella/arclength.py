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


def project_polyline(points, vertices, *, ambiguity_distance=0.01,
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


def project_arclength(points, vertices, *, ambiguity_distance=0.01,
                      branch_separation=None, chunk_size=2048,
                      smoothing_width=None, method="soft"):
    """Smooth path progress in physical-arclength units.

    The coordinate is the Gaussian-weighted mean of s over the *continuous*
    reference polyline, with analytically integrated infinite endpoint rays:
    s(q) = integral s exp(-|q-r(s)|²/(2h²)) ds / integral exp(...) ds.

    This is a continuous-path analogue of a soft path collective variable.
    It equals geometric projection on a straight line, but has no finite-area
    point masses at polygon corners. It is not exactly nearest-point arclength
    on a bent path; h is an explicit, geometry-based coordinate definition.
    The original 2D energies must still be used for unbiasing. Normal distance
    remains distance to the geometric path. Window IDs never enter the map.
    """
    from scipy.special import log_ndtr, logsumexp
    hard = project_polyline(points, vertices, ambiguity_distance=ambiguity_distance,
                            branch_separation=branch_separation, chunk_size=chunk_size)
    if method == "polyline":
        hard["projection_method"] = "polyline"
        hard["smoothing_width"] = 0.
        return hard
    if method != "soft":
        raise ValueError("Projection method must be 'soft' or 'polyline'")
    vertices = np.asarray(vertices, float)
    points = np.asarray(points, float)
    delta = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    h = float(.5 * np.median(lengths) if smoothing_width is None else smoothing_width)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Smoothing width must be positive")
    tangents = delta / lengths[:, None]
    starts = hard["vertex_s"][:-1]
    soft_s = np.empty(len(points)); remote_mass = np.empty(len(points))
    separation = max(4*h, branch_separation if branch_separation is not None else 3*np.median(lengths))
    for begin in range(0, len(points), chunk_size):
        q = points[begin:begin+chunk_size]
        relative = q[:, None] - vertices[:-1]
        t = np.einsum("nkd,kd->nk", relative, tangents)
        normal2 = np.maximum(np.sum(relative**2, axis=2) - t**2, 0)
        alpha = -t / h
        beta = (lengths - t) / h
        alpha[:, 0] = -np.inf
        beta[:, -1] = np.inf
        # Stable Gaussian interval mass: use survival probabilities in the
        # positive tail to avoid subtracting two values rounded to one.
        flip = alpha > 0
        upper = np.where(flip, -alpha, beta)
        lower = np.where(flip, -beta, alpha)
        log_upper, log_lower = log_ndtr(upper), log_ndtr(lower)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_z = log_upper + np.log(-np.expm1(log_lower-log_upper))
        log_phi_a = -.5*alpha**2 - .5*np.log(2*np.pi)
        log_phi_b = -.5*beta**2 - .5*np.log(2*np.pi)
        mean_local = t + h*(np.exp(log_phi_a-log_z) - np.exp(log_phi_b-log_z))
        means = starts + mean_local
        log_mass = -.5*normal2/h**2 + log_z
        weights = np.exp(log_mass-logsumexp(log_mass, axis=1)[:, None])
        soft_s[begin:begin+len(q)] = np.sum(weights*means, axis=1)
        remote = np.abs(means-hard["s"][begin:begin+len(q), None]) > separation
        remote_mass[begin:begin+len(q)] = np.sum(weights*remote, axis=1)
    if not np.all(np.isfinite(soft_s)):
        raise ValueError("Non-finite soft projection; inspect path scale and smoothing width")
    return {**hard, "hard_s": hard["s"], "s": soft_s,
            "ambiguous": hard["ambiguous"] | (remote_mass > .05),
            "remote_branch_weight": remote_mass,
            "projection_method": "soft", "smoothing_width": h}
