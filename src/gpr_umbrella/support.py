"""Sampled-support geometry shared by 2D reconstruction and pathways."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def sampled_support_mask(
    points: np.ndarray,
    window_means: np.ndarray,
    lengthscale: float | np.ndarray,
    radius: float = 0.5,
) -> np.ndarray:
    """Return the union of kernel-scaled balls around sampled window means.

    A point is supported when its distance to at least one sampled mean is no
    larger than ``radius`` after dividing each coordinate by the corresponding
    GP lengthscale.  In two dimensions this produces circles for an isotropic
    kernel and axis-aligned ellipses for an anisotropic kernel, with semiaxes
    ``radius * lengthscale``.

    The default radius is half a GP lengthscale. The definition is local: it
    neither fills a convex hull nor bridges gaps between disconnected groups
    of windows.
    """
    points = np.asarray(points, dtype=float)
    window_means = np.asarray(window_means, dtype=float)
    if points.ndim != 2 or window_means.ndim != 2:
        raise ValueError("points and window_means must be two-dimensional arrays")
    if points.shape[1] != window_means.shape[1]:
        raise ValueError("points and window_means must have the same dimension")
    if len(window_means) == 0:
        raise ValueError("At least one sampled window mean is required")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(window_means)):
        raise ValueError("points and window_means must contain only finite values")

    try:
        lengthscale = np.broadcast_to(
            np.asarray(lengthscale, dtype=float), (points.shape[1],)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lengthscale must be scalar or match the point dimension"
        ) from exc
    if not np.all(np.isfinite(lengthscale)) or np.any(lengthscale <= 0):
        raise ValueError("lengthscale must contain finite positive values")
    if isinstance(radius, (bool, np.bool_)):
        raise ValueError("support radius must be finite and positive")
    try:
        radius = float(radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("support radius must be finite and positive") from exc
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("support radius must be finite and positive")

    tree = cKDTree(window_means / lengthscale)
    nearest_distance, _ = tree.query(points / lengthscale, k=1)
    return nearest_distance <= float(radius)
