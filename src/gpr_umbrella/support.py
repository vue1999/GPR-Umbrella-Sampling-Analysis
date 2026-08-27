"""Sampled-support geometry shared by 2D reconstruction and pathways."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
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


def values_at_window_means(results: dict, field: np.ndarray) -> np.ndarray:
    """Interpolate a gridded field at the GP observation locations."""
    interpolator = RegularGridInterpolator(
        (results["gx"], results["gy"]), np.asarray(field, dtype=float),
        bounds_error=False, fill_value=np.nan,
    )
    points = np.asarray(results["means"], dtype=float).copy()
    points[:, 0] = np.clip(points[:, 0], results["gx"][0], results["gx"][-1])
    points[:, 1] = np.clip(points[:, 1], results["gy"][0], results["gy"][-1])
    values = np.asarray(interpolator(points), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite grid values at the sampled window means")
    return values


def window_anchored_display_policy(results: dict) -> dict:
    """Return observation-anchored PMF and uncertainty display limits.

    The PMF interval contains the complete range at sampled window means plus
    a fixed 25 percent margin. Uncertainty intervals contain all values at the
    window means plus the same margin. These limits keep extrapolative edge
    excursions from flattening meaningful detail in the sampled region.
    """
    pmf_at_means = values_at_window_means(results, results["pmf"])
    reference = float(np.min(pmf_at_means))
    span = float(np.ptp(pmf_at_means))

    calibrated_at_means = values_at_window_means(
        results, results["pmf_std_calibrated"]
    )
    typical_sigma = float(np.median(calibrated_at_means))
    padding = (
        0.25 * span if span > 1e-12 else max(2.0 * typical_sigma, 1e-9)
    )

    uncertainty_limits = {}
    for key in ("pmf_std_raw", "pmf_std_calibrated"):
        at_means = values_at_window_means(results, results[key])
        uncertainty_limits[key] = (
            0.0, max(1.25 * float(np.max(at_means)), 1e-9)
        )

    return {
        "pmf_reference": reference,
        "pmf": np.asarray(results["pmf"], dtype=float) - reference,
        "pmf_limits": (-padding, span + padding),
        "uncertainty_limits": uncertainty_limits,
    }


def path_valid_mask(results: dict, display_policy: dict | None = None) -> np.ndarray:
    """Return finite, geometrically supported, non-warning PMF cells."""
    display = (
        window_anchored_display_policy(results)
        if display_policy is None else display_policy
    )
    lower, upper = display["pmf_limits"]
    tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
    pmf = np.asarray(display["pmf"], dtype=float)
    support = np.asarray(results["support_mask"], dtype=bool)
    if pmf.shape != support.shape:
        raise ValueError("PMF and support mask must have matching shapes")
    return (
        support
        & np.isfinite(pmf)
        & (pmf >= lower - tolerance)
        & (pmf <= upper + tolerance)
    )
