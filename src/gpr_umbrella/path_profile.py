"""Evaluation and covariance-aware errors for arbitrary two-dimensional paths."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve


def _posterior_mean_and_variance(results: dict, points: np.ndarray):
    """Evaluate the latent GP at arbitrary points without a dense covariance."""
    from .integration_2d import _k_f_grad

    state = results.get("_gp_state")
    if not isinstance(state, dict):
        raise ValueError("GP state is required for arbitrary-point path evaluation")
    cross = _k_f_grad(
        points, state["X"], state["sigma_f"], state["lengthscale"]
    )
    solved = cho_solve((state["cho_factor"], state["lower"]), cross.T)
    mean = cross @ state["alpha"] - float(state["pmf_reference_mean"])
    variance = state["sigma_f"] ** 2 - np.einsum(
        "ij,ji->i", cross, solved
    )
    return mean, np.clip(variance, 0.0, np.inf)


def relative_uncertainty(
    results: dict,
    points: np.ndarray,
    latent_variance: np.ndarray,
    reference: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, calibrated, and configured sigma of F-F(reference)."""
    from .integration_2d import posterior_covariance_2d

    covariance = posterior_covariance_2d(
        results, points, points[[reference]], calibrated=False
    ).ravel()
    variance = np.clip(
        latent_variance + latent_variance[reference] - 2.0 * covariance,
        0.0,
        np.inf,
    )
    variance[reference] = 0.0
    raw = np.sqrt(variance)
    calibrated = raw * results["loo_calibration_factor"]
    default = calibrated if results["default_uncertainty"] == "calibrated" else raw
    return raw, calibrated, default


def evaluate_path_profile(
    results: dict,
    points: np.ndarray,
    metric_scale: np.ndarray,
    *,
    energy: np.ndarray | None = None,
    latent_variance: np.ndarray | None = None,
) -> dict:
    """Return energies, landmarks, and correlated errors for one path."""
    points = np.asarray(points, dtype=float)
    metric_scale = np.asarray(metric_scale, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("A path must have shape (n_points >= 2, 2)")
    if energy is None or latent_variance is None:
        predicted_energy, predicted_variance = _posterior_mean_and_variance(
            results, points
        )
        energy = predicted_energy if energy is None else energy
        latent_variance = (
            predicted_variance if latent_variance is None else latent_variance
        )
    energy = np.asarray(energy, dtype=float)
    latent_variance = np.asarray(latent_variance, dtype=float)
    if energy.shape != (len(points),) or latent_variance.shape != (len(points),):
        raise ValueError("energy and latent_variance must match the path length")

    segment = np.linalg.norm(np.diff(points, axis=0) / metric_scale, axis=1)
    if np.any(segment <= 0.0):
        raise ValueError("A path cannot contain consecutive duplicate points")
    s = np.r_[0.0, np.cumsum(segment)]
    minimum = int(np.argmin(energy))
    maximum = int(np.argmax(energy))
    relative_start = energy - energy[0]
    relative_minimum = energy - energy[minimum]
    start_raw, start_calibrated, start_default = relative_uncertainty(
        results, points, latent_variance, 0
    )
    min_raw, min_calibrated, min_default = relative_uncertainty(
        results, points, latent_variance, minimum
    )

    return {
        "s": s,
        "x": points[:, 0],
        "y": points[:, 1],
        "pmf": energy,
        "pmf_rel": relative_start,
        "pmf_rel_path_min": relative_minimum,
        "sigma_raw": start_raw,
        "sigma_calibrated": start_calibrated,
        "sigma": start_default,
        "sigma_from_path_min_raw": min_raw,
        "sigma_from_path_min_calibrated": min_calibrated,
        "sigma_from_path_min": min_default,
        "start_xy": tuple(points[0]),
        "end_xy": tuple(points[-1]),
        "ts_xy": tuple(points[maximum]),
        "ts_s": float(s[maximum]),
        "path_min_index": minimum,
        "path_min_xy": tuple(points[minimum]),
        "path_min_s": float(s[minimum]),
        "path_min_pmf": float(energy[minimum]),
        "bottleneck_energy": float(energy[maximum]),
        "path_metric_length": float(s[-1]),
        "barrier": float(relative_minimum[maximum]),
        "barrier_err_raw": float(min_raw[maximum]),
        "barrier_err_calibrated": float(min_calibrated[maximum]),
        "barrier_err": float(min_default[maximum]),
        "delta_f": float(relative_start[-1]),
        "delta_f_err_raw": float(start_raw[-1]),
        "delta_f_err_calibrated": float(start_calibrated[-1]),
        "delta_f_err": float(start_default[-1]),
    }
