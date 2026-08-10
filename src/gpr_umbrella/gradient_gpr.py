"""Gaussian-process reconstruction from free-energy-gradient observations.

This module contains the bias-independent reconstruction layer shared by
umbrella integration and instantaneous-collective-force (ICF) estimators.  It
conditions a Gaussian process with an anisotropic squared-exponential kernel
on vector gradient observations and predicts a scalar free-energy surface up
to an additive constant.

The observation order is window/point major: for ``N`` locations and ``D``
collective variables, gradients are flattened as
``[g_0,0, ..., g_0,D-1, g_1,0, ...]``.  The supplied ``(N*D, N*D)`` noise
covariance must use the same order.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


__all__ = [
    "fit_gradient_gp",
    "fit_icf_gp",
    "posterior_covariance_gradient_gp",
    "predict_gradient_gp",
]


def _as_observations(
    points: np.ndarray,
    gradients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize gradient observations to ``(N, D)`` arrays."""
    points = np.asarray(points, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if gradients.ndim == 1:
        gradients = gradients[:, None]
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, n_CVs)")
    if gradients.shape != points.shape:
        raise ValueError("gradients must have the same shape as points")
    if points.shape[0] < 2:
        raise ValueError("At least two gradient-observation points are required")
    if points.shape[1] < 1:
        raise ValueError("At least one collective variable is required")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("gradients must contain only finite values")
    span = np.ptp(points, axis=0)
    if np.any(span <= 0.0):
        raise ValueError("Gradient-observation points must span every CV coordinate")
    return points, gradients


def _as_noise_covariance(
    covariance: np.ndarray,
    size: int,
) -> np.ndarray:
    """Validate a full gradient-observation noise covariance matrix."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError(
            f"gradient_noise_cov must have shape ({size}, {size})"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("gradient_noise_cov must contain only finite values")
    scale = max(float(np.max(np.abs(covariance))), 1.0)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("gradient_noise_cov must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-10 * scale
    if eigenvalues[0] < -tolerance:
        raise ValueError("gradient_noise_cov must be positive semidefinite")
    if eigenvalues[0] < 0.0:
        # Remove only roundoff-sized negative modes.  This also gives Cholesky
        # a deterministic input on platforms with slightly different LAPACK.
        values, vectors = np.linalg.eigh(covariance)
        covariance = (vectors * np.clip(values, 0.0, np.inf)) @ vectors.T
        covariance = 0.5 * (covariance + covariance.T)
    return covariance


def _as_lengthscale(
    value: float | tuple[float, ...] | np.ndarray,
    dimensions: int,
    name: str,
) -> np.ndarray:
    """Broadcast and validate one positive lengthscale per CV."""
    try:
        lengthscale = np.broadcast_to(
            np.asarray(value, dtype=float), (dimensions,)
        ).copy()
    except ValueError as exc:
        raise ValueError(
            f"{name} must be scalar or have shape ({dimensions},)"
        ) from exc
    if np.any(~np.isfinite(lengthscale)) or np.any(lengthscale <= 0.0):
        raise ValueError(f"{name} values must be finite and positive")
    return lengthscale


def _as_query_points(points: np.ndarray, dimensions: int) -> np.ndarray:
    """Normalize query coordinates while keeping 1D calls convenient."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 0 and dimensions == 1:
        points = points.reshape(1, 1)
    elif points.ndim == 1:
        if dimensions == 1:
            points = points[:, None]
        elif points.shape == (dimensions,):
            points = points[None, :]
    if points.ndim != 2 or points.shape[1] != dimensions:
        raise ValueError(f"query points must have shape (n_points, {dimensions})")
    if not np.all(np.isfinite(points)):
        raise ValueError("query points must contain only finite values")
    return points


def _se(
    points_a: np.ndarray,
    points_b: np.ndarray,
    sigma_f: float,
    lengthscale: np.ndarray,
) -> np.ndarray:
    """Anisotropic squared-exponential covariance between point sets."""
    difference = points_a[:, None, :] - points_b[None, :, :]
    squared_distance = np.sum((difference / lengthscale) ** 2, axis=2)
    return sigma_f**2 * np.exp(-0.5 * squared_distance)


def _k_f_grad(
    prediction_points: np.ndarray,
    observation_points: np.ndarray,
    sigma_f: float,
    lengthscale: np.ndarray,
) -> np.ndarray:
    """Covariance of scalar values with gradients at observation points."""
    n_prediction = prediction_points.shape[0]
    n_observation, dimensions = observation_points.shape
    difference = (
        prediction_points[:, None, :] - observation_points[None, :, :]
    )
    base = _se(
        prediction_points, observation_points, sigma_f, lengthscale
    )
    covariance = difference / lengthscale**2 * base[:, :, None]
    return covariance.reshape(n_prediction, n_observation * dimensions)


def _k_grad_grad(
    points_a: np.ndarray,
    points_b: np.ndarray,
    sigma_f: float,
    lengthscale: np.ndarray,
) -> np.ndarray:
    """Covariance between gradient components at two point sets."""
    n_a, dimensions = points_a.shape
    n_b = points_b.shape[0]
    difference = points_a[:, None, :] - points_b[None, :, :]
    base = _se(points_a, points_b, sigma_f, lengthscale)
    inverse_squared = 1.0 / lengthscale**2

    diagonal = np.zeros((n_a, n_b, dimensions, dimensions))
    indices = np.arange(dimensions)
    diagonal[:, :, indices, indices] = inverse_squared[None, None, :]
    left = (difference * inverse_squared)[:, :, :, None]
    right = (difference * inverse_squared)[:, :, None, :]
    blocks = (diagonal - left * right) * base[:, :, None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(
        n_a * dimensions, n_b * dimensions
    )


def _with_relative_diagonal_jitter(
    covariance: np.ndarray,
    relative: float = 1e-8,
) -> np.ndarray:
    """Return a copy with component-wise, unit-covariant diagonal jitter."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    diagonal = np.diag(covariance)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("covariance must have a finite positive diagonal")
    result = covariance.copy()
    result[np.diag_indices_from(result)] += relative * diagonal
    return result


def _negative_log_marginal_likelihood(
    sigma_f: float,
    lengthscale: np.ndarray,
    points: np.ndarray,
    observations: np.ndarray,
    noise_covariance: np.ndarray,
) -> float:
    """Negative log marginal likelihood for gradient observations."""
    if sigma_f <= 0.0 or np.any(lengthscale <= 0.0):
        return 1e12
    covariance = _k_grad_grad(
        points, points, sigma_f, lengthscale
    ) + noise_covariance
    covariance = _with_relative_diagonal_jitter(covariance)
    try:
        factor, lower = cho_factor(covariance, lower=True)
        alpha = cho_solve((factor, lower), observations)
    except np.linalg.LinAlgError:
        return 1e12
    return float(
        0.5 * observations @ alpha
        + np.sum(np.log(np.diag(factor)))
        + 0.5 * observations.size * np.log(2.0 * np.pi)
    )


def _fit_hyperparameters(
    points: np.ndarray,
    observations: np.ndarray,
    noise_covariance: np.ndarray,
    lengthscale_initial: np.ndarray,
    lengthscale_upper: np.ndarray,
    sigma_f_initial: float,
    fixed_lengthscale: float | tuple[float, ...] | np.ndarray | None,
    fixed_sigma_f: float | None,
    optimize: bool,
) -> tuple[float, np.ndarray, bool]:
    """Fit or select GP hyperparameters in dimensionless log coordinates."""
    dimensions = points.shape[1]
    lengthscale_initial = _as_lengthscale(
        lengthscale_initial, dimensions, "initial lengthscale"
    )
    lengthscale_upper = _as_lengthscale(
        lengthscale_upper, dimensions, "upper lengthscale"
    )
    if not np.isfinite(sigma_f_initial) or sigma_f_initial <= 0.0:
        raise ValueError("Initial GP signal scale must be finite and positive")

    fixed_ell = None
    if fixed_lengthscale is not None:
        fixed_ell = _as_lengthscale(
            fixed_lengthscale, dimensions, "fixed_lengthscale"
        )
    if fixed_sigma_f is not None:
        fixed_sigma_f = float(fixed_sigma_f)
        if not np.isfinite(fixed_sigma_f) or fixed_sigma_f <= 0.0:
            raise ValueError("fixed_sigma_f must be finite and positive")

    if not optimize or (fixed_sigma_f is not None and fixed_ell is not None):
        return (
            fixed_sigma_f if fixed_sigma_f is not None else sigma_f_initial,
            fixed_ell if fixed_ell is not None else lengthscale_initial,
            True,
        )

    lengthscale_lower = lengthscale_upper / 3000.0
    bounds: list[tuple[float, float]] = []
    if fixed_sigma_f is None:
        bounds.append((np.log(1e-3), np.log(1e3)))
    if fixed_ell is None:
        bounds.extend(
            zip(
                np.log(lengthscale_lower / lengthscale_initial),
                np.log(lengthscale_upper / lengthscale_initial),
            )
        )

    def unpack(log_ratios: np.ndarray) -> tuple[float, np.ndarray]:
        cursor = 0
        if fixed_sigma_f is None:
            sigma_f = sigma_f_initial * np.exp(log_ratios[cursor])
            cursor += 1
        else:
            sigma_f = fixed_sigma_f
        if fixed_ell is None:
            lengthscale = (
                lengthscale_initial
                * np.exp(log_ratios[cursor:cursor + dimensions])
            )
        else:
            lengthscale = fixed_ell
        return float(sigma_f), np.asarray(lengthscale, dtype=float)

    def objective(log_ratios: np.ndarray) -> float:
        sigma_f, lengthscale = unpack(log_ratios)
        return _negative_log_marginal_likelihood(
            sigma_f,
            lengthscale,
            points,
            observations,
            noise_covariance,
        )

    starts = []
    for sigma_scale, lengthscale_scale in (
        (1.0, 1.0),
        (0.5, 0.5),
        (2.0, 2.0),
        (1.0, 0.5),
        (0.5, 2.0),
    ):
        values = []
        if fixed_sigma_f is None:
            values.append(np.log(sigma_scale))
        if fixed_ell is None:
            values.extend([np.log(lengthscale_scale)] * dimensions)
        start = np.asarray(values, dtype=float)
        for index, (lower, upper) in enumerate(bounds):
            start[index] = np.clip(start[index], lower, upper)
        starts.append(start)

    best = None
    best_value = np.inf
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.success and result.fun < best_value:
            best = result.x
            best_value = float(result.fun)
    if best is None:
        return (
            fixed_sigma_f if fixed_sigma_f is not None else sigma_f_initial,
            fixed_ell if fixed_ell is not None else lengthscale_initial,
            False,
        )
    sigma_f, lengthscale = unpack(best)
    return sigma_f, lengthscale, True


def _leave_one_point_out_z(
    precision: np.ndarray,
    observations: np.ndarray,
    dimensions: int,
) -> np.ndarray:
    """Whiten leave-one-location-out vector residuals."""
    weighted = precision @ observations
    z_scores = []
    for start in range(0, observations.size, dimensions):
        block = slice(start, start + dimensions)
        precision_block = 0.5 * (
            precision[block, block] + precision[block, block].T
        )
        block_factor = cho_factor(precision_block, lower=True)
        conditional_covariance = cho_solve(
            block_factor, np.eye(dimensions)
        )
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )
        residual = cho_solve(block_factor, weighted[block])
        whitening_factor = np.linalg.cholesky(conditional_covariance)
        z_scores.extend(np.linalg.solve(whitening_factor, residual))
    return np.asarray(z_scores)


def _gp_state(model: dict) -> dict:
    """Return and minimally validate fitted state from a public model dict."""
    if not isinstance(model, dict) or "_gp_state" not in model:
        raise ValueError("model does not contain fitted gradient-GP state")
    state = model["_gp_state"]
    required = {
        "X", "sigma_f", "lengthscale", "cho_factor", "lower", "alpha"
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("model contains incomplete gradient-GP state")
    return state


def fit_gradient_gp(
    points: np.ndarray,
    gradients: np.ndarray,
    gradient_noise_cov: np.ndarray,
    *,
    optimize_hyperparams: bool = True,
    fixed_lengthscale: float | tuple[float, ...] | np.ndarray | None = None,
    fixed_sigma_f: float | None = None,
    calibrate_uncertainty: bool = True,
    allow_uncertainty_downscaling: bool = False,
) -> dict:
    """Fit a scalar GP to free-energy-gradient observations.

    Parameters
    ----------
    points, gradients : array-like
        Observation locations and values of ``grad F`` with shape ``(N, D)``.
        One-dimensional arrays are accepted as shorthand for ``(N, 1)``.
    gradient_noise_cov : array-like
        Full statistical covariance of the flattened gradient observations,
        with shape ``(N*D, N*D)``.  Correlations between CV components and
        observation locations are retained.
    optimize_hyperparams : bool
        Optimize free GP hyperparameters by marginal likelihood.
    fixed_lengthscale : float or array-like, optional
        Fix one positive lengthscale per CV.  A scalar is broadcast.
    fixed_sigma_f : float, optional
        Fix the positive GP signal scale in energy units.
    calibrate_uncertainty : bool
        Select LOO-scaled uncertainty as the default reported by
        :func:`predict_gradient_gp`.  Raw and calibrated predictions remain
        available regardless of this setting.
    allow_uncertainty_downscaling : bool
        Permit pointwise LOO calibration to make posterior uncertainty smaller
        than the raw GP value.  The safe default caps the applied factor at one
        because coherent or time-correlated observation errors can look
        artificially predictable to leave-one-location-out validation.

    Returns
    -------
    dict
        Fitted hyperparameters, vector LOO diagnostics, the original
        observations, and private state used by prediction helpers.
    """
    points, gradients = _as_observations(points, gradients)
    if not isinstance(allow_uncertainty_downscaling, (bool, np.bool_)):
        raise ValueError("allow_uncertainty_downscaling must be Boolean")
    n_points, dimensions = points.shape
    noise_covariance = _as_noise_covariance(
        gradient_noise_cov, n_points * dimensions
    )
    observations = gradients.reshape(n_points * dimensions)

    span = np.ptp(points, axis=0)
    lengthscale_initial = 0.5 * span
    lengthscale_upper = 3.0 * span
    scaled_gradients = gradients * lengthscale_initial[None, :]
    noise_diagonal = np.diag(noise_covariance).reshape(n_points, dimensions)
    noise_energy_variance = (
        noise_diagonal * lengthscale_initial[None, :] ** 2
    )
    sigma_f_initial = float(np.sqrt(
        np.mean(scaled_gradients**2) + np.mean(noise_energy_variance)
    ))
    if not np.isfinite(sigma_f_initial):
        raise ValueError("Could not derive a finite GP energy scale from the data")
    if sigma_f_initial == 0.0:
        if fixed_sigma_f is None:
            raise ValueError(
                "The gradient observations and their sampling covariance are "
                "exactly zero, so the GP signal scale is unidentifiable; "
                "provide fixed_sigma_f"
            )
        sigma_f_initial = float(fixed_sigma_f)

    sigma_f, lengthscale, fit_ok = _fit_hyperparameters(
        points,
        observations,
        noise_covariance,
        lengthscale_initial,
        lengthscale_upper,
        sigma_f_initial,
        fixed_lengthscale,
        fixed_sigma_f,
        optimize_hyperparams,
    )

    prior_gradient_covariance = _k_grad_grad(
        points, points, sigma_f, lengthscale
    )
    observation_covariance = _with_relative_diagonal_jitter(
        prior_gradient_covariance + noise_covariance
    )
    factor, lower = cho_factor(observation_covariance, lower=True)
    alpha = cho_solve((factor, lower), observations)
    precision = cho_solve(
        (factor, lower), np.eye(n_points * dimensions)
    )
    loo_z = _leave_one_point_out_z(precision, observations, dimensions)
    uncapped_calibration_factor = float(np.std(loo_z, ddof=1))
    if (
        not np.isfinite(uncapped_calibration_factor)
        or uncapped_calibration_factor <= 0.0
    ):
        uncapped_calibration_factor = 1.0
    calibration_factor = (
        uncapped_calibration_factor
        if allow_uncertainty_downscaling
        else max(1.0, uncapped_calibration_factor)
    )

    return {
        "points": points,
        "gradients": gradients,
        "gradient_noise_cov": noise_covariance,
        "n_points": n_points,
        "n_dimensions": dimensions,
        "sigma_f": sigma_f,
        "lengthscale": lengthscale,
        "fit_ok": fit_ok,
        "loo_z": loo_z,
        "loo_calibration_factor": calibration_factor,
        "loo_calibration_factor_uncapped": uncapped_calibration_factor,
        "uncertainty_downscaling_allowed": bool(
            allow_uncertainty_downscaling
        ),
        "default_uncertainty": (
            "calibrated" if calibrate_uncertainty else "raw"
        ),
        "observation_kind": "free_energy_gradient",
        "_gp_state": {
            "X": points,
            "sigma_f": sigma_f,
            "lengthscale": lengthscale,
            "cho_factor": factor,
            "lower": lower,
            "alpha": alpha,
        },
    }


def fit_icf_gp(
    points: np.ndarray,
    collective_forces: np.ndarray,
    force_noise_cov: np.ndarray,
    *,
    force_convention: str,
    optimize_hyperparams: bool = True,
    fixed_lengthscale: float | tuple[float, ...] | np.ndarray | None = None,
    fixed_sigma_f: float | None = None,
    calibrate_uncertainty: bool = True,
    allow_uncertainty_downscaling: bool = False,
) -> dict:
    """Fit the gradient GP to pre-averaged ICF observations.

    ``force_convention`` is required so no sign convention is inferred:

    ``"free_energy_gradient"``
        ``collective_forces`` already contains ``grad F``.
    ``"thermodynamic_force"``
        ``collective_forces`` contains the conventional force ``-grad F`` and
        is negated before fitting.

    This function expects collective-force estimates that already include any
    CV metric/Jacobian correction required by the ICF estimator, together with
    their statistical covariance.  It does not derive collective forces from
    Cartesian atomic forces.

    Pointwise LOO calibration is capped at a factor of one by default.  Set
    ``allow_uncertainty_downscaling=True`` only when the validation units are
    demonstrably independent enough to justify a smaller uncertainty.
    """
    if force_convention == "free_energy_gradient":
        gradients = np.asarray(collective_forces, dtype=float)
    elif force_convention == "thermodynamic_force":
        gradients = -np.asarray(collective_forces, dtype=float)
    else:
        raise ValueError(
            "force_convention must be 'free_energy_gradient' or "
            "'thermodynamic_force'"
        )
    model = fit_gradient_gp(
        points,
        gradients,
        force_noise_cov,
        optimize_hyperparams=optimize_hyperparams,
        fixed_lengthscale=fixed_lengthscale,
        fixed_sigma_f=fixed_sigma_f,
        calibrate_uncertainty=calibrate_uncertainty,
        allow_uncertainty_downscaling=allow_uncertainty_downscaling,
    )
    model["observation_kind"] = "instantaneous_collective_force"
    model["force_convention"] = force_convention
    model["collective_forces"] = np.asarray(
        collective_forces, dtype=float
    ).copy()
    return model


def posterior_covariance_gradient_gp(
    model: dict,
    points_a: np.ndarray,
    points_b: np.ndarray | None = None,
    *,
    calibrated: bool = False,
) -> np.ndarray:
    """Return posterior covariance between arbitrary scalar-F query points."""
    state = _gp_state(model)
    dimensions = state["X"].shape[1]
    points_a = _as_query_points(points_a, dimensions)
    same_points = points_b is None
    points_b = (
        points_a if same_points else _as_query_points(points_b, dimensions)
    )
    covariance_a = _k_f_grad(
        points_a,
        state["X"],
        state["sigma_f"],
        state["lengthscale"],
    )
    covariance_b = _k_f_grad(
        points_b,
        state["X"],
        state["sigma_f"],
        state["lengthscale"],
    )
    solved_b = cho_solve(
        (state["cho_factor"], state["lower"]), covariance_b.T
    )
    posterior = _se(
        points_a,
        points_b,
        state["sigma_f"],
        state["lengthscale"],
    ) - covariance_a @ solved_b
    if same_points:
        posterior = 0.5 * (posterior + posterior.T)
    if calibrated:
        posterior *= model["loo_calibration_factor"] ** 2
    return posterior


def predict_gradient_gp(
    model: dict,
    points: np.ndarray,
    *,
    reference: np.ndarray | float | None = None,
    calibrated: bool | None = None,
    prediction_batch_size: int = 10_000,
) -> dict:
    """Predict the scalar free energy at arbitrary points.

    If ``reference`` is supplied, both the mean and uncertainty describe
    ``F(points) - F(reference)`` and therefore have a defined additive
    constant.  With ``reference=None``, the returned mean is the latent GP mean
    under its zero-mean prior and ``std_raw`` is its pointwise latent standard
    deviation.

    ``calibrated=None`` follows the default selected when fitting.  Raw and
    LOO-scaled standard deviations are always returned.
    """
    state = _gp_state(model)
    dimensions = state["X"].shape[1]
    points = _as_query_points(points, dimensions)
    if (
        isinstance(prediction_batch_size, bool)
        or not isinstance(prediction_batch_size, (int, np.integer))
        or prediction_batch_size < 1
    ):
        raise ValueError("prediction_batch_size must be a positive integer")

    if calibrated is None:
        calibrated = model.get("default_uncertainty") == "calibrated"
    if not isinstance(calibrated, (bool, np.bool_)):
        raise ValueError("calibrated must be True, False, or None")

    n_prediction = points.shape[0]
    mean = np.empty(n_prediction, dtype=float)
    variance = np.empty(n_prediction, dtype=float)
    reference_point = None
    reference_mean = 0.0
    reference_variance = 0.0
    solved_reference = None

    if reference is not None:
        reference_point = _as_query_points(reference, dimensions)
        if reference_point.shape[0] != 1:
            raise ValueError("reference must specify exactly one point")
        reference_covariance = _k_f_grad(
            reference_point,
            state["X"],
            state["sigma_f"],
            state["lengthscale"],
        )
        solved_reference = cho_solve(
            (state["cho_factor"], state["lower"]),
            reference_covariance.T,
        ).ravel()
        reference_mean = (reference_covariance @ state["alpha"]).item()
        reference_variance = (
            state["sigma_f"] ** 2
            - (reference_covariance @ solved_reference).item()
        )

    for start in range(0, n_prediction, prediction_batch_size):
        block = slice(start, min(start + prediction_batch_size, n_prediction))
        cross_covariance = _k_f_grad(
            points[block],
            state["X"],
            state["sigma_f"],
            state["lengthscale"],
        )
        solved = cho_solve(
            (state["cho_factor"], state["lower"]), cross_covariance.T
        )
        mean[block] = cross_covariance @ state["alpha"] - reference_mean
        latent_variance = np.clip(
            state["sigma_f"] ** 2
            - np.einsum("ij,ji->i", cross_covariance, solved),
            0.0,
            np.inf,
        )
        if reference_point is None:
            variance[block] = latent_variance
        else:
            prior_to_reference = _se(
                points[block],
                reference_point,
                state["sigma_f"],
                state["lengthscale"],
            ).ravel()
            posterior_to_reference = (
                prior_to_reference - cross_covariance @ solved_reference
            )
            variance[block] = np.clip(
                latent_variance
                + reference_variance
                - 2.0 * posterior_to_reference,
                0.0,
                np.inf,
            )

    std_raw = np.sqrt(variance)
    calibration_factor = float(model["loo_calibration_factor"])
    std_calibrated = std_raw * calibration_factor
    std = std_calibrated if calibrated else std_raw
    return {
        "points": points,
        "mean": mean,
        "pmf_mean": mean,
        "variance_raw": variance,
        "std_raw": std_raw,
        "std_calibrated": std_calibrated,
        "std": std,
        "pmf_std": std,
        "calibrated": bool(calibrated),
        "reference_point": (
            None if reference_point is None else reference_point[0].copy()
        ),
    }
