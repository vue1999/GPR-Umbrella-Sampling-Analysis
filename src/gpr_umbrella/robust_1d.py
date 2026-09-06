"""Resolution-aware SE fitting using the package's existing covariance kernels.

Supports linear observations of F, including finite free-energy differences.
No additional fitted white-noise term can conceal sampling/model failures.
Hyperparameter grids are explicit and saved, not a single local optimiser.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .integration_1d import k_base, k_f_fprime


def covariance_reference(cov, reference=0):
    cov = np.asarray(cov)
    return cov - cov[:, [reference]] - cov[[reference], :] + cov[reference, reference]


def fit_linear_observations(
    nodes,
    operator,
    values,
    noise_covariance,
    grid,
    *,
    resolution=None,
    lengthscales=None,
    amplitudes=None,
):
    """Fit y=A F(nodes) with correlated observation errors.

    Integrating a derivative across an interval is represented by a difference
    of endpoint values, avoiding the false assumption that a finite difference
    is an exact point derivative. A discrete log-uniform SE hyperparameter
    mixture propagates both conditional and between-hyperparameter variance.
    Its bounds are part of the prior and are returned for sensitivity checks.
    """
    nodes, values, grid = [np.asarray(a, dtype=float) for a in (nodes, values, grid)]
    operator, noise = [np.asarray(a, dtype=float) for a in (operator, noise_covariance)]
    n = values.size
    if (
        nodes.ndim != 1
        or values.ndim != 1
        or grid.ndim != 1
        or len(nodes) < 3
        or len(grid) < 2
        or n < 2
        or np.any(np.diff(nodes) <= 0)
        or np.any(np.diff(grid) <= 0)
        or operator.shape != (n, len(nodes))
        or noise.shape != (n, n)
        or not all(
            np.all(np.isfinite(a)) for a in (nodes, values, grid, operator, noise)
        )
    ):
        raise ValueError("Invalid GP observations, covariance, or ordered coordinates")
    if not np.allclose(noise, noise.T) or np.linalg.eigvalsh(
        noise
    ).min() < -1e-10 * np.max(np.diag(noise)):
        raise ValueError(
            "Observation covariance must be symmetric positive semidefinite"
        )
    if np.any(np.diag(noise) <= 0):
        raise ValueError("Observation variances must be positive")
    span = np.ptp(nodes)
    resolution = float(
        resolution if resolution is not None else np.median(np.diff(nodes))
    )
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be positive")
    if lengthscales is None:
        lengthscales = np.geomspace(resolution, max(3 * span, 2 * resolution), 35)
    if amplitudes is None:
        # This API's typical operator is a secant slope (units inverse length).
        # Least-squares reconstruction also supports dimensionless differences.
        signal = np.ptp(np.linalg.lstsq(operator, values, rcond=None)[0])
        scale = max(
            signal,
            np.sqrt(np.mean(np.diag(noise))) / np.linalg.norm(operator, axis=1).mean(),
        )
        amplitudes = np.geomspace(scale / 30, scale * 30, 25)
    lengthscales, amplitudes = [
        np.asarray(v, dtype=float) for v in (lengthscales, amplitudes)
    ]
    if any(
        v.ndim != 1 or len(v) == 0 or np.any(~np.isfinite(v)) or np.any(v <= 0)
        for v in (lengthscales, amplitudes)
    ):
        raise ValueError("Hyperparameters must be finite positive arrays")
    if np.min(lengthscales) < resolution * (1 - 1e-10):
        raise ValueError("Lengthscale grid includes unresolved sub-spacing values")
    # Equal weights require a regular log grid (singleton means fixed).
    for axis in (lengthscales, amplitudes):
        if np.any(np.diff(axis) <= 0):
            raise ValueError("Hyperparameter grids must be strictly increasing")
        if len(axis) > 2 and not np.allclose(
            np.diff(np.log(axis)), np.diff(np.log(axis))[0]
        ):
            raise ValueError("Use log-uniform hyperparameter grids")
    records = []
    jitter = np.mean(np.diag(noise)) * 1e-9
    for ell in lengthscales:
        base = operator @ k_base(nodes, nodes, 1.0, ell) @ operator.T
        for amplitude in amplitudes:
            kernel = amplitude**2 * base
            ky = kernel + noise + np.eye(n) * jitter
            try:
                factor = cho_factor(ky, lower=True)
                inv = cho_solve(factor, np.eye(n))
            except np.linalg.LinAlgError:
                continue
            alpha = inv @ values
            nll = 0.5 * (
                values @ alpha
                + 2 * np.log(np.diag(factor[0])).sum()
                + n * np.log(2 * np.pi)
            )
            loo_var = 1 / np.diag(inv)
            loo_z = alpha * np.sqrt(loo_var)
            records.append(
                (
                    float(ell),
                    float(amplitude),
                    float(nll),
                    float(np.sqrt(np.mean(loo_z**2))),
                    float(np.max(np.abs(loo_z))),
                )
            )
    if not records:
        raise ValueError("No numerically valid GP candidate")
    table = np.array(records)
    log_weight = -table[:, 2]
    weights = np.exp(log_weight - logsumexp(log_weight))
    best = int(np.argmax(weights))
    # Retain effectively all mass; avoid hundreds of negligible dense matrices.
    selected = np.flatnonzero(weights > weights.max() * 1e-5)
    weights_retained = weights[selected] / weights[selected].sum()
    mean = np.zeros(len(grid))
    second = np.zeros((len(grid), len(grid)))
    dmean = np.zeros(len(grid))
    dsecond = np.zeros(len(grid))
    components = []
    best_result = None
    for idx, weight in zip(selected, weights_retained):
        ell, amplitude = table[idx, :2]
        kernel = operator @ k_base(nodes, nodes, amplitude, ell) @ operator.T
        ky = kernel + noise + np.eye(n) * jitter
        factor = cho_factor(ky, lower=True)
        inv = cho_solve(factor, np.eye(n))
        alpha = inv @ values
        cross = k_base(grid, nodes, amplitude, ell) @ operator.T
        fm = cross @ alpha
        fc = k_base(grid, grid, amplitude, ell) - cross @ cho_solve(factor, cross.T)
        fm -= fm[0]
        fc = covariance_reference(fc)
        dcross = -k_f_fprime(grid, nodes, amplitude, ell) @ operator.T
        dm = dcross @ alpha
        dv = amplitude**2 / ell**2 - np.einsum(
            "ij,ji->i", dcross, cho_solve(factor, dcross.T)
        )
        mean += weight * fm
        second += weight * (fc + np.outer(fm, fm))
        dmean += weight * dm
        dsecond += weight * (np.maximum(dv, 0) + dm**2)
        components.append(
            {
                "weight": float(weight),
                "mean": fm,
                "covariance": fc,
                "lengthscale": float(ell),
                "sigma_f": float(amplitude),
            }
        )
        if idx == best:
            loo_var = 1 / np.diag(inv)
            best_result = {
                "loo_means": values - alpha * loo_var,
                "loo_stds": np.sqrt(loo_var),
                "loo_z": alpha * np.sqrt(loo_var),
                "training_residuals": values - kernel @ alpha,
                "pmf_std_raw": np.sqrt(np.maximum(np.diag(fc), 0)),
            }
    covariance = second - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    issues = []
    boundary_mass = float(
        weights[
            (table[:, 0] == lengthscales[0]) | (table[:, 0] == lengthscales[-1])
        ].sum()
    )
    amplitude_boundary_mass = float(
        weights[(table[:, 1] == amplitudes[0]) | (table[:, 1] == amplitudes[-1])].sum()
    )
    if len(lengthscales) > 1 and boundary_mass > 0.2:
        issues.append("lengthscale_prior_boundary")
    if len(amplitudes) > 1 and amplitude_boundary_mass > 0.2:
        issues.append("amplitude_prior_boundary")
    if table[best, 3] > 2 or table[best, 4] > 4:
        issues.append("poor_raw_leave_one_out")
    # Joint conditional checks of adjacent held-out intervals, not only LOO.
    ell, amplitude = table[best, :2]
    ky = (
        operator @ k_base(nodes, nodes, amplitude, ell) @ operator.T
        + noise
        + np.eye(n) * jitter
    )
    inv = cho_solve(cho_factor(ky, lower=True), np.eye(n))
    alpha = inv @ values
    blocked = []
    for hold in np.array_split(np.arange(n), min(8, max(2, n // 3))):
        precision = inv[np.ix_(hold, hold)]
        residual = np.linalg.solve(precision, alpha[hold])
        blocked.append(float(np.sqrt(residual @ precision @ residual / len(hold))))
    if max(blocked) > 3:
        issues.append("poor_blocked_cross_validation")
    return {
        "x_star": grid,
        "observed_support": [float(nodes[0]), float(nodes[-1])],
        "pmf_mean": mean,
        "pmf_covariance": covariance,
        "pmf_std": np.sqrt(np.maximum(np.diag(covariance), 0)),
        "deriv_mean": dmean,
        "deriv_std": np.sqrt(np.maximum(dsecond - dmean**2, 0)),
        "sigma_f": float(table[best, 1]),
        "lengthscale": float(table[best, 0]),
        "hyperparameter_table": np.column_stack([table, weights]),
        "hyperparameter_columns": [
            "lengthscale",
            "sigma_f",
            "nll",
            "loo_rms",
            "loo_max_abs",
            "weight",
        ],
        "components": components,
        "quality_issues": issues,
        "blocked_cv_rms": blocked,
        "lengthscale_boundary_mass": boundary_mass,
        "amplitude_boundary_mass": amplitude_boundary_mass,
        "retained_hyperparameter_mass": float(weights[selected].sum()),
        **best_result,
    }


def barrier_posterior(
    result, reactant_interval, transition_interval, *, draws=1000, seed=2026
):
    """Sample extrema in explicit physical basins, including their selection.

    Hyperparameters are drawn from the discrete mixture, then full correlated
    GP profiles are sampled. Never add pointwise standard errors in quadrature.
    """
    x = result["x_star"]
    if not isinstance(draws, (int, np.integer)) or draws < 2:
        raise ValueError("At least two integer posterior draws are required")
    observed_lo, observed_hi = result["observed_support"]
    if any(
        lo < observed_lo or hi > observed_hi
        for lo, hi in (reactant_interval, transition_interval)
    ):
        raise ValueError(
            "Barrier intervals must lie inside observed support, not GP extrapolation"
        )
    masks = [
        (x >= lo) & (x <= hi) for lo, hi in (reactant_interval, transition_interval)
    ]
    if (
        any(not m.any() for m in masks)
        or reactant_interval[1] >= transition_interval[0]
    ):
        raise ValueError(
            "Provide non-overlapping ordered reactant and transition intervals on the grid"
        )
    if any(
        lo < x[0] or hi > x[-1] or lo >= hi
        for lo, hi in (reactant_interval, transition_interval)
    ):
        raise ValueError("Barrier intervals must lie inside sampled profile support")
    rng = np.random.default_rng(seed)
    components = result["components"]
    allocation = rng.multinomial(draws, [c["weight"] for c in components])
    selected = np.flatnonzero(masks[0] | masks[1])
    rmask, tmask = masks[0][selected], masks[1][selected]
    barriers = []
    component_barriers = []
    reactant_edges = []
    transition_edges = []
    for c, count in zip(components, allocation):
        component_barriers.append(
            float(c["mean"][masks[1]].max() - c["mean"][masks[0]].min())
        )
        if count == 0:
            continue
        cov = c["covariance"][np.ix_(selected, selected)]
        eig, vec = np.linalg.eigh((cov + cov.T) / 2)
        samples = (
            c["mean"][selected]
            + rng.normal(size=(count, len(eig))) @ (vec * np.sqrt(np.maximum(eig, 0))).T
        )
        reactant_samples, transition_samples = samples[:, rmask], samples[:, tmask]
        ri, ti = (
            np.argmin(reactant_samples, axis=1),
            np.argmax(transition_samples, axis=1),
        )
        reactant_edges.extend((ri == 0) | (ri == reactant_samples.shape[1] - 1))
        transition_edges.extend((ti == 0) | (ti == transition_samples.shape[1] - 1))
        barriers.extend(transition_samples.max(axis=1) - reactant_samples.min(axis=1))
    barriers = np.asarray(barriers)
    return {
        "mean": float(barriers.mean()),
        "std": float(barriers.std(ddof=1)),
        "median": float(np.median(barriers)),
        "ci95": np.quantile(barriers, [0.025, 0.975]).tolist(),
        "posterior_mean_profile_barrier": float(
            result["pmf_mean"][masks[1]].max() - result["pmf_mean"][masks[0]].min()
        ),
        "hyperparameter_barrier_range": [
            min(component_barriers),
            max(component_barriers),
        ],
        "reactant_interval": list(reactant_interval),
        "transition_interval": list(transition_interval),
        "reactant_extremum_boundary_probability": float(np.mean(reactant_edges)),
        "transition_extremum_boundary_probability": float(np.mean(transition_edges)),
        "draws": draws,
        "seed": seed,
    }


def profile_landmarks(result, *, prominence=0.02):
    """Exploratory minima/maximum pairs; not automatic physical basin labels.

    Errors here condition on the displayed extrema. Use barrier_posterior with
    physical basin intervals to include extrema selection uncertainty.
    """
    from scipy.signal import find_peaks

    x, f, cov = result["x_star"], result["pmf_mean"], result["pmf_covariance"]
    minima = find_peaks(-f, prominence=prominence)[0]
    maxima = find_peaks(f, prominence=prominence)[0]
    pairs = []
    for ts in maxima:
        preceding = minima[minima < ts]
        reference = int(preceding[-1]) if len(preceding) else 0
        variance = cov[ts, ts] + cov[reference, reference] - 2 * cov[ts, reference]
        pairs.append(
            {
                "reference_s": float(x[reference]),
                "maximum_s": float(x[ts]),
                "rise": float(f[ts] - f[reference]),
                "conditional_sigma": float(np.sqrt(max(variance, 0))),
                "reference_at_support_boundary": reference == 0,
                "interpretation": "exploratory profile feature; physical basins and extrema uncertainty not established",
            }
        )
    maximum = int(np.argmax(f))
    return {
        "minima_s": x[minima].tolist(),
        "maxima_s": x[maxima].tolist(),
        "candidate_rises": pairs,
        "prominence": prominence,
        "forward_rise_from_first_supported_bin": float(f[maximum] - f[0]),
        "forward_rise_conditional_sigma": float(
            np.sqrt(max(cov[maximum, maximum] + cov[0, 0] - 2 * cov[maximum, 0], 0))
        ),
        "full_profile_range": float(np.ptp(f)),
        "note": "Forward rise and full range are not necessarily an activation barrier",
    }
