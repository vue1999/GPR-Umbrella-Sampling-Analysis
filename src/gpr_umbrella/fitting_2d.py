"""Noise-aware derivative-GP fitting, with data-scaled numerical regularization.

Sampling covariance, optional statistical discrepancy, and numerical jitter
are separate quantities. Parameter/prior scales transform with the CV and
energy units; there are no material-specific energy or length defaults.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def force_scales(y, sampling_cov, dimensions=2):
    g = np.asarray(y).reshape(-1, dimensions)
    variance = np.diag(sampling_cov).reshape(-1, dimensions)
    return np.sqrt(np.mean(g * g + variance, axis=0))


def numerical_jitter(sampling_cov, y, relative=1e-8):
    """Fixed for the fit: never a function of GP amplitude or fitted noise."""
    if not np.isfinite(relative) or relative <= 0:
        raise ValueError("relative jitter must be finite and positive")
    diagonal = np.diag(sampling_cov)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal < 0):
        raise ValueError("Sampling covariance must have finite nonnegative diagonal")
    scales = force_scales(y, sampling_cov)
    # Constant trajectories may have exactly zero sampling variance. A small
    # data-derived fallback still stays independent of fitted hyperparameters.
    baseline = np.maximum(diagonal, np.tile(1e-6 * scales**2, len(y) // 2))
    return relative * baseline


def training_covariance(signal, sampling_cov, y, extra_noise=None):
    covariance = np.array(signal + sampling_cov, copy=True)
    jitter = numerical_jitter(sampling_cov, y)
    eta = np.zeros(2) if extra_noise is None else np.asarray(extra_noise)
    covariance[np.diag_indices_from(covariance)] += jitter + np.tile(eta**2, len(y) // 2)
    return covariance


def objective(log_parameters, X, y, sampling_cov, noise_scale=None):
    """Penalized NLL and analytic derivatives w.r.t. log physical parameters.

    The optional half-normal prior uses one user/data-derived scale per CV.
    A fixed unit-dependent additive likelihood constant is removed solely to
    make optimizer stopping criteria invariant to unit conversions.
    """
    from .integration_2d import _k_grad_grad, _se

    p = np.exp(log_parameters)
    sf, ell = p[0], p[1:3]
    extra = len(p) == 5
    eta = p[3:] if extra else np.zeros(2)
    K = _k_grad_grad(X, X, sf, ell)
    Ky = training_covariance(K, sampling_cov, y, eta)
    try:
        cf = cho_factor(Ky, lower=True)
        alpha = cho_solve(cf, y)
    except np.linalg.LinAlgError:
        return 1e100, np.zeros_like(p)
    scales = force_scales(y, sampling_cov)
    positive = scales > 0
    normalization = (len(X) * np.log(scales[positive]).sum())
    value = .5 * y @ alpha + np.log(np.diag(cf[0])).sum() - normalization
    W = cho_solve(cf, np.eye(len(y))) - np.outer(alpha, alpha)
    r = X[:, None, :] - X[None, :, :]
    inv = ell**-2
    product = (r * inv)[:, :, :, None] * (r * inv)[:, :, None, :]
    diagonal = np.diag(inv)[None, None, :, :]
    k = _se(X, X, sf, ell)
    derivatives = [2 * K]
    for d in range(2):
        term = (diagonal - product) * (r[:, :, d]**2 * inv[d])[:, :, None, None]
        correction = np.zeros((2, 2)); correction[d, d] = -2 * inv[d]
        indicator = np.zeros(2); indicator[d] = 1
        term += correction + 2 * product * (indicator[:, None] + indicator[None, :])
        derivatives.append((k[:, :, None, None] * term).transpose(0, 2, 1, 3).reshape(len(y), len(y)))
    gradient = [.5 * np.sum(W * derivative) for derivative in derivatives]
    if extra:
        prior = (eta / noise_scale)**2
        value += .5 * prior.sum()
        gradient.extend(eta[d]**2 * np.diag(W)[d::2].sum() + prior[d] for d in range(2))
    return float(value), np.asarray(gradient)


def fit_hyperparameters(X, y, sampling_cov, ell_init, ell_upper, sigma_f_init,
                        fixed_lengthscale, fixed_sigma_f, optimize, *,
                        fit_extra_noise=False, extra_noise_scale=None,
                        sigma_f_max=None):
    """Fit free parameters in dimensionless log ratios with six restarts."""
    ell_init = np.broadcast_to(np.asarray(ell_init, float), (2,)).copy()
    ell_upper = np.broadcast_to(np.asarray(ell_upper, float), (2,))
    fixed_ell = None if fixed_lengthscale is None else np.broadcast_to(np.asarray(fixed_lengthscale, float), (2,)).copy()
    for name, values in [('initial lengths', ell_init), ('upper lengths', ell_upper),
                         ('initial amplitude', sigma_f_init), ('fixed lengths', fixed_ell),
                         ('fixed amplitude', fixed_sigma_f), ('amplitude cap', sigma_f_max)]:
        if values is not None and (np.any(~np.isfinite(values)) or np.any(np.asarray(values) <= 0)):
            raise ValueError(name + ' must be finite and positive')
    if sigma_f_max is not None and fixed_sigma_f is not None and fixed_sigma_f > sigma_f_max:
        raise ValueError('fixed_sigma_f exceeds sigma_f_max')
    if fit_extra_noise and not optimize:
        raise ValueError('fit_extra_noise requires optimize_hyperparams=True')
    if extra_noise_scale is not None and not fit_extra_noise:
        raise ValueError('extra_noise_scale requires fit_extra_noise=True')
    scale = force_scales(y, sampling_cov)
    scale = np.where(scale > 0, scale, sigma_f_init / ell_init)
    if extra_noise_scale is not None:
        scale = np.broadcast_to(np.asarray(extra_noise_scale, float), (2,)).copy()
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('extra_noise_scale must be finite and positive')
    sf_upper = sigma_f_init * 1e3
    if sigma_f_max is not None:
        sf_upper = min(sf_upper, sigma_f_max)
    sf_lower = min(sigma_f_init * 1e-3, sf_upper * 1e-3)
    base = np.r_[sigma_f_init, ell_init, scale] if fit_extra_noise else np.r_[sigma_f_init, ell_init]
    lower = np.r_[sf_lower, ell_upper / 3000., scale * 1e-6] if fit_extra_noise else np.r_[sf_lower, ell_upper / 3000.]
    upper = np.r_[sf_upper, ell_upper, scale * 5.] if fit_extra_noise else np.r_[sf_upper, ell_upper]
    free = np.array(([0] if fixed_sigma_f is None else []) + ([1, 2] if fixed_ell is None else []) + ([3, 4] if fit_extra_noise else []), int)
    initial = base.copy()
    if fixed_sigma_f is not None: initial[0] = fixed_sigma_f
    if fixed_ell is not None: initial[1:3] = fixed_ell
    if not optimize or not len(free):
        if sigma_f_max is not None and initial[0] > sigma_f_max:
            raise ValueError('Unoptimized amplitude exceeds sigma_f_max')
        return initial[0], initial[1:3], np.zeros(2), True, dict(trials=[],extra_noise_scale=scale if fit_extra_noise else None,sigma_f_max=sigma_f_max,bound_hits=[])
    trials = []
    starts = [(1., [.08, .2], .01), (.5, [.2, .4], .05),
              (1., [.5, .5], .15), (1., [1., 1.], .3),
              (2., [2., 1.], .6), (1., [1., 2.], 1.)]
    bounds = np.column_stack([np.log(lower / base), np.log(upper / base)])
    for sf_ratio, ell_ratio, noise_ratio in starts:
        p = initial.copy()
        if fixed_sigma_f is None: p[0] *= sf_ratio
        if fixed_ell is None: p[1:3] *= ell_ratio
        if fit_extra_noise: p[3:] *= noise_ratio
        ratios = np.log(p / base)
        ratios[free] = np.clip(ratios[free], bounds[free, 0], bounds[free, 1])
        def fun(v):
            full = ratios.copy(); full[free] = v
            value, grad = objective(np.log(base) + full, X, y, sampling_cov, scale)
            return value, grad[free]
        result = minimize(fun, ratios[free], jac=True, method='L-BFGS-B',
                          bounds=bounds[free], options=dict(maxiter=600, ftol=1e-11, gtol=1e-6, maxls=50))
        final = ratios.copy(); final[free] = result.x
        parameters = base * np.exp(final)
        trials.append(dict(parameters=parameters.tolist(),objective=float(result.fun),
                           success=bool(result.success),iterations=int(result.nit),message=str(result.message)))
    valid = [t for t in trials if t['success'] and np.isfinite(t['objective']) and t['objective'] < 1e90]
    if not valid:
        raise RuntimeError('All six GP optimization starts failed; no initial-guess fit returned')
    best = min(valid, key=lambda t: t['objective']); p = np.array(best['parameters'])
    names = ['sigma_f', 'ell0', 'ell1', 'extra_noise0', 'extra_noise1']
    hits = [names[i] + ('_lower' if p[i] <= lower[i] * 1.001 else '_upper') for i in free
            if p[i] <= lower[i] * 1.001 or p[i] >= upper[i] / 1.001]
    metadata = dict(trials=trials,objective=best['objective'],bounds=np.column_stack([lower,upper]).tolist(),
                    extra_noise_scale=scale.tolist() if fit_extra_noise else None,
                    sigma_f_max=sigma_f_max,bound_hits=hits,
                    objective_definition='NLL minus fixed unit-normalization constant plus optional half-normal noise penalty')
    return float(p[0]),p[1:3],p[3:] if fit_extra_noise else np.zeros(2),True,metadata
