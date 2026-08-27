"""Two-dimensional GPR umbrella integration.

Generalizes the 1D umbrella-integration scheme in
:mod:`gpr_umbrella.integration_1d`
to a separable-bias multi-CV setup, e.g. the H-H distance x relative-z 2D
umbrella sampling in ``desorption_2dUS``.

Method
------
Each window ``w`` applies an independent harmonic restraint per CV,

    U_bias = sum_d 0.5 * kappa_d * (x_d - c_{w,d})**2 ,

so the umbrella estimate of the free-energy gradient at the window's mean
sampled position is, component-wise,

    grad F_d(<x>_w) ~= kappa_d * (c_{w,d} - <x_d>_w) .

We condition a Gaussian process with a separable squared-exponential kernel on
these *gradient* observations (a derivative-observation GP) and predict the
scalar PMF F(x) up to an additive constant on a dense grid, with calibrated
uncertainties.  This is the direct D-dimensional analogue of the 1D code:
identical kernels, marginal-likelihood hyperparameter fit, and LOO calibration.

Within each window, non-overlapping batch means estimate the full covariance
of the two sampled means.  This retains both autocorrelation inflation and the
cross-component covariance before propagation through the force constants.
"""
from __future__ import annotations

from pathlib import Path
import os
import glob

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from .integration_1d import (
    _extract_window_index,
    _kj_per_mol_to_energy_factor,
    compute_tau_int,
)

from .support import (
    path_valid_mask as build_path_valid_mask,
    sampled_support_mask,
    window_anchored_display_policy,
)

# ---------------------------------------------------------------------------
# Data loading (2D PLUMED COLVAR + per-window centre/kappa files)
# ---------------------------------------------------------------------------

def load_plumed_colvar_2d(
    colvar_dir: str,
    kappa_dir: str | None = None,
    cv_cols: tuple[int, int] = (1, 2),
    kappa_in_kj_per_mol: bool = False,
    energy_unit: str = "eV",
) -> dict:
    """Load 2D umbrella data from PLUMED COLVAR + ``window_centers_kappa`` files.

    Expects per window:
      * ``COLVAR_window_<i>.dat`` with columns ``time, cv0, cv1`` (cols set by
        *cv_cols*), written by ``ui_md_umbrella_2d.py``.
      * ``window_centers_kappa_<i>.txt`` (searched in *kappa_dir*, or alongside
        the COLVAR files) with one data line
        ``c0, c1, kappa0, kappa1``.

    When ``kappa_in_kj_per_mol`` is true, force constants are converted from
    kJ/mol/CV_unit² to ``energy_unit/CV_unit²`` before reconstruction.

    Returns a dict with arrays keyed by window:
      ``centers`` (N, 2), ``kappa`` (N, 2), ``means`` (N, 2), ``vars`` (N, 2),
      ``n_samples`` (N,), ``all_positions`` (list of (n_w, 2) arrays).
    """
    colvar_files = glob.glob(os.path.join(colvar_dir, "COLVAR_window_*.dat"))
    if not colvar_files:
        colvar_files = glob.glob(os.path.join(colvar_dir, "COLVAR*.dat"))
    colvar_files = sorted(colvar_files, key=_extract_window_index)
    if not colvar_files:
        raise ValueError(f"No COLVAR files found in {colvar_dir}")

    kdir = kappa_dir if kappa_dir is not None else colvar_dir

    all_positions: list[np.ndarray] = []
    centers: list[list[float]] = []
    kappas: list[list[float]] = []
    for cf in colvar_files:
        w = _extract_window_index(cf)
        data = np.loadtxt(cf, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_positions.append(data[:, list(cv_cols)])

        kfile = os.path.join(kdir, f"window_centers_kappa_{w}.txt")
        if not os.path.exists(kfile):
            raise ValueError(f"Missing centre/kappa file for window {w}: {kfile}")
        c0 = c1 = k0 = k1 = None
        with open(kfile) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", " ").split()]
                c0, c1, k0, k1 = (float(parts[0]), float(parts[1]),
                                  float(parts[2]), float(parts[3]))
                break
        if c0 is None:
            raise ValueError(f"No data line in {kfile}")
        centers.append([c0, c1])
        kappas.append([k0, k1])

    centers = np.asarray(centers, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    if kappa_in_kj_per_mol:
        kappas = kappas * _kj_per_mol_to_energy_factor(energy_unit)

    means = np.array([p.mean(axis=0) for p in all_positions])
    variances = np.array([p.var(axis=0, ddof=1) for p in all_positions])
    n_samples = np.array([len(p) for p in all_positions], dtype=float)

    return {
        "window_files": colvar_files,
        "centers": centers,
        "kappa": kappas,
        "means": means,
        "vars": variances,
        "n_samples": n_samples,
        "all_positions": all_positions,
    }


def estimate_mean_covariance(
    positions: np.ndarray,
    tau_int: np.ndarray,
    *,
    batch_factor: float = 5.0,
    min_batches: int = 4,
) -> tuple[np.ndarray, int]:
    """Estimate the covariance matrix of a correlated multivariate mean.

    The trajectory is split into non-overlapping batches longer than the
    slowest estimated correlation time.  If ``B`` batch means are available,
    ``cov(batch_means) / B`` estimates ``cov(mean(positions))`` and preserves
    correlations between CV components.  The returned integer is the batch
    length used. Very short trajectories use a multivariate Bartlett/Newey-West
    long-run covariance estimate, which also retains lagged cross-component
    correlations.
    """
    positions = np.asarray(positions, dtype=float)
    tau_int = np.asarray(tau_int, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (n_samples, n_CVs)")
    n, d = positions.shape
    if n < 2:
        raise ValueError("At least two samples are required per window")
    if tau_int.shape != (d,):
        raise ValueError(f"tau_int must have shape ({d},)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values")
    if not np.all(np.isfinite(tau_int)) or np.any(tau_int <= 0):
        raise ValueError("tau_int values must be finite and positive")
    if batch_factor <= 0 or min_batches < 2:
        raise ValueError("batch_factor must be positive and min_batches >= 2")

    target = max(1, int(np.ceil(batch_factor * float(np.max(tau_int)))))
    largest = max(1, n // min_batches)
    batch_size = min(target, largest)
    n_batches = n // batch_size

    if target <= largest and n_batches >= min_batches and batch_size > 1:
        used = positions[:n_batches * batch_size]
        batch_means = used.reshape(n_batches, batch_size, d).mean(axis=1)
        mean_cov = np.atleast_2d(np.cov(batch_means, rowvar=False, ddof=1))
        mean_cov = mean_cov / n_batches
    else:
        centered = positions - positions.mean(axis=0)
        max_hac_lag = min(n - 1, target)
        long_run_cov = centered.T @ centered / n
        for lag in range(1, max_hac_lag + 1):
            lag_cov = centered[:-lag].T @ centered[lag:] / n
            bartlett_weight = 1.0 - lag / (max_hac_lag + 1.0)
            long_run_cov += bartlett_weight * (lag_cov + lag_cov.T)
        mean_cov = long_run_cov / n

    # Symmetrize and project roundoff/small-sample negative modes onto the PSD
    # cone in correlation coordinates. Performing the projection after
    # component-wise normalization makes it invariant to independent CV-unit
    # changes. Zero-variance components stay exactly zero; GP training adds
    # its own unit-covariant numerical jitter later.
    mean_cov = 0.5 * (mean_cov + mean_cov.T)
    diagonal = np.clip(np.diag(mean_cov), 0.0, np.inf)
    positive = diagonal > 0.0
    if np.any(positive):
        scales = np.sqrt(diagonal[positive])
        normalized = mean_cov[np.ix_(positive, positive)] / np.outer(scales, scales)
        normalized = 0.5 * (normalized + normalized.T)
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        normalized = (
            eigenvectors * np.clip(eigenvalues, 0.0, np.inf)
        ) @ eigenvectors.T
        # Restore a unit diagonal after clipping so the original component
        # variances are retained while only invalid correlation modes change.
        normalized_scale = np.sqrt(np.clip(np.diag(normalized), 0.0, np.inf))
        nonzero = normalized_scale > 0.0
        normalized[np.ix_(nonzero, nonzero)] /= np.outer(
            normalized_scale[nonzero], normalized_scale[nonzero]
        )
        projected = np.zeros_like(mean_cov)
        projected[np.ix_(positive, positive)] = normalized * np.outer(scales, scales)
        mean_cov = projected
    else:
        mean_cov = np.zeros_like(mean_cov)
    return mean_cov, batch_size


def _gradient_noise_covariance(
    all_positions: list[np.ndarray],
    kappa: np.ndarray,
    tau: np.ndarray,
    *,
    include_cross_component: bool,
    batch_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the block-diagonal covariance of all gradient observations."""
    kappa = np.asarray(kappa, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if kappa.ndim != 2:
        raise ValueError("kappa must have shape (n_windows, n_CVs)")
    n_windows, d = kappa.shape
    if len(all_positions) != n_windows:
        raise ValueError("all_positions and kappa must contain the same windows")
    if tau.shape != (n_windows, d):
        raise ValueError(f"tau must have shape ({n_windows}, {d})")
    blocks = []
    mean_covariances = []
    batch_sizes = []
    for w, positions in enumerate(all_positions):
        mean_cov, batch_size = estimate_mean_covariance(
            positions, tau[w], batch_factor=batch_factor
        )
        if not include_cross_component:
            mean_cov = np.diag(np.diag(mean_cov))
        scale = np.diag(kappa[w])
        blocks.append(scale @ mean_cov @ scale)
        mean_covariances.append(mean_cov)
        batch_sizes.append(batch_size)

    noise_cov = np.zeros((n_windows * d, n_windows * d), dtype=float)
    for w, block in enumerate(blocks):
        sl = slice(w * d, (w + 1) * d)
        noise_cov[sl, sl] = block
    return noise_cov, np.asarray(mean_covariances), np.asarray(batch_sizes)


def _grid_support_mask(
    points: np.ndarray,
    training_points: np.ndarray,
    lengthscale: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Mark the union of kernel-scaled neighborhoods of sampled means."""
    return sampled_support_mask(points, training_points, lengthscale, radius)


# ---------------------------------------------------------------------------
# Separable squared-exponential kernel and its derivatives (D dims)
# ---------------------------------------------------------------------------

def _se(Xa: np.ndarray, Xb: np.ndarray, sigma_f: float,
        ell: np.ndarray) -> np.ndarray:
    """Base SE covariance matrix between point sets Xa (Na,D) and Xb (Nb,D)."""
    diff = Xa[:, None, :] - Xb[None, :, :]          # (Na, Nb, D)
    sqdist = np.sum((diff / ell) ** 2, axis=2)      # (Na, Nb)
    return sigma_f**2 * np.exp(-0.5 * sqdist)


def _k_f_grad(Xs: np.ndarray, Xt: np.ndarray, sigma_f: float,
              ell: np.ndarray) -> np.ndarray:
    """Cov(f(Xs), d f(Xt)/dXt_j).  Returns (M, N*D).

    dk/dXt_j = (r_j / ell_j^2) k, with r = Xs - Xt.
    """
    M, N, D = Xs.shape[0], Xt.shape[0], Xt.shape[1]
    r = Xs[:, None, :] - Xt[None, :, :]             # (M, N, D)
    k = _se(Xs, Xt, sigma_f, ell)                   # (M, N)
    out = (r / ell**2) * k[:, :, None]              # (M, N, D)
    return out.reshape(M, N * D)


def _k_grad_grad(Xa: np.ndarray, Xb: np.ndarray, sigma_f: float,
                 ell: np.ndarray) -> np.ndarray:
    """Cov(d f(Xa)/dXa_i, d f(Xb)/dXb_j).  Returns (Na*D, Nb*D).

    d2k/dXa_i dXb_j = k * [ delta_ij/ell_i^2 - r_i r_j /(ell_i^2 ell_j^2) ],
    with r = Xa - Xb.
    """
    Na, D = Xa.shape
    Nb = Xb.shape[0]
    r = Xa[:, None, :] - Xb[None, :, :]             # (Na, Nb, D)
    k = _se(Xa, Xb, sigma_f, ell)                   # (Na, Nb)
    inv2 = 1.0 / ell**2                             # (D,)
    # delta_ij / ell_i^2 term
    delta = np.zeros((Na, Nb, D, D))
    idx = np.arange(D)
    delta[:, :, idx, idx] = inv2[None, None, :]
    # r_i r_j /(ell_i^2 ell_j^2)
    ri = (r * inv2)[:, :, :, None]                  # (Na,Nb,D,1)
    rj = (r * inv2)[:, :, None, :]                  # (Na,Nb,1,D)
    block = (delta - ri * rj) * k[:, :, None, None]  # (Na,Nb,D,D)
    return block.transpose(0, 2, 1, 3).reshape(Na * D, Nb * D)


# ---------------------------------------------------------------------------
# Hyperparameters via marginal likelihood on the gradient observations
# ---------------------------------------------------------------------------

def _with_relative_diagonal_jitter(
    covariance: np.ndarray,
    relative: float = 1e-8,
) -> np.ndarray:
    """Return a copy with component-wise, unit-covariant diagonal jitter."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(relative) or relative <= 0:
        raise ValueError("relative jitter must be finite and positive")
    result = covariance.copy()
    diagonal = np.diag(result)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError("covariance must have a finite positive diagonal")
    result[np.diag_indices_from(result)] += relative * diagonal
    return result

def _nll(params: np.ndarray, X: np.ndarray, y: np.ndarray,
         noise_cov: np.ndarray) -> float:
    sigma_f, ell = params[0], params[1:]
    if sigma_f <= 0 or np.any(ell <= 0):
        return 1e12
    n = len(y)
    Kgg = _k_grad_grad(X, X, sigma_f, ell)
    Ky = _with_relative_diagonal_jitter(Kgg + noise_cov)
    try:
        L, low = cho_factor(Ky, lower=True)
        alpha = cho_solve((L, low), y)
        return float(0.5 * y @ alpha + np.sum(np.log(np.diag(L)))
                     + 0.5 * n * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        return 1e12


def _fit_hyperparameters_2d(X, y, noise_cov, ell_init, ell_upper, sigma_f_init,
                            fixed_lengthscale, fixed_sigma_f, optimize):
    D = X.shape[1]
    ell_init = np.atleast_1d(ell_init).astype(float)
    if ell_init.size == 1:
        ell_init = np.full(D, ell_init[0])
    ell_upper = np.broadcast_to(np.asarray(ell_upper, dtype=float), (D,))
    ell_lower = ell_upper / 3000.0

    if (
        ell_init.shape != (D,)
        or np.any(~np.isfinite(ell_init))
        or np.any(ell_init <= 0)
        or np.any(~np.isfinite(ell_upper))
        or np.any(ell_upper <= ell_lower)
        or not np.isfinite(sigma_f_init)
        or sigma_f_init <= 0
    ):
        raise ValueError("Initial GP hyperparameter scales must be finite and positive")

    fixed_ell = None
    if fixed_lengthscale is not None:
        try:
            fixed_ell = np.broadcast_to(
                np.asarray(fixed_lengthscale, dtype=float), (D,)
            ).copy()
        except ValueError as exc:
            raise ValueError(f"fixed_lengthscale must be scalar or have shape ({D},)") from exc
        if np.any(~np.isfinite(fixed_ell)) or np.any(fixed_ell <= 0):
            raise ValueError("fixed_lengthscale values must be finite and positive")
    if fixed_sigma_f is not None:
        fixed_sigma_f = float(fixed_sigma_f)
        if not np.isfinite(fixed_sigma_f) or fixed_sigma_f <= 0:
            raise ValueError("fixed_sigma_f must be finite and positive")

    # Nothing free to optimize (both fixed, or optimization disabled).
    if not optimize or (fixed_sigma_f is not None and fixed_lengthscale is not None):
        sf = fixed_sigma_f if fixed_sigma_f is not None else sigma_f_init
        ell = fixed_ell if fixed_ell is not None else ell_init
        return sf, ell, True

    # Optimize logarithmic, dimensionless ratios. This keeps both bounds and
    # optimizer geometry unchanged under energy or per-CV unit conversions.
    def unpack(log_ratios):
        cursor = 0
        if fixed_sigma_f is None:
            sf = sigma_f_init * np.exp(log_ratios[cursor])
            cursor += 1
        else:
            sf = fixed_sigma_f
        if fixed_ell is None:
            ell = ell_init * np.exp(log_ratios[cursor:cursor + D])
        else:
            ell = fixed_ell
        return float(sf), np.asarray(ell, dtype=float)

    def pack_nll(log_ratios):
        sf, ell = unpack(log_ratios)
        return _nll(np.concatenate([[sf], ell]), X, y, noise_cov)

    # Build bounds for the dimensionless ratios.
    starts = []
    bounds = []
    if fixed_sigma_f is None:
        bounds.append((np.log(1e-3), np.log(1e3)))
    if fixed_ell is None:
        bounds += list(zip(np.log(ell_lower / ell_init),
                           np.log(ell_upper / ell_init)))

    def make_x0(scale_sf, scale_ell):
        x0 = []
        if fixed_sigma_f is None:
            x0.append(np.log(scale_sf))
        if fixed_ell is None:
            x0 += [np.log(scale_ell)] * D
        x0 = np.asarray(x0, dtype=float)
        for index, (lower, upper) in enumerate(bounds):
            x0[index] = np.clip(x0[index], lower, upper)
        return np.array(x0)

    for s_sf, s_ell in [(1, 1), (0.5, 0.5), (2, 2), (1, 0.5), (0.5, 2)]:
        starts.append(make_x0(s_sf, s_ell))

    best, best_nll = None, np.inf
    for x0 in starts:
        res = minimize(pack_nll, x0=x0, method="L-BFGS-B", bounds=bounds)
        if res.success and res.fun < best_nll:
            best_nll, best = res.fun, res.x
    if best is None:
        fallback_sigma = fixed_sigma_f if fixed_sigma_f is not None else sigma_f_init
        fallback_ell = fixed_ell if fixed_ell is not None else ell_init
        return fallback_sigma, fallback_ell, False
    sf, ell = unpack(best)
    return sf, ell, True


def posterior_covariance_2d(
    results: dict,
    Xa: np.ndarray,
    Xb: np.ndarray | None = None,
    *,
    calibrated: bool = False,
) -> np.ndarray:
    """Posterior covariance between arbitrary points on a fitted 2D surface.

    Only the requested ``len(Xa) x len(Xb)`` matrix is formed.  This is used by
    pathway error propagation without retaining a full grid-grid covariance.
    """
    state = results.get("_gp_state")
    if state is None:
        raise ValueError("results does not contain fitted GP state")
    Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
    same_points = Xb is None
    Xb = Xa if same_points else np.atleast_2d(np.asarray(Xb, dtype=float))
    dimensions = state["X"].shape[1]
    if Xa.shape[1] != dimensions or Xb.shape[1] != dimensions:
        raise ValueError(f"query points must have shape (n_points, {dimensions})")
    K_a = _k_f_grad(Xa, state["X"], state["sigma_f"], state["lengthscale"])
    K_b = _k_f_grad(Xb, state["X"], state["sigma_f"], state["lengthscale"])
    solved_b = cho_solve((state["cho_factor"], state["lower"]), K_b.T)
    covariance = _se(
        Xa, Xb, state["sigma_f"], state["lengthscale"]
    ) - K_a @ solved_b
    if same_points:
        covariance = 0.5 * (covariance + covariance.T)
    if calibrated:
        covariance *= results["loo_calibration_factor"] ** 2
    return covariance


def _leave_one_window_out_z(
    precision: np.ndarray,
    observations: np.ndarray,
    dimensions: int,
) -> np.ndarray:
    """Whiten leave-one-window-out residuals using precision-matrix blocks."""
    weighted = precision @ observations
    z_scores = []
    for start in range(0, len(observations), dimensions):
        sl = slice(start, start + dimensions)
        precision_block = 0.5 * (
            precision[sl, sl] + precision[sl, sl].T
        )
        block_factor = cho_factor(precision_block, lower=True)
        conditional_covariance = cho_solve(
            block_factor, np.eye(dimensions)
        )
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )
        residual = cho_solve(block_factor, weighted[sl])
        chol = np.linalg.cholesky(conditional_covariance)
        z_scores.extend(np.linalg.solve(chol, residual))
    return np.asarray(z_scores)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconstruct_pmf_2d(
    colvar_dir: str | None = None,
    *,
    data: dict | None = None,
    kappa_dir: str | None = None,
    cv_cols: tuple[int, int] = (1, 2),
    kappa_in_kj_per_mol: bool = False,
    cv_units: tuple[str, str] = ("A", "A"),
    cv_names: tuple[str, str] = ("hh", "relz"),
    energy_unit: str = "eV",
    grid_n: tuple[int, int] = (60, 60),
    prediction_batch_size: int = 10_000,
    optimize_hyperparams: bool = True,
    fixed_lengthscale: float | tuple[float, float] | None = None,
    fixed_sigma_f: float | None = None,
    max_lag: int = 1000,
    acf_threshold: float = 0.05,
    include_cross_component_covariance: bool = True,
    covariance_batch_factor: float = 5.0,
    calibrate_uncertainty: bool = True,
    restrict_to_sampled_support: bool = True,
    support_radius: float = 0.5,
    output_dir: str | None = None,
    output_prefix: str | None = None,
    plot: bool = True,
    plot_diagnostics: bool = True,
    find_lowest_barrier: bool = False,
    path_endpoints=None,
    path_metric_scale: tuple[float, float] | None = None,
    path_reference: np.ndarray | None = None,
    path_mode: str = "search",
    path_corridor_radius: float | None = None,
    path_aligned_marginal: bool = False,
    thermal_energy: float | None = None,
    perpendicular_points: int = 201,
    perpendicular_width: float | None = None,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict:
    """Reconstruct a 2D PMF from umbrella windows via gradient-observation GPR.

    Provide either *colvar_dir* (loaded with :func:`load_plumed_colvar_2d`) or a
    pre-loaded *data* dict in the same schema (handy for synthetic tests).
    For either source, ``kappa_in_kj_per_mol=True`` converts a copied force-
    constant array into the selected numerical ``energy_unit``.
    ``cv_units`` contains one unit per coordinate; force constants and GP
    gradients are interpreted in ``energy_unit / cv_units[d]²`` and
    ``energy_unit / cv_units[d]``, respectively.

    By default plots, path search, and transverse integration are restricted
    to the union of kernel-scaled neighborhoods around sampled window means.
    ``support_radius`` gives the radius in GP lengthscales. ``path_mode``
    selects free search between required endpoints, corridor search using a
    reference trajectory, or direct evaluation of that fixed trajectory.
    """
    if (colvar_dir is None) == (data is None):
        raise ValueError("Provide exactly one of colvar_dir or data.")
    try:
        grid_values = np.asarray(grid_n, dtype=float)
    except (TypeError, ValueError):
        grid_values = np.array([], dtype=float)
    if (
        grid_values.shape != (2,)
        or np.asarray(grid_n).dtype == np.dtype(bool)
        or not np.all(np.isfinite(grid_values))
        or np.any(grid_values != np.floor(grid_values))
        or np.any(grid_values < 2)
    ):
        raise ValueError("grid_n must contain two integers >= 2")
    grid_n = tuple(int(value) for value in grid_values)
    if prediction_batch_size < 1:
        raise ValueError("prediction_batch_size must be positive")
    if path_aligned_marginal and thermal_energy is None:
        raise ValueError(
            "thermal_energy is required when path_aligned_marginal=True"
        )

    if data is None:
        data = load_plumed_colvar_2d(
            colvar_dir, kappa_dir=kappa_dir, cv_cols=cv_cols,
            kappa_in_kj_per_mol=kappa_in_kj_per_mol,
            energy_unit=energy_unit,
        )
        base_dir = os.path.dirname(os.path.abspath(colvar_dir).rstrip("/"))
    else:
        data = dict(data)
        if kappa_in_kj_per_mol:
            data["kappa"] = (
                np.asarray(data["kappa"], dtype=float)
                * _kj_per_mol_to_energy_factor(energy_unit)
            )
        base_dir = output_dir or "."

    centers = data["centers"]
    kappa = data["kappa"]
    means = data["means"]
    variances = data["vars"]
    n_samples = data["n_samples"]
    all_positions = data["all_positions"]
    N, D = centers.shape
    if D != 2:
        raise ValueError(f"reconstruct_pmf_2d expects 2 CVs, got {D}")

    # Effective sample size per window per CV component
    tau = np.array([[compute_tau_int(p[:, d], max_lag=max_lag,
                                     acf_threshold=acf_threshold)
                     for d in range(D)] for p in all_positions])
    n_eff = n_samples[:, None] / (2.0 * tau)

    # Gradient observations and their full statistical covariance.  Each
    # window contributes a 2x2 block, including cross-CV covariance by default.
    grad = kappa * (centers - means)                       # (N, D)
    noise_cov, mean_covariances, covariance_batch_sizes = (
        _gradient_noise_covariance(
            all_positions,
            kappa,
            tau,
            include_cross_component=include_cross_component_covariance,
            batch_factor=covariance_batch_factor,
        )
    )
    grad_err = np.sqrt(np.maximum(np.diag(noise_cov), 0.0)).reshape(N, D)

    if verbose:
        print("=" * 72)
        print("2D GPR UMBRELLA INTEGRATION")
        print("=" * 72)
        print(f"Windows: {N}   CVs: {cv_names}   units: {cv_units}")
        print(
            f"{cv_names[0]} range: {centers[:,0].min():.3f}.."
            f"{centers[:,0].max():.3f} {cv_units[0]}"
        )
        print(
            f"{cv_names[1]} range: {centers[:,1].min():.3f}.."
            f"{centers[:,1].max():.3f} {cv_units[1]}"
        )
        mean_abs_grad = np.mean(np.abs(grad), axis=0)
        print("mean |gradient component|: "
              f"{mean_abs_grad[0]:.4f} {energy_unit}/{cv_units[0]}, "
              f"{mean_abs_grad[1]:.4f} {energy_unit}/{cv_units[1]}")
        print(f"mean tau_int: {tau.mean():.1f}   mean N_eff: {n_eff.mean():.0f}")

    # Derivative observations live at the window means (cf. the 1D code);
    # the prediction grid still spans the window centres.
    X = means
    y = grad.reshape(N * D)                                 # stacked gradient

    span = X.max(axis=0) - X.min(axis=0)
    if np.any(~np.isfinite(span)) or np.any(span <= 0):
        raise ValueError("Sampled window means must span both CV coordinates")
    ell_init = 0.5 * span
    ell_upper = 3.0 * span
    gradient_energy_scale = grad * ell_init[None, :]
    noise_energy_variance = (
        np.diag(noise_cov).reshape(N, D) * ell_init[None, :] ** 2
    )
    energy_scale_squared = float(
        np.mean(gradient_energy_scale**2) + np.mean(noise_energy_variance)
    )
    sigma_f_init = float(np.sqrt(energy_scale_squared))
    if not np.isfinite(sigma_f_init):
        raise ValueError("Could not derive a finite GP energy scale from the data")
    if sigma_f_init == 0.0:
        if fixed_sigma_f is None:
            raise ValueError(
                "The gradient observations and their sampling covariance are "
                "exactly zero, so the GP signal scale is unidentifiable; "
                "provide fixed_sigma_f"
            )
        sigma_f_init = float(fixed_sigma_f)

    sigma_f, ell, ok = _fit_hyperparameters_2d(
        X, y, noise_cov, ell_init, ell_upper, sigma_f_init,
        fixed_lengthscale, fixed_sigma_f, optimize_hyperparams,
    )
    if verbose:
        hyperparameters = (
            f"hyperparams: sigma_f={sigma_f:.4f} {energy_unit}  "
            f"ell=({ell[0]:.3f} {cv_units[0]}, {ell[1]:.3f} {cv_units[1]})"
        )
        print(hyperparameters + ("" if ok else "  [fit FAILED -> initial estimates]"))

    # Train
    Kgg = _k_grad_grad(X, X, sigma_f, ell)
    Ky = _with_relative_diagonal_jitter(Kgg + noise_cov)
    L, low = cho_factor(Ky, lower=True)
    alpha = cho_solve((L, low), y)

    # Prediction grid. A restricted grid includes complete boundary ellipses
    # instead of clipping them at the extrema of the sampled window means.
    if restrict_to_sampled_support:
        if isinstance(support_radius, (bool, np.bool_)):
            raise ValueError("support_radius must be finite and positive")
        try:
            support_radius = float(support_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "support_radius must be finite and positive"
            ) from exc
        if not np.isfinite(support_radius) or support_radius <= 0:
            raise ValueError(
                "support_radius must be finite and positive when support "
                "restriction is enabled"
            )
        lower = X.min(axis=0) - support_radius * ell
        upper = X.max(axis=0) + support_radius * ell
    else:
        lower = centers.min(axis=0)
        upper = centers.max(axis=0)
    gx = np.linspace(lower[0], upper[0], grid_n[0])
    gy = np.linspace(lower[1], upper[1], grid_n[1])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Xs = np.column_stack([GX.ravel(), GY.ravel()])
    if restrict_to_sampled_support:
        support_mask = _grid_support_mask(Xs, X, ell, support_radius)
    else:
        support_mask = np.ones(len(Xs), dtype=bool)
    supported_indices = np.flatnonzero(support_mask)
    if len(supported_indices) == 0:
        raise ValueError(
            "No prediction-grid points lie in the sampled-support region; "
            "increase grid_n or support_radius"
        )

    # Predict in chunks.  Peak working storage is O(batch_size * N * D), not
    # O(grid_size^2) or even O(grid_size * N * D).
    n_grid = len(Xs)
    f_mean = np.empty(n_grid, dtype=float)
    f_var = np.empty(n_grid, dtype=float)
    for start in range(0, n_grid, prediction_batch_size):
        sl = slice(start, min(start + prediction_batch_size, n_grid))
        K_chunk = _k_f_grad(Xs[sl], X, sigma_f, ell)
        solved_chunk = cho_solve((L, low), K_chunk.T)
        f_mean[sl] = K_chunk @ alpha
        f_var[sl] = np.clip(
            sigma_f**2 - np.einsum("ij,ji->i", K_chunk, solved_chunk),
            0,
            np.inf,
        )

    # PMF is defined up to a constant. The reference is the lowest supported
    # prediction; opting out of support restriction makes every grid cell valid.
    ref = int(supported_indices[np.argmin(f_mean[supported_indices])])
    pmf = f_mean - f_mean[ref]
    K_ref = _k_f_grad(Xs[[ref]], X, sigma_f, ell)
    solved_ref = cho_solve((L, low), K_ref.T).ravel()
    posterior_cov_ref = np.empty(n_grid, dtype=float)
    for start in range(0, n_grid, prediction_batch_size):
        sl = slice(start, min(start + prediction_batch_size, n_grid))
        K_chunk = _k_f_grad(Xs[sl], X, sigma_f, ell)
        prior_cov_ref = _se(Xs[sl], Xs[[ref]], sigma_f, ell).ravel()
        posterior_cov_ref[sl] = prior_cov_ref - K_chunk @ solved_ref
    var_diff = f_var + f_var[ref] - 2.0 * posterior_cov_ref
    # F(x_ref) - F(x_ref) is exactly zero. Enforce that identity before the
    # square root instead of exposing platform-dependent cancellation noise.
    var_diff[ref] = 0.0
    pmf_std_raw = np.sqrt(np.clip(var_diff, 0, np.inf))

    # Leave one complete window (both correlated CV observations) out at a
    # time, then whiten its 2-vector residual with the conditional covariance.
    Ky_inv = cho_solve((L, low), np.eye(N * D))
    loo_z = _leave_one_window_out_z(Ky_inv, y, D)
    cal_factor = float(loo_z.std(ddof=1))
    if not np.isfinite(cal_factor) or cal_factor <= 0:
        cal_factor = 1.0
    pmf_std_calibrated = pmf_std_raw * cal_factor
    if verbose:
        print(f"LOO z-score std (calibration): "
              f"{cal_factor:.3f}{'' if calibrate_uncertainty else ' (reported, not default)'}")
        displayed_std = pmf_std_calibrated if calibrate_uncertainty else pmf_std_raw
        supported_pmf = pmf[support_mask]
        supported_std = displayed_std[support_mask]
        print(
            f"Supported PMF range: {supported_pmf.min():.3f}.."
            f"{supported_pmf.max():.3f} {energy_unit}  "
            f"max sigma: {supported_std.max():.3f}"
        )

    results = {
        "centers": centers, "kappa": kappa, "means": means,
        "vars": variances, "n_samples": n_samples, "all_positions": all_positions,
        "grad": grad, "grad_err": grad_err, "gradient_noise_cov": noise_cov,
        "mean_covariances": mean_covariances,
        "covariance_batch_sizes": covariance_batch_sizes,
        "tau": tau, "n_eff": n_eff,
        "sigma_f": sigma_f, "lengthscale": ell, "fit_ok": ok,
        "gx": gx, "gy": gy, "GX": GX, "GY": GY,
        "pmf": pmf.reshape(grid_n),
        "latent_variance_raw": f_var.reshape(grid_n),
        "pmf_std_raw": pmf_std_raw.reshape(grid_n),
        "pmf_std_calibrated": pmf_std_calibrated.reshape(grid_n),
        "support_mask": support_mask.reshape(grid_n),
        "restrict_to_sampled_support": bool(restrict_to_sampled_support),
        "support_kind": (
            "union_of_kernel_ellipses"
            if restrict_to_sampled_support else "full_rectangle"
        ),
        "support_radius": (
            float(support_radius) if restrict_to_sampled_support else None
        ),
        "support_ellipse_semiaxes": (
            float(support_radius) * ell if restrict_to_sampled_support else None
        ),
        "loo_z": loo_z, "loo_calibration_factor": cal_factor,
        "default_uncertainty": "calibrated" if calibrate_uncertainty else "raw",
        "cv_names": cv_names, "cv_units": cv_units, "energy_unit": energy_unit,
        "_gp_state": {
            "X": X, "sigma_f": sigma_f, "lengthscale": ell,
            "cho_factor": L, "lower": low, "alpha": alpha,
            "pmf_reference_index": ref,
            "pmf_reference_point": Xs[ref].copy(),
            "pmf_reference_mean": float(f_mean[ref]),
        },
    }

    # Plotting and pathfinding consume the same observation-anchored policy.
    display_policy = window_anchored_display_policy(results)
    valid_mask = build_path_valid_mask(results, display_policy)
    results["path_valid_mask"] = valid_mask
    results["path_valid_kind"] = (
        "sampled_support_and_window_anchored_pmf_range"
    )

    if output_prefix is None:
        output_prefix = (os.path.basename(base_dir) if base_dir not in (None, ".")
                         else "gpr_2d")
    out = output_dir or base_dir or "."
    Path(out).mkdir(parents=True, exist_ok=True)

    if save_outputs:
        flat = np.column_stack([GX.ravel(), GY.ravel(),
                                pmf.ravel(), pmf_std_raw,
                                pmf_std_calibrated, support_mask.astype(int),
                                valid_mask.ravel().astype(int)])
        pmf_path = os.path.join(out, f"{output_prefix}_pmf_2d.dat")
        np.savetxt(pmf_path, flat,
                   header=f"{cv_names[0]}({cv_units[0]}) {cv_names[1]}({cv_units[1]}) "
                          f"PMF({energy_unit}) sigma_raw({energy_unit}) "
                          f"sigma_calibrated({energy_unit}) supported path_valid",
                   fmt="%.6f")
        results["pmf_path"] = pmf_path
        if verbose:
            print(f"wrote {pmf_path}")

    if plot:
        from .plotting_2d import plot_pmf_2d
        fig_path = os.path.join(out, f"{output_prefix}_pmf_2d.png")
        plot_pmf_2d(results, output_path=fig_path)
        results["figure_path"] = fig_path
        if verbose:
            print(f"wrote {fig_path}")

    if plot_diagnostics:
        from .plotting_2d import plot_diagnostics_2d
        diag_path = os.path.join(out, f"{output_prefix}_diagnostics_2d.png")
        plot_diagnostics_2d(results, output_path=diag_path,
                            output_prefix=output_prefix)
        results["diagnostics_path"] = diag_path
        if verbose:
            print(f"wrote {diag_path}")

    if find_lowest_barrier or path_aligned_marginal:
        from .pathways import (
            find_lowest_barrier_path,
            save_lowest_barrier_path,
            save_path_aligned_marginal,
        )
        path_result = find_lowest_barrier_path(
            results,
            endpoints=path_endpoints,
            metric_scale=path_metric_scale,
            reference_path=path_reference,
            path_mode=path_mode,
            corridor_radius=path_corridor_radius,
            path_aligned_marginal=path_aligned_marginal,
            thermal_energy=thermal_energy,
            perpendicular_points=perpendicular_points,
            perpendicular_width=perpendicular_width,
        )
        results["lowest_barrier_path"] = path_result
        if verbose:
            print(f"Lowest-barrier path: {len(path_result['minima'])} minima; "
                  f"{tuple(round(v,2) for v in path_result['start_xy'])} -> "
                  f"{tuple(round(v,2) for v in path_result['end_xy'])}")
            print(f"     barrier   = {path_result['barrier']:.3f} +/- "
                  f"{path_result['barrier_err']:.3f} {energy_unit}  "
                  f"(TS at {tuple(round(v,2) for v in path_result['ts_xy'])})")
            print(f"     reaction dF = {path_result['delta_f']:.3f} +/- "
                  f"{path_result['delta_f_err']:.3f} {energy_unit}")
        if save_outputs:
            path_data = os.path.join(out, f"{output_prefix}_lowest_barrier_path.dat")
            save_lowest_barrier_path(path_result, path_data)
            results["lowest_barrier_path_file"] = path_data
            if verbose:
                print(f"wrote {path_data}")
        if path_aligned_marginal:
            marginal = path_result["path_aligned_marginal"]
            results["path_aligned_marginal"] = marginal
            if verbose:
                print(
                    "Path-aligned marginal PMF: "
                    f"{marginal['pmf'].min():.3f}..{marginal['pmf'].max():.3f} "
                    f"{energy_unit} at kBT={marginal['thermal_energy']:.5g} "
                    f"{energy_unit}"
                )
            if save_outputs:
                marginal_data = os.path.join(
                    out, f"{output_prefix}_path_aligned_pmf_1d.dat"
                )
                save_path_aligned_marginal(marginal, marginal_data)
                results["path_aligned_marginal_path"] = marginal_data
                if verbose:
                    print(f"wrote {marginal_data}")
        if plot:
            from .plotting_2d import plot_lowest_barrier_path
            path_fig = os.path.join(out, f"{output_prefix}_lowest_barrier_path.png")
            plot_lowest_barrier_path(results, path_result, output_path=path_fig,
                                     output_prefix=output_prefix)
            results["lowest_barrier_path_figure"] = path_fig
            if verbose:
                print(f"wrote {path_fig}")

    return results
