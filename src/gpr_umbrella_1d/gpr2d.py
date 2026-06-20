"""Multi-dimensional GPR umbrella integration (currently 2D).

Generalizes the 1D umbrella-integration scheme in :mod:`gpr_umbrella_1d.gpr`
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

The 1D autocorrelation estimator is reused for the effective sample size.
"""
from __future__ import annotations

from pathlib import Path
import os
import glob

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

from .gpr import _extract_window_index, compute_tau_int, KJ_PER_MOL_PER_EV


# ---------------------------------------------------------------------------
# Data loading (2D PLUMED COLVAR + per-window centre/kappa files)
# ---------------------------------------------------------------------------

def load_plumed_colvar_2d(
    colvar_dir: str,
    kappa_dir: str | None = None,
    cv_cols: tuple[int, int] = (1, 2),
    kappa_in_kj_per_mol: bool = False,
) -> dict:
    """Load 2D umbrella data from PLUMED COLVAR + ``window_centers_kappa`` files.

    Expects per window:
      * ``COLVAR_window_<i>.dat`` with columns ``time, cv0, cv1`` (cols set by
        *cv_cols*), written by ``ui_md_umbrella_2d.py``.
      * ``window_centers_kappa_<i>.txt`` (searched in *kappa_dir*, or alongside
        the COLVAR files) with one data line
        ``c0, c1, kappa0, kappa1``.

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
        kappas = kappas / KJ_PER_MOL_PER_EV

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

def _nll(params: np.ndarray, X: np.ndarray, y: np.ndarray,
         errors: np.ndarray) -> float:
    D = X.shape[1]
    sigma_f, ell = params[0], params[1:]
    if sigma_f <= 0 or np.any(ell <= 0.01):
        return 1e12
    n = len(y)
    Kgg = _k_grad_grad(X, X, sigma_f, ell)
    Ky = Kgg + np.diag(errors**2) + 1e-8 * np.eye(n)
    try:
        L, low = cho_factor(Ky)
        alpha = cho_solve((L, low), y)
        return float(0.5 * y @ alpha + np.sum(np.log(np.diag(L)))
                     + 0.5 * n * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        return 1e12


def _fit_hyperparameters_2d(X, y, errors, ell_init, ell_upper, sigma_f_init,
                            fixed_lengthscale, fixed_sigma_f, optimize):
    D = X.shape[1]
    ell_init = np.atleast_1d(ell_init).astype(float)
    if ell_init.size == 1:
        ell_init = np.full(D, ell_init[0])

    # Nothing free to optimize (both fixed, or optimization disabled).
    if not optimize or (fixed_sigma_f is not None and fixed_lengthscale is not None):
        sf = fixed_sigma_f if fixed_sigma_f is not None else sigma_f_init
        ell = (np.full(D, fixed_lengthscale) if fixed_lengthscale is not None
               else ell_init)
        return sf, ell, True

    def pack_nll(p):
        if fixed_sigma_f is not None:
            sf = fixed_sigma_f
            ell = p
        else:
            sf = p[0]
            ell = p[1:]
        if fixed_lengthscale is not None:
            ell = np.full(D, fixed_lengthscale)
        return _nll(np.concatenate([[sf], ell]), X, y, errors)

    # Build parameter vector layout and bounds
    starts = []
    bounds = []
    if fixed_sigma_f is None:
        bounds.append((1e-3, 1e3))
    if fixed_lengthscale is None:
        bounds += [(0.02, ell_upper)] * D

    def make_x0(scale_sf, scale_ell):
        x0 = []
        if fixed_sigma_f is None:
            x0.append(np.clip(sigma_f_init * scale_sf, 1e-3, 1e3))
        if fixed_lengthscale is None:
            x0 += list(np.clip(ell_init * scale_ell, 0.02, ell_upper))
        return np.array(x0)

    for s_sf, s_ell in [(1, 1), (0.5, 0.5), (2, 2), (1, 0.5), (0.5, 2)]:
        starts.append(make_x0(s_sf, s_ell))

    best, best_nll = None, np.inf
    for x0 in starts:
        res = minimize(pack_nll, x0=x0, method="L-BFGS-B", bounds=bounds)
        if res.success and res.fun < best_nll:
            best_nll, best = res.fun, res.x
    if best is None:
        return sigma_f_init, ell_init, False

    if fixed_sigma_f is not None:
        sf = fixed_sigma_f
        ell = best
    else:
        sf = best[0]
        ell = best[1:]
    if fixed_lengthscale is not None:
        ell = np.full(D, fixed_lengthscale)
    return float(sf), np.asarray(ell, dtype=float), True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gpr_umbrella_integration_2d(
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
    optimize_hyperparams: bool = True,
    fixed_lengthscale: float | None = None,
    fixed_sigma_f: float | None = None,
    max_lag: int = 1000,
    acf_threshold: float = 0.05,
    calibrate_uncertainty: bool = True,
    output_dir: str | None = None,
    output_prefix: str | None = None,
    plot: bool = True,
    plot_diagnostics: bool = True,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict:
    """Reconstruct a 2D PMF from umbrella windows via gradient-observation GPR.

    Provide either *colvar_dir* (loaded with :func:`load_plumed_colvar_2d`) or a
    pre-loaded *data* dict in the same schema (handy for synthetic tests).
    """
    if (colvar_dir is None) == (data is None):
        raise ValueError("Provide exactly one of colvar_dir or data.")

    if data is None:
        data = load_plumed_colvar_2d(
            colvar_dir, kappa_dir=kappa_dir, cv_cols=cv_cols,
            kappa_in_kj_per_mol=kappa_in_kj_per_mol,
        )
        base_dir = os.path.dirname(os.path.abspath(colvar_dir).rstrip("/"))
    else:
        base_dir = output_dir or "."

    centers = data["centers"]
    kappa = data["kappa"]
    means = data["means"]
    variances = data["vars"]
    n_samples = data["n_samples"]
    all_positions = data["all_positions"]
    N, D = centers.shape
    if D != 2:
        raise ValueError(f"gpr_umbrella_integration_2d expects 2 CVs, got {D}")

    # Effective sample size per window per CV component
    tau = np.array([[compute_tau_int(p[:, d], max_lag=max_lag,
                                     acf_threshold=acf_threshold)
                     for d in range(D)] for p in all_positions])
    n_eff = n_samples[:, None] / (2.0 * tau)

    # Gradient observations and their statistical errors (per component)
    grad = kappa * (centers - means)                       # (N, D)
    grad_err = kappa * np.sqrt(variances / np.maximum(n_eff, 1.0))
    grad_err = np.where(grad_err > 0, grad_err, 1e-12)

    if verbose:
        print("=" * 72)
        print("2D GPR UMBRELLA INTEGRATION")
        print("=" * 72)
        print(f"Windows: {N}   CVs: {cv_names}   units: {cv_units}")
        print(f"{cv_names[0]} range: {centers[:,0].min():.3f}..{centers[:,0].max():.3f}")
        print(f"{cv_names[1]} range: {centers[:,1].min():.3f}..{centers[:,1].max():.3f}")
        print(f"mean |grad|: {np.linalg.norm(grad, axis=1).mean():.4f} "
              f"{energy_unit}/{cv_units[0]}")
        print(f"mean tau_int: {tau.mean():.1f}   mean N_eff: {n_eff.mean():.0f}")

    # Derivative observations live at the window means (cf. the 1D code);
    # the prediction grid still spans the window centres.
    X = means
    y = grad.reshape(N * D)                                 # stacked gradient
    errors = grad_err.reshape(N * D)

    span = X.max(axis=0) - X.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    ell_init = np.maximum(0.5 * span, 0.05)
    ell_upper = float(3.0 * span.max())
    sigma_f_init = max(np.linalg.norm(grad, axis=1).std() * ell_init.mean(), 1e-3)

    sigma_f, ell, ok = _fit_hyperparameters_2d(
        X, y, errors, ell_init, ell_upper, sigma_f_init,
        fixed_lengthscale, fixed_sigma_f, optimize_hyperparams,
    )
    if verbose:
        print(f"hyperparams: sigma_f={sigma_f:.4f}  ell={np.round(ell,3)}"
              + ("" if ok else "  [fit FAILED -> initial estimates]"))

    # Train
    Kgg = _k_grad_grad(X, X, sigma_f, ell)
    Ky = Kgg + np.diag(errors**2) + 1e-8 * np.eye(N * D)
    L, low = cho_factor(Ky)
    alpha = cho_solve((L, low), y)

    # Prediction grid
    gx = np.linspace(centers[:, 0].min(), centers[:, 0].max(), grid_n[0])
    gy = np.linspace(centers[:, 1].min(), centers[:, 1].max(), grid_n[1])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    Xs = np.column_stack([GX.ravel(), GY.ravel()])         # (M, 2)

    K_fg = _k_f_grad(Xs, X, sigma_f, ell)                  # (M, N*D)
    f_mean = K_fg @ alpha
    K_ss = _se(Xs, Xs, sigma_f, ell)
    cov_f = K_ss - K_fg @ cho_solve((L, low), K_fg.T)
    f_var = np.clip(np.diag(cov_f), 0, np.inf)

    # PMF defined up to a constant: reference at the global predicted minimum
    ref = int(np.argmin(f_mean))
    pmf = f_mean - f_mean[ref]
    var_diff = f_var + f_var[ref] - 2.0 * cov_f[:, ref]
    pmf_std = np.sqrt(np.clip(var_diff, 0, np.inf))

    # LOO calibration on the gradient observations (component-wise z-scores)
    Ky_inv = cho_solve((L, low), np.eye(N * D))
    diag_inv = np.maximum(np.diag(Ky_inv), 1e-15)
    loo_resid = (Ky_inv @ y) / diag_inv
    loo_std = np.sqrt(1.0 / diag_inv)
    loo_z = loo_resid / np.maximum(loo_std, 1e-15)
    cal_factor = None
    if calibrate_uncertainty and np.isfinite(loo_z.std()) and loo_z.std() > 0:
        cal_factor = float(loo_z.std())
        pmf_std = pmf_std * cal_factor
    if verbose:
        print(f"LOO z-score std (calibration): "
              f"{loo_z.std():.3f}{'' if cal_factor else ' (not applied)'}")
        print(f"PMF range: {pmf.min():.3f}..{pmf.max():.3f} {energy_unit}  "
              f"max sigma: {pmf_std.max():.3f}")

    results = {
        "centers": centers, "kappa": kappa, "means": means,
        "vars": variances, "n_samples": n_samples, "all_positions": all_positions,
        "grad": grad, "grad_err": grad_err, "tau": tau, "n_eff": n_eff,
        "sigma_f": sigma_f, "lengthscale": ell, "fit_ok": ok,
        "gx": gx, "gy": gy, "GX": GX, "GY": GY,
        "pmf": pmf.reshape(grid_n), "pmf_std": pmf_std.reshape(grid_n),
        "loo_z": loo_z, "uncertainty_calibration_factor": cal_factor,
        "cv_names": cv_names, "cv_units": cv_units, "energy_unit": energy_unit,
    }

    if output_prefix is None:
        output_prefix = (os.path.basename(base_dir) if base_dir not in (None, ".")
                         else "gpr2d")
    out = output_dir or base_dir or "."
    Path(out).mkdir(parents=True, exist_ok=True)

    if save_outputs:
        flat = np.column_stack([GX.ravel(), GY.ravel(),
                                pmf.ravel(), pmf_std.ravel()])
        pmf_path = os.path.join(out, f"{output_prefix}_pmf2d_gpr.dat")
        np.savetxt(pmf_path, flat,
                   header=f"{cv_names[0]}({cv_units[0]}) {cv_names[1]}({cv_units[1]}) "
                          f"PMF({energy_unit}) sigma({energy_unit})", fmt="%.6f")
        results["pmf_path"] = pmf_path
        if verbose:
            print(f"wrote {pmf_path}")

    if plot:
        from .plotting2d import plot_pmf_2d
        fig_path = os.path.join(out, f"{output_prefix}_pmf2d_gpr.png")
        plot_pmf_2d(results, output_path=fig_path)
        results["figure_path"] = fig_path
        if verbose:
            print(f"wrote {fig_path}")

    if plot_diagnostics:
        from .plotting2d import plot_diagnostics_2d
        diag_path = os.path.join(out, f"{output_prefix}_diagnostics2d.png")
        plot_diagnostics_2d(results, output_path=diag_path,
                            output_prefix=output_prefix)
        results["diagnostics_path"] = diag_path
        if verbose:
            print(f"wrote {diag_path}")

    return results
