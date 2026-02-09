from __future__ import annotations

from pathlib import Path
import os
import glob
import re
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

KJ_PER_MOL_PER_EV = 96.485


def _extract_window_index(filepath: str) -> int:
    """Extract numeric window index from filename.

    Handles patterns like ``window_5.ui_dat``, ``COLVAR_window_5.dat``, etc.
    """
    name = os.path.splitext(os.path.basename(filepath))[0]
    matches = re.findall(r'(\d+)', name)
    if matches:
        return int(matches[-1])
    return 0


def load_window_data(
    data_folder: str,
    usecols: tuple[int, ...] = (1, 2, 3),
    kappa_in_kj_per_mol: bool = True,
) -> dict:
    """Load umbrella window data from window_*.ui_dat files.

    Parameters
    ----------
    data_folder : str
        Directory containing ``window_*.ui_dat`` files.
    usecols : tuple of int
        Column indices to load (position, centre, kappa).
    kappa_in_kj_per_mol : bool
        If True (default), kappa in the files is in kJ/mol/CV_unit² and will
        be converted to eV/CV_unit².  Set to False if kappa is already in
        eV/CV_unit².
    """
    window_files = glob.glob(os.path.join(data_folder, "window_*.ui_dat"))
    window_files = sorted(window_files, key=_extract_window_index)
    if len(window_files) == 0:
        raise ValueError(f"No window files found in {data_folder}")

    kappa_list: list[float] = []
    x_centers: list[float] = []
    x_means: list[float] = []
    x_vars: list[float] = []
    n_samples_list: list[int] = []
    all_positions: list[np.ndarray] = []

    for wfile in window_files:
        data = np.loadtxt(wfile, comments="#", usecols=usecols)
        positions = data[:, 0]
        eq_position = data[0, 1]
        kappa_raw = data[0, 2]

        if kappa_in_kj_per_mol:
            kappa_list.append(kappa_raw / KJ_PER_MOL_PER_EV)
        else:
            kappa_list.append(kappa_raw)
        x_centers.append(eq_position)
        x_means.append(float(np.mean(positions)))
        x_vars.append(float(np.var(positions, ddof=1)))
        n_samples_list.append(len(positions))
        all_positions.append(positions)

    return {
        "window_files": window_files,
        "kappa": np.array(kappa_list, dtype=float),
        "x_centers": np.array(x_centers, dtype=float),
        "x_means": np.array(x_means, dtype=float),
        "x_vars": np.array(x_vars, dtype=float),
        "n_samples": np.array(n_samples_list, dtype=float),
        "all_positions": all_positions,
    }


def load_plumed_colvar_data(
    colvar_dir: str,
    kappa: float | None = None,
    kappa_dir: str | None = None,
    centers: str | np.ndarray | list | None = None,
    kappa_in_kj_per_mol: bool = False,
    cv_col: int = 1,
) -> dict:
    """Load umbrella window data from PLUMED COLVAR files.

    Parameters
    ----------
    colvar_dir : str
        Directory containing ``COLVAR_window_*.dat`` files.
    kappa : float, optional
        Single force constant for all windows (eV/CV_unit² by default).
    kappa_dir : str, optional
        Directory with per-window ``window_centers_kappa_*.txt`` files.
        Each file contains one data line: ``center, kappa``.
    centers : str, array-like, or None
        Window centres.  Can be a path to a text file (one centre per line)
        or an array of floats.  Required when *kappa* is a single value.
    kappa_in_kj_per_mol : bool
        If True, kappa values are in kJ/mol/CV_unit².  Default False.
    cv_col : int
        Column index (0-based) for the CV in COLVAR files.  Default 1.

    Returns
    -------
    dict
        Same keys as :func:`load_window_data` plus ``cv_name``.
    """
    colvar_files = glob.glob(os.path.join(colvar_dir, "COLVAR_window_*.dat"))
    if not colvar_files:
        colvar_files = glob.glob(os.path.join(colvar_dir, "COLVAR*.dat"))
    colvar_files = sorted(colvar_files, key=_extract_window_index)
    if len(colvar_files) == 0:
        raise ValueError(f"No COLVAR files found in {colvar_dir}")

    # Try to read CV name from PLUMED header
    cv_name: str | None = None
    with open(colvar_files[0]) as fh:
        for line in fh:
            if line.startswith("#! FIELDS"):
                fields = line.strip().split()
                # fields = ["#!", "FIELDS", "time", "cv_name", ...]
                if len(fields) > cv_col + 2:
                    cv_name = fields[cv_col + 2]
                break

    # Load positions from each COLVAR file
    all_positions: list[np.ndarray] = []
    for cf in colvar_files:
        data = np.loadtxt(cf, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_positions.append(data[:, cv_col])

    n_windows = len(colvar_files)

    # ---- resolve kappa & centres ----
    if kappa_dir is not None:
        kappa_files = glob.glob(os.path.join(kappa_dir, "window_centers_kappa_*.txt"))
        kappa_files = sorted(kappa_files, key=_extract_window_index)
        if len(kappa_files) != n_windows:
            raise ValueError(
                f"Number of kappa files ({len(kappa_files)}) does not match "
                f"number of COLVAR files ({n_windows})"
            )
        _centers: list[float] = []
        _kappas: list[float] = []
        for kf in kappa_files:
            with open(kf) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.split(",")
                    _centers.append(float(parts[0]))
                    _kappas.append(float(parts[1]))
                    break
        x_centers = np.array(_centers)
        kappa_arr = np.array(_kappas)

    elif kappa is not None:
        kappa_arr = np.full(n_windows, float(kappa))
        if centers is not None:
            if isinstance(centers, (str, Path)):
                x_centers = np.loadtxt(str(centers), comments="#").ravel()
            else:
                x_centers = np.asarray(centers, dtype=float)
            if len(x_centers) != n_windows:
                raise ValueError(
                    f"Number of centres ({len(x_centers)}) does not match "
                    f"number of COLVAR files ({n_windows})"
                )
        else:
            raise ValueError(
                "Window centres must be provided via 'centers' when using a "
                "single kappa value.  Pass a file path or array of centres."
            )
    else:
        raise ValueError("Provide either 'kappa' (single value) or 'kappa_dir'.")

    # Convert kappa to eV/CV^2 if given in kJ/mol
    if kappa_in_kj_per_mol:
        kappa_arr = kappa_arr / KJ_PER_MOL_PER_EV

    x_means = np.array([np.mean(p) for p in all_positions])
    x_vars = np.array([np.var(p, ddof=1) for p in all_positions])
    n_samples = np.array([len(p) for p in all_positions], dtype=float)

    return {
        "window_files": colvar_files,
        "kappa": kappa_arr,
        "x_centers": x_centers,
        "x_means": x_means,
        "x_vars": x_vars,
        "n_samples": n_samples,
        "all_positions": all_positions,
        "cv_name": cv_name,
    }


def compute_tau_int(
    positions: np.ndarray,
    max_lag: int = 1000,
    acf_threshold: float = 0.05,
) -> float:
    """Estimate integrated autocorrelation time using FFT.

    Uses Sokal's automatic windowing procedure: the summation is truncated
    at the first lag *M* where ``M >= C * tau_int(M)``, with ``C = 5`` by
    default.  A secondary hard cutoff is applied when the ACF drops below
    *acf_threshold* for robustness against noisy tails.
    """
    x = positions - np.mean(positions)
    n = len(x)
    if n < 4:
        return 0.5
    max_lag = min(max_lag, max(1, n // 4))

    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] == 0:
        return 0.5
    acf = acf / acf[0]
    acf = acf[:max_lag]

    # Sokal automatic windowing: stop when lag >= C * running tau_int
    sokal_c = 5.0
    tau_int = 0.5
    for lag in range(1, len(acf)):
        if acf[lag] < acf_threshold:
            break
        tau_int += acf[lag]
        if lag >= sokal_c * tau_int:
            break

    return max(tau_int, 0.5)


# ---------------------------------------------------------------------------
# Kernel functions for Squared-Exponential GP
# ---------------------------------------------------------------------------

def k_base(
    x1: np.ndarray,
    x2: np.ndarray,
    sigma_f: float,
    ell: float,
) -> np.ndarray:
    """SE covariance: Cov(f(x1), f(x2))."""
    x1, x2 = np.atleast_1d(x1)[:, None], np.atleast_1d(x2)[None, :]
    sqdist = (x1 - x2) ** 2
    return sigma_f**2 * np.exp(-0.5 * sqdist / ell**2)


def k_f_fprime(
    x_star: np.ndarray,
    x: np.ndarray,
    sigma_f: float,
    ell: float,
) -> np.ndarray:
    r"""Cross-covariance: Cov(f(x*), f'(x)) = dk/dx.

    For the SE kernel k(x*, x) = sigma_f^2 exp(-0.5*(x*-x)^2 / ell^2),
    the derivative with respect to the *second* argument x is:

        dk/dx = (x* - x) / ell^2  *  k(x*, x)

    This is the correct form when training observations are f'(x) = df/dx.
    """
    x_star = np.atleast_1d(x_star)[:, None]
    x = np.atleast_1d(x)[None, :]
    delta = x_star - x
    base = sigma_f**2 * np.exp(-0.5 * delta**2 / ell**2)
    return (delta / ell**2) * base


def k_fprime_fprime(
    x1: np.ndarray,
    x2: np.ndarray,
    sigma_f: float,
    ell: float,
) -> np.ndarray:
    r"""Derivative-derivative covariance: Cov(f'(x1), f'(x2)) = d^2k/dx1 dx2."""
    x1, x2 = np.atleast_1d(x1), np.atleast_1d(x2)
    sqdist = (x1[:, None] - x2[None, :]) ** 2
    base = sigma_f**2 * np.exp(-0.5 * sqdist / ell**2)
    return (1.0 / ell**2 - sqdist / ell**4) * base


# ---------------------------------------------------------------------------
# Marginal likelihood & LOO
# ---------------------------------------------------------------------------

def neg_log_marginal_likelihood(
    params: np.ndarray,
    x_train: np.ndarray,
    y: np.ndarray,
    errors: np.ndarray,
) -> float:
    """Negative log marginal likelihood for hyperparameter optimisation."""
    sigma_f, ell = params
    if ell <= 0.01 or sigma_f <= 0:
        return 1e10

    n = len(x_train)
    K_dd = k_fprime_fprime(x_train, x_train, sigma_f, ell)
    Sigma_y = np.diag(errors**2)
    Ky = K_dd + Sigma_y + 1e-8 * np.eye(n)

    try:
        L, low = cho_factor(Ky)
        alpha = cho_solve((L, low), y)
        data_fit = 0.5 * y.T @ alpha
        complexity = np.sum(np.log(np.diag(L)))
        constant = 0.5 * n * np.log(2 * np.pi)
        return float(data_fit + complexity + constant)
    except np.linalg.LinAlgError:
        return 1e10


def leave_one_out_cv(
    x_train: np.ndarray,
    y: np.ndarray,
    errors: np.ndarray,
    sigma_f: float,
    ell: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic leave-one-out cross-validation via matrix inverse.

    Returns (loo_means, loo_stds, loo_z).
    """
    n = len(x_train)
    K_dd = k_fprime_fprime(x_train, x_train, sigma_f, ell)
    Sigma_y = np.diag(errors**2)
    Ky = K_dd + Sigma_y + 1e-8 * np.eye(n)

    # Use Cholesky to get the inverse stably
    L, low = cho_factor(Ky)
    Ky_inv = cho_solve((L, low), np.eye(n))
    alpha = Ky_inv @ y

    diag_inv = np.diag(Ky_inv)
    # Guard against zero / negative diagonal (ill-conditioned)
    diag_inv = np.maximum(diag_inv, 1e-15)

    loo_means = y - alpha / diag_inv
    loo_vars = 1.0 / diag_inv
    loo_stds = np.sqrt(np.maximum(loo_vars, 0.0))

    loo_residuals = y - loo_means
    loo_z = loo_residuals / np.maximum(loo_stds, 1e-15)

    return loo_means, loo_stds, loo_z


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gpr_umbrella_integration(
    data_folder: str | None = None,
    colvar_dir: str | None = None,
    kappa: float | None = None,
    kappa_dir: str | None = None,
    centers: str | np.ndarray | list | None = None,
    kappa_in_kj_per_mol: bool = False,
    cv_col: int = 1,
    cv_unit: str = "nm",
    energy_unit: str = "eV",
    output_prefix: str | None = None,
    output_dir: str | None = None,
    optimize_hyperparams: bool = True,
    max_lag: int = 1000,
    acf_threshold: float = 0.05,
    n_star: int = 200,
    plot: bool = True,
    save_fig: bool = True,
    show: bool = False,
    figure_dpi: int = 150,
    figure_path: str | None = None,
    save_outputs: bool = True,
    verbose: bool = True,
):
    """Run GPR umbrella integration.

    Data can be supplied in two ways:

    1. **Preprocessed window files** -- pass *data_folder* pointing to a
       directory of ``window_*.ui_dat`` files.
    2. **Raw PLUMED COLVAR files** -- pass *colvar_dir* together with force
       constant information (*kappa* + *centers*, or *kappa_dir*).

    Units
    -----
    Internally, force constants are stored in ``energy_unit / cv_unit^2``
    (default eV/nm^2).  Set *cv_unit* and *energy_unit* to control axis
    labels and output headers.
    """
    if data_folder is not None and colvar_dir is not None:
        raise ValueError("Specify either data_folder or colvar_dir, not both.")

    if colvar_dir is not None:
        data = load_plumed_colvar_data(
            colvar_dir=colvar_dir,
            kappa=kappa,
            kappa_dir=kappa_dir,
            centers=centers,
            kappa_in_kj_per_mol=kappa_in_kj_per_mol,
            cv_col=cv_col,
        )
        base_dir = os.path.dirname(os.path.abspath(colvar_dir).rstrip("/"))
    elif data_folder is not None:
        data_folder = str(data_folder)
        if not os.path.isdir(data_folder):
            raise ValueError(f"data_folder does not exist: {data_folder}")
        data = load_window_data(data_folder)
        base_dir = os.path.dirname(os.path.abspath(data_folder).rstrip("/"))
    else:
        raise ValueError("Must provide either data_folder or colvar_dir.")

    if output_prefix is None:
        output_prefix = os.path.basename(base_dir) if base_dir else "gpr"

    if output_dir is None:
        output_dir = base_dir or "."

    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    deriv_unit = f"{energy_unit}/{cv_unit}"
    kappa_unit_label = f"{energy_unit}/{cv_unit}\u00b2"

    step = 0

    if verbose:
        print("=" * 80)
        print("GPR UMBRELLA INTEGRATION ANALYSIS")
        if colvar_dir is not None:
            print(f"COLVAR directory: {colvar_dir}")
        else:
            print(f"Data folder: {data_folder}")
        print("=" * 80)

    kappa_vals = data["kappa"]
    x_centers = data["x_centers"]
    x_means = data["x_means"]
    x_vars = data["x_vars"]
    n_samples = data["n_samples"]
    all_positions = data["all_positions"]
    cv_name = data.get("cv_name")

    step += 1
    if verbose:
        print(f"\n{step}. DATA LOADED")
        print(f"   Number of windows: {len(kappa_vals)}")
        if cv_name:
            print(f"   CV name (from COLVAR header): {cv_name}")
        print(f"   Reaction coordinate range: {x_centers.min():.4f} to {x_centers.max():.4f} {cv_unit}")
        print(f"   Mean samples per window: {n_samples.mean():.0f}")
        print(f"   Mean force constant: {kappa_vals.mean():.4f} {kappa_unit_label}")

    tau_ints = np.array([
        compute_tau_int(pos, max_lag=max_lag, acf_threshold=acf_threshold)
        for pos in all_positions
    ])
    n_eff = n_samples / (2 * tau_ints)

    step += 1
    if verbose:
        print(f"\n{step}. AUTOCORRELATION ESTIMATED")
        print(f"   Mean tau_int: {tau_ints.mean():.1f}")
        print(f"   tau_int range: {tau_ints.min():.1f} to {tau_ints.max():.1f}")
        print(f"   Mean N_eff: {n_eff.mean():.0f}")
        print(f"   Effective autocorr_factor: {(1 / (2 * tau_ints.mean())):.4f}")

    # kappa is already in energy_unit / cv_unit^2
    derivatives = kappa_vals * (x_centers - x_means)
    derivative_errors_stat = kappa_vals * np.sqrt(x_vars / n_eff)

    step += 1
    if verbose:
        print(f"\n{step}. DERIVATIVES COMPUTED")
        print(f"   Mean derivative: {derivatives.mean():.4f} {deriv_unit}")
        print(f"   Derivative std: {derivatives.std():.4f} {deriv_unit}")
        print(f"   Mean statistical error: {derivative_errors_stat.mean():.4f} {deriv_unit}")

    if len(x_centers) > 1:
        window_spacing = np.diff(np.sort(x_centers)).mean()
    else:
        window_spacing = 0.1

    # Adaptive bounds based on data range
    cv_range = x_centers.max() - x_centers.min() if len(x_centers) > 1 else 1.0
    ell_upper = max(3.0 * cv_range, 5.0)

    ell_init = max(2 * window_spacing, 0.1)
    deriv_std_val = derivatives.std()
    if deriv_std_val <= 0:
        deriv_std_val = 1e-6
    sigma_f_init = ell_init * deriv_std_val

    if optimize_hyperparams:
        bounds = [(0.001, 100.0), (0.02, ell_upper)]

        # Multi-start optimisation for robustness
        best_nll = np.inf
        best_params = (sigma_f_init, ell_init)

        starting_points = [
            [sigma_f_init, ell_init],
            [0.5 * sigma_f_init, 0.5 * ell_init],
            [2.0 * sigma_f_init, 2.0 * ell_init],
            [sigma_f_init, 0.5 * cv_range],
            [0.1, 1.0],
        ]

        for x0 in starting_points:
            # Clamp to bounds
            x0 = [
                np.clip(x0[0], bounds[0][0], bounds[0][1]),
                np.clip(x0[1], bounds[1][0], bounds[1][1]),
            ]
            result = minimize(
                neg_log_marginal_likelihood,
                x0=x0,
                args=(x_centers, derivatives, derivative_errors_stat),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if result.success and result.fun < best_nll:
                best_nll = result.fun
                best_params = tuple(result.x)

        if best_nll < np.inf:
            sigma_f_opt, ell_opt = best_params
        else:
            sigma_f_opt, ell_opt = sigma_f_init, ell_init
            warnings.warn(
                "Hyperparameter optimisation failed for all starting points; "
                "falling back to initial estimates.  Results may be unreliable.",
                stacklevel=2,
            )
            if verbose:
                print(f"\n   WARNING: Hyperparameter optimisation failed, using initial estimates")
    else:
        sigma_f_opt, ell_opt = sigma_f_init, ell_init

    derivative_errors = np.where(derivative_errors_stat > 0, derivative_errors_stat, 1e-12)

    step += 1
    if verbose:
        print(f"\n{step}. HYPERPARAMETERS")
        print(f"   Initial sigma_f: {sigma_f_init:.4f} {energy_unit}")
        print(f"   Initial ell: {ell_init:.3f} {cv_unit}")
        print(f"   Optimised sigma_f: {sigma_f_opt:.4f} {energy_unit}")
        print(f"   Optimised ell: {ell_opt:.3f} {cv_unit}")
        print(f"   Mean derivative error: {derivative_errors.mean():.4f} {deriv_unit}")

    x_train = x_centers
    y = derivatives

    K_dd = k_fprime_fprime(x_train, x_train, sigma_f_opt, ell_opt)
    Sigma_y = np.diag(derivative_errors**2)
    Ky = K_dd + Sigma_y + 1e-8 * np.eye(len(x_train))

    L_chol, low = cho_factor(Ky)
    alpha = cho_solve((L_chol, low), y)

    x_star = np.linspace(x_centers.min(), x_centers.max(), n_star)

    K_star_d = k_f_fprime(x_star, x_train, sigma_f_opt, ell_opt)
    f_mean = K_star_d @ alpha

    K_ss = k_base(x_star, x_star, sigma_f_opt, ell_opt)
    cov_f = K_ss - K_star_d @ cho_solve((L_chol, low), K_star_d.T)
    f_std = np.sqrt(np.clip(np.diag(cov_f), 0, np.inf))

    ref_idx = 0
    cov_ref = cov_f[:, ref_idx]
    var_ref = cov_f[ref_idx, ref_idx]
    var_f = np.diag(cov_f)
    var_diff = var_f + var_ref - 2.0 * cov_ref
    std_diff = np.sqrt(np.clip(var_diff, 0, np.inf))
    f_mean_diff = f_mean - f_mean[ref_idx]

    K_dstar_dtrain = k_fprime_fprime(x_star, x_train, sigma_f_opt, ell_opt)
    deriv_mean_star = K_dstar_dtrain @ alpha

    K_dd_ss = k_fprime_fprime(x_star, x_star, sigma_f_opt, ell_opt)
    cov_deriv = K_dd_ss - K_dstar_dtrain @ cho_solve((L_chol, low), K_dstar_dtrain.T)
    deriv_std = np.sqrt(np.clip(np.diag(cov_deriv), 0, np.inf))

    step += 1
    if verbose:
        print(f"\n{step}. GPR COMPLETED")
        print(f"   Mean PMF uncertainty: {std_diff.mean():.4f} {energy_unit}")
        print(f"   Max PMF uncertainty: {std_diff.max():.4f} {energy_unit}")
        print(f"   Mean derivative uncertainty: {deriv_std.mean():.4f} {deriv_unit}")

    # Training residuals: posterior mean at training points vs observed
    deriv_pred_train = K_dd @ alpha
    residuals = y - deriv_pred_train
    std_residuals = residuals / derivative_errors

    step += 1
    if verbose:
        print(f"\n{step}. TRAINING RESIDUAL VALIDATION")
        print(f"   Std of residuals: {residuals.std():.4f} {deriv_unit}")
        print(f"   Std of standardised residuals: {std_residuals.std():.2f} (target: ~1)")
        print(f"   Max |standardised residual|: {np.abs(std_residuals).max():.2f}")
        print(f"   Percent outside \u00b12 sigma: {100 * np.mean(np.abs(std_residuals) > 2):.1f}%")

    loo_means, loo_stds, loo_z = leave_one_out_cv(
        x_train, y, derivative_errors, sigma_f_opt, ell_opt
    )

    step += 1
    if verbose:
        print(f"\n{step}. LEAVE-ONE-OUT CROSS-VALIDATION")
        print(f"   Mean LOO residual: {(y - loo_means).mean():.4f} {deriv_unit}")
        print(f"   Std of LOO standardised residuals: {loo_z.std():.2f} (target: 1.0)")
        print(f"   Max |LOO z-score|: {np.abs(loo_z).max():.2f}")
        print(f"   Percent outside \u00b12 sigma: {100 * np.mean(np.abs(loo_z) > 2):.1f}%")
        print(f"   Percent outside \u00b13 sigma: {100 * np.mean(np.abs(loo_z) > 3):.1f}%")

    results = {
        "x_centers": x_centers,
        "x_means": x_means,
        "x_vars": x_vars,
        "derivatives": derivatives,
        "derivative_errors": derivative_errors,
        "derivative_errors_stat": derivative_errors_stat,
        "kappa": kappa_vals,
        "n_samples": n_samples,
        "tau_ints": tau_ints,
        "n_eff": n_eff,
        "lengthscale": ell_opt,
        "sigma_f": sigma_f_opt,
        "sigma_f_kJ": sigma_f_opt * KJ_PER_MOL_PER_EV,
        "cv_unit": cv_unit,
        "energy_unit": energy_unit,
        "deriv_unit": deriv_unit,
        "cv_name": cv_name,
        "x_star": x_star,
        "pmf_mean": f_mean_diff,
        "pmf_std": std_diff,
        "pmf_f_std": f_std,
        "deriv_mean": deriv_mean_star,
        "deriv_std": deriv_std,
        "training_residuals": residuals,
        "training_std_residuals": std_residuals,
        "loo_means": loo_means,
        "loo_stds": loo_stds,
        "loo_z": loo_z,
    }

    fig_path = None
    if plot:
        from .plotting import plot_diagnostics

        fig = plot_diagnostics(results, output_prefix=output_prefix)
        if save_fig:
            if figure_path is None:
                figure_path = os.path.join(output_dir, f"{output_prefix}_gpr_analysis.png")
            fig.savefig(figure_path, dpi=figure_dpi, bbox_inches="tight")
            fig_path = figure_path
            step += 1
            if verbose:
                print(f"\n{step}. FIGURE SAVED: {figure_path}")
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)

    pmf_path = None
    deriv_path = None
    if save_outputs:
        pmf_data = np.column_stack([x_star, f_mean_diff, std_diff])
        deriv_data = np.column_stack([x_star, deriv_mean_star, deriv_std])

        pmf_path = os.path.join(output_dir, f"{output_prefix}_pmf_gpr.dat")
        deriv_path = os.path.join(output_dir, f"{output_prefix}_deriv_gpr.dat")

        np.savetxt(
            pmf_path, pmf_data,
            header=f"x({cv_unit}) PMF({energy_unit}) uncertainty({energy_unit})",
            fmt="%.6f",
        )
        np.savetxt(
            deriv_path, deriv_data,
            header=f"x({cv_unit}) dF/dx({deriv_unit}) uncertainty({deriv_unit})",
            fmt="%.6f",
        )

        if verbose:
            print(f"   PMF data saved: {pmf_path}")
            print(f"   Derivative data saved: {deriv_path}")

    results["figure_path"] = fig_path
    results["pmf_path"] = pmf_path
    results["deriv_path"] = deriv_path

    return results
