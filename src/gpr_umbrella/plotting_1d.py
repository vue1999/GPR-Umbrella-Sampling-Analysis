from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Colour-blind-friendly palette derived from Wong (Nat. Methods 8, 441, 2011).
PALETTE = {
    "pmf":        "#0072B2",   # blue
    "pmf_band":   "#0072B2",
    "force":      "#D55E00",   # vermillion
    "force_band": "#D55E00",
    "sampling":   "#009E73",   # green
    "residual":   "#CC79A7",   # rose
    "loo":        "#9467BD",   # purple
    "tau":        "#8C564B",   # brown
    "guide":      "#555555",   # grey for reference lines
    "warn":       "#B22222",   # firebrick for ±2σ thresholds
}


def apply_plot_style() -> None:
    """Set a consistent, publication-friendly Matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.alpha":        0.18,
        "grid.linestyle":    "-",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "axes.titleweight":  "semibold",
        "axes.titlepad":     6,
        "legend.fontsize":   8.5,
        "legend.frameon":    False,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "font.family":       "DejaVu Sans",
    })


def _band(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
          color: str, label: str | None = None) -> None:
    """Two-tone ±1σ / ±2σ uncertainty band."""
    ax.fill_between(x, mean - 2 * std, mean + 2 * std,
                    color=color, alpha=0.12, linewidth=0,
                    label=label)
    ax.fill_between(x, mean - std,     mean + std,
                    color=color, alpha=0.22, linewidth=0)


def _raw_band_outline(ax: plt.Axes, x: np.ndarray, mean: np.ndarray,
                      std_raw: np.ndarray, color: str, label: str) -> None:
    """Dashed outline of the un-calibrated ±2σ envelope (no fill)."""
    ax.plot(x, mean + 2 * std_raw, linestyle="--", color=color,
            linewidth=0.9, alpha=0.7, label=label)
    ax.plot(x, mean - 2 * std_raw, linestyle="--", color=color,
            linewidth=0.9, alpha=0.7)


def _annotate(ax: plt.Axes, text: str) -> None:
    """Centred annotation placed below the axes so it never covers the data."""
    ax.text(0.5, -0.19, text,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))


def _plot_cv_distributions(ax: plt.Axes, all_positions) -> None:
    """Per-window CV distributions as overlaid Gaussian KDEs.

    Curves are coloured by window index (viridis low→high) so
    adjacent-window overlap — the key umbrella-sampling diagnostic —
    is read directly from how much neighbouring curves intersect.
    """
    from scipy.stats import gaussian_kde

    n_w = len(all_positions)
    cmap = plt.get_cmap("viridis")
    lo = min(p.min() for p in all_positions)
    hi = max(p.max() for p in all_positions)
    pad = 0.03 * (hi - lo) if hi > lo else 0.1
    grid = np.linspace(lo - pad, hi + pad, 400)

    for i, pos in enumerate(all_positions):
        if np.std(pos) < 1e-12:
            continue
        try:
            kde = gaussian_kde(pos)
        except (np.linalg.LinAlgError, ValueError):
            continue
        density = kde(grid)
        color = cmap(i / max(n_w - 1, 1))
        ax.fill_between(grid, 0, density, color=color, alpha=0.30, linewidth=0)
        ax.plot(grid, density, color=color, linewidth=0.7, alpha=0.85)


def _bar_with_ref_lines(ax: plt.Axes, values: np.ndarray, color: str,
                        ref_lines=(-2, 0, 2)) -> None:
    """Per-window bar plot with ±2σ reference lines."""
    idx = np.arange(len(values))
    ax.bar(idx, values, color=color, edgecolor="white", linewidth=0.4)
    for ref in ref_lines:
        ax.axhline(ref,
                   color=PALETTE["warn"] if ref else "black",
                   linestyle="--" if ref else "-",
                   linewidth=0.7 if ref else 0.8,
                   alpha=0.75 if ref else 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_diagnostics(results: dict, output_prefix: str | None = None) -> plt.Figure:
    """Diagnostic figure for GPR umbrella integration results.

    Layout (8 panels in 3 rows: 2 + 3 + 3):

        Row 1 (large):  PMF | Mean force
        Row 2 (small):  Sampling check | CV distribution | Autocorrelation τ_int
        Row 3 (small):  Training residuals | LOO z per window | LOO z histogram

    When ``uncertainty_calibration_factor`` is set the top panels show
    the calibrated ±σ band as a filled region and the raw GP ±2σ as a
    dashed outline so the calibration impact is visible directly.
    """
    apply_plot_style()

    cv_unit     = results.get("cv_unit", "nm")
    energy_unit = results.get("energy_unit", "eV")
    deriv_unit  = results.get("deriv_unit", f"{energy_unit}/{cv_unit}")
    kappa_unit  = f"{energy_unit}/{cv_unit}²"

    x_star          = results["x_star"]
    f_mean_diff     = results["pmf_mean"]
    std_diff        = results["pmf_std"]
    deriv_mean_star = results["deriv_mean"]
    deriv_std       = results["deriv_std"]
    x_train         = results["x_means"]
    y               = results["derivatives"]
    derivative_errors = results["derivative_errors"]
    x_means         = results["x_means"]
    x_centers       = results["x_centers"]
    x_vars          = results["x_vars"]
    n_samples       = results["n_samples"]
    std_residuals   = results["training_std_residuals"]
    loo_z           = results["loo_z"]
    tau_ints        = results["tau_ints"]
    all_positions   = results.get("all_positions")
    cal             = results.get("uncertainty_calibration_factor")
    std_diff_raw    = results.get("pmf_std_raw")
    deriv_std_raw   = results.get("deriv_std_raw")

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, top=0.91, bottom=0.05,
                          left=0.06, right=0.985,
                          hspace=0.42, wspace=0.9,
                          height_ratios=[1.6, 1.0, 1.0])

    # ------------------------------------------------------------------
    # Row 1 — main physical results (large)
    # ------------------------------------------------------------------
    if cal is None:
        band_label = "GP ±1σ / ±2σ"
    else:
        band_label = f"calibrated ±1σ / ±2σ  (= GP σ × {cal:.2f})"

    # PMF ---------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:3])
    _band(ax, x_star, f_mean_diff, std_diff, PALETTE["pmf_band"], label=band_label)
    if cal is not None and std_diff_raw is not None:
        _raw_band_outline(ax, x_star, f_mean_diff, std_diff_raw,
                          PALETTE["pmf"], label="GP ±2σ (uncalibrated)")
    ax.plot(x_star, f_mean_diff, color=PALETTE["pmf"], linewidth=2.0,
            label="GP posterior mean")
    ax.set_xlabel(f"Reaction coordinate ({cv_unit})")
    ax.set_ylabel(f"ΔF ({energy_unit})")
    ax.set_title("Free-energy profile")
    ax.legend(loc="best")
    if cal is None:
        ann = f"max GP σ = {std_diff.max():.3g} {energy_unit}"
    else:
        ann = (f"max calibrated σ = {std_diff.max():.3g} {energy_unit}  "
               f"   max GP σ = {std_diff_raw.max():.3g} {energy_unit}")
    _annotate(ax, ann)

    # Mean force --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3:6])
    _band(ax, x_star, deriv_mean_star, deriv_std, PALETTE["force_band"],
          label=band_label)
    if cal is not None and deriv_std_raw is not None:
        _raw_band_outline(ax, x_star, deriv_mean_star, deriv_std_raw,
                          PALETTE["force"], label="GP ±2σ (uncalibrated)")
    ax.plot(x_star, deriv_mean_star, color=PALETTE["force"], linewidth=2.0,
            label="GP posterior mean")
    ax.errorbar(x_train, y, yerr=2 * derivative_errors,
                fmt="o", markersize=3.5, markerfacecolor="white",
                markeredgecolor="#222", markeredgewidth=0.7,
                ecolor="#222", elinewidth=0.7, capsize=0,
                alpha=0.9, label="UI estimates ±2σ")
    ax.axhline(0, color=PALETTE["guide"], linewidth=0.6, alpha=0.6)
    ax.set_xlabel(f"Reaction coordinate ({cv_unit})")
    ax.set_ylabel(f"dF/dx ({deriv_unit})")
    ax.set_title("Mean force")
    ax.legend(loc="best")
    _annotate(ax,
              f"σ_f = {results['sigma_f']:.3g} {energy_unit}     "
              f"ℓ = {results['lengthscale']:.3g} {cv_unit}")

    # ------------------------------------------------------------------
    # Row 2 — sampling diagnostics (small)
    # ------------------------------------------------------------------

    # Window sampling check --------------------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    sample_se = np.sqrt(x_vars / n_samples)
    rc_lo, rc_hi = x_centers.min(), x_centers.max()
    pad = 0.03 * (rc_hi - rc_lo) if rc_hi > rc_lo else 0.1
    ref = np.array([rc_lo - pad, rc_hi + pad])
    ax.plot(ref, ref, linestyle="--", color=PALETTE["guide"],
            linewidth=0.9, alpha=0.7, label="x = centre")
    ax.errorbar(x_centers, x_means, yerr=2 * sample_se, fmt="o",
                markersize=3.5, color=PALETTE["sampling"],
                ecolor=PALETTE["sampling"], elinewidth=0.8, capsize=0,
                alpha=0.9, label="⟨x⟩ ± 2 SE")
    ax.set_xlabel(f"Window centre ({cv_unit})")
    ax.set_ylabel(f"Mean sampled position ({cv_unit})")
    ax.set_title("Window sampling check")
    ax.legend(loc="best")

    # Per-window CV distribution ---------------------------------------
    ax = fig.add_subplot(gs[1, 2:4])
    if all_positions is not None and len(all_positions) > 0:
        _plot_cv_distributions(ax, all_positions)
    ax.set_xlabel(f"Reaction coordinate ({cv_unit})")
    ax.set_ylabel("Sample density")
    ax.set_title("Per-window CV distribution")
    ax.set_yticks([])

    # Autocorrelation time ---------------------------------------------
    ax = fig.add_subplot(gs[1, 4:6])
    idx = np.arange(len(x_train))
    ax.bar(idx, tau_ints, color=PALETTE["tau"], edgecolor="white", linewidth=0.4)
    ax.axhline(tau_ints.mean(), color=PALETTE["guide"], linestyle="--",
               linewidth=0.9, label=f"mean = {tau_ints.mean():.1f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax.set_title("Autocorrelation time")
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ------------------------------------------------------------------
    # Row 3 — model-fit diagnostics (small)
    # ------------------------------------------------------------------

    # Training residuals -----------------------------------------------
    ax = fig.add_subplot(gs[2, 0:2])
    _bar_with_ref_lines(ax, std_residuals, PALETTE["residual"])
    ax.set_xlabel("Window index")
    ax.set_ylabel("Standardised residual")
    ax.set_title(f"Training residuals  (std = {std_residuals.std():.2f})")

    # Per-window LOO z-scores ------------------------------------------
    ax = fig.add_subplot(gs[2, 2:4])
    _bar_with_ref_lines(ax, loo_z, PALETTE["loo"])
    n_out = int(np.sum(np.abs(loo_z) > 2))
    ax.set_xlabel("Window index")
    ax.set_ylabel("LOO z-score")
    ax.set_title(f"Per-window LOO  ({n_out} outside ±2σ)")

    # LOO z-score histogram (calibration check) ------------------------
    ax = fig.add_subplot(gs[2, 4:6])
    bins = max(8, int(np.sqrt(len(loo_z)) * 2))
    ax.hist(loo_z, bins=bins, density=True, color=PALETTE["loo"], alpha=0.55,
            edgecolor="white", linewidth=0.5, label="LOO z-scores")
    z_grid = np.linspace(min(-4, loo_z.min()), max(4, loo_z.max()), 200)
    ax.plot(z_grid, np.exp(-z_grid ** 2 / 2) / np.sqrt(2 * np.pi),
            color="black", linestyle="--", linewidth=1.2, label="N(0,1)")
    ax.set_xlabel("LOO z-score")
    ax.set_ylabel("Density")
    ax.set_title(f"Calibration check  (std = {loo_z.std():.2f})")
    ax.legend(loc="best")

    # ------------------------------------------------------------------
    # Title block
    # ------------------------------------------------------------------
    title = "GPR umbrella integration"
    if output_prefix:
        title = f"{title} — {output_prefix}"

    setup = (
        f"{len(x_centers)} windows, "
        f"reaction coordinate {x_centers.min():.3g}–{x_centers.max():.3g} {cv_unit}, "
        f"mean force constant κ = {results['kappa'].mean():.3g} {kappa_unit}"
    )
    if cal is not None:
        setup += (f"  ·  calibrated σ = GP σ × {cal:.2f}  "
                  f"(scale factor = LOO z-score std)")

    fig.text(0.5, 0.975, title, ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945, setup, ha="center", va="top",
             fontsize=9.5, color="#333333")

    return fig
