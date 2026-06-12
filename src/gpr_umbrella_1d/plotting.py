from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Colour-blind-friendly palette derived from Wong (Nat. Methods 8, 441, 2011).
PALETTE = {
    "pmf":       "#0072B2",   # blue
    "pmf_band":  "#0072B2",
    "force":     "#D55E00",   # vermillion
    "force_band":"#D55E00",
    "sampling":  "#009E73",   # green
    "deriv_err": "#56B4E9",   # sky blue
    "residual":  "#CC79A7",   # rose
    "loo":       "#9467BD",   # purple
    "tau":       "#8C564B",   # brown
    "guide":     "#555555",   # grey for reference lines
    "warn":      "#B22222",   # firebrick for ±2σ thresholds
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


def _integer_xaxis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _band(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, std: np.ndarray,
          color: str, label: str | None = None) -> None:
    """Two-tone ±1σ / ±2σ uncertainty band."""
    ax.fill_between(x, mean - 2 * std, mean + 2 * std,
                    color=color, alpha=0.12, linewidth=0,
                    label=label)
    ax.fill_between(x, mean - std,     mean + std,
                    color=color, alpha=0.22, linewidth=0)


def plot_diagnostics(results: dict, output_prefix: str | None = None) -> plt.Figure:
    """Create a diagnostic figure for GPR umbrella integration results.

    Layout (4 columns × 2 rows):

        Row 1: PMF | Mean force | Sampling deviation | Derivative error per window
        Row 2: Training residuals | LOO z-score per window | LOO z histogram | τ_int per window

    A compact statistics strip is shown above the panels.
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
    x_train         = results["x_centers"]
    y               = results["derivatives"]
    derivative_errors = results["derivative_errors"]
    x_means         = results["x_means"]
    x_vars          = results["x_vars"]
    n_samples       = results["n_samples"]
    std_residuals   = results["training_std_residuals"]
    loo_z           = results["loo_z"]
    tau_ints        = results["tau_ints"]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, top=0.88, bottom=0.07,
                          left=0.05, right=0.985,
                          hspace=0.42, wspace=0.32)

    # ------------------------------------------------------------------
    # Row 1 — main physical results
    # ------------------------------------------------------------------

    # PMF ---------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _band(ax, x_star, f_mean_diff, std_diff, PALETTE["pmf_band"], label="±1σ / ±2σ")
    ax.plot(x_star, f_mean_diff, color=PALETTE["pmf"], linewidth=1.8, label="GP posterior mean")
    ax.set_xlabel(f"Reaction coordinate ({cv_unit})")
    ax.set_ylabel(f"ΔF ({energy_unit})")
    ax.set_title("Free-energy profile")
    ax.legend(loc="best")
    # Surface the largest PMF uncertainty in-panel rather than in a stats strip.
    ax.text(0.97, 0.04,
            f"max σ[ΔF] = {std_diff.max():.3g} {energy_unit}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # Mean force --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _band(ax, x_star, deriv_mean_star, deriv_std, PALETTE["force_band"], label="±1σ / ±2σ")
    ax.plot(x_star, deriv_mean_star, color=PALETTE["force"], linewidth=1.8,
            label="GP posterior mean")
    ax.errorbar(x_train, y, yerr=2 * derivative_errors,
                fmt="o", markersize=3.2, markerfacecolor="white",
                markeredgecolor="#222", markeredgewidth=0.7,
                ecolor="#222", elinewidth=0.7, capsize=0,
                alpha=0.9, label="UI estimates ±2σ")
    ax.axhline(0, color=PALETTE["guide"], linewidth=0.6, alpha=0.6)
    ax.set_xlabel(f"Reaction coordinate ({cv_unit})")
    ax.set_ylabel(f"dF/dx ({deriv_unit})")
    ax.set_title("Mean force")
    ax.legend(loc="best")
    # GP hyperparameters live with the panel they shape.
    ax.text(0.97, 0.04,
            f"σ_f = {results['sigma_f']:.3g} {energy_unit}\n"
            f"ℓ = {results['lengthscale']:.3g} {cv_unit}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # Mean position vs window centre ----------------------------------
    # Sampling diagnostic: each point is the mean CV value sampled in a
    # window plotted against where the umbrella was centred.  Points on
    # the dashed y = x line mean the umbrella held the system at its
    # nominal centre; vertical departures show biased sampling, which is
    # what propagates into the dF/dx estimate.
    ax = fig.add_subplot(gs[0, 2])
    sample_se = np.sqrt(x_vars / n_samples)
    rc_lo, rc_hi = x_train.min(), x_train.max()
    pad = 0.03 * (rc_hi - rc_lo) if rc_hi > rc_lo else 0.1
    ref = np.array([rc_lo - pad, rc_hi + pad])
    ax.plot(ref, ref, linestyle="--", color=PALETTE["guide"],
            linewidth=0.9, alpha=0.7, label="x = window centre")
    ax.errorbar(x_train, x_means, yerr=2 * sample_se, fmt="o",
                markersize=3.8, color=PALETTE["sampling"],
                ecolor=PALETTE["sampling"], elinewidth=0.8, capsize=0,
                alpha=0.9, label=f"⟨x⟩ ± 2 SE")
    ax.set_xlabel(f"Window centre ({cv_unit})")
    ax.set_ylabel(f"Mean sampled position ({cv_unit})")
    ax.set_title("Window sampling check")
    ax.legend(loc="best")

    # Per-window derivative error --------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    idx = np.arange(len(x_train))
    ax.bar(idx, derivative_errors, color=PALETTE["deriv_err"],
           edgecolor="white", linewidth=0.4)
    ax.axhline(derivative_errors.mean(), color=PALETTE["guide"], linestyle="--",
               linewidth=0.9, label=f"mean = {derivative_errors.mean():.3g}")
    ax.set_xlabel("Window index")
    ax.set_ylabel(f"σ[dF/dx] ({deriv_unit})")
    ax.set_title("Statistical derivative error")
    ax.legend(loc="best")
    _integer_xaxis(ax)

    # ------------------------------------------------------------------
    # Row 2 — model diagnostics
    # ------------------------------------------------------------------

    # Training residuals -----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(idx, std_residuals, color=PALETTE["residual"],
           edgecolor="white", linewidth=0.4)
    for ref in (-2, 0, 2):
        ax.axhline(ref,
                   color=PALETTE["warn"] if ref else "black",
                   linestyle="--" if ref else "-",
                   linewidth=0.6 if ref else 0.8,
                   alpha=0.7 if ref else 1.0)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Standardised residual")
    ax.set_title(f"Training residuals  (std = {std_residuals.std():.2f})")
    _integer_xaxis(ax)

    # LOO z-scores -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(idx, loo_z, color=PALETTE["loo"], edgecolor="white", linewidth=0.4)
    for ref in (-2, 0, 2):
        ax.axhline(ref,
                   color=PALETTE["warn"] if ref else "black",
                   linestyle="--" if ref else "-",
                   linewidth=0.6 if ref else 0.8,
                   alpha=0.7 if ref else 1.0)
    ax.set_xlabel("Window index")
    ax.set_ylabel("LOO z-score")
    ax.set_title(f"Leave-one-out CV  (std = {loo_z.std():.2f})")
    _integer_xaxis(ax)

    # LOO z histogram --------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    bins = max(8, int(np.sqrt(len(loo_z)) * 2))
    ax.hist(loo_z, bins=bins, density=True, color=PALETTE["loo"], alpha=0.55,
            edgecolor="white", linewidth=0.5, label="LOO z-scores")
    z_grid = np.linspace(min(-4, loo_z.min()), max(4, loo_z.max()), 200)
    ax.plot(z_grid, np.exp(-z_grid ** 2 / 2) / np.sqrt(2 * np.pi),
            color="black", linestyle="--", linewidth=1.2, label="N(0,1)")
    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    ax.set_title("LOO z-score distribution")
    ax.legend(loc="best")

    # Integrated autocorrelation time ----------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(idx, tau_ints, color=PALETTE["tau"], edgecolor="white", linewidth=0.4)
    ax.axhline(tau_ints.mean(), color=PALETTE["guide"], linestyle="--",
               linewidth=0.9, label=f"mean = {tau_ints.mean():.1f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax.set_title("Integrated autocorrelation time")
    ax.legend(loc="best")
    _integer_xaxis(ax)

    # ------------------------------------------------------------------
    # Title block — basic run context only.  Quantitative stats sit on
    # the panel they describe.
    # ------------------------------------------------------------------
    title = "GPR umbrella integration"
    if output_prefix:
        title = f"{title} — {output_prefix}"

    setup = (
        f"{len(x_train)} windows, "
        f"reaction coordinate {x_train.min():.3g}–{x_train.max():.3g} {cv_unit}, "
        f"mean force constant κ = {results['kappa'].mean():.3g} {kappa_unit}"
    )
    cal = results.get("uncertainty_calibration_factor")
    if cal is not None:
        setup += f"  ·  uncertainties calibrated ×{cal:.2f}"

    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.928, setup, ha="center", va="top",
             fontsize=9.5, color="#333333")

    return fig
