"""Plotting for 2D GPR umbrella integration."""
from __future__ import annotations

import numpy as np

from .plotting import PALETTE, apply_plot_style


def plot_pmf_2d(results: dict, output_path: str | None = None, show: bool = False):
    """Filled-contour PMF + per-point uncertainty, with window centres overlaid."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GX, GY = results["GX"], results["GY"]
    pmf, pmf_std = results["pmf"], results["pmf_std"]
    centers = results["centers"]
    cvn, cvu = results["cv_names"], results["cv_units"]
    eu = results["energy_unit"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    cf = axes[0].contourf(GX, GY, pmf, levels=30, cmap="viridis")
    axes[0].contour(GX, GY, pmf, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    axes[0].scatter(centers[:, 0], centers[:, 1], c="white", edgecolors="k",
                    s=28, label="window centres")
    fig.colorbar(cf, ax=axes[0], label=f"PMF ({eu})")
    axes[0].set_title("2D PMF (GPR umbrella integration)")
    axes[0].legend(loc="upper right", fontsize=8)

    cs = axes[1].contourf(GX, GY, pmf_std, levels=30, cmap="magma")
    axes[1].scatter(centers[:, 0], centers[:, 1], c="cyan", edgecolors="k", s=18)
    fig.colorbar(cs, ax=axes[1], label=f"PMF uncertainty ({eu})")
    axes[1].set_title("Calibrated 1-sigma uncertainty")

    for ax in axes:
        ax.set_xlabel(f"{cvn[0]} ({cvu[0]})")
        ax.set_ylabel(f"{cvn[1]} ({cvu[1]})")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _window_scatter(ax, centers, cmap="viridis"):
    """Overlay window centres coloured by index (low→high)."""
    n = len(centers)
    ax.scatter(centers[:, 0], centers[:, 1],
               c=np.arange(n), cmap=cmap, s=18,
               edgecolors="k", linewidths=0.4, zorder=5)


def plot_diagnostics_2d(results: dict, output_path: str | None = None,
                        output_prefix: str | None = None,
                        show: bool = False):
    """Eight-panel diagnostics figure for 2D GPR umbrella integration.

    Layout (3 rows):

        Row 1 (large):  2D PMF | calibrated ±1σ uncertainty
        Row 2:  window drift (centre→mean) | mean-force field | autocorrelation τ
        Row 3:  window overlap map | per-observation LOO z | LOO z histogram

    The row-2/3 panels are the umbrella-sampling health checks that the single
    PMF plot cannot show: whether windows hold their targets (drift), whether
    neighbours overlap (overlap map), how correlated the samples are (τ), and
    whether the GP uncertainties are calibrated (LOO z).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import MaxNLocator

    apply_plot_style()

    GX, GY = results["GX"], results["GY"]
    pmf, pmf_std = results["pmf"], results["pmf_std"]
    centers = results["centers"]
    means = results["means"]
    grad = results["grad"]
    variances = results["vars"]
    tau = results["tau"]
    loo_z = np.asarray(results["loo_z"]).ravel()
    cal = results.get("uncertainty_calibration_factor")
    cvn, cvu = results["cv_names"], results["cv_units"]
    eu = results["energy_unit"]
    N = len(centers)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 6, top=0.90, bottom=0.06,
                          left=0.06, right=0.985,
                          hspace=0.34, wspace=0.9,
                          height_ratios=[1.5, 1.0, 1.0])

    xlab, ylab = f"{cvn[0]} ({cvu[0]})", f"{cvn[1]} ({cvu[1]})"

    # ------------------------------------------------------------------
    # Row 1 — reconstructed surface (large)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:3])
    cf = ax.contourf(GX, GY, pmf, levels=30, cmap="viridis")
    ax.contour(GX, GY, pmf, levels=15, colors="k", linewidths=0.3, alpha=0.35)
    _window_scatter(ax, centers)
    fig.colorbar(cf, ax=ax, label=f"PMF ({eu})")
    ax.set_title("2D PMF (GPR umbrella integration)")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    ax = fig.add_subplot(gs[0, 3:6])
    cs = ax.contourf(GX, GY, pmf_std, levels=30, cmap="magma")
    ax.scatter(centers[:, 0], centers[:, 1], c="cyan", edgecolors="k",
               s=16, linewidths=0.4)
    fig.colorbar(cs, ax=ax, label=f"σ ({eu})")
    band = "calibrated 1σ" if cal else "GP 1σ"
    ax.set_title(f"Uncertainty ({band})")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    # ------------------------------------------------------------------
    # Row 2 — sampling diagnostics
    # ------------------------------------------------------------------
    # Window drift: arrow from restraint centre to mean sampled position.
    ax = fig.add_subplot(gs[1, 0:2])
    drift = means - centers
    dmag = np.linalg.norm(drift, axis=1)
    ax.quiver(centers[:, 0], centers[:, 1], drift[:, 0], drift[:, 1],
              dmag, cmap="plasma", angles="xy", scale_units="xy", scale=1.0,
              width=0.006, alpha=0.9)
    ax.scatter(centers[:, 0], centers[:, 1], c=PALETTE["guide"], s=6, zorder=5)
    ax.set_title(f"Window drift  (max |Δ| = {dmag.max():.2f} {cvu[0]})")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    # Mean-force field: the gradient observations the GP integrates.
    ax = fig.add_subplot(gs[1, 2:4])
    ax.contourf(GX, GY, pmf, levels=20, cmap="viridis", alpha=0.35)
    gmag = np.linalg.norm(grad, axis=1)
    ax.quiver(centers[:, 0], centers[:, 1], grad[:, 0], grad[:, 1],
              angles="xy", color=PALETTE["force"], width=0.006,
              alpha=0.9)
    ax.set_title(f"Mean-force ∇F  (mean |∇F| = {gmag.mean():.2f} {eu}/{cvu[0]})")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    # Autocorrelation time per window (max over the two CV components).
    ax = fig.add_subplot(gs[1, 4:6])
    tau_w = np.asarray(tau).max(axis=1) if np.ndim(tau) == 2 else np.asarray(tau)
    idx = np.arange(N)
    ax.bar(idx, tau_w, color=PALETTE["tau"], edgecolor="white", linewidth=0.4)
    ax.axhline(tau_w.mean(), color=PALETTE["guide"], linestyle="--",
               linewidth=0.9, label=f"mean = {tau_w.mean():.1f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}$ (max CV)")
    ax.set_title("Autocorrelation time")
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ------------------------------------------------------------------
    # Row 3 — model-fit diagnostics
    # ------------------------------------------------------------------
    # Overlap map: ±1σ sampling ellipses at each mean; overlapping neighbours
    # = well-connected windows (the key 2D umbrella-sampling requirement).
    ax = fig.add_subplot(gs[2, 0:2])
    sd = np.sqrt(np.clip(variances, 0, np.inf))
    cmap = plt.get_cmap("viridis")
    for i in range(N):
        ax.add_patch(Ellipse(means[i], width=2 * sd[i, 0], height=2 * sd[i, 1],
                             facecolor=cmap(i / max(N - 1, 1)), alpha=0.35,
                             edgecolor=cmap(i / max(N - 1, 1)), linewidth=0.6))
    ax.scatter(means[:, 0], means[:, 1], c="k", s=3, zorder=5)
    ax.autoscale_view()
    ax.set_title("Window overlap (±1σ sampling)")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    # Per-observation LOO z (2 per window: one per CV component).
    ax = fig.add_subplot(gs[2, 2:4])
    ax.bar(np.arange(len(loo_z)), loo_z, color=PALETTE["loo"],
           edgecolor="white", linewidth=0.4)
    for ref in (-2, 0, 2):
        ax.axhline(ref, color=PALETTE["warn"] if ref else "black",
                   linestyle="--" if ref else "-",
                   linewidth=0.7 if ref else 0.8, alpha=0.75 if ref else 1.0)
    n_out = int(np.sum(np.abs(loo_z) > 2))
    ax.set_xlabel("Gradient observation index")
    ax.set_ylabel("LOO z-score")
    ax.set_title(f"Per-observation LOO  ({n_out} outside ±2σ)")

    # LOO z histogram vs N(0,1) — the calibration check.
    ax = fig.add_subplot(gs[2, 4:6])
    bins = max(8, int(np.sqrt(len(loo_z)) * 2))
    ax.hist(loo_z, bins=bins, density=True, color=PALETTE["loo"], alpha=0.55,
            edgecolor="white", linewidth=0.5, label="LOO z-scores")
    zg = np.linspace(min(-4, loo_z.min()), max(4, loo_z.max()), 200)
    ax.plot(zg, np.exp(-zg ** 2 / 2) / np.sqrt(2 * np.pi),
            color="black", linestyle="--", linewidth=1.2, label="N(0,1)")
    ax.set_xlabel("LOO z-score")
    ax.set_ylabel("Density")
    ax.set_title(f"Calibration check  (std = {loo_z.std():.2f})")
    ax.legend(loc="best")

    # ------------------------------------------------------------------
    # Title block
    # ------------------------------------------------------------------
    title = "2D GPR umbrella integration"
    if output_prefix:
        title = f"{title} — {output_prefix}"
    ell = np.atleast_1d(results["lengthscale"])
    setup = (
        f"{N} windows  ·  σ_f = {results['sigma_f']:.3g} {eu}  ·  "
        f"ℓ = ({ell[0]:.3g}, {ell[-1]:.3g}) ({cvu[0]}, {cvu[1]})  ·  "
        f"PMF 0–{pmf.max():.3g} {eu}"
    )
    if cal:
        setup += f"  ·  calibrated σ = GP σ × {cal:.2f}"
    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.935, setup, ha="center", va="top",
             fontsize=9.5, color="#333333")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
