"""Plotting for 2D GPR umbrella integration."""
from __future__ import annotations

import numpy as np

from .plotting_1d import PALETTE, apply_plot_style, _band, _annotate


def plot_pmf_2d(results: dict, output_path: str | None = None, show: bool = False):
    """Plot the reconstructed PMF plus raw and LOO-scaled uncertainties."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GX, GY = results["GX"], results["GY"]
    mask = ~results["support_mask"]
    pmf = np.ma.masked_where(mask, results["pmf"])
    pmf_std_raw = np.ma.masked_where(mask, results["pmf_std_raw"])
    pmf_std_calibrated = np.ma.masked_where(mask, results["pmf_std_calibrated"])
    centers = results["centers"]
    cvn, cvu = results["cv_names"], results["cv_units"]
    eu = results["energy_unit"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    cf = axes[0].contourf(GX, GY, pmf, levels=30, cmap="viridis")
    axes[0].contour(GX, GY, pmf, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    axes[0].scatter(centers[:, 0], centers[:, 1], c="white", edgecolors="k",
                    s=28, label="window centres")
    fig.colorbar(cf, ax=axes[0], label=f"PMF ({eu})")
    axes[0].set_title("2D PMF (GPR umbrella integration)")
    axes[0].legend(loc="upper right", fontsize=8)

    cs = axes[1].contourf(GX, GY, pmf_std_raw, levels=30, cmap="magma")
    axes[1].scatter(centers[:, 0], centers[:, 1], c="cyan", edgecolors="k", s=18)
    fig.colorbar(cs, ax=axes[1], label=f"PMF uncertainty ({eu})")
    axes[1].set_title("Raw GP 1σ uncertainty")

    cs = axes[2].contourf(GX, GY, pmf_std_calibrated, levels=30, cmap="magma")
    axes[2].scatter(centers[:, 0], centers[:, 1], c="cyan", edgecolors="k", s=18)
    fig.colorbar(cs, ax=axes[2], label=f"PMF uncertainty ({eu})")
    factor = results["loo_calibration_factor"]
    axes[2].set_title(f"LOO-scaled 1σ uncertainty (×{factor:.2f})")

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


def _chaikin(pts, iters=2):
    """Chaikin corner-cutting for display only: round a polyline into a smooth
    curve with fixed endpoints (turns the 8-connected staircase into a line)."""
    pts = np.asarray(pts, dtype=float)
    for _ in range(iters):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(pts[-1])
        pts = np.array(out)
    return pts


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
    mask = ~results["support_mask"]
    pmf = np.ma.masked_where(mask, results["pmf"])
    uncertainty_key = f"pmf_std_{results['default_uncertainty']}"
    pmf_std = np.ma.masked_where(mask, results[uncertainty_key])
    centers = results["centers"]
    means = results["means"]
    grad = results["grad"]
    variances = results["vars"]
    tau = results["tau"]
    loo_z = np.asarray(results["loo_z"]).ravel()
    cal = results["loo_calibration_factor"]
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
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax = fig.add_subplot(gs[0, 3:6])
    cs = ax.contourf(GX, GY, pmf_std, levels=30, cmap="magma")
    ax.scatter(centers[:, 0], centers[:, 1], c="cyan", edgecolors="k",
               s=16, linewidths=0.4)
    fig.colorbar(cs, ax=ax, label=f"σ ({eu})")
    band = f"{results['default_uncertainty']} 1σ"
    ax.set_title(f"Uncertainty ({band})")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # ------------------------------------------------------------------
    # Row 2 — sampling diagnostics
    # ------------------------------------------------------------------
    # Window drift: arrow from restraint centre to mean sampled position.
    ax = fig.add_subplot(gs[1, 0:2])
    drift = means - centers
    ell = np.asarray(results["lengthscale"])
    dmag = np.linalg.norm(drift / ell, axis=1)
    ax.quiver(centers[:, 0], centers[:, 1], drift[:, 0], drift[:, 1],
              dmag, cmap="plasma", angles="xy", scale_units="xy", scale=1.0,
              width=0.006, alpha=0.9)
    ax.scatter(centers[:, 0], centers[:, 1], c=PALETTE["guide"], s=6, zorder=5)
    ax.set_title(f"Window drift  (max normalized |Δ| = {dmag.max():.2f})")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Mean-force field: the gradient observations the GP integrates.
    ax = fig.add_subplot(gs[1, 2:4])
    ax.contourf(GX, GY, pmf, levels=20, cmap="viridis", alpha=0.35)
    # dF/d(x/ell) = ell*dF/dx has one common energy unit even when the two
    # CV axes do not. Draw its direction in the dimensionless GP metric, then
    # convert the display displacement back to the corresponding CV units.
    scaled_grad = grad * ell
    gmag = np.linalg.norm(scaled_grad, axis=1)
    unit_direction = np.divide(
        scaled_grad,
        gmag[:, None],
        out=np.zeros_like(scaled_grad),
        where=gmag[:, None] > 0,
    )
    scaled_span = np.ptp(centers / ell, axis=0)
    arrow_length = 0.12 * max(float(np.max(scaled_span)), 1.0)
    arrows = unit_direction * ell * arrow_length
    quiver = ax.quiver(
        centers[:, 0], centers[:, 1], arrows[:, 0], arrows[:, 1], gmag,
        angles="xy", scale_units="xy", scale=1.0, cmap="plasma", width=0.006,
        alpha=0.9,
    )
    fig.colorbar(quiver, ax=ax, label=f"|ell · grad F| ({eu})")
    ax.set_title(
        f"Mean-force direction (mean metric magnitude = {gmag.mean():.2f} {eu})"
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

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
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

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
    setup += f"  ·  LOO-scaled σ = raw GP σ × {cal:.2f}"
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


def plot_lowest_barrier_path(results: dict, path_result: dict,
                             output_path: str | None = None,
                             output_prefix: str | None = None,
                             show: bool = False):
    """Plot a lowest-barrier grid path and its 1D energy profile.

    Left:   PMF contour, located minima, and the lowest-barrier path with its
            transition state marked.
    Centre: free energy along the path (relative to the start minimum) with
            the selected ±1σ / ±2σ uncertainty band.
    Right:  when requested, the separately referenced path-aligned marginal
            PMF obtained by perpendicular Boltzmann integration.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_plot_style()

    GX, GY = results["GX"], results["GY"]
    pmf = np.ma.masked_where(~results["support_mask"], results["pmf"])
    centers = results["centers"]
    cvn, cvu = path_result["cv_names"], path_result["cv_units"]
    eu = path_result["energy_unit"]
    s, E_rel, sig = path_result["s"], path_result["pmf_rel"], path_result["sigma"]
    ts_s = path_result["ts_s"]
    barrier, berr = path_result["barrier"], path_result["barrier_err"]
    marginal = path_result.get("path_aligned_marginal")

    ncols = 3 if marginal is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(19 if ncols == 3 else 14, 5.6))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)

    # --- Left: surface + path ---------------------------------------------
    ax = axes[0]
    ax.grid(False)
    cf = ax.contourf(GX, GY, pmf, levels=30, cmap="viridis")
    ax.contour(GX, GY, pmf, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    ax.scatter(centers[:, 0], centers[:, 1], c="white", edgecolors="k",
               s=18, linewidths=0.4, alpha=0.5, zorder=4)
    fig.colorbar(cf, ax=ax, label=f"PMF ({eu})")
    smooth = _chaikin(np.column_stack([path_result["x"], path_result["y"]]), iters=2)
    ax.plot(smooth[:, 0], smooth[:, 1], color="black", linestyle="--",
            linewidth=1.4, dash_capstyle="round", zorder=5, label="lowest-barrier path")
    if path_result.get("endpoints_adjusted", False):
        nominal = np.asarray([
            path_result["nominal_start_xy"], path_result["nominal_end_xy"]
        ])
        selected = np.asarray([
            path_result["start_xy"], path_result["end_xy"]
        ])
        for requested, relocated in zip(nominal, selected):
            ax.plot(
                [requested[0], relocated[0]], [requested[1], relocated[1]],
                color=PALETTE["guide"], linewidth=0.9, alpha=0.8, zorder=5,
            )
        ax.scatter(
            nominal[:, 0], nominal[:, 1], marker="x", c=PALETTE["guide"],
            linewidths=1.5, s=55, zorder=6, label="requested endpoints",
        )
    ax.scatter(*path_result["start_xy"], c=PALETTE["sampling"], edgecolors="k", s=70,
               zorder=6, label="start")
    ax.scatter(*path_result["end_xy"], c=PALETTE["pmf"], edgecolors="k", s=70,
               zorder=6, label="end")
    ax.scatter(*path_result["ts_xy"], marker="*", c=PALETTE["warn"], edgecolors="k",
               s=200, zorder=7, label="TS")
    ax.set_xlabel(f"{cvn[0]} ({cvu[0]})")
    ax.set_ylabel(f"{cvn[1]} ({cvu[1]})")
    ax.set_title("Lowest-barrier grid path")
    ax.legend(loc="best")
    _annotate(ax, f"TS candidate: {path_result['ts_xy'][0]:.2g} {cvu[0]}, "
                  f"{path_result['ts_xy'][1]:.2g} {cvu[1]}   ·   "
                  f"{len(path_result['minima'])} minima")

    # --- Right: energy profile along the path -----------------------------
    ax = axes[1]
    sigma_label = path_result["default_uncertainty"]
    _band(ax, s, E_rel, sig, PALETTE["guide"], label=f"{sigma_label} ±1σ / ±2σ")
    ax.plot(s, E_rel, color="black", linewidth=1.8, label="ΔF along path")
    ax.axhline(0, color=PALETTE["guide"], linewidth=0.6, alpha=0.6)
    ax.axvline(ts_s, color=PALETTE["warn"], linestyle="--", linewidth=0.9,
               alpha=0.75)
    ax.scatter([ts_s], [barrier], marker="*", c=PALETTE["warn"],
               edgecolors="k", s=190, zorder=6, label="TS")
    ax.set_xlabel("Dimensionless metric arclength s")
    ax.set_ylabel(f"Relative free energy ({eu})")
    ax.set_title("Free energy along the grid path (start-referenced)")
    ax.legend(loc="best")
    _annotate(ax,
              f"barrier = {barrier:.3g} ± {berr:.2g} {eu}     "
              f"ΔF = {path_result['delta_f']:.3g} ± "
              f"{path_result['delta_f_err']:.2g} {eu}")

    if marginal is not None:
        ax = axes[2]
        _band(
            ax,
            marginal["s"],
            marginal["pmf"],
            marginal["sigma"],
            PALETTE["pmf_band"],
            label=f"{marginal['default_uncertainty']} ±1σ / ±2σ",
        )
        ax.plot(
            marginal["s"],
            marginal["pmf"],
            color=PALETTE["pmf"],
            linewidth=1.8,
            label="A(s) - min A(s)",
        )
        ax.axhline(0, color=PALETTE["guide"], linewidth=0.6, alpha=0.6)
        ax.set_xlabel("Dimensionless metric arclength s")
        ax.set_ylabel(f"Marginal free energy ({eu})")
        ax.set_title("Perpendicular Boltzmann marginal (minimum-referenced)")
        ax.legend(loc="best")
        _annotate(
            ax,
            f"kBT = {marginal['thermal_energy']:.3g} {eu}  ·  "
            f"{int(np.median(marginal['perpendicular_samples']))} "
            "median samples/station",
        )

    # Header block (matches the diagnostics figure so the two read as a set)
    title = "2D GPR lowest-barrier path"
    if output_prefix:
        title = f"{title} — {output_prefix}"
    ell = np.atleast_1d(results["lengthscale"])
    metric = np.atleast_1d(path_result["metric_scale"])
    setup = (
        f"{len(centers)} windows  ·  σ_f = {results['sigma_f']:.3g} {eu}  ·  "
        f"ℓ_GP = ({ell[0]:.3g} {cvu[0]}, {ell[-1]:.3g} {cvu[1]})  ·  "
        f"path metric = ({metric[0]:.3g} {cvu[0]}, "
        f"{metric[-1]:.3g} {cvu[1]})  ·  "
        f"minimax path over the GP surface"
    )
    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.925, setup, ha="center", va="top",
             fontsize=9.5, color="#333333")
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
