"""Plotting for 2D GPR umbrella integration."""
from __future__ import annotations

import numpy as np

from .plotting_1d import PALETTE, apply_plot_style, _band, _annotate


def _values_at_window_means(results: dict, field: np.ndarray) -> np.ndarray:
    """Interpolate a gridded field at the GP observation locations."""
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (results["gx"], results["gy"]), np.asarray(field, dtype=float),
        bounds_error=False, fill_value=np.nan,
    )
    points = np.asarray(results["means"], dtype=float).copy()
    points[:, 0] = np.clip(points[:, 0], results["gx"][0], results["gx"][-1])
    points[:, 1] = np.clip(points[:, 1], results["gy"][0], results["gy"][-1])
    values = np.asarray(interpolator(points), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite grid values at the sampled window means")
    return values


def _window_anchored_display_policy(results: dict) -> dict:
    """Choose robust, observation-anchored colour limits for 2D fields.

    The GP is constrained by mean-force observations at the sampled window
    means, whereas large excursions near the edge of sampled support are much
    more extrapolative. The normal PMF colour range therefore contains the
    complete min--max range at the window means, with a 25% margin. The
    uncertainty panels similarly contain every uncertainty evaluated at a
    window mean. Values beyond these limits are retained and drawn with the
    warning colour. Reconstruction also intersects this normal PMF range with
    geometric sampling support to define endpoint and path validity.
    """
    pmf_at_means = _values_at_window_means(results, results["pmf"])
    reference = float(np.min(pmf_at_means))
    span = float(np.ptp(pmf_at_means))

    calibrated_at_means = _values_at_window_means(
        results, results["pmf_std_calibrated"]
    )
    typical_sigma = float(np.median(calibrated_at_means))
    if span > 1e-12:
        padding = 0.25 * span
    else:
        padding = max(2.0 * typical_sigma, 1e-9)

    uncertainty_limits = {}
    for key in ("pmf_std_raw", "pmf_std_calibrated"):
        at_means = _values_at_window_means(results, results[key])
        uncertainty_limits[key] = (0.0, max(1.25 * float(np.max(at_means)), 1e-9))

    return {
        "pmf_reference": reference,
        "pmf": np.asarray(results["pmf"], dtype=float) - reference,
        "pmf_limits": (-padding, span + padding),
        "uncertainty_limits": uncertainty_limits,
    }


def _bounded_contourf(ax, GX, GY, values, limits, cmap_name, *,
                      levels=30, warn_below=True, alpha=1.0):
    """Draw fixed-level contours and mark values beyond the limits in red."""
    import matplotlib.pyplot as plt

    lower, upper = limits
    contour_levels = np.linspace(lower, upper, levels)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_over(PALETTE["warn"])
    extend = "max"
    if warn_below:
        cmap.set_under(PALETTE["warn"])
        extend = "both"
    return ax.contourf(
        GX, GY, values, levels=contour_levels, cmap=cmap, extend=extend,
        alpha=alpha,
    )


def _bounded_contours(ax, GX, GY, values, limits, *, levels=15, alpha=0.4):
    """Draw contour lines only within the normal display interval."""
    lower, upper = limits
    line_levels = np.linspace(lower, upper, levels + 2)[1:-1]
    return ax.contour(
        GX, GY, values, levels=line_levels, colors="k", linewidths=0.3,
        alpha=alpha,
    )


def plot_pmf_2d(results: dict, output_path: str | None = None,
                show: bool = False, show_targets: bool = True):
    """Plot the reconstructed PMF plus raw and LOO-scaled uncertainties.

    Sampled window means are the GP observation locations and therefore the
    primary markers.  When *show_targets* is true, restraint targets are also
    shown as faint crosses to provide drift context.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GX, GY = results["GX"], results["GY"]
    mask = ~results["support_mask"]
    display = _window_anchored_display_policy(results)
    pmf = np.ma.masked_where(mask, display["pmf"])
    pmf_std_raw = np.ma.masked_where(mask, results["pmf_std_raw"])
    pmf_std_calibrated = np.ma.masked_where(mask, results["pmf_std_calibrated"])
    centers = results["centers"]
    means = results["means"]
    cvn, cvu = results["cv_names"], results["cv_units"]
    eu = results["energy_unit"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    cf = _bounded_contourf(
        axes[0], GX, GY, pmf, display["pmf_limits"], "viridis"
    )
    _bounded_contours(axes[0], GX, GY, pmf, display["pmf_limits"])
    _window_scatter(
        axes[0], means, centers, mean_color="white", mean_size=28,
        show_targets=show_targets,
    )
    fig.colorbar(
        cf, ax=axes[0],
        label=f"PMF − sampled-window minimum ({eu}); red = outside display range",
    )
    axes[0].set_title("2D PMF (window-anchored display)")
    axes[0].legend(loc="upper right", fontsize=8)

    cs = _bounded_contourf(
        axes[1], GX, GY, pmf_std_raw,
        display["uncertainty_limits"]["pmf_std_raw"], "magma",
        warn_below=False,
    )
    _window_scatter(
        axes[1], means, centers, mean_color="cyan", mean_size=18,
        show_targets=show_targets,
    )
    fig.colorbar(
        cs, ax=axes[1],
        label=f"PMF uncertainty ({eu}); red = above display range",
    )
    axes[1].set_title("Raw GP 1σ uncertainty")

    cs = _bounded_contourf(
        axes[2], GX, GY, pmf_std_calibrated,
        display["uncertainty_limits"]["pmf_std_calibrated"], "magma",
        warn_below=False,
    )
    _window_scatter(
        axes[2], means, centers, mean_color="cyan", mean_size=18,
        show_targets=show_targets,
    )
    fig.colorbar(
        cs, ax=axes[2],
        label=f"PMF uncertainty ({eu}); red = above display range",
    )
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


def _window_scatter(ax, means, centers=None, cmap="viridis", *,
                    mean_color=None, mean_size=18, show_targets=True):
    """Overlay GP observation locations and optional restraint targets.

    The derivative observations passed to the GP live at the sampled means,
    not at the nominal restraint targets.  Means are therefore drawn as the
    prominent circular markers.  Targets, when requested, are faint crosses
    intended only to make target-to-mean drift visible.
    """
    means = np.asarray(means)
    n = len(means)
    if show_targets and centers is not None:
        centers = np.asarray(centers)
        ax.scatter(
            centers[:, 0], centers[:, 1], marker="x", color=PALETTE["guide"],
            s=max(16, 0.8 * mean_size), linewidths=0.7, alpha=0.38,
            zorder=4, label="restraint targets",
        )
    colors = np.arange(n) if mean_color is None else mean_color
    return ax.scatter(
        means[:, 0], means[:, 1], c=colors,
        cmap=cmap if mean_color is None else None,
        s=mean_size, edgecolors="k", linewidths=0.4, zorder=5,
        label="sampled means (GP observations)",
    )


def plot_diagnostics_2d(results: dict, output_path: str | None = None,
                        output_prefix: str | None = None,
                        show: bool = False, show_targets: bool = True):
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
    display = _window_anchored_display_policy(results)
    pmf = np.ma.masked_where(mask, display["pmf"])
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
    cf = _bounded_contourf(ax, GX, GY, pmf, display["pmf_limits"], "viridis")
    _bounded_contours(ax, GX, GY, pmf, display["pmf_limits"], alpha=0.35)
    _window_scatter(ax, means, centers, show_targets=show_targets)
    fig.colorbar(
        cf, ax=ax,
        label=f"PMF − sampled-window minimum ({eu}); red = outside display range",
    )
    ax.set_title("2D PMF (window-anchored display)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax = fig.add_subplot(gs[0, 3:6])
    cs = _bounded_contourf(
        ax, GX, GY, pmf_std, display["uncertainty_limits"][uncertainty_key],
        "magma", warn_below=False,
    )
    _window_scatter(
        ax, means, centers, mean_color="cyan", mean_size=16,
        show_targets=show_targets,
    )
    fig.colorbar(cs, ax=ax, label=f"σ ({eu}); red = above display range")
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
    _window_scatter(
        ax, means, centers, mean_color="white", mean_size=18,
        show_targets=show_targets,
    )
    ax.set_title(f"Window drift  (max normalized |Δ| = {dmag.max():.2f})")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Mean-force field: the gradient observations the GP integrates.
    ax = fig.add_subplot(gs[1, 2:4])
    _bounded_contourf(
        ax, GX, GY, pmf, display["pmf_limits"], "viridis",
        levels=20, alpha=0.35,
    )
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
    scaled_span = np.ptp(means / ell, axis=0)
    arrow_length = 0.12 * max(float(np.max(scaled_span)), 1.0)
    arrows = unit_direction * ell * arrow_length
    quiver = ax.quiver(
        means[:, 0], means[:, 1], arrows[:, 0], arrows[:, 1], gmag,
        angles="xy", scale_units="xy", scale=1.0, cmap="plasma", width=0.006,
        alpha=0.9, label="mean-force observations",
    )
    _window_scatter(
        ax, means, centers, mean_color="white", mean_size=13,
        show_targets=show_targets,
    )
    fig.colorbar(quiver, ax=ax, label=f"|ell · grad F| ({eu})")
    ax.set_title(
        f"Mean-force direction at sampled means "
        f"(mean metric magnitude = {gmag.mean():.2f} {eu})"
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
        f"PMF display {display['pmf_limits'][0]:.3g}–"
        f"{display['pmf_limits'][1]:.3g} {eu} (window-anchored)"
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
                             show: bool = False,
                             show_targets: bool = True):
    """Plot a lowest-barrier grid path and its 1D energy profile.

    Left:   PMF contour, located minima, and the lowest-barrier path with its
            transition state marked.
    Centre: free energy along the path (relative to its visited minimum) with
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
    display = _window_anchored_display_policy(results)
    pmf = np.ma.masked_where(~results["support_mask"], display["pmf"])
    centers = results["centers"]
    means = results["means"]
    cvn, cvu = path_result["cv_names"], path_result["cv_units"]
    eu = path_result["energy_unit"]
    s = path_result["s"]
    E_rel = path_result["pmf_rel_path_min"]
    sig = path_result["sigma_from_path_min"]
    ts_s = path_result["ts_s"]
    barrier, berr = path_result["barrier"], path_result["barrier_err"]
    marginal = path_result.get("path_aligned_marginal")

    ncols = 3 if marginal is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(19 if ncols == 3 else 14, 5.6))
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)

    # --- Left: surface + path ---------------------------------------------
    ax = axes[0]
    ax.grid(False)
    cf = _bounded_contourf(ax, GX, GY, pmf, display["pmf_limits"], "viridis")
    _bounded_contours(ax, GX, GY, pmf, display["pmf_limits"])
    _window_scatter(
        ax, means, centers, mean_color="white", mean_size=18,
        show_targets=show_targets,
    )
    fig.colorbar(
        cf, ax=ax,
        label=f"PMF − sampled-window minimum ({eu}); red = outside display range",
    )
    mode = path_result.get("path_mode", "search")
    path_label = "lowest-barrier path" if mode == "search" else "selected path"
    ax.plot(path_result["x"], path_result["y"], color="black", linestyle="--",
            linewidth=1.4, dash_capstyle="round", zorder=5, label=path_label)
    reference = path_result.get("reference_path")
    if reference is not None:
        reference = np.asarray(reference)
        ax.plot(
            reference[:, 0], reference[:, 1], color=PALETTE["guide"],
            linestyle=":", linewidth=1.5, zorder=4,
            label="reference trajectory",
        )
    if path_result.get("explicit_endpoints", False):
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
    ax.scatter(*path_result["path_min_xy"], marker="v", c=PALETTE["guide"],
               edgecolors="k", s=65, zorder=7, label="path minimum")
    ax.set_xlabel(f"{cvn[0]} ({cvu[0]})")
    ax.set_ylabel(f"{cvn[1]} ({cvu[1]})")
    ax.set_title(
        "Lowest-barrier grid path" if mode == "search"
        else f"Selected path ({mode})"
    )
    ax.legend(loc="best")
    _annotate(ax, f"TS candidate: {path_result['ts_xy'][0]:.2g} {cvu[0]}, "
                  f"{path_result['ts_xy'][1]:.2g} {cvu[1]}   ·   "
                  f"{len(path_result['minima'])} minima")

    # --- Right: energy profile along the path -----------------------------
    ax = axes[1]
    sigma_label = path_result["default_uncertainty"]
    _band(ax, s, E_rel, sig, PALETTE["guide"], label=f"{sigma_label} ±1σ / ±2σ")
    ax.plot(s, E_rel, color="black", linewidth=1.8, label="PMF - path minimum")
    ax.axhline(0, color=PALETTE["guide"], linewidth=0.6, alpha=0.6)
    ax.axvline(ts_s, color=PALETTE["warn"], linestyle="--", linewidth=0.9,
               alpha=0.75)
    ax.scatter([ts_s], [barrier], marker="*", c=PALETTE["warn"],
               edgecolors="k", s=190, zorder=6, label="TS")
    ax.set_xlabel("Dimensionless metric arclength s")
    ax.set_ylabel(f"Relative free energy ({eu})")
    ax.set_title("Free energy along the grid path (path-minimum-referenced)")
    ax.legend(loc="best")
    _annotate(ax,
              f"barrier (max - min) = {barrier:.3g} ± {berr:.2g} {eu}     "
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
        f"path mode = {path_result.get('path_mode', 'search')}"
    )
    if path_result.get("path_uncertainty_weight", 0.0) > 0:
        setup += (
            "  ·  UCB weight = "
            f"{path_result['path_uncertainty_weight']:.3g}σ"
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
