from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def plot_diagnostics(results: dict, output_prefix: str | None = None):
    apply_plot_style()

    cv_unit = results.get("cv_unit", "nm")
    energy_unit = results.get("energy_unit", "eV")
    deriv_unit = results.get("deriv_unit", f"{energy_unit}/{cv_unit}")
    kappa_unit_label = f"{energy_unit}/{cv_unit}\u00b2"

    x_star = results["x_star"]
    f_mean_diff = results["pmf_mean"]
    std_diff = results["pmf_std"]
    deriv_mean_star = results["deriv_mean"]
    deriv_std = results["deriv_std"]
    x_train = results["x_centers"]
    y = results["derivatives"]
    derivative_errors = results["derivative_errors"]
    x_means = results["x_means"]
    x_vars = results["x_vars"]
    n_samples = results["n_samples"]
    std_residuals = results["training_std_residuals"]
    loo_z = results["loo_z"]
    tau_ints = results["tau_ints"]

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_star, f_mean_diff, color="#1f77b4", linewidth=2, label="GP mean: Delta F(x)")
    ax1.fill_between(x_star, f_mean_diff - 2 * std_diff, f_mean_diff + 2 * std_diff,
                     alpha=0.25, color="#1f77b4", label="plus/minus 2 sigma")
    ax1.set_xlabel(f"Reaction Coordinate ({cv_unit})")
    ax1.set_ylabel(f"Delta F ({energy_unit})")
    ax1.set_title("PMF")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_star, deriv_mean_star, color="#ff7f0e", linewidth=2, label="GP mean: dF/dx")
    ax2.fill_between(x_star, deriv_mean_star - 2 * deriv_std, deriv_mean_star + 2 * deriv_std,
                     alpha=0.25, color="#ff7f0e", label="plus/minus 2 sigma")
    ax2.errorbar(x_train, y, yerr=2 * derivative_errors, fmt="o", color="black", markersize=4,
                 alpha=0.6, label="UI estimates plus/minus 2 sigma")
    ax2.set_xlabel(f"Reaction Coordinate ({cv_unit})")
    ax2.set_ylabel(f"dF/dx ({deriv_unit})")
    ax2.set_title("Mean Force")
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.errorbar(x_train, x_means, yerr=2 * np.sqrt(x_vars / n_samples), fmt="o",
                 alpha=0.6, markersize=4, label="Mean position plus/minus 2 sigma")
    ax3.plot(x_train, x_train, linestyle="--", color="black", alpha=0.5, label="Window centers")
    ax3.set_xlabel(f"Window Center ({cv_unit})")
    ax3.set_ylabel(f"Mean Position ({cv_unit})")
    ax3.set_title("Position Sampling")
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(np.arange(len(x_train)), derivative_errors, alpha=0.7, color="#2ca02c")
    ax4.axhline(derivative_errors.mean(), color="red", linestyle="--",
                label=f"Mean: {derivative_errors.mean():.4f}")
    ax4.set_xlabel("Window Index")
    ax4.set_ylabel(f"Derivative Error ({deriv_unit})")
    ax4.set_title("Derivative Error per Window")
    ax4.legend(loc="best")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(np.arange(len(std_residuals)), std_residuals, alpha=0.7, color="#d62728")
    ax5.axhline(0, color="black", linewidth=1)
    ax5.axhline(2, color="red", linestyle="--", alpha=0.5)
    ax5.axhline(-2, color="red", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Window Index")
    ax5.set_ylabel("Standardized Residual")
    ax5.set_title(f"Training Residuals (std={std_residuals.std():.2f})")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(np.arange(len(loo_z)), loo_z, alpha=0.7, color="#9467bd")
    ax6.axhline(0, color="black", linewidth=1)
    ax6.axhline(2, color="red", linestyle="--", alpha=0.5)
    ax6.axhline(-2, color="red", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Window Index")
    ax6.set_ylabel("LOO Z-score")
    ax6.set_title(f"LOO Cross-Validation (std={loo_z.std():.2f})")

    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(loo_z, bins=15, density=True, alpha=0.6, color="#9467bd", label="LOO z-scores")
    x_norm = np.linspace(-4, 4, 100)
    ax7.plot(x_norm, 1 / np.sqrt(2 * np.pi) * np.exp(-x_norm**2 / 2), "k--", linewidth=2,
             label="N(0,1)")
    ax7.set_xlabel("Z-score")
    ax7.set_ylabel("Density")
    ax7.set_title("LOO Z-score Distribution")
    ax7.legend(loc="best")

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.bar(np.arange(len(tau_ints)), tau_ints, alpha=0.7, color="#8c564b")
    ax8.axhline(tau_ints.mean(), color="red", linestyle="--",
                label=f"Mean: {tau_ints.mean():.1f}")
    ax8.set_xlabel("Window Index")
    ax8.set_ylabel("tau_int")
    ax8.set_title("Autocorrelation Time per Window")
    ax8.legend(loc="best")

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    summary_text = (
        "ANALYSIS SUMMARY\n"
        "----------------------------\n\n"
        f"Windows: {len(x_train)}\n"
        f"RC range: {x_train.min():.3f} to {x_train.max():.3f} {cv_unit}\n"
        f"Mean kappa: {results['kappa'].mean():.4f} {kappa_unit_label}\n\n"
        f"Signal std sigma_f: {results['sigma_f']:.4f} {energy_unit}\n"
        f"Lengthscale ell: {results['lengthscale']:.4f} {cv_unit}\n\n"
        f"Mean PMF uncertainty: {std_diff.mean():.4f} {energy_unit}\n"
        f"Mean derivative uncertainty: {deriv_std.mean():.4f} {deriv_unit}\n\n"
        f"Training residual std: {std_residuals.std():.2f}\n"
        f"LOO z-score std: {loo_z.std():.2f} (target: 1.0)\n"
        f"LOO outside +/-2 sigma: {100 * np.mean(np.abs(loo_z) > 2):.1f}%\n"
    )
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f4f1ea", alpha=0.8))

    title = "GPR Umbrella Integration"
    if output_prefix:
        title = f"{title}: {output_prefix}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    return fig
