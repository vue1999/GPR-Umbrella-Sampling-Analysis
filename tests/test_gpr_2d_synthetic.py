"""Synthetic recovery test for the 2D gradient-observation GPR.

Builds umbrella windows on a known analytic 2D PMF (a double-Gaussian well),
draws biased samples per window, and checks that the reconstructed PMF matches
the truth to within a tolerance after aligning the additive constant.
"""
import numpy as np

from gpr_umbrella import reconstruct_pmf_2d
from gpr_umbrella.plotting_2d import (
    _window_anchored_display_policy,
    plot_diagnostics_2d,
    plot_lowest_barrier_path,
    plot_pmf_2d,
)


def true_pmf(x, y):
    """Two basins separated by a barrier; smooth and well-behaved."""
    return (-1.5 * np.exp(-((x + 1.0) ** 2 + (y + 0.5) ** 2) / 0.8)
            - 1.2 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) / 0.8))


def true_grad(x, y):
    g1 = -1.5 * np.exp(-((x + 1.0) ** 2 + (y + 0.5) ** 2) / 0.8)
    g2 = -1.2 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) / 0.8)
    dx = g1 * (-2 * (x + 1.0) / 0.8) + g2 * (-2 * (x - 1.0) / 0.8)
    dy = g1 * (-2 * (y + 0.5) / 0.8) + g2 * (-2 * (y - 0.6) / 0.8)
    return np.array([dx, dy])


def build_synthetic_data(n_per_side=6, n_samples=4000, kappa=(20.0, 20.0),
                         seed=0):
    rng = np.random.default_rng(seed)
    cx = np.linspace(-2.2, 2.2, n_per_side)
    cy = np.linspace(-2.0, 2.0, n_per_side)
    kappa = np.asarray(kappa, dtype=float)

    centers, kappas, positions = [], [], []
    for x0 in cx:
        for y0 in cy:
            c = np.array([x0, y0])
            # Sample around the biased minimum: with a stiff harmonic bias the
            # window distribution is ~ Gaussian centred near c, shifted by the
            # local PMF gradient (linear-response): mean ~= c - grad/kappa.
            shift = true_grad(x0, y0) / kappa
            mean = c - shift
            cov = np.diag(1.0 / kappa)        # width set by the restraint
            samp = rng.multivariate_normal(mean, cov, size=n_samples)
            centers.append(c)
            kappas.append(kappa.copy())
            positions.append(samp)

    return {
        "window_files": [f"w{i}" for i in range(len(centers))],
        "centers": np.array(centers),
        "kappa": np.array(kappas),
        "means": np.array([p.mean(axis=0) for p in positions]),
        "vars": np.array([p.var(axis=0, ddof=1) for p in positions]),
        "n_samples": np.array([len(p) for p in positions], dtype=float),
        "all_positions": positions,
    }


def test_2d_pmf_recovery():
    data = build_synthetic_data()
    res = reconstruct_pmf_2d(
        data=data, plot=False, plot_diagnostics=False, save_outputs=False,
        verbose=False, cv_names=("x", "y"), cv_units=("u", "u"),
    )
    GX, GY, pmf = res["GX"], res["GY"], res["pmf"]

    truth = true_pmf(GX, GY)
    truth = truth - truth.min()
    pred = pmf - pmf.min()

    # Compare over the interior (avoid extrapolation at grid edges).
    sl = (slice(5, -5), slice(5, -5))
    rmse = np.sqrt(np.mean((pred[sl] - truth[sl]) ** 2))
    assert rmse < 0.15, f"2D PMF RMSE too high: {rmse:.3f}"


def test_barrier_location():
    """The reconstructed barrier (max along the basin-connecting line) sits
    near the analytic saddle at x~0."""
    data = build_synthetic_data()
    res = reconstruct_pmf_2d(
        data=data, plot=False, plot_diagnostics=False, save_outputs=False,
        verbose=False,
    )
    gx, gy, pmf = res["gx"], res["gy"], res["pmf"]
    j0 = np.argmin(np.abs(gy - 0.0))
    line = pmf[:, j0]
    # The inter-basin saddle is a local max *between* the wells (x~-1 and x~+1),
    # not the global max (which sits on the far-field plateau at the edges).
    interior = np.abs(gx) < 0.9
    x_barrier = gx[interior][np.argmax(line[interior])]
    assert abs(x_barrier) < 0.6, f"barrier x={x_barrier:.2f} far from saddle"


def test_lowest_barrier_path_recovers_saddle(tmp_path):
    """The minimax path connects the wells through the analytic saddle (x~0),
    recovers a sensible barrier, and writes both the data file and figure."""
    data = build_synthetic_data()
    res = reconstruct_pmf_2d(
        data=data, output_dir=str(tmp_path), output_prefix="path",
        support_radius=1.0,
        plot=True, plot_diagnostics=False, find_lowest_barrier=True,
        path_aligned_marginal=True, thermal_energy=0.15,
        perpendicular_points=31,
        save_outputs=True, verbose=False, cv_names=("x", "y"),
    )
    path = res["lowest_barrier_path"]
    assert abs(path["ts_xy"][0]) < 0.5, f"TS x={path['ts_xy'][0]:.2f} off saddle"
    assert 0.7 < path["barrier"] < 1.2, f"barrier {path['barrier']:.2f} eV"
    assert (tmp_path / "path_pmf_2d.dat").exists()
    assert (tmp_path / "path_pmf_2d.png").exists()
    assert (tmp_path / "path_lowest_barrier_path.dat").exists()
    assert (tmp_path / "path_lowest_barrier_path.png").exists()
    assert (tmp_path / "path_path_aligned_pmf_1d.dat").exists()
    marginal = res["path_aligned_marginal"]
    assert marginal["pmf"].min() == 0.0
    assert np.all(np.isfinite(marginal["pmf"]))
    assert np.all(np.isfinite(marginal["sigma"]))


def test_diagnostics_figure(tmp_path):
    """The 8-panel diagnostics figure is produced and written to disk."""
    data = build_synthetic_data()
    res = reconstruct_pmf_2d(
        data=data, output_dir=str(tmp_path), output_prefix="diag",
        plot=False, plot_diagnostics=True, save_outputs=False, verbose=False,
    )
    diag = tmp_path / "diag_diagnostics_2d.png"
    assert res["diagnostics_path"] == str(diag)
    assert diag.exists() and diag.stat().st_size > 0
    # The diagnostics rely on these fields being exported from the results dict.
    for key in ("vars", "all_positions", "grad", "tau", "loo_z"):
        assert key in res


def _scatter_offsets(ax, label):
    matches = [artist for artist in ax.collections
               if artist.get_label() == label]
    assert len(matches) == 1
    return np.asarray(matches[0].get_offsets())


def test_2d_display_range_is_anchored_at_sampled_means():
    """Edge excursions are flagged without changing the meaningful colour scale."""
    gx = np.array([0.0, 1.0, 2.0])
    gy = np.array([0.0, 1.0, 2.0])
    pmf = np.array([
        [0.0, 0.5, 100.0],
        [0.7, 2.0, 90.0],
        [-80.0, 70.0, 60.0],
    ])
    raw = np.array([
        [0.1, 0.2, 20.0],
        [0.2, 0.3, 18.0],
        [15.0, 16.0, 17.0],
    ])
    results = {
        "gx": gx,
        "gy": gy,
        "means": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "pmf": pmf,
        "pmf_std_raw": raw,
        "pmf_std_calibrated": 2.0 * raw,
    }

    display = _window_anchored_display_policy(results)
    lower, upper = display["pmf_limits"]
    assert lower < 0.0 < 2.0 < upper
    assert display["pmf"][0, 2] > upper
    assert display["pmf"][2, 0] < lower
    assert display["uncertainty_limits"]["pmf_std_raw"][1] < raw[0, 2]

    shifted = dict(results, pmf=pmf + 123.0)
    shifted_display = _window_anchored_display_policy(shifted)
    np.testing.assert_allclose(shifted_display["pmf"], display["pmf"])
    np.testing.assert_allclose(shifted_display["pmf_limits"], display["pmf_limits"])


def test_2d_plots_mark_gp_observations_at_sampled_means():
    """Landscape markers and force arrows use sampled means, not targets."""
    from matplotlib.quiver import Quiver

    data = build_synthetic_data(n_per_side=4, n_samples=600)
    res = reconstruct_pmf_2d(
        data=data, plot=False, plot_diagnostics=False, save_outputs=False,
        find_lowest_barrier=True, verbose=False, grid_n=(24, 24),
        optimize_hyperparams=False, fixed_sigma_f=1.0,
        fixed_lengthscale=(1.0, 1.0),
    )
    assert not np.allclose(res["means"], res["centers"])

    pmf_fig = plot_pmf_2d(res)
    for ax in pmf_fig.axes[:3]:
        np.testing.assert_allclose(
            _scatter_offsets(ax, "sampled means (GP observations)"),
            res["means"],
        )
        np.testing.assert_allclose(
            _scatter_offsets(ax, "restraint targets"), res["centers"]
        )

    pmf_without_targets = plot_pmf_2d(res, show_targets=False)
    for ax in pmf_without_targets.axes[:3]:
        np.testing.assert_allclose(
            _scatter_offsets(ax, "sampled means (GP observations)"),
            res["means"],
        )
        assert all(artist.get_label() != "restraint targets"
                   for artist in ax.collections)

    diagnostics_fig = plot_diagnostics_2d(res)
    force_ax = next(
        ax for ax in diagnostics_fig.axes
        if ax.get_title().startswith("Mean-force direction")
    )
    force_quiver = next(
        artist for artist in force_ax.collections
        if isinstance(artist, Quiver)
        and artist.get_label() == "mean-force observations"
    )
    np.testing.assert_allclose(
        np.column_stack([force_quiver.X, force_quiver.Y]), res["means"]
    )

    path_fig = plot_lowest_barrier_path(res, res["lowest_barrier_path"])
    path_ax = next(
        ax for ax in path_fig.axes if ax.get_title() == "Lowest-barrier grid path"
    )
    np.testing.assert_allclose(
        _scatter_offsets(path_ax, "sampled means (GP observations)"),
        res["means"],
    )
    path_lines = [
        line for line in path_ax.lines
        if line.get_label() == "lowest-barrier path"
    ]
    assert len(path_lines) == 1
    np.testing.assert_allclose(
        path_lines[0].get_xdata(), res["lowest_barrier_path"]["x"]
    )
    np.testing.assert_allclose(
        path_lines[0].get_ydata(), res["lowest_barrier_path"]["y"]
    )
