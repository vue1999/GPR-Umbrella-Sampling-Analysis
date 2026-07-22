"""Synthetic recovery test for the 2D gradient-observation GPR.

Builds umbrella windows on a known analytic 2D PMF (a double-Gaussian well),
draws biased samples per window, and checks that the reconstructed PMF matches
the truth to within a tolerance after aligning the additive constant.
"""
import numpy as np

from gpr_umbrella import reconstruct_pmf_2d


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
