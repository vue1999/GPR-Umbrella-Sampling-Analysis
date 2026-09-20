"""Synthetic recovery test for the 2D gradient-observation GPR.

Builds umbrella windows on a known analytic 2D PMF (a double-Gaussian well),
draws biased samples per window, and checks that the reconstructed PMF matches
the truth to within a tolerance after aligning the additive constant.
"""
import numpy as np
import pytest

from gpr_umbrella import reconstruct_pmf_2d, find_lowest_barrier_path
from gpr_umbrella.pathways import save_lowest_barrier_path
from gpr_umbrella.support import window_anchored_display_policy
from gpr_umbrella.plotting_2d import (
    plot_diagnostics_2d,
    plot_lowest_barrier_path,
    plot_pmf_2d,
)


# The runnable example owns the shared analytic surface and sampling generator.
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_demo_spec = spec_from_file_location(
    "synthetic_2d_demo", Path(__file__).resolve().parents[1] / "examples/run_synthetic_2d_demo.py")
_demo = module_from_spec(_demo_spec)
_demo_spec.loader.exec_module(_demo)
build_synthetic_data, true_pmf = _demo.build_synthetic_data, _demo.true_pmf


@pytest.fixture(scope="module")
def synthetic_fit(tmp_path_factory):
    return reconstruct_pmf_2d(
        data=build_synthetic_data(), support_radius=1.,
        output_dir=str(tmp_path_factory.mktemp("synthetic2d")),
        plot=True, plot_diagnostics=True, save_outputs=True, verbose=False,
        cv_names=("x", "y"), cv_units=("u", "u"))


def test_2d_pmf_and_saddle_recovery(synthetic_fit):
    res = synthetic_fit
    for key in ("pmf_path", "figure_path", "diagnostics_path", "fit_metadata_path"):
        assert Path(res[key]).stat().st_size > 0
    truth = true_pmf(res["GX"], res["GY"])
    pred = res["pmf"] - res["pmf"].min()
    truth -= truth.min()
    sl = (slice(5, -5), slice(5, -5))
    assert np.sqrt(np.mean((pred[sl] - truth[sl])**2)) < .15
    gx, gy = res["gx"], res["gy"]
    interior = np.abs(gx) < .9
    x_peak = gx[interior][np.argmax(pred[interior, np.argmin(np.abs(gy))])]
    assert abs(x_peak) < .6


def test_standalone_path_recovers_saddle_and_saves(synthetic_fit, tmp_path):
    path = find_lowest_barrier_path(synthetic_fit, endpoints=((-1., -.5), (1., .6)))
    assert abs(path["ts_xy"][0]) < .5
    assert .7 < path["energy_range"] < 1.2
    data_file, image_file = tmp_path / "path.dat", tmp_path / "path.png"
    save_lowest_barrier_path(path, str(data_file))
    plot_lowest_barrier_path(synthetic_fit, path, output_path=str(image_file))
    assert data_file.is_file() and image_file.stat().st_size > 0


def test_diagnostics_keep_all_sampling_and_model_panels(synthetic_fit, tmp_path):
    res = dict(synthetic_fit, covariance_block_size=4000, extra_noise=np.array([.1, .2]))
    output = tmp_path / "diagnostics.png"
    figure = plot_diagnostics_2d(res, output_path=str(output))
    titles = [ax.get_title() for ax in figure.axes]
    for panel in ("2D PMF", "Uncertainty", "Window drift", "Mean-force direction",
                  "Autocorrelation time", "Window overlap", "Per-observation LOO", "Calibration check"):
        assert sum(title.startswith(panel) for title in titles) == 1
    header = " ".join(text.get_text() for text in figure.texts)
    assert "4000" in header and "extra gradient SD" in header and "0.1" in header and "0.2" in header
    assert "ℓ" in header and "σ_f" in header
    assert output.stat().st_size > 0


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

    display = window_anchored_display_policy(results)
    lower, upper = display["pmf_limits"]
    assert lower < 0.0 < 2.0 < upper
    assert display["pmf"][0, 2] > upper
    assert display["pmf"][2, 0] < lower
    assert display["uncertainty_limits"]["pmf_std_raw"][1] < raw[0, 2]

    shifted = dict(results, pmf=pmf + 123.0)
    shifted_display = window_anchored_display_policy(shifted)
    np.testing.assert_allclose(shifted_display["pmf"], display["pmf"])
    np.testing.assert_allclose(shifted_display["pmf_limits"], display["pmf_limits"])


def test_2d_plots_mark_gp_observations_at_sampled_means():
    """Landscape markers and force arrows use sampled means, not targets."""
    from matplotlib.quiver import Quiver

    data = build_synthetic_data(n_per_side=4, n_samples=600)
    res = reconstruct_pmf_2d(
        data=data, plot=False, plot_diagnostics=False, save_outputs=False,
        verbose=False, grid_n=(24, 24),
        support_radius=1.0,
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

    path = find_lowest_barrier_path(res, endpoints=((-1., -.5), (1., .6)))
    path_fig = plot_lowest_barrier_path(res, path)
    path_ax = next(
        ax for ax in path_fig.axes if ax.get_title() == "Exact minimum-range grid path"
    )
    np.testing.assert_allclose(
        _scatter_offsets(path_ax, "sampled means (GP observations)"),
        res["means"],
    )
    path_lines = [
        line for line in path_ax.lines
        if line.get_label() == "minimum-range path"
    ]
    assert len(path_lines) == 1
    np.testing.assert_allclose(
        path_lines[0].get_xdata(), path["x"]
    )
    np.testing.assert_allclose(
        path_lines[0].get_ydata(), path["y"]
    )
