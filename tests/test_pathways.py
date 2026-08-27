"""Focused tests for lowest-barrier paths and path-aligned marginals."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import label

from gpr_umbrella.integration_2d import (
    _k_f_grad,
    _k_grad_grad,
    posterior_covariance_2d,
)
from gpr_umbrella.path_graph import minimum_energy_interval, minimum_range_path
from gpr_umbrella.pathways import (
    _supported_segment,
    find_lowest_barrier_path,
    save_lowest_barrier_path,
    save_path_aligned_marginal,
)


def _gp_state(training_points, gradients, *, sigma_f=3.0, lengthscale=(1.5, 1.5)):
    """Build the small derivative-observation GP state used by these tests."""
    training_points = np.asarray(training_points, dtype=float)
    gradients = np.asarray(gradients, dtype=float).reshape(-1)
    lengthscale = np.asarray(lengthscale, dtype=float)
    covariance = _k_grad_grad(
        training_points, training_points, sigma_f, lengthscale
    )
    factor, lower = cho_factor(covariance + 1e-8 * np.eye(len(gradients)))
    alpha = cho_solve((factor, lower), gradients)
    return {
        "X": training_points,
        "sigma_f": sigma_f,
        "lengthscale": lengthscale,
        "cho_factor": factor,
        "lower": lower,
        "alpha": alpha,
        "pmf_reference_index": 0,
        "pmf_reference_point": training_points[0].copy(),
        "pmf_reference_mean": 0.0,
    }


def _latent_variance_on_grid(state, gx, gy):
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    query = np.column_stack([GX.ravel(), GY.ravel()])
    cross = _k_f_grad(
        query, state["X"], state["sigma_f"], state["lengthscale"]
    )
    solved = cho_solve((state["cho_factor"], state["lower"]), cross.T)
    variance = state["sigma_f"] ** 2 - np.einsum("ij,ji->i", cross, solved)
    return np.clip(variance, 0.0, np.inf).reshape(GX.shape)


def _analytic_results():
    """A compact GP approximation to F(x,y)=0.4(x+1)+0.6y^2."""
    train_x = np.linspace(-1.0, 1.0, 7)
    train_y = np.linspace(-2.0, 2.0, 7)
    TX, TY = np.meshgrid(train_x, train_y, indexing="ij")
    training = np.column_stack([TX.ravel(), TY.ravel()])
    gradients = np.column_stack([
        np.full(len(training), 0.4),
        1.2 * training[:, 1],
    ])
    state = _gp_state(training, gradients)

    gx = np.linspace(-1.0, 1.0, 11)
    gy = np.linspace(-2.0, 2.0, 21)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    pmf = 0.4 * (GX + 1.0) + 0.6 * GY**2
    return {
        "gx": gx,
        "gy": gy,
        "centers": training.copy(),
        "pmf": pmf,
        "support_mask": np.ones_like(pmf, dtype=bool),
        "support_radius": 0.01,
        "latent_variance_raw": _latent_variance_on_grid(state, gx, gy),
        "lengthscale": state["lengthscale"].copy(),
        "loo_calibration_factor": 1.7,
        "default_uncertainty": "raw",
        "cv_names": ("distance", "angle"),
        "cv_units": ("nm", "rad"),
        "energy_unit": "kJ/mol",
        "_gp_state": state,
    }



def _brute_minimum_interval(pmf, support, start, end):
    levels = np.unique(pmf[support])
    endpoint_low = min(pmf[start], pmf[end])
    endpoint_high = max(pmf[start], pmf[end])
    candidates = []
    for lower in levels[levels <= endpoint_low]:
        for upper in levels[levels >= endpoint_high]:
            allowed = support & (pmf >= lower) & (pmf <= upper)
            components, _ = label(allowed, structure=np.ones((3, 3), dtype=np.uint8))
            if allowed[start] and components[start] == components[end]:
                candidates.append((float(upper - lower), float(upper), -float(lower)))
                break
    if not candidates:
        raise AssertionError("Test graph unexpectedly disconnects its endpoints")
    width, upper, negative_lower = min(candidates)
    return -negative_lower, upper, width


def test_minimum_range_differs_from_minimum_peak():
    pmf = np.full((3, 5), np.nan)
    support = np.zeros_like(pmf, dtype=bool)
    start, end = (1, 0), (1, 4)
    support[start] = support[end] = True
    pmf[start] = pmf[end] = 5.0
    support[0, 1:4] = True
    pmf[0, 1:4] = [-10.0, 6.0, 6.0]
    support[2, 1:4] = True
    pmf[2, 1:4] = 10.0

    path, lower, upper = minimum_range_path(
        pmf, start, end, 1.0, 1.0, support,
    )

    assert (lower, upper) == pytest.approx((5.0, 10.0))
    assert all(index[0] != 0 for index in path)
    assert max(pmf[index] for index in path) - min(pmf[index] for index in path) == 5.0


@pytest.mark.parametrize("seed", range(12))
def test_minimum_range_interval_matches_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    pmf = rng.normal(size=(4, 4))
    support = rng.random((4, 4)) > 0.3
    support[0, :] = True
    support[:, -1] = True
    start, end = (0, 0), (3, 3)

    lower, upper, allowed = minimum_energy_interval(pmf, start, end, support)
    expected_lower, expected_upper, expected_width = _brute_minimum_interval(
        pmf, support, start, end
    )

    assert upper - lower == pytest.approx(expected_width)
    assert upper == pytest.approx(expected_upper)
    assert lower == pytest.approx(expected_lower)
    assert allowed[start] and allowed[end]


def test_fixed_gradient_alignment_is_secondary_to_exact_range():
    pmf = np.zeros((5, 3))
    support = np.zeros_like(pmf, dtype=bool)
    support[:, 1:] = True
    gradient = np.zeros(pmf.shape + (2,))
    gradient[:, 1, 1] = 1.0
    gradient[:, 2, 0] = 1.0
    start, end = (0, 1), (4, 1)

    path, lower, upper = minimum_range_path(
        pmf, start, end, 1.0, 1.0, support, gradient_scaled=gradient,
    )

    assert (lower, upper) == (0.0, 0.0)
    assert any(index[1] == 2 for index in path)
    assert all(pmf[index] == 0.0 for index in path)


@pytest.mark.parametrize(
    "metric_scale",
    [(np.nan, 1.0), (np.inf, 1.0), (1.0, -np.inf)],
)
def test_path_metric_scale_must_be_finite(metric_scale):
    with pytest.raises(ValueError, match="finite positive"):
        find_lowest_barrier_path(
            _analytic_results(),
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            metric_scale=metric_scale,
        )
def test_isolated_supported_normal_point_is_not_extended_into_extrapolation():
    coordinate = np.linspace(-1.0, 1.0, 5)
    points = np.column_stack([np.zeros(5), coordinate])
    supported = np.array([False, False, True, False, False])

    with pytest.raises(ValueError, match="supported perpendicular interval"):
        _supported_segment(coordinate, points, supported)


def test_supported_segment_stops_at_an_internal_gap():
    coordinate = np.linspace(-2.0, 2.0, 5)
    points = np.column_stack([np.zeros(5), coordinate])
    supported = np.array([True, False, True, True, True])

    kept_coordinate, _ = _supported_segment(coordinate, points, supported)
    np.testing.assert_array_equal(kept_coordinate, np.array([0.0, 1.0, 2.0]))


def test_barrier_uses_path_range_and_covariance_between_extrema():
    """Barrier and uncertainty are referenced to the lowest visited point."""
    gx = np.array([0.0, 1.0, 2.0, 3.0])
    gy = np.array([0.0])
    # A remote derivative observation leaves an almost-prior, strongly
    # correlated surface over these nearby query points.
    state = _gp_state(
        [[100.0, 100.0]], [[0.0, 0.0]], sigma_f=1.0,
        lengthscale=(5.0, 5.0),
    )
    points = np.column_stack([gx, np.zeros_like(gx)])
    covariance = posterior_covariance_2d({"_gp_state": state}, points)
    results = {
        "gx": gx,
        "gy": gy,
        "pmf": np.array([[0.3], [1.0], [-0.2], [0.1]]),
        "support_mask": np.ones((4, 1), dtype=bool),
        "support_radius": 0.01,
        "latent_variance_raw": np.diag(covariance)[:, None],
        "lengthscale": np.array([5.0, 5.0]),
        "loo_calibration_factor": 2.0,
        "default_uncertainty": "calibrated",
        "cv_names": ("x", "y"),
        "cv_units": ("nm", "rad"),
        "energy_unit": "eV",
        "_gp_state": state,
    }

    path = find_lowest_barrier_path(
        results, endpoints=((0.0, 0.0), (3.0, 0.0))
    )
    expected_raw = np.sqrt(
        covariance[1, 1] + covariance[2, 2] - 2.0 * covariance[1, 2]
    )
    independent_error = np.sqrt(covariance[1, 1] + covariance[2, 2])

    assert path["path_min_index"] == 2
    assert path["path_min_xy"] == pytest.approx((2.0, 0.0))
    assert path["barrier"] == pytest.approx(1.2)
    assert path["delta_f"] == pytest.approx(-0.2)
    np.testing.assert_allclose(path["pmf_rel_path_min"], [0.5, 1.2, 0.0, 0.3])
    assert path["sigma_from_path_min_raw"][2] == 0.0
    assert path["barrier_err_raw"] == pytest.approx(expected_raw)
    assert path["barrier_err_calibrated"] == pytest.approx(2.0 * expected_raw)
    assert path["barrier_err"] == pytest.approx(2.0 * expected_raw)
    assert path["barrier_err_raw"] < 0.25 * independent_error


@pytest.mark.parametrize("thermal_energy", [None, 0.0, -0.1, np.nan])
def test_path_aligned_marginal_requires_positive_thermal_energy(thermal_energy):
    results = _analytic_results()
    with pytest.raises(ValueError, match="thermal_energy"):
        find_lowest_barrier_path(
            results,
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            metric_scale=(1.0, 1.0),
            path_aligned_marginal=True,
            thermal_energy=thermal_energy,
            perpendicular_points=11,
            perpendicular_width=2.0,
        )


@pytest.mark.parametrize("perpendicular_points", [-1, 0, 1, 2])
def test_path_aligned_marginal_requires_at_least_three_points(
    perpendicular_points,
):
    results = _analytic_results()
    with pytest.raises(ValueError, match="perpendicular_points"):
        find_lowest_barrier_path(
            results,
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            metric_scale=(1.0, 1.0),
            path_aligned_marginal=True,
            thermal_energy=0.4,
            perpendicular_points=perpendicular_points,
            perpendicular_width=2.0,
        )


@pytest.fixture(scope="module")
def analytic_marginal_path():
    return find_lowest_barrier_path(
        _analytic_results(),
        endpoints=((-1.0, 0.0), (1.0, 0.0)),
        metric_scale=(1.0, 1.0),
        path_aligned_marginal=True,
        thermal_energy=0.4,
        perpendicular_points=61,
        perpendicular_width=2.0,
    )


def _relative_marginal_profile(marginal):
    profile = np.asarray(marginal.get("pmf_rel", marginal["pmf"]), dtype=float)
    return profile - np.min(profile)


def test_harmonic_perpendicular_integration_recovers_relative_pmf(
    analytic_marginal_path,
):
    r"""For F(s,u)=V(s)+ku²/2, integrating u only adds a constant."""
    marginal = analytic_marginal_path["path_aligned_marginal"]
    profile = _relative_marginal_profile(marginal)
    if "x" in marginal:
        expected = 0.4 * (np.asarray(marginal["x"]) + 1.0)
    else:
        # metric_scale=(1,1), and this path runs horizontally from x=-1.
        expected = 0.4 * np.asarray(marginal["s"])
    expected -= expected.min()

    np.testing.assert_allclose(profile, expected, atol=2.5e-2, rtol=0.0)
    assert profile.min() == pytest.approx(0.0, abs=1e-12)
    for key in ("sigma_raw", "sigma_calibrated", "sigma"):
        values = np.asarray(marginal[key])
        assert values.shape == profile.shape
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)


def test_corner_stations_with_zero_normal_measure_are_omitted():
    results = _analytic_results()
    results["gx"] = np.linspace(-1.0, 1.0, 11)
    results["gy"] = np.linspace(-1.0, 1.0, 11)
    results["GX"], results["GY"] = np.meshgrid(
        results["gx"], results["gy"], indexing="ij"
    )
    results["pmf"] = np.zeros((11, 11))
    results["support_mask"] = np.ones((11, 11), dtype=bool)
    results["latent_variance_raw"] = _latent_variance_on_grid(
        results["_gp_state"], results["gx"], results["gy"]
    )

    path = find_lowest_barrier_path(
        results,
        endpoints=((-1.0, -1.0), (1.0, 1.0)),
        metric_scale=(1.0, 1.0),
        path_aligned_marginal=True,
        thermal_energy=0.4,
        perpendicular_points=31,
    )
    marginal = path["path_aligned_marginal"]

    # Boundary stations with a zero-measure normal interval are omitted, while
    # all remaining stations stay finite. The fixed gradient-aware tie-break
    # determines which endpoint geometry is encountered.
    assert len(marginal["s"]) < len(path["s"])
    assert np.all(np.isfinite(marginal["pmf"]))
    assert not (
        np.allclose([marginal["x"][0], marginal["y"][0]], path["start_xy"])
        and np.allclose(
            [marginal["x"][-1], marginal["y"][-1]], path["end_xy"]
        )
    )


def test_coarse_normal_sampling_resolves_one_cell_wide_support():
    results = _analytic_results()
    middle = int(np.argmin(np.abs(results["gy"])))
    results["support_mask"] = np.zeros_like(results["pmf"], dtype=bool)
    results["support_mask"][:, middle] = True

    path = find_lowest_barrier_path(
        results,
        endpoints=((-1.0, 0.0), (1.0, 0.0)),
        metric_scale=(1.0, 1.0),
        path_aligned_marginal=True,
        thermal_energy=0.4,
        perpendicular_points=3,
        perpendicular_width=2.0,
    )
    marginal = path["path_aligned_marginal"]

    # The initial samples are u=(-2, 0, 2), only one of which lies in the
    # supported grid-cell strip. Refinement must discover its finite width
    # rather than dropping every path station or adding an unsupported point.
    assert len(marginal["s"]) == len(path["s"])
    assert np.all(marginal["perpendicular_samples"] >= 2)
    assert np.all(marginal["u_min"] < 0.0)
    assert np.all(marginal["u_max"] > 0.0)
    assert np.all(np.isfinite(marginal["pmf"]))


def test_path_marginal_is_stable_to_an_underflow_sized_energy_shift(
    analytic_marginal_path,
):
    shifted = _analytic_results()
    energy_shift = 10_000.0
    shifted["pmf"] = shifted["pmf"] + energy_shift
    # Arbitrary-point PMF predictions are referenced by subtracting this GP
    # value.  Making it negative adds the same large constant everywhere and
    # would make a direct exp(-F/kBT) calculation underflow to zero.
    shifted["_gp_state"]["pmf_reference_mean"] -= energy_shift
    shifted["pmf_reference_mean"] = shifted["_gp_state"]["pmf_reference_mean"]
    shifted_path = find_lowest_barrier_path(
        shifted,
        endpoints=((-1.0, 0.0), (1.0, 0.0)),
        metric_scale=(1.0, 1.0),
        path_aligned_marginal=True,
        thermal_energy=0.4,
        perpendicular_points=61,
        perpendicular_width=2.0,
    )

    baseline = _relative_marginal_profile(
        analytic_marginal_path["path_aligned_marginal"]
    )
    shifted_profile = _relative_marginal_profile(
        shifted_path["path_aligned_marginal"]
    )
    assert np.all(np.isfinite(shifted_profile))
    np.testing.assert_allclose(shifted_profile, baseline, atol=2e-9, rtol=0.0)


def test_path_savers_keep_each_cv_unit_in_mixed_unit_headers(
    tmp_path, analytic_marginal_path,
):
    path_file = tmp_path / "case_lowest_barrier_path.dat"
    marginal_file = tmp_path / "case_path_aligned_pmf_1d.dat"
    save_lowest_barrier_path(analytic_marginal_path, str(path_file))
    save_path_aligned_marginal(
        analytic_marginal_path["path_aligned_marginal"], str(marginal_file)
    )

    for output in (path_file, marginal_file):
        header = output.read_text()
        assert "distance(nm)" in header
        assert "angle(rad)" in header
        assert "kJ/mol" in header




def test_search_requires_explicit_endpoints():
    with pytest.raises(ValueError, match="requires two explicit endpoints"):
        find_lowest_barrier_path(_analytic_results())


def test_requested_endpoints_only_snap_to_nearest_grid_cells():
    results = _analytic_results()
    path = find_lowest_barrier_path(
        results,
        endpoints=((-0.93, 0.07), (0.91, -0.08)),
    )

    assert path["nominal_start_xy"] == (-0.93, 0.07)
    assert path["nominal_end_xy"] == (0.91, -0.08)
    assert path["start_xy"] == pytest.approx((-1.0, 0.0))
    assert path["end_xy"] == pytest.approx((1.0, 0.0))
    assert path["start_endpoint"]["selection"] == "nearest_grid_cell"
    assert path["end_endpoint"]["selection"] == "nearest_grid_cell"


def test_path_avoids_cells_excluded_by_path_valid_mask():
    results = _analytic_results()
    middle = int(np.argmin(np.abs(results["gy"])))
    blocked = (len(results["gx"]) // 2, middle)
    results["path_valid_mask"] = results["support_mask"].copy()
    results["path_valid_mask"][blocked] = False

    path = find_lowest_barrier_path(
        results, endpoints=((-1.0, 0.0), (1.0, 0.0))
    )
    path_indices = [
        (int(np.argmin(np.abs(results["gx"] - x))),
         int(np.argmin(np.abs(results["gy"] - y))))
        for x, y in zip(path["x"], path["y"])
    ]
    assert blocked not in path_indices
    assert all(results["path_valid_mask"][index] for index in path_indices)


def test_invalid_and_disconnected_endpoints_fail_explicitly():
    results = _analytic_results()
    middle = int(np.argmin(np.abs(results["gy"])))
    results["path_valid_mask"] = results["support_mask"].copy()
    results["path_valid_mask"][0, middle] = False
    with pytest.raises(ValueError, match="start endpoint.*outside"):
        find_lowest_barrier_path(
            results, endpoints=((-1.0, 0.0), (1.0, 0.0))
        )

    results = _analytic_results()
    support = np.zeros_like(results["pmf"], dtype=bool)
    support[:4, :] = True
    support[7:, :] = True
    results["support_mask"] = support
    with pytest.raises(ValueError, match="disconnected path-valid"):
        find_lowest_barrier_path(
            results, endpoints=((-1.0, 0.0), (1.0, 0.0))
        )


def test_coincident_endpoint_grid_cells_are_rejected():
    with pytest.raises(ValueError, match="collapse to one grid point"):
        find_lowest_barrier_path(
            _analytic_results(), endpoints=((-1.0, 0.0), (-0.99, 0.01))
        )


def test_corridor_uses_reference_endpoints_and_stays_inside_radius():
    results = _analytic_results()
    reference = np.array([[-1.0, 0.0], [1.0, 0.0]])
    path = find_lowest_barrier_path(
        results,
        metric_scale=(1.0, 1.0),
        reference_path=reference,
        path_mode="corridor",
        corridor_radius=0.21,
    )

    assert path["path_mode"] == "corridor"
    assert path["nominal_start_xy"] == pytest.approx(reference[0])
    assert path["nominal_end_xy"] == pytest.approx(reference[-1])
    assert path["max_reference_distance"] <= 0.21 + 1e-12
    assert np.max(np.abs(path["y"])) <= 0.21 + 1e-12


def test_fixed_reference_path_is_evaluated_without_grid_search():
    results = _analytic_results()
    reference = np.array([[-1.0, 0.0], [0.0, 0.4], [1.0, 0.0]])
    path = find_lowest_barrier_path(
        results,
        reference_path=reference,
        path_mode="fixed",
        metric_scale=(1.0, 1.0),
    )

    assert path["path_mode"] == "fixed"
    assert path["path_objective"] == "fixed_reference_trajectory"
    assert path["start_xy"] == pytest.approx(tuple(reference[0]))
    assert path["end_xy"] == pytest.approx(tuple(reference[-1]))
    assert path["barrier"] == pytest.approx(
        np.max(path["pmf"]) - np.min(path["pmf"])
    )
    assert any(np.allclose([x, y], reference[1])
               for x, y in zip(path["x"], path["y"]))


def test_fixed_reference_path_must_remain_path_valid_between_vertices():
    results = _analytic_results()
    results["path_valid_mask"] = np.broadcast_to(
        np.abs(results["gy"])[None, :] < 0.3, results["pmf"].shape
    ).copy()
    with pytest.raises(ValueError, match="leaves the path-valid region"):
        find_lowest_barrier_path(
            results,
            reference_path=np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            path_mode="fixed",
            metric_scale=(1.0, 1.0),
        )


def test_reference_modes_validate_required_and_disallowed_inputs():
    results = _analytic_results()
    reference = np.array([[-1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="requires reference_path"):
        find_lowest_barrier_path(results, path_mode="fixed")
    with pytest.raises(ValueError, match="uses the reference trajectory endpoints"):
        find_lowest_barrier_path(
            results,
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            reference_path=reference,
            path_mode="corridor",
            corridor_radius=0.2,
        )
    with pytest.raises(ValueError, match="does not overlap"):
        find_lowest_barrier_path(
            results,
            reference_path=np.array([[-1.0, 20.0], [1.0, 20.0]]),
            path_mode="corridor",
            corridor_radius=0.1,
            metric_scale=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="only valid with corridor"):
        find_lowest_barrier_path(
            results,
            reference_path=reference,
            path_mode="fixed",
            corridor_radius=0.1,
        )


def test_saved_path_header_describes_fixed_objective_without_weights(tmp_path):
    path = find_lowest_barrier_path(
        _analytic_results(), endpoints=((-1.0, 0.0), (1.0, 0.0))
    )
    output = tmp_path / "path.dat"
    save_lowest_barrier_path(path, str(output))
    header = output.read_text()
    assert "minimum_pmf_range_then_fixed_gradient_aligned_cost" in header
    assert "selected energy interval" in header
    assert "gradient weight" not in header
    assert "uncertainty weight" not in header
