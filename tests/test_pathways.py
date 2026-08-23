"""Focused tests for lowest-barrier paths and path-aligned marginals."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from gpr_umbrella.integration_2d import (
    _k_f_grad,
    _k_grad_grad,
    posterior_covariance_2d,
)
from gpr_umbrella.pathways import (
    _minimax_path,
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


def test_minimax_tie_break_uses_the_matching_axis_step_lengths():
    """Cheap x motion must win over the equal-barrier y-detour.

    The two supported corridors have two diagonal steps apiece.  One adds two
    axis-0 (x) steps and the other adds two axis-1 (y) steps, so swapping the
    axial costs selects the opposite corridor.
    """
    support = np.array([
        [1, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
    ], dtype=bool)
    path = _minimax_path(
        np.zeros((4, 4)), (0, 0), (2, 2),
        dx_scaled=0.2, dy_scaled=2.0, support_mask=support,
    )

    axial_x = sum(a[0] != b[0] and a[1] == b[1] for a, b in zip(path, path[1:]))
    axial_y = sum(a[0] == b[0] and a[1] != b[1] for a, b in zip(path, path[1:]))
    assert (axial_x, axial_y) == (2, 0)


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


def test_gradient_alignment_is_a_secondary_objective_after_exact_bottleneck():
    pmf = np.zeros((5, 3))
    support = np.zeros_like(pmf, dtype=bool)
    support[:, 1:] = True
    gradient = np.zeros(pmf.shape + (2,))
    gradient[:, 1, 1] = 1.0
    gradient[:, 2, 0] = 1.0
    start, end = (0, 1), (4, 1)

    shortest = _minimax_path(
        pmf, start, end, 1.0, 1.0, support,
        gradient_scaled=gradient, gradient_alignment_weight=0.0,
    )
    aligned = _minimax_path(
        pmf, start, end, 1.0, 1.0, support,
        gradient_scaled=gradient, gradient_alignment_weight=1.0,
    )

    assert all(index[1] == 1 for index in shortest)
    assert any(index[1] == 2 for index in aligned)
    assert max(pmf[index] for index in shortest) == 0.0
    assert max(pmf[index] for index in aligned) == 0.0


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf, True])
def test_gradient_alignment_weight_must_be_finite_and_non_negative(weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        find_lowest_barrier_path(
            _analytic_results(),
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            gradient_alignment_weight=weight,
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
            gradient_alignment_weight=0.0,
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
        gradient_alignment_weight=0.0,
        path_aligned_marginal=True,
        thermal_energy=0.4,
        perpendicular_points=31,
    )
    marginal = path["path_aligned_marginal"]

    # A normal to the diagonal touches the square only at each corner, so A(s)
    # is undefined at those two zero-measure stations. Interior stations remain
    # well-defined and should be returned instead of failing the whole profile.
    assert len(marginal["s"]) == len(path["s"]) - 2
    assert marginal["s"][0] > path["s"][0]
    assert marginal["s"][-1] < path["s"][-1]
    assert np.all(np.isfinite(marginal["pmf"]))
    assert not np.allclose(
        [marginal["x"][0], marginal["y"][0]], path["start_xy"]
    )
    assert not np.allclose(
        [marginal["x"][-1], marginal["y"][-1]], path["end_xy"]
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


def test_requested_endpoints_move_to_deepest_supported_local_minima():
    results = _analytic_results()
    results["lengthscale"] = np.ones(2)
    results["support_radius"] = 0.5
    middle = int(np.argmin(np.abs(results["gy"])))
    results["pmf"] = np.full_like(results["pmf"], 10.0)
    results["pmf"][:, middle] = 2.0
    start_index = int(np.argmin(np.abs(results["gx"] + 0.6)))
    end_index = int(np.argmin(np.abs(results["gx"] - 0.6)))
    results["pmf"][start_index, middle] = -2.0
    results["pmf"][end_index, middle] = -1.0

    path = find_lowest_barrier_path(
        results,
        endpoints=((-1.0, 0.0), (1.0, 0.0)),
    )

    assert path["nominal_start_xy"] == (-1.0, 0.0)
    assert path["nominal_end_xy"] == (1.0, 0.0)
    assert path["start_xy"] == pytest.approx((-0.6, 0.0))
    assert path["end_xy"] == pytest.approx((0.6, 0.0))
    assert path["start_endpoint"]["selection"] == "grid_local_minimum"
    assert path["end_endpoint"]["selection"] == "grid_local_minimum"
    assert path["endpoints_adjusted"] is True
    assert path["endpoint_search_radius"] == pytest.approx(0.5)
    path_indices = [
        (int(np.argmin(np.abs(results["gx"] - x))),
         int(np.argmin(np.abs(results["gy"] - y))))
        for x, y in zip(path["x"], path["y"])
    ]
    assert all(results["support_mask"][index] for index in path_indices)

def test_window_centre_mode_requires_explicit_endpoints():
    with pytest.raises(ValueError, match="requires two explicit endpoints"):
        find_lowest_barrier_path(_analytic_results(), adjust_endpoints=False)


def test_endpoint_minimum_search_can_be_disabled_in_favor_of_window_centres(
    tmp_path,
):
    results = _analytic_results()
    results["lengthscale"] = np.ones(2)
    results["centers"] = np.array([
        [-0.8, 0.0],
        [0.0, 1.0],
        [0.8, 0.0],
    ])
    middle = int(np.argmin(np.abs(results["gy"])))
    results["pmf"] = np.full_like(results["pmf"], 10.0)
    results["pmf"][:, middle] = 2.0
    results["pmf"][2, middle] = -2.0
    results["pmf"][-3, middle] = -1.0

    path = find_lowest_barrier_path(
        results,
        endpoints=((-1.0, 0.0), (1.0, 0.0)),
        adjust_endpoints=False,
    )

    assert path["start_xy"] == pytest.approx((-0.8, 0.0))
    assert path["end_xy"] == pytest.approx((0.8, 0.0))
    assert path["start_endpoint"]["window_index"] == 0
    assert path["end_endpoint"]["window_index"] == 2
    assert path["start_endpoint"]["selection"] == "nearest_window_center"
    assert path["end_endpoint"]["selection"] == "nearest_window_center"
    assert path["endpoint_selection"] == "window_centre"
    assert path["endpoints_adjusted"] is False
    assert path["endpoint_search_radius"] is None

    output = tmp_path / "window_centres_path.dat"
    save_lowest_barrier_path(path, str(output))
    header = output.read_text()
    assert "endpoint selection = nearest restraint-window centres" in header
    assert "window indices = (0, 2)" in header
    assert "gradient weight = 1" in header
    assert "gradient alignment: median |cos|" in header


def test_window_centre_endpoint_must_be_path_valid():
    results = _analytic_results()
    results["centers"] = np.array([[-1.0, 0.0], [1.0, 0.0]])
    results["path_valid_mask"] = results["support_mask"].copy()
    start = (0, int(np.argmin(np.abs(results["gy"]))))
    results["path_valid_mask"][start] = False

    with pytest.raises(ValueError, match="window centre.*path-valid"):
        find_lowest_barrier_path(
            results,
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            adjust_endpoints=False,
        )


def _test_edge_weight(
    gradient, start, end, di, dj, dx, dy, gradient_floor, alignment_weight,
):
    distance = float(np.hypot(di * dx, dj * dy))
    if gradient is None or alignment_weight == 0.0:
        return distance
    local = 0.5 * (gradient[start] + gradient[end])
    magnitude = float(np.linalg.norm(local))
    direction = np.array([di * dx, dj * dy]) / distance
    cosine = float(np.dot(local, direction) / max(magnitude, np.finfo(float).tiny))
    sine_squared = max(0.0, 1.0 - min(1.0, cosine * cosine))
    reliability = magnitude**2 / (magnitude**2 + gradient_floor**2)
    return distance * (1.0 + alignment_weight * reliability * sine_squared)


def _independent_graph_optimum(
    pmf, support, start, end, dx, dy, gradient=None, alignment_weight=0.0,
):
    """Solve the discrete lexicographic objective via SciPy graph routines."""
    shape = pmf.shape
    node_count = pmf.size
    start_flat = np.ravel_multi_index(start, shape)
    end_flat = np.ravel_multi_index(end, shape)
    steps = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    for threshold in np.unique(pmf[support]):
        allowed = support & (pmf <= threshold)
        if not allowed[start] or not allowed[end]:
            continue
        gradient_floor = 1.0
        if gradient is not None:
            typical = float(np.median(np.linalg.norm(gradient[allowed], axis=1)))
            if typical > 0.0:
                gradient_floor = 0.1 * typical
        rows, columns, weights = [], [], []
        for i, j in np.argwhere(allowed):
            source = np.ravel_multi_index((i, j), shape)
            for di, dj in steps:
                ni, nj = i + di, j + dj
                if 0 <= ni < shape[0] and 0 <= nj < shape[1] and allowed[ni, nj]:
                    rows.append(source)
                    columns.append(np.ravel_multi_index((ni, nj), shape))
                    weights.append(_test_edge_weight(
                        gradient, (i, j), (ni, nj), di, dj, dx, dy,
                        gradient_floor, alignment_weight,
                    ))
        graph = csr_matrix((weights, (rows, columns)), shape=(node_count, node_count))
        distances = dijkstra(graph, directed=True, indices=start_flat)
        if np.isfinite(distances[end_flat]):
            return float(threshold), float(distances[end_flat]), gradient_floor
    raise AssertionError("Test support unexpectedly disconnects its endpoints")


@pytest.mark.parametrize("seed", range(12))
def test_minimax_path_matches_independent_global_graph_optimum(seed):
    """Barrier and length are globally optimal on the finite 8-neighbor graph."""
    rng = np.random.default_rng(seed)
    pmf = rng.normal(size=(4, 4))
    support = rng.random((4, 4)) > 0.25
    support[0, :] = True
    support[:, -1] = True
    start, end = (0, 0), (3, 3)
    dx, dy = 0.7, 1.3

    path = _minimax_path(pmf, start, end, dx, dy, support)
    returned_bottleneck = max(pmf[index] for index in path)
    returned_length = sum(
        np.hypot((b[0] - a[0]) * dx, (b[1] - a[1]) * dy)
        for a, b in zip(path, path[1:])
    )
    expected_bottleneck, expected_length, _ = _independent_graph_optimum(
        pmf, support, start, end, dx, dy
    )

    assert returned_bottleneck == pytest.approx(expected_bottleneck)
    assert returned_length == pytest.approx(expected_length)


@pytest.mark.parametrize("seed", range(6))
def test_gradient_weighted_path_matches_independent_global_graph_optimum(seed):
    rng = np.random.default_rng(100 + seed)
    pmf = rng.normal(size=(4, 4))
    support = rng.random((4, 4)) > 0.25
    support[0, :] = True
    support[:, -1] = True
    gradient = rng.normal(size=pmf.shape + (2,))
    start, end = (0, 0), (3, 3)
    dx, dy, weight = 0.7, 1.3, 1.0

    path = _minimax_path(
        pmf, start, end, dx, dy, support,
        gradient_scaled=gradient, gradient_alignment_weight=weight,
    )
    returned_bottleneck = max(pmf[index] for index in path)
    expected_bottleneck, expected_cost, gradient_floor = (
        _independent_graph_optimum(
            pmf, support, start, end, dx, dy,
            gradient=gradient, alignment_weight=weight,
        )
    )
    returned_cost = sum(
        _test_edge_weight(
            gradient, a, b, b[0] - a[0], b[1] - a[1], dx, dy,
            gradient_floor, weight,
        )
        for a, b in zip(path, path[1:])
    )

    assert returned_bottleneck == pytest.approx(expected_bottleneck)
    assert returned_cost == pytest.approx(expected_cost)


def test_path_avoids_cells_excluded_by_the_path_valid_mask():
    results = _analytic_results()
    middle = int(np.argmin(np.abs(results["gy"])))
    blocked = (len(results["gx"]) // 2, middle)
    results["path_valid_mask"] = results["support_mask"].copy()
    results["path_valid_mask"][blocked] = False

    path = find_lowest_barrier_path(
        results,
        endpoints=((-1.0, 0.0), (1.0, 0.0)),
        adjust_endpoints=False,
    )
    path_indices = [
        (int(np.argmin(np.abs(results["gx"] - x))),
         int(np.argmin(np.abs(results["gy"] - y))))
        for x, y in zip(path["x"], path["y"])
    ]
    assert blocked not in path_indices
    assert all(results["path_valid_mask"][index] for index in path_indices)


def test_disconnected_endpoint_neighborhoods_fail_with_actionable_error():
    results = _analytic_results()
    results["lengthscale"] = np.ones(2)
    support = np.zeros_like(results["pmf"], dtype=bool)
    support[:4, :] = True
    support[7:, :] = True
    results["support_mask"] = support

    with pytest.raises(ValueError, match="same connected path-valid"):
        find_lowest_barrier_path(
            results,
            endpoints=((-1.0, 0.0), (1.0, 0.0)),
            endpoint_search_radius=0.25,
        )
