"""Grid search, fixed references, support, and covariance-aware path differences."""
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
    find_lowest_barrier_path,
    save_lowest_barrier_path,
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
    assert path["energy_range"] == pytest.approx(1.2)
    assert path["delta_f"] == pytest.approx(-0.2)
    np.testing.assert_allclose(path["pmf_rel_path_min"], [0.5, 1.2, 0.0, 0.3])
    assert path["sigma_from_path_min_raw"][2] == 0.0
    assert path["energy_range_err_raw"] == pytest.approx(expected_raw)
    assert path["energy_range_err_calibrated"] == pytest.approx(2.0 * expected_raw)
    assert path["energy_range_err"] == pytest.approx(2.0 * expected_raw)
    assert path["energy_range_err_raw"] < 0.25 * independent_error
    endpoint_sigma = np.sqrt(covariance[1, 1] + covariance[0, 0] - 2 * covariance[1, 0])
    assert path["endpoint_to_max"] == pytest.approx(.7)
    assert path["endpoint_to_max_err_raw"] == pytest.approx(endpoint_sigma)
    assert path["endpoint_to_max_err"] == pytest.approx(2 * endpoint_sigma)
    assert path["endpoint_to_max"] != path["energy_range"]


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
    assert path["energy_range"] == pytest.approx(
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


def test_path_output_retains_mixed_coordinate_units(tmp_path):
    path = find_lowest_barrier_path(
        _analytic_results(), endpoints=((-1., 0.), (1., 0.)))
    output = tmp_path / "path.dat"
    save_lowest_barrier_path(path, str(output))
    header = output.read_text()
    assert all(unit in header for unit in ("distance(nm)", "angle(rad)", "kJ/mol"))
