"""Focused numerical tests for the two-dimensional GP covariance path."""

from __future__ import annotations

import numpy as np
import pytest

import gpr_umbrella.integration_2d as integration_2d

from gpr_umbrella.integration_2d import (
    _grid_support_mask,
    _gradient_noise_covariance,
    _k_f_grad,
    _k_grad_grad,
    _leave_one_window_out_z,
    _nll,
    _se,
    _with_relative_diagonal_jitter,
    estimate_mean_covariance,
    posterior_covariance_2d,
    reconstruct_pmf_2d,
)


def _small_correlated_data() -> dict:
    """Four windows with deterministic, correlated two-CV fluctuations."""
    centers = np.array([
        [-1.0, -0.8],
        [-1.0, 0.9],
        [1.1, -0.8],
        [1.1, 0.9],
    ])
    kappa = np.array([
        [3.0, 5.0],
        [4.0, 6.0],
        [5.0, 4.0],
        [6.0, 3.0],
    ])
    gradients = np.array([
        [-0.35, 0.20],
        [0.15, 0.45],
        [-0.25, -0.30],
        [0.40, -0.10],
    ])
    means = centers - gradients / kappa

    u = np.array([-1.0, 1.0, -0.4, 0.4, -1.4, 1.4, -0.2, 0.2])
    orthogonal = np.array([0.3, -0.3, -0.7, 0.7, 0.2, -0.2, 0.5, -0.5])
    base = np.column_stack([u, 0.75 * u + 0.25 * orthogonal])
    positions = [
        means[w] + (0.025 + 0.004 * w) * base
        for w in range(len(centers))
    ]
    return {
        "centers": centers,
        "kappa": kappa,
        "means": np.array([p.mean(axis=0) for p in positions]),
        "vars": np.array([p.var(axis=0, ddof=1) for p in positions]),
        "n_samples": np.array([len(p) for p in positions], dtype=float),
        "all_positions": positions,
    }


def _rescale_energy(data: dict, factor: float) -> dict:
    scaled = dict(data)
    scaled["kappa"] = np.asarray(data["kappa"]) * factor
    return scaled


def _rescale_coordinates(data: dict, factors) -> dict:
    factors = np.asarray(factors, dtype=float)
    scaled = dict(data)
    scaled["centers"] = np.asarray(data["centers"]) * factors
    scaled["means"] = np.asarray(data["means"]) * factors
    scaled["vars"] = np.asarray(data["vars"]) * factors**2
    scaled["kappa"] = np.asarray(data["kappa"]) / factors**2
    scaled["all_positions"] = [
        np.asarray(positions) * factors for positions in data["all_positions"]
    ]
    return scaled


def _reconstruct_small(data: dict, **kwargs) -> dict:
    options = {
        "grid_n": (6, 5),
        "plot": False,
        "plot_diagnostics": False,
        "save_outputs": False,
        "verbose": False,
        "calibrate_uncertainty": False,
    }
    options.update(kwargs)
    return reconstruct_pmf_2d(data=data, **options)


def test_mean_and_gradient_noise_retain_cross_component_covariance() -> None:
    first = np.array([
        [-2.0, -3.0],
        [2.0, 3.0],
        [-1.0, -2.0],
        [1.0, 2.0],
        [-0.5, -1.5],
        [0.5, 1.5],
        [-1.5, -2.5],
        [1.5, 2.5],
    ])
    second = first @ np.array([[0.6, -0.3], [0.2, 0.8]])
    positions = [first, second]
    tau = np.full((2, 2), 0.5)
    kappa = np.array([[2.0, 3.0], [5.0, 7.0]])

    noise, mean_covariances, batch_sizes = _gradient_noise_covariance(
        positions,
        kappa,
        tau,
        include_cross_component=True,
        batch_factor=1.0,
    )

    np.testing.assert_array_equal(batch_sizes, np.ones(2, dtype=int))
    for window in range(2):
        sl = slice(2 * window, 2 * window + 2)
        expected = (
            np.diag(kappa[window])
            @ mean_covariances[window]
            @ np.diag(kappa[window])
        )
        np.testing.assert_allclose(noise[sl, sl], expected, rtol=1e-14, atol=0.0)
        assert abs(noise[sl.start, sl.start + 1]) > 0.0

    # Windows are statistically independent, while the two observations from
    # each individual window retain their propagated covariance.
    np.testing.assert_array_equal(noise[:2, 2:], np.zeros((2, 2)))
    np.testing.assert_array_equal(noise[2:, :2], np.zeros((2, 2)))

    diagonal_noise, diagonal_means, _ = _gradient_noise_covariance(
        positions,
        kappa,
        tau,
        include_cross_component=False,
        batch_factor=1.0,
    )
    np.testing.assert_array_equal(diagonal_means[:, 0, 1], np.zeros(2))
    np.testing.assert_array_equal(diagonal_means[:, 1, 0], np.zeros(2))
    np.testing.assert_array_equal(
        diagonal_noise - np.diag(np.diag(diagonal_noise)),
        np.zeros_like(diagonal_noise),
    )

    # A perfectly constant trajectory remains exactly zero; the GP system adds
    # its own unit-covariant numerical jitter at factorization time.
    constant_covariance, _ = estimate_mean_covariance(
        np.ones((8, 2)), np.array([0.5, 0.5]), batch_factor=1.0
    )
    np.testing.assert_array_equal(constant_covariance, np.zeros((2, 2)))


def test_short_trajectory_hac_fallback_retains_lagged_cross_covariance() -> None:
    """Cross-CV correlation at nonzero lag must survive the fallback path."""
    rng = np.random.default_rng(12)
    first = rng.normal(size=64)
    first -= first.mean()
    second = np.roll(first, 1)
    second -= second.mean()
    # Remove its zero-lag projection exactly. Any recovered cross-covariance
    # therefore comes from the lagged terms, not the ordinary sample matrix.
    second -= first * (first @ second) / (first @ first)
    second -= second.mean()
    positions = np.column_stack([first, second])
    assert abs(np.cov(positions, rowvar=False, ddof=1)[0, 1]) < 1e-14

    covariance, batch_size = estimate_mean_covariance(
        positions,
        np.array([4.0, 4.0]),
        batch_factor=5.0,
        min_batches=4,
    )

    # target batch size 20 exceeds floor(64 / 4) = 16, selecting the HAC
    # fallback rather than non-overlapping batch means.
    assert batch_size == 16
    correlation = covariance[0, 1] / np.sqrt(
        covariance[0, 0] * covariance[1, 1]
    )
    assert covariance[0, 1] > 0.005
    assert correlation > 0.5


def test_zero_variance_cv_does_not_inherit_other_components_covariance_floor() -> None:
    positions = np.column_stack([
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.zeros(5),
    ])
    covariance, _ = estimate_mean_covariance(
        positions, np.array([0.5, 0.5]), batch_factor=1.0
    )
    rescaled_covariance, _ = estimate_mean_covariance(
        positions * np.array([1e6, 1.0]),
        np.array([0.5, 0.5]),
        batch_factor=1.0,
    )

    assert rescaled_covariance[0, 0] == pytest.approx(
        covariance[0, 0] * 1e12, rel=2e-14
    )
    assert covariance[1, 1] == 0.0
    assert rescaled_covariance[1, 1] == 0.0
    assert covariance[0, 1] == covariance[1, 0] == 0.0
    assert rescaled_covariance[0, 1] == rescaled_covariance[1, 0] == 0.0


def test_rank_deficient_window_geometry_is_supported_locally() -> None:
    training = np.array([
        [-1.0, -2.0],
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
    ])
    query = np.array([[0.5, 0.5], [0.0, 1.1]])

    mask = _grid_support_mask(
        query,
        training,
        lengthscale=np.array([1.0, 1.0]),
        radius=1.0,
    )

    np.testing.assert_array_equal(mask, [True, False])


def test_reconstruction_uses_sampled_neighborhoods_by_default() -> None:
    results = _reconstruct_small(
        _small_correlated_data(),
        grid_n=(10, 9),
        optimize_hyperparams=False,
        fixed_lengthscale=(0.8, 1.1),
        fixed_sigma_f=1.2,
    )

    assert results["restrict_to_sampled_support"] is True
    assert results["support_kind"] == "union_of_kernel_ellipses"
    assert results["support_radius"] == pytest.approx(0.5)
    np.testing.assert_allclose(results["support_ellipse_semiaxes"], (0.4, 0.55))
    assert np.any(results["support_mask"])
    assert np.any(~results["support_mask"])
    reference = results["_gp_state"]["pmf_reference_index"]
    assert results["support_mask"].ravel()[reference]
    assert results["pmf"].ravel()[reference] == pytest.approx(0.0, abs=1e-14)


def test_reconstruction_can_explicitly_use_the_full_rectangle() -> None:
    results = _reconstruct_small(
        _small_correlated_data(),
        optimize_hyperparams=False,
        fixed_lengthscale=(0.8, 1.1),
        fixed_sigma_f=1.2,
        restrict_to_sampled_support=False,
        support_radius=None,
    )

    assert np.all(results["support_mask"])
    assert results["restrict_to_sampled_support"] is False
    assert results["support_kind"] == "full_rectangle"
    assert results["support_radius"] is None

def test_nll_uses_the_complete_observation_covariance() -> None:
    X = np.array([[-0.7, 0.1], [0.2, -0.4], [0.9, 0.8]])
    y = np.array([-0.3, 0.2, 0.5, -0.4, 0.1, 0.7])
    params = np.array([1.3, 0.8, 1.1])
    noise = np.zeros((6, 6))
    blocks = (
        np.array([[0.08, 0.025], [0.025, 0.05]]),
        np.array([[0.06, -0.018], [-0.018, 0.07]]),
        np.array([[0.04, 0.012], [0.012, 0.09]]),
    )
    for window, block in enumerate(blocks):
        sl = slice(2 * window, 2 * window + 2)
        noise[sl, sl] = block

    Ky = _with_relative_diagonal_jitter(
        _k_grad_grad(X, X, params[0], params[1:]) + noise
    )
    sign, log_determinant = np.linalg.slogdet(Ky)
    assert sign > 0
    expected = (
        0.5 * y @ np.linalg.solve(Ky, y)
        + 0.5 * log_determinant
        + 0.5 * len(y) * np.log(2.0 * np.pi)
    )
    np.testing.assert_allclose(_nll(params, X, y, noise), expected, rtol=2e-13)

    diagonal_only = np.diag(np.diag(noise))
    assert not np.isclose(
        _nll(params, X, y, noise),
        _nll(params, X, y, diagonal_only),
        rtol=1e-7,
        atol=1e-9,
    )


def test_block_loo_matches_explicit_conditional_distributions() -> None:
    rng = np.random.default_rng(22)
    factor = rng.normal(size=(6, 6))
    covariance = factor @ factor.T + 0.7 * np.eye(6)
    observations = rng.normal(size=6)
    actual = _leave_one_window_out_z(
        np.linalg.inv(covariance), observations, dimensions=2
    )

    expected = []
    all_indices = np.arange(6)
    for start in range(0, 6, 2):
        held_out = all_indices[start:start + 2]
        retained = np.delete(all_indices, held_out)
        cross = covariance[np.ix_(held_out, retained)]
        retained_covariance = covariance[np.ix_(retained, retained)]
        conditional_mean = cross @ np.linalg.solve(
            retained_covariance, observations[retained]
        )
        conditional_covariance = (
            covariance[np.ix_(held_out, held_out)]
            - cross
            @ np.linalg.solve(
                retained_covariance, covariance[np.ix_(retained, held_out)]
            )
        )
        residual = observations[held_out] - conditional_mean
        expected.extend(
            np.linalg.solve(np.linalg.cholesky(conditional_covariance), residual)
        )

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_mixed_derivative_kernel_matches_finite_differences() -> None:
    xa = np.array([[0.35, -0.45]])
    xb = np.array([[-0.25, 0.70]])
    sigma_f = 1.4
    lengthscale = np.array([0.65, 1.25])
    analytic = _k_grad_grad(xa, xb, sigma_f, lengthscale).reshape(2, 2)

    step = 1e-4
    finite_difference = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            delta_a = np.zeros_like(xa)
            delta_b = np.zeros_like(xb)
            delta_a[0, i] = step
            delta_b[0, j] = step
            finite_difference[i, j] = (
                _se(xa + delta_a, xb + delta_b, sigma_f, lengthscale)[0, 0]
                - _se(xa + delta_a, xb - delta_b, sigma_f, lengthscale)[0, 0]
                - _se(xa - delta_a, xb + delta_b, sigma_f, lengthscale)[0, 0]
                + _se(xa - delta_a, xb - delta_b, sigma_f, lengthscale)[0, 0]
            ) / (4.0 * step**2)

    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-7, atol=2e-7)


def test_posterior_covariance_and_reference_state_drive_pmf_uncertainty() -> None:
    results = reconstruct_pmf_2d(
        data=_small_correlated_data(),
        grid_n=(4, 3),
        optimize_hyperparams=False,
        fixed_lengthscale=(0.8, 1.1),
        fixed_sigma_f=1.2,
        covariance_batch_factor=1.0,
        restrict_to_sampled_support=True,
        support_radius=1.0,
        plot=False,
        plot_diagnostics=False,
        save_outputs=False,
        verbose=False,
    )
    state = results["_gp_state"]
    points = np.column_stack([results["GX"].ravel(), results["GY"].ravel()])

    # Training uses the complete block covariance, not only its diagonal.
    training_covariance = _with_relative_diagonal_jitter(
        _k_grad_grad(
            state["X"], state["X"], state["sigma_f"], state["lengthscale"]
        )
        + results["gradient_noise_cov"]
    )
    expected_alpha = np.linalg.solve(training_covariance, results["grad"].ravel())
    np.testing.assert_allclose(state["alpha"], expected_alpha, rtol=2e-10, atol=2e-11)

    covariance = posterior_covariance_2d(results, points)
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=2e-14)
    assert np.linalg.eigvalsh(covariance).min() > -2e-11

    K_query = _k_f_grad(
        points, state["X"], state["sigma_f"], state["lengthscale"]
    )
    solved = np.linalg.solve(training_covariance, K_query.T)
    expected_covariance = (
        _se(points, points, state["sigma_f"], state["lengthscale"])
        - K_query @ solved
    )
    np.testing.assert_allclose(covariance, expected_covariance, rtol=3e-11, atol=2e-12)
    np.testing.assert_allclose(
        np.diag(covariance), results["latent_variance_raw"].ravel(), atol=2e-12
    )

    reference = state["pmf_reference_index"]
    np.testing.assert_allclose(state["pmf_reference_point"], points[reference])
    posterior_mean = K_query @ state["alpha"]
    np.testing.assert_allclose(
        state["pmf_reference_mean"], posterior_mean[reference], atol=2e-13
    )
    np.testing.assert_allclose(
        results["pmf"].ravel(), posterior_mean - state["pmf_reference_mean"],
        rtol=2e-12,
        atol=2e-13,
    )

    relative_variance = np.clip(
        np.diag(covariance)
        + covariance[reference, reference]
        - 2.0 * covariance[:, reference],
        0.0,
        np.inf,
    )
    np.testing.assert_allclose(
        results["pmf_std_raw"].ravel() ** 2,
        relative_variance,
        rtol=2e-9,
        atol=2e-12,
    )
    # A diagonal-only propagation would incorrectly assign uncertainty even
    # to F(reference) - F(reference), which is known exactly.
    diagonal_only_at_reference = np.sqrt(2.0 * covariance[reference, reference])
    assert diagonal_only_at_reference > 0.0
    assert results["pmf_std_raw"].ravel()[reference] < 1e-8

    scaled_results = dict(results)
    scaled_results["loo_calibration_factor"] = 2.5
    calibrated = posterior_covariance_2d(scaled_results, points, calibrated=True)
    np.testing.assert_allclose(calibrated, covariance * 2.5**2, rtol=2e-14)

    left, right = points[:4], points[4:]
    np.testing.assert_allclose(
        posterior_covariance_2d(results, left, right),
        posterior_covariance_2d(results, right, left).T,
        rtol=2e-12,
        atol=2e-13,
    )


def test_pmf_reference_is_lowest_point_within_support(monkeypatch) -> None:
    selected_index = 2

    def one_supported_grid_point(points, training_points, lengthscale, radius):
        support = np.zeros(len(points), dtype=bool)
        support[selected_index] = True
        return support

    monkeypatch.setattr(
        integration_2d, "_grid_support_mask", one_supported_grid_point
    )
    results = reconstruct_pmf_2d(
        data=_small_correlated_data(),
        grid_n=(4, 3),
        optimize_hyperparams=False,
        fixed_lengthscale=(0.8, 1.1),
        fixed_sigma_f=1.2,
        covariance_batch_factor=1.0,
        restrict_to_sampled_support=True,
        support_radius=1.0,
        plot=False,
        plot_diagnostics=False,
        save_outputs=False,
        verbose=False,
    )

    state = results["_gp_state"]
    query = np.column_stack([results["GX"].ravel(), results["GY"].ravel()])
    posterior_mean = _k_f_grad(
        query, state["X"], state["sigma_f"], state["lengthscale"]
    ) @ state["alpha"]

    # This deliberately supported point is not the unrestricted GP minimum;
    # choosing the latter would reference the PMF to an extrapolated cell.
    assert np.argmin(posterior_mean) != selected_index
    assert state["pmf_reference_index"] == selected_index
    assert results["support_mask"].ravel()[selected_index]
    assert results["pmf"].ravel()[selected_index] == pytest.approx(0.0, abs=1e-14)
    assert results["pmf"].ravel().min() < 0.0


def test_optimized_reconstruction_is_invariant_to_energy_unit_rescaling() -> None:
    factor = 96.485
    base = _reconstruct_small(
        _small_correlated_data(),
        energy_unit="eV",
        optimize_hyperparams=True,
        restrict_to_sampled_support=False,
        support_radius=None,
    )
    rescaled = _reconstruct_small(
        _rescale_energy(_small_correlated_data(), factor),
        energy_unit="kJ/mol",
        optimize_hyperparams=True,
        restrict_to_sampled_support=False,
        support_radius=None,
    )

    assert base["fit_ok"] and rescaled["fit_ok"]
    np.testing.assert_allclose(
        rescaled["lengthscale"], base["lengthscale"], rtol=3e-3, atol=2e-5
    )
    assert rescaled["sigma_f"] / factor == pytest.approx(
        base["sigma_f"], rel=3e-3, abs=2e-6
    )
    np.testing.assert_allclose(
        rescaled["pmf"] / factor,
        base["pmf"],
        rtol=3e-3,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.sqrt(rescaled["latent_variance_raw"]) / factor,
        np.sqrt(base["latent_variance_raw"]),
        rtol=4e-3,
        atol=3e-5,
    )
    np.testing.assert_array_equal(rescaled["support_mask"], base["support_mask"])
    assert rescaled["loo_calibration_factor"] == pytest.approx(
        base["loo_calibration_factor"], rel=5e-3
    )


def test_independent_cv_rescaling_with_fixed_hyperparameters_is_invariant() -> None:
    coordinate_factors = np.array([7.0, 0.01])
    lengthscale = np.array([0.8, 1.1])
    base = _reconstruct_small(
        _small_correlated_data(),
        optimize_hyperparams=False,
        fixed_lengthscale=lengthscale,
        fixed_sigma_f=1.2,
        restrict_to_sampled_support=True,
        support_radius=1.5,
    )
    rescaled = _reconstruct_small(
        _rescale_coordinates(_small_correlated_data(), coordinate_factors),
        optimize_hyperparams=False,
        fixed_lengthscale=lengthscale * coordinate_factors,
        fixed_sigma_f=1.2,
        restrict_to_sampled_support=True,
        support_radius=1.5,
    )

    np.testing.assert_allclose(
        rescaled["gx"] / coordinate_factors[0], base["gx"], rtol=2e-14
    )
    np.testing.assert_allclose(
        rescaled["gy"] / coordinate_factors[1], base["gy"], rtol=2e-14
    )
    np.testing.assert_allclose(rescaled["pmf"], base["pmf"], rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(
        rescaled["pmf_std_raw"], base["pmf_std_raw"], rtol=3e-5, atol=2e-7
    )
    np.testing.assert_array_equal(rescaled["support_mask"], base["support_mask"])


def test_optimized_reconstruction_is_invariant_to_independent_cv_units() -> None:
    coordinate_factors = np.array([7.0, 0.01])
    base = _reconstruct_small(
        _small_correlated_data(),
        optimize_hyperparams=True,
        restrict_to_sampled_support=False,
        support_radius=None,
    )
    rescaled = _reconstruct_small(
        _rescale_coordinates(_small_correlated_data(), coordinate_factors),
        optimize_hyperparams=True,
        restrict_to_sampled_support=False,
        support_radius=None,
    )

    assert base["fit_ok"] and rescaled["fit_ok"]
    np.testing.assert_allclose(
        rescaled["lengthscale"] / coordinate_factors,
        base["lengthscale"],
        rtol=3e-3,
        atol=2e-5,
    )
    assert rescaled["sigma_f"] == pytest.approx(
        base["sigma_f"], rel=3e-3, abs=2e-6
    )
    np.testing.assert_allclose(
        rescaled["pmf"], base["pmf"], rtol=3e-3, atol=3e-5
    )
    np.testing.assert_allclose(
        rescaled["pmf_std_raw"],
        base["pmf_std_raw"],
        rtol=4e-3,
        atol=3e-5,
    )
    np.testing.assert_array_equal(rescaled["support_mask"], base["support_mask"])


def test_no_optimize_default_hyperparameters_follow_independent_cv_rescaling() -> None:
    coordinate_factors = np.array([7.0, 0.01])
    base = _reconstruct_small(
        _small_correlated_data(),
        optimize_hyperparams=False,
        restrict_to_sampled_support=True,
        support_radius=1.5,
    )
    rescaled = _reconstruct_small(
        _rescale_coordinates(_small_correlated_data(), coordinate_factors),
        optimize_hyperparams=False,
        restrict_to_sampled_support=True,
        support_radius=1.5,
    )

    np.testing.assert_allclose(
        rescaled["lengthscale"] / coordinate_factors,
        base["lengthscale"],
        rtol=2e-12,
        atol=2e-14,
    )
    assert rescaled["sigma_f"] == pytest.approx(base["sigma_f"], rel=2e-12)
    np.testing.assert_allclose(rescaled["pmf"], base["pmf"], rtol=3e-5, atol=2e-7)
    np.testing.assert_array_equal(rescaled["support_mask"], base["support_mask"])


def test_data_input_honors_kj_kappa_conversion_without_mutating_caller() -> None:
    factor = 1.0 / 96.485
    raw_data = _small_correlated_data()
    original_kappa = raw_data["kappa"].copy()
    explicitly_converted = _rescale_energy(raw_data, factor)

    converted_by_api = _reconstruct_small(
        raw_data,
        kappa_in_kj_per_mol=True,
        energy_unit="eV",
        optimize_hyperparams=False,
        fixed_lengthscale=(0.8, 1.1),
        fixed_sigma_f=1.2,
        support_radius=1.0,
    )
    expected = _reconstruct_small(
        explicitly_converted,
        kappa_in_kj_per_mol=False,
        energy_unit="eV",
        optimize_hyperparams=False,
        fixed_lengthscale=(0.8, 1.1),
        fixed_sigma_f=1.2,
        support_radius=1.0,
    )

    np.testing.assert_array_equal(raw_data["kappa"], original_kappa)
    np.testing.assert_allclose(converted_by_api["kappa"], expected["kappa"])
    np.testing.assert_allclose(converted_by_api["grad"], expected["grad"])
    np.testing.assert_allclose(
        converted_by_api["gradient_noise_cov"], expected["gradient_noise_cov"]
    )
    np.testing.assert_allclose(converted_by_api["pmf"], expected["pmf"])


def test_zero_signal_data_requires_an_explicit_signal_scale() -> None:
    centers = np.array([
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ])
    positions = [np.repeat(point[None, :], 8, axis=0) for point in centers]
    data = {
        "centers": centers,
        "kappa": np.ones_like(centers),
        "means": centers.copy(),
        "vars": np.zeros_like(centers),
        "n_samples": np.full(len(centers), 8.0),
        "all_positions": positions,
    }

    with pytest.raises(ValueError, match="signal scale is unidentifiable"):
        _reconstruct_small(data, optimize_hyperparams=True, support_radius=1.0)

    result = _reconstruct_small(
        data,
        optimize_hyperparams=True,
        fixed_sigma_f=1.0,
        support_radius=1.0,
    )
    np.testing.assert_allclose(result["pmf"], 0.0, atol=0.0)
    assert np.all(np.isfinite(result["pmf_std_raw"]))
