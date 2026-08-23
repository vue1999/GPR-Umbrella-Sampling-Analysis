"""Synthetic checks for the bias-independent gradient-observation GP core."""

from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella.gradient_gpr import (
    fit_gradient_gp,
    fit_icf_gp,
    posterior_covariance_gradient_gp,
    predict_gradient_gp,
)


def _fit_1d(*, calibrate_uncertainty: bool = False) -> dict:
    points = np.linspace(-2.0, 2.0, 13)
    gradients = points
    noise = np.eye(points.size) * 1.0e-6
    return fit_gradient_gp(
        points,
        gradients,
        noise,
        optimize_hyperparams=False,
        fixed_sigma_f=3.0,
        fixed_lengthscale=1.4,
        calibrate_uncertainty=calibrate_uncertainty,
    )


def test_1d_gradient_gp_recovers_a_relative_quadratic_pmf() -> None:
    model = _fit_1d()
    query = np.linspace(-1.8, 1.8, 19)
    prediction = predict_gradient_gp(model, query, reference=0.0)
    expected = 0.5 * query**2

    np.testing.assert_allclose(prediction["pmf_mean"], expected, atol=0.035)
    assert prediction["points"].shape == (query.size, 1)
    assert prediction["reference_point"] == pytest.approx(np.array([0.0]))
    assert np.all(prediction["pmf_std"] >= 0.0)


def test_2d_gradient_gp_recovers_an_anisotropic_quadratic() -> None:
    x = np.linspace(-1.5, 1.5, 5)
    y = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    gradients = np.column_stack([points[:, 0], 2.0 * points[:, 1]])
    size = gradients.size
    model = fit_gradient_gp(
        points,
        gradients,
        np.eye(size) * 1.0e-6,
        optimize_hyperparams=False,
        fixed_sigma_f=3.0,
        fixed_lengthscale=(1.3, 1.0),
        calibrate_uncertainty=False,
    )

    query = np.array([
        [-1.0, -0.5],
        [-0.5, 0.5],
        [0.0, 0.0],
        [0.75, -0.25],
        [1.0, 0.5],
    ])
    prediction = predict_gradient_gp(model, query, reference=(0.0, 0.0))
    expected = 0.5 * query[:, 0] ** 2 + query[:, 1] ** 2
    np.testing.assert_allclose(prediction["mean"], expected, atol=0.055)
    assert model["n_dimensions"] == 2
    assert model["gradients"].shape == points.shape


def test_relative_prediction_variance_uses_covariance_with_reference() -> None:
    model = _fit_1d()
    query = np.array([-1.0, 0.0, 1.0])
    prediction = predict_gradient_gp(model, query, reference=0.0)
    covariance = posterior_covariance_gradient_gp(
        model, query, np.array([0.0])
    ).ravel()
    latent = np.diag(posterior_covariance_gradient_gp(model, query))
    reference_variance = posterior_covariance_gradient_gp(
        model, np.array([0.0])
    )[0, 0]
    expected = latent + reference_variance - 2.0 * covariance

    np.testing.assert_allclose(prediction["variance_raw"], expected, atol=1e-12)
    assert prediction["variance_raw"][1] == pytest.approx(0.0, abs=1e-12)


def test_calibration_scales_standard_deviation_and_covariance_consistently() -> None:
    model = _fit_1d(calibrate_uncertainty=True)
    query = np.array([-0.8, 0.2, 1.1])
    prediction = predict_gradient_gp(model, query)
    factor = model["loo_calibration_factor"]

    assert prediction["calibrated"] is True
    np.testing.assert_allclose(
        prediction["std_calibrated"], prediction["std_raw"] * factor
    )
    np.testing.assert_allclose(
        posterior_covariance_gradient_gp(model, query, calibrated=True),
        posterior_covariance_gradient_gp(model, query) * factor**2,
    )


def test_pointwise_loo_cannot_shrink_icf_uncertainty_without_opt_in() -> None:
    points = np.linspace(-1.0, 1.0, 15)
    # A coherent force offset is predictable point-to-point but remains a
    # systematic PMF error; pointwise LOO must not use it to claim smaller
    # uncertainty by default.
    gradient_with_coherent_offset = points + 0.3
    thermodynamic_force = -gradient_with_coherent_offset
    noise = np.eye(points.size) * 0.3**2
    options = {
        "force_convention": "thermodynamic_force",
        "optimize_hyperparams": False,
        "fixed_lengthscale": 0.7,
        "fixed_sigma_f": 2.0,
        "calibrate_uncertainty": True,
    }

    safe = fit_icf_gp(points, thermodynamic_force, noise, **options)
    assert safe["loo_calibration_factor_uncapped"] < 1.0
    assert safe["loo_calibration_factor"] == pytest.approx(1.0)
    assert safe["uncertainty_downscaling_allowed"] is False
    safe_prediction = predict_gradient_gp(safe, points, reference=0.0)
    np.testing.assert_allclose(
        safe_prediction["std_calibrated"], safe_prediction["std_raw"]
    )

    opted_in = fit_icf_gp(
        points,
        thermodynamic_force,
        noise,
        allow_uncertainty_downscaling=True,
        **options,
    )
    assert opted_in["loo_calibration_factor"] == pytest.approx(
        opted_in["loo_calibration_factor_uncapped"]
    )
    assert opted_in["loo_calibration_factor"] < 1.0
    assert opted_in["uncertainty_downscaling_allowed"] is True


def test_icf_force_convention_is_explicit_and_sign_safe() -> None:
    points = np.linspace(-2.0, 2.0, 11)
    gradient = points
    noise = np.eye(points.size) * 1.0e-5
    common = {
        "optimize_hyperparams": False,
        "fixed_sigma_f": 3.0,
        "fixed_lengthscale": 1.2,
        "calibrate_uncertainty": False,
    }
    from_gradient = fit_icf_gp(
        points,
        gradient,
        noise,
        force_convention="free_energy_gradient",
        **common,
    )
    from_force = fit_icf_gp(
        points,
        -gradient,
        noise,
        force_convention="thermodynamic_force",
        **common,
    )

    np.testing.assert_allclose(from_gradient["gradients"], from_force["gradients"])
    np.testing.assert_allclose(
        predict_gradient_gp(from_gradient, points, reference=0.0)["mean"],
        predict_gradient_gp(from_force, points, reference=0.0)["mean"],
    )
    assert from_force["observation_kind"] == "instantaneous_collective_force"
    assert from_force["force_convention"] == "thermodynamic_force"

    with pytest.raises(ValueError, match="force_convention"):
        fit_icf_gp(
            points,
            gradient,
            noise,
            force_convention="guess",
            **common,
        )


def test_full_noise_covariance_is_retained_and_affects_the_fit() -> None:
    points = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    gradients = np.column_stack([points[:, 0], points[:, 1]])
    diagonal = np.eye(gradients.size) * 0.04
    correlated = diagonal.copy()
    for point in range(points.shape[0]):
        block = slice(2 * point, 2 * point + 2)
        correlated[block, block] = np.array([[0.04, 0.03], [0.03, 0.04]])
    kwargs = {
        "optimize_hyperparams": False,
        "fixed_sigma_f": 2.0,
        "fixed_lengthscale": (1.0, 1.0),
        "calibrate_uncertainty": False,
    }
    full_model = fit_gradient_gp(points, gradients, correlated, **kwargs)
    diagonal_model = fit_gradient_gp(points, gradients, diagonal, **kwargs)

    np.testing.assert_allclose(full_model["gradient_noise_cov"], correlated)
    full_prediction = predict_gradient_gp(
        full_model, np.array([[0.4, -0.2]]), reference=(0.0, 0.0)
    )
    diagonal_prediction = predict_gradient_gp(
        diagonal_model, np.array([[0.4, -0.2]]), reference=(0.0, 0.0)
    )
    assert not np.isclose(
        full_prediction["variance_raw"][0],
        diagonal_prediction["variance_raw"][0],
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    ("points", "gradients", "noise", "message"),
    [
        (
            np.array([[0.0], [1.0]]),
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            np.eye(2),
            "same shape",
        ),
        (
            np.array([[0.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.eye(2),
            "span every CV",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[0.0], [1.0]]),
            np.eye(3),
            "must have shape",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0, 0.2], [0.0, 1.0]]),
            "symmetric",
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            "positive semidefinite",
        ),
    ],
)
def test_fit_gradient_gp_rejects_invalid_observations(
    points: np.ndarray,
    gradients: np.ndarray,
    noise: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_gradient_gp(
            points,
            gradients,
            noise,
            optimize_hyperparams=False,
            fixed_sigma_f=1.0,
            fixed_lengthscale=1.0,
        )


def test_prediction_validates_reference_and_batch_size() -> None:
    model = _fit_1d()
    with pytest.raises(ValueError, match="exactly one point"):
        predict_gradient_gp(model, [0.0, 1.0], reference=[0.0, 1.0])
    with pytest.raises(ValueError, match="positive integer"):
        predict_gradient_gp(model, [0.0], prediction_batch_size=0)
