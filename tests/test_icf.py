"""Synthetic tests for paper-convention instantaneous collective forces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpr_umbrella.icf import (
    aggregate_icf_observations,
    assess_icf_validity,
    load_icf_data,
    reconstruct_pmf_icf,
)


def _write_plumed_table(
    path: Path,
    fields: tuple[str, ...],
    rows: np.ndarray,
) -> Path:
    lines = ["#! FIELDS " + " ".join(fields)]
    lines.extend(" ".join(f"{value:.12g}" for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _validity() -> dict:
    return assess_icf_validity(
        bias_depends_only_on_modeled_cvs=True,
        quasi_equilibrium=True,
        physical_force_excludes_bias=True,
        metric_correction_included=True,
    )


def test_named_icf_loader_applies_the_paper_sign_and_records_assumptions(
    tmp_path: Path,
) -> None:
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "path.s", "path_s.icf"),
        np.array([
            [0.0, -1.0, 2.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, -2.0],
        ]),
    )
    result = load_icf_data(
        path,
        cv_fields=("path.s",),
        icf_fields=("path_s.icf",),
        bias_depends_only_on_modeled_cvs=True,
        quasi_equilibrium=True,
        physical_force_excludes_bias=True,
        metric_correction_included=True,
    )

    np.testing.assert_allclose(result["icf"].ravel(), [2.0, 0.0, -2.0])
    np.testing.assert_allclose(result["gradients"].ravel(), [-2.0, 0.0, 2.0])
    assert result["force_convention"] == "thermodynamic_force"
    assert result["validity"]["paper_equation_8_applicable"] is True
    assert result["validity"]["paper_assumptions_satisfied"] is True
    assert result["validity"]["convergence_assessed"] is False


def test_icf_loader_deduplicates_restarts_and_retains_parser_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "COLVAR"
    path.write_text(
        """#! FIELDS time path.s path_s.icf
0 -1 2
1 0 0
interrupted row
#! FIELDS path_s.icf time path.s
-0.5 1 0.25
-2 2 1
""",
        encoding="utf-8",
    )

    result = load_icf_data(
        path,
        cv_fields=("path.s",),
        icf_fields=("path_s.icf",),
    )

    np.testing.assert_allclose(result["time"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result["positions"].ravel(), [-1.0, 0.25, 1.0])
    np.testing.assert_allclose(result["icf"].ravel(), [2.0, -0.5, -2.0])
    np.testing.assert_allclose(result["gradients"].ravel(), [-2.0, 0.5, 2.0])
    assert result["table"]["duplicate_records_replaced"] == 1
    assert result["table"]["malformed_records_ignored"] == 1
    assert result["table"]["files"] == (str(path),)


def test_exact_icf_gp_recovers_a_harmonic_pmf_with_unambiguous_sign() -> None:
    curvature = 2.0
    positions = np.linspace(-1.0, 1.0, 15)[:, None]
    thermodynamic_force = -curvature * positions
    result = reconstruct_pmf_icf(
        data={
            "positions": positions,
            "icf": thermodynamic_force,
            "cv_names": ("s",),
            "validity": _validity(),
        },
        force_errors=0.03,
        grid_n=61,
        optimize_hyperparams=False,
        fixed_lengthscale=0.7,
        fixed_sigma_f=2.0,
        calibrate_uncertainty=False,
        require_paper_assumptions=True,
    )

    np.testing.assert_allclose(result["gradients"], curvature * positions)
    x = result["x_star"]
    expected = 0.5 * curvature * x**2
    expected -= expected.min()
    assert np.corrcoef(result["pmf"], expected)[0, 1] > 0.98
    assert result["method"] == "icf_gradient_gpr"
    assert result["prediction"]["reference_point"].shape == (1,)
    assert "converged" not in result


def test_local_icf_aggregation_propagates_vector_mean_covariance() -> None:
    positions = np.array([-1.0, -0.9, -0.1, 0.1, 0.9, 1.0])[:, None]
    forces = -2.0 * positions + np.array([0.1, -0.1, 0.05, -0.05, 0.1, -0.1])[:, None]
    result = aggregate_icf_observations(
        {
            "positions": positions,
            "collective_forces": forces,
            "time": np.arange(len(positions), dtype=float),
            "validity": _validity(),
        },
        bins=3,
        min_samples=2,
    )

    assert result["positions"].shape == (3, 1)
    assert result["force_noise_cov"].shape == (3, 3)
    assert np.all(np.diag(result["force_noise_cov"]) >= 0.0)
    np.testing.assert_allclose(result["gradients"], -result["collective_forces"])
    np.testing.assert_array_equal(result["counts"], [2, 2, 2])


def test_exact_gp_guard_rejects_production_scale_data_without_aggregation() -> None:
    positions = np.linspace(-1.0, 1.0, 12)[:, None]
    with pytest.raises(ValueError, match="above max_exact_observations"):
        reconstruct_pmf_icf(
            data={
                "positions": positions,
                "icf": -positions,
                "validity": _validity(),
            },
            force_errors=0.1,
            max_exact_observations=10,
        )


def test_paper_assumptions_are_required_by_default_with_explicit_opt_out() -> None:
    unknown = assess_icf_validity()
    assert unknown["paper_equation_8_applicable"] is None
    assert unknown["paper_assumptions_satisfied"] is None

    with pytest.raises(ValueError, match="Paper assumptions"):
        reconstruct_pmf_icf(
            data={
                "positions": np.array([[-1.0], [1.0]]),
                "icf": np.array([[1.0], [-1.0]]),
                "validity": unknown,
            },
            force_errors=0.1,
            optimize_hyperparams=False,
            fixed_lengthscale=1.0,
            fixed_sigma_f=1.0,
        )

    opted_out = reconstruct_pmf_icf(
        data={
            "positions": np.array([[-1.0], [1.0]]),
            "icf": np.array([[1.0], [-1.0]]),
            "validity": unknown,
        },
        force_errors=0.1,
        require_paper_assumptions=False,
        optimize_hyperparams=False,
        fixed_lengthscale=1.0,
        fixed_sigma_f=1.0,
        calibrate_uncertainty=False,
        grid_n=5,
    )
    assert opted_out["icf_validity"]["paper_assumptions_satisfied"] is None


def test_icf_unit_scaling_couples_energy_and_coordinate_factors(
    tmp_path: Path,
) -> None:
    kj_per_mol_per_ev = 8.31446261815324e-3 / 8.617333262145e-5
    energy_factor = 1.0 / kj_per_mol_per_ev
    cv_factor = 10.0
    positions_nm = np.linspace(-0.1, 0.1, 7)
    positions_angstrom = positions_nm * cv_factor
    forces_ev_per_angstrom = -2.0 * positions_angstrom
    forces_kj_per_mol_per_nm = (
        forces_ev_per_angstrom * cv_factor / energy_factor
    )
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "z", "z.icf"),
        np.column_stack([
            np.arange(len(positions_nm), dtype=float),
            positions_nm,
            forces_kj_per_mol_per_nm,
        ]),
    )
    validity_options = {
        "bias_depends_only_on_modeled_cvs": True,
        "quasi_equilibrium": True,
        "physical_force_excludes_bias": True,
        "metric_correction_included": True,
    }

    loaded = load_icf_data(
        path,
        cv_fields=("z",),
        icf_fields=("z.icf",),
        cv_factors=cv_factor,
        energy_factor=energy_factor,
        **validity_options,
    )
    np.testing.assert_allclose(
        loaded["positions"].ravel(), positions_angstrom
    )
    np.testing.assert_allclose(
        loaded["collective_forces"].ravel(), forces_ev_per_angstrom
    )
    np.testing.assert_allclose(
        loaded["gradients"].ravel(), -forces_ev_per_angstrom
    )
    np.testing.assert_allclose(
        loaded["icf_factors"], [energy_factor / cv_factor]
    )
    assert loaded["unit_provenance"]["icf_factors_source"] == (
        "derived_energy_factor_over_cv_factors"
    )

    result = reconstruct_pmf_icf(
        path,
        cv_fields=("z",),
        icf_fields=("z.icf",),
        cv_factors=cv_factor,
        energy_factor=energy_factor,
        force_errors=0.03,
        grid_n=11,
        optimize_hyperparams=False,
        fixed_lengthscale=0.7,
        fixed_sigma_f=2.0,
        calibrate_uncertainty=False,
        **validity_options,
    )
    np.testing.assert_allclose(result["points"], positions_angstrom[:, None])
    np.testing.assert_allclose(
        result["collective_forces"].ravel(), forces_ev_per_angstrom
    )
    np.testing.assert_allclose(
        result["gradients"].ravel(), -forces_ev_per_angstrom
    )
    assert result["energy_factor"] == energy_factor
    assert result["unit_provenance"]["force_scaling_relation"] == (
        "icf_factor = energy_factor / cv_factor"
    )


def test_icf_unit_scaling_rejects_ambiguous_force_conversion(
    tmp_path: Path,
) -> None:
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "s", "s.icf"),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, -1.0]]),
    )
    with pytest.raises(ValueError, match="energy_factor.*icf_factors"):
        load_icf_data(
            path,
            cv_fields=("s",),
            icf_fields=("s.icf",),
            energy_factor=0.5,
            icf_factors=1.0,
        )


def test_exact_gp_guard_counts_scalar_gradient_components() -> None:
    positions = np.column_stack([
        np.linspace(-1.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
    ])
    with pytest.raises(
        ValueError,
        match=(
            "6 observation points with 12 scalar gradient components, above "
            "max_exact_observations=10"
        ),
    ):
        reconstruct_pmf_icf(
            data={
                "positions": positions,
                "icf": -positions,
                "validity": _validity(),
            },
            force_errors=0.1,
            max_exact_observations=10,
        )


def test_automatic_reference_is_chosen_on_sampled_support(monkeypatch) -> None:
    positions = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    calls: list[np.ndarray] = []

    def fake_predict(
        model,
        points,
        *,
        reference=None,
        calibrated=True,
        prediction_batch_size=10_000,
    ):
        del model, calibrated, prediction_batch_size
        points = np.asarray(points, dtype=float)
        calls.append(points.copy())
        mean = np.sum(points, axis=1)
        reference_point = None
        if reference is not None:
            reference_point = np.asarray(reference, dtype=float)
            mean = mean - np.sum(reference_point)
        zeros = np.zeros(len(points))
        return {
            "mean": mean,
            "std": zeros,
            "std_raw": zeros,
            "std_calibrated": zeros,
            "reference_point": reference_point,
        }

    monkeypatch.setattr(
        "gpr_umbrella.gradient_gpr.predict_gradient_gp", fake_predict
    )
    result = reconstruct_pmf_icf(
        data={
            "positions": positions,
            "icf": -positions,
            "validity": _validity(),
        },
        force_errors=0.1,
        prediction_points=np.array([[0.0, 0.0], [1.0, 1.0]]),
        optimize_hyperparams=False,
        fixed_lengthscale=1.0,
        fixed_sigma_f=1.0,
        calibrate_uncertainty=False,
    )

    np.testing.assert_allclose(calls[0], positions)
    np.testing.assert_allclose(
        result["prediction"]["reference_point"], positions[0]
    )
    assert result["reference_source"] == (
        "lowest_prediction_on_observation_support"
    )

    explicit = reconstruct_pmf_icf(
        data={
            "positions": positions,
            "icf": -positions,
            "validity": _validity(),
        },
        force_errors=0.1,
        prediction_points=np.array([[0.0, 0.0], [1.0, 1.0]]),
        reference=np.array([2.0, 2.0]),
        optimize_hyperparams=False,
        fixed_lengthscale=1.0,
        fixed_sigma_f=1.0,
        calibrate_uncertainty=False,
    )
    np.testing.assert_allclose(
        explicit["prediction"]["reference_point"], [2.0, 2.0]
    )
    assert explicit["reference_source"] == "explicit"
