"""Focused tests for the metadynamics adapter and its validity boundaries."""
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

import gpr_umbrella.metadynamics as metadynamics
from gpr_umbrella.metadynamics import (
    analyze_metadynamics,
    load_hills_diagnostics,
    load_metadynamics_data,
    run_sum_hills,
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


def test_rbias_and_extra_bias_make_dimensionless_log_weights(
    tmp_path: Path,
) -> None:
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "wall.bias", "path.s", "metad.rbias"),
        np.array([
            [0.0, 0.05, -1.0, 0.10],
            [1.0, 0.00, 0.0, 0.20],
            [2.0, 0.10, 1.0, 0.30],
        ]),
    )

    result = load_metadynamics_data(
        path,
        cv_fields=("path.s",),
        rbias_field="metad.rbias",
        extra_bias_fields=("wall.bias",),
        thermal_energy=0.5,
        bias_energy_factor=2.0,
    )

    np.testing.assert_allclose(result["positions"].ravel(), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(result["log_weights"], [0.6, 0.8, 1.6])
    assert result["weight_source"] == "normalized_rbias"
    assert result["validity"]["normalized_time_dependent_bias"] is True
    assert result["bias_energy_conversion"] == {
        "factor_into_thermal_energy_unit": 2.0,
        "applied": True,
        "input_unit_inferred": False,
        "unit_contract": (
            "Each bias-energy column is multiplied by bias_energy_factor to "
            "reach the unit of thermal_energy; factor=1 explicitly asserts "
            "the units already match."
        ),
    }
    assert "converged" not in result


def test_restart_deduplication_and_parser_provenance_are_retained(
    tmp_path: Path,
) -> None:
    path = tmp_path / "COLVAR"
    path.write_text(
        """#! FIELDS time s metad.rbias metad.bias wall.bias
0 0.0 0.1 1.1 0.01
1 0.5 0.2 1.2 0.02
malformed interrupted row
#! FIELDS wall.bias metad.bias s time metad.rbias
0.03 1.3 0.75 1 0.3
0.04 1.4 1.0 2 0.4
""",
        encoding="utf-8",
    )

    result = load_metadynamics_data(
        path,
        cv_fields=("s",),
        rbias_field="metad.rbias",
        extra_bias_fields=("wall.bias",),
        thermal_energy=0.1,
    )

    np.testing.assert_allclose(result["time"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result["positions"].ravel(), [0.0, 0.75, 1.0])
    np.testing.assert_allclose(result["log_weights"], [1.1, 3.3, 4.4])
    assert result["table"]["duplicate_records_replaced"] == 1
    assert result["table"]["malformed_records_ignored"] == 1
    assert result["table"]["files"] == (str(path),)
    assert result["rbias_replaces_raw_bias_field"] == "metad.bias"
    assert set(result["accounted_bias_fields"]) == {
        "metad.bias", "wall.bias"
    }

    with pytest.raises(ValueError, match="expected 5 values"):
        load_metadynamics_data(
            path,
            cv_fields=("s",),
            rbias_field="metad.rbias",
            extra_bias_fields=("wall.bias",),
            thermal_energy=0.1,
            strict=True,
        )


def test_unlisted_bias_fields_fail_unless_explicitly_audited(
    tmp_path: Path,
) -> None:
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        (
            "time", "s", "metad.bias", "metad.rbias",
            "wall.bias", "restraint.bias",
        ),
        np.array([
            [0.0, 0.0, 9.0, 0.1, 0.01, 0.001],
            [1.0, 1.0, 8.0, 0.2, 0.02, 0.002],
        ]),
    )

    with pytest.raises(ValueError, match=r"unlisted bias fields.*restraint\.bias"):
        load_metadynamics_data(
            path,
            cv_fields=("s",),
            rbias_field="metad.rbias",
            extra_bias_fields=("wall.bias",),
            thermal_energy=0.1,
        )

    audited = load_metadynamics_data(
        path,
        cv_fields=("s",),
        rbias_field="metad.rbias",
        extra_bias_fields=("wall.bias",),
        thermal_energy=0.1,
        allow_unlisted_bias_fields=True,
    )
    assert audited["unlisted_bias_fields"] == ("restraint.bias",)
    # The matching raw metad.bias is replaced by rbias, never double counted.
    np.testing.assert_allclose(audited["log_weights"], [1.1, 2.2])

    with pytest.raises(ValueError, match="must not be added again"):
        load_metadynamics_data(
            path,
            cv_fields=("s",),
            rbias_field="metad.rbias",
            extra_bias_fields=("metad.bias", "wall.bias", "restraint.bias"),
            thermal_energy=0.1,
        )


def test_raw_time_dependent_bias_requires_explicit_quasistatic_opt_in(
    tmp_path: Path,
) -> None:
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "s", "metad.bias"),
        np.array([[0.0, 0.0, 0.1], [1.0, 1.0, 0.2]]),
    )
    with pytest.raises(ValueError, match="Raw time-dependent"):
        load_metadynamics_data(
            path,
            cv_fields=("s",),
            raw_bias_field="metad.bias",
            thermal_energy=0.025,
        )

    with pytest.raises(ValueError, match="explicit finite start_time"):
        load_metadynamics_data(
            path,
            cv_fields=("s",),
            raw_bias_field="metad.bias",
            allow_quasistatic_bias=True,
            thermal_energy=0.1,
        )

    accepted = load_metadynamics_data(
        path,
        cv_fields=("s",),
        raw_bias_field="metad.bias",
        allow_quasistatic_bias=True,
        thermal_energy=0.1,
        start_time=0.0,
        stop_time=1.0,
    )
    assert accepted["validity"]["quasistatic_bias_assumed"] is True
    assert accepted["validity"]["quasistatic_segment_explicitly_bounded"] is True
    assert accepted["validity"]["assumption_verified_by_code"] is False
    assert accepted["selection"]["explicit_finite_time_range"] is True


def test_all_weight_sources_support_bounded_time_and_stride_selection(
    tmp_path: Path,
) -> None:
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "s", "logw"),
        np.column_stack([
            np.arange(6, dtype=float),
            np.linspace(-1.0, 1.0, 6),
            np.arange(10.0, 16.0),
        ]),
    )

    result = load_metadynamics_data(
        path,
        cv_fields=("s",),
        logweight_field="logw",
        start_time=1.0,
        stop_time=4.0,
        stride=2,
    )

    np.testing.assert_allclose(result["time"], [1.0, 3.0])
    np.testing.assert_allclose(result["positions"].ravel(), [-0.6, 0.2])
    np.testing.assert_allclose(result["log_weights"], [11.0, 13.0])
    assert result["selection"] == {
        "requested_start_time": 1.0,
        "requested_stop_time": 4.0,
        "stride": 2,
        "n_records_before_selection": 6,
        "n_records_within_time_bounds": 4,
        "n_records_selected": 2,
        "selected_start_time": 1.0,
        "selected_stop_time": 3.0,
        "explicit_finite_time_range": True,
        "bounds_inclusive": True,
        "selection_order": (
            "restart_deduplication", "time_bounds", "stride"
        ),
    }

    for options, message in (
        ({"stride": 0}, "positive integer"),
        ({"stride": True}, "positive integer"),
        ({"start_time": np.inf}, "start_time must be finite"),
        ({"start_time": 3.0, "stop_time": 2.0}, "greater than start_time"),
        ({"start_time": 4.0, "stop_time": 5.0, "stride": 2}, "At least two"),
    ):
        with pytest.raises(ValueError, match=message):
            load_metadynamics_data(
                path,
                cv_fields=("s",),
                logweight_field="logw",
                **options,
            )


def test_reweighted_analysis_is_stable_and_reports_evidence_not_a_verdict(
    tmp_path: Path,
) -> None:
    time = np.arange(12, dtype=float)
    coordinate = np.linspace(-1.0, 1.0, len(time))
    # The large additive constant must not overflow or alter normalized weights.
    log_weights = 1000.0 + np.log(np.linspace(1.0, 2.0, len(time)))
    path = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "s", "logw"),
        np.column_stack([time, coordinate, log_weights]),
    )

    result = analyze_metadynamics(
        path,
        cv_fields=("s",),
        logweight_field="logw",
        thermal_energy=0.025,
        bins=4,
        n_blocks=3,
    )

    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.all(np.isfinite(result["free_energy"][result["support_mask"]]))
    assert 1.0 <= result["global_ess"] <= len(time)
    assert len(result["time_blocks"]) == 3
    assert result["convergence_evidence"]["successive_block_shape_rms"].shape == (2,)
    assert "converged" not in result
    assert "converged" not in result["convergence_evidence"]


def test_extreme_logweights_preserve_tiny_bins_in_full_and_block_fes(
    tmp_path: Path,
) -> None:
    pattern_positions = np.array([0.25, 0.25, 1.25, 2.25, 2.25])
    pattern_log_weights = np.array([-100.0, -101.0, 0.0, -100.0, -102.0])
    positions = np.tile(pattern_positions, 2)
    log_weights = 1000.0 + np.tile(pattern_log_weights, 2)
    path = _write_plumed_table(
        tmp_path / "COLVAR_extreme",
        ("time", "s", "logw"),
        np.column_stack([np.arange(10.0), positions, log_weights]),
    )
    result = analyze_metadynamics(
        path,
        cv_fields=("s",),
        logweight_field="logw",
        thermal_energy=0.025,
        bins=(np.array([0.0, 1.0, 2.0, 3.0]),),
        n_blocks=2,
    )

    assert np.all(result["support_mask"])
    assert np.all(np.isfinite(result["free_energy"]))
    assert result["probability_mass"][0] > 0.0
    assert result["probability_mass"][2] > 0.0
    assert np.all(np.isfinite(result["log_probability_mass"]))
    first_log_mass = -100.0 + np.log1p(np.exp(-1.0))
    last_log_mass = -100.0 + np.log1p(np.exp(-2.0))
    np.testing.assert_allclose(
        result["free_energy"],
        [-0.025 * first_log_mass, 0.0, -0.025 * last_log_mass],
    )
    first_ess = 2.0 * (1.0 + np.exp(-1.0)) ** 2 / (
        1.0 + np.exp(-2.0)
    )
    last_ess = 2.0 * (1.0 + np.exp(-2.0)) ** 2 / (
        1.0 + np.exp(-4.0)
    )
    np.testing.assert_allclose(result["local_ess"], [first_ess, 2.0, last_ess])
    np.testing.assert_allclose(
        result["local_weight_leverage"],
        [
            1.0 / (2.0 * (1.0 + np.exp(-1.0))),
            0.5,
            1.0 / (2.0 * (1.0 + np.exp(-2.0))),
        ],
    )
    assert all(np.all(block["support_mask"]) for block in result["time_blocks"])
    np.testing.assert_allclose(
        result["convergence_evidence"]["successive_block_shape_rms"],
        [0.0],
        atol=1e-14,
    )


def test_extreme_logweights_are_reduced_per_flat_nd_bin() -> None:
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    result = metadynamics._histogram_analysis(
        np.array([[0.25, 0.25], [1.25, 1.25], [2.25, 2.25]]),
        np.array([-100.0, 0.0, -100.0]),
        bins=(edges, edges),
        ranges=None,
        thermal_energy=0.025,
    )

    diagonal = (np.arange(3), np.arange(3))
    assert np.all(result["support_mask"][diagonal])
    assert np.all(np.isfinite(result["free_energy"][diagonal]))
    assert np.all(result["probability_mass"][diagonal] > 0.0)
    assert np.count_nonzero(result["support_mask"]) == 3
    assert np.all(np.isnan(
        result["local_weight_leverage"][~result["support_mask"]]
    ))


def test_hills_diagnostics_and_sum_hills_use_separate_safe_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hills = _write_plumed_table(
        tmp_path / "HILLS",
        ("time", "s", "sigma_s", "height", "biasf"),
        np.array([
            [0.0, -1.0, 0.1, 1.0, 10.0],
            [2.0, 0.0, 0.1, 0.5, 10.0],
            [4.0, 1.0, 0.1, 0.25, 10.0],
        ]),
    )
    diagnostics = load_hills_diagnostics(hills)
    assert diagnostics["n_hills"] == 3
    assert diagnostics["walkers"][0]["cv_names"] == ("s",)
    assert diagnostics["walkers"][0]["median_deposition_interval"] == 2.0
    assert diagnostics["walkers"][0]["height_ratio_final_to_initial"] == 0.25

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(metadynamics.subprocess, "run", fake_run)
    completed = run_sum_hills(
        hills,
        output_file=tmp_path / "fes.dat",
        bins=(50,),
        minimum=(-1.0,),
        maximum=(1.0,),
        stride=10,
        thermal_energy=0.025,
    )

    assert completed.returncode == 0
    command, options = calls[0]
    assert command[:2] == ["plumed", "sum_hills"]
    assert command[command.index("--hills") + 1] == str(hills)
    assert command[command.index("--bin") + 1] == "50"
    assert "--mintozero" in command
    assert "shell" not in options
    assert options == {"check": True, "capture_output": True, "text": True}


def test_analysis_applies_the_same_bias_conversion_to_hills_heights(
    tmp_path: Path,
) -> None:
    colvar = _write_plumed_table(
        tmp_path / "COLVAR",
        ("time", "s", "logw"),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
    )
    hills = _write_plumed_table(
        tmp_path / "HILLS",
        ("time", "s", "sigma_s", "height"),
        np.array([[0.0, 0.0, 0.1, 1.5], [1.0, 1.0, 0.1, 0.5]]),
    )

    result = analyze_metadynamics(
        colvar,
        cv_fields=("s",),
        logweight_field="logw",
        thermal_energy=0.025,
        bias_energy_factor=2.0,
        bins=2,
        n_blocks=1,
        hills_files=hills,
    )

    hills_result = result["hills_diagnostics"]
    assert hills_result["bias_energy_factor"] == pytest.approx(2.0)
    np.testing.assert_allclose(hills_result["walkers"][0]["heights"], [3.0, 1.0])
