"""Contract tests for the unified biased-sampling CLI."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

import gpr_umbrella.cli_biased as cli_biased
from gpr_umbrella.biased_sampling import thermal_energy


def _assert_no_converged_key(value):
    if isinstance(value, dict):
        assert "converged" not in value
        for item in value.values():
            _assert_no_converged_key(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_converged_key(item)


def _fake_opes_result():
    return {
        "method": "direct_total_bias_reweighting",
        "opes_data": {
            "time": np.arange(4.0),
            "bias_fields": ("opes.bias", "wall.bias"),
            "available_bias_fields": ("opes.bias", "wall.bias"),
            "unlisted_bias_fields": (),
            "table": {
                "n_records": 5,
                "files": ("COLVAR.0", "COLVAR.1"),
                "fields_headers_seen": 2,
                "duplicate_records_replaced": 1,
                "malformed_records_ignored": 0,
            },
            "selection": {
                "requested_start_time": 10.0,
                "requested_stop_time": 40.0,
                "stride": 2,
                "n_records_before_selection": 5,
                "n_records_within_time_bounds": 4,
                "n_records_selected": 4,
                "selected_start_time": 10.0,
                "selected_stop_time": 40.0,
                "explicit_finite_time_range": True,
                "bounds_inclusive": True,
                "selection_order": (
                    "restart_deduplication", "time_bounds", "stride"
                ),
            },
        },
        "pmf": {
            "bin_centers": np.array([0.5, 1.5]),
            "pmf": np.array([0.2, 0.0]),
            "density": np.array([0.25, 0.75]),
            "local_ess": np.array([2.0, 1.2]),
            "local_maximum_weight_fraction": np.array([0.5, 0.9]),
            "support_mask": np.array([True, True]),
            "importance_weight_ess": 2.5,
            "maximum_weight_fraction": 0.6,
        },
        "cumulative": {"snapshots": [{"status": "ok"}, {"status": "ok"}]},
        "blocks": {
            "n_blocks": 2,
            "successive_comparisons": [{
                "status": "ok", "offset_aligned_rms": 0.03,
                "common_support_bins": 2,
            }],
            "landmark_support": [
                {
                    "basin_a_status": "ok", "basin_b_status": "ok",
                    "population_delta_f_status": "ok",
                },
                {
                    "basin_a_status": "ok",
                    "basin_b_status": "insufficient_support",
                    "population_delta_f_status": "unavailable",
                },
            ],
        },
        "regions": {
            "regions": {
                "transition": {
                    "status": "ok", "n_records": 2, "weight_mass": 0.1,
                    "kish_ess": 1.2, "maximum_local_leverage": 0.9,
                }
            }
        },
        "transitions": {
            "initial_state": "basin_a",
            "final_state": "basin_a",
            "forward_events": 1,
            "reverse_events": 1,
            "completed_round_trips": 1,
            "last_event_time": 30.0,
            "basin_evidence": {
                "basin_a": {
                    "n_qualifying_records": 2,
                    "last_qualifying_time": 40.0,
                },
                "basin_b": {
                    "n_qualifying_records": 1,
                    "last_qualifying_time": 20.0,
                },
            },
        },
        "opes_diagnostics": {
            "series": {
                "opes.rct": {"final": 0.1, "tail_drift": 0.01},
                "opes.neff": {"final": 12.0, "tail_drift": 2.0},
            }
        },
        "rct_used_for_reweighting": False,
    }


def _fake_metad_result():
    return {
        "weight_source": "quasistatic_raw_bias",
        "time": np.arange(4.0),
        "global_ess": 3.2,
        "global_ess_fraction": 0.8,
        "maximum_weight_fraction": 0.4,
        "convergence_evidence": {
            "n_time_blocks": 2,
            "successive_block_shape_rms": np.array([np.nan, 0.03]),
        },
        "validity": {
            "normalized_time_dependent_bias": False,
            "quasistatic_bias_assumed": True,
            "quasistatic_segment_explicitly_bounded": True,
        },
        "selection": {
            "requested_start_time": 100.0,
            "requested_stop_time": 700.0,
            "stride": 2,
            "n_records_before_selection": 10,
            "n_records_within_time_bounds": 7,
            "n_records_selected": 4,
            "selected_start_time": 100.0,
            "selected_stop_time": 700.0,
            "explicit_finite_time_range": True,
            "bounds_inclusive": True,
            "selection_order": (
                "restart_deduplication", "time_bounds", "stride"
            ),
        },
        "bin_centers": (np.array([-0.5, 0.5]),),
        "free_energy": np.array([0.0, 0.1]),
        "local_ess": np.array([2.0, 1.5]),
        "local_weight_leverage": np.array([0.5, 0.8]),
        "counts": np.array([2, 2]),
        "support_mask": np.array([True, True]),
        "hills_diagnostics": {"n_files": 2, "n_hills": 8},
    }


def _fake_icf_result():
    return {
        "method": "icf_gradient_gpr",
        "n_points": 3,
        "n_dimensions": 1,
        "prediction_points": np.array([[-1.0], [0.0], [1.0]]),
        "sigma_f": 1.2,
        "lengthscale": np.array([0.5]),
        "loo_calibration_factor": 1.1,
        "aggregation": {"kind": "local_spatial_bins"},
        "source_file": "COLVAR",
        "cv_names": ("path.s",),
        "icf_fields": ("path_s.icf",),
        "force_convention": "thermodynamic_force",
        "gradient_relation": "grad_A = -conditional_mean(ICF)",
        "reference_source": "lowest_prediction_on_observation_support",
        "unit_provenance": {
            "cv_factors": np.array([10.0]),
            "energy_factor": 0.5,
            "icf_factors": np.array([0.05]),
            "icf_factors_source": "derived_energy_factor_over_cv_factors",
            "force_scaling_relation": "icf_factor = energy_factor / cv_factor",
        },
        "icf_validity": {
            "paper_assumptions_satisfied": True,
            "paper_equation_8_applicable": True,
            "icf_definition_complete": True,
        },
        "x_star": np.array([-1.0, 0.0, 1.0]),
        "pmf": np.array([1.0, 0.0, 1.0]),
        "pmf_std": np.array([0.2, 0.0, 0.2]),
        "pmf_std_raw": np.array([0.1, 0.0, 0.1]),
        "pmf_std_calibrated": np.array([0.2, 0.0, 0.2]),
    }


def test_help_lists_all_three_subcommands(capsys):
    with pytest.raises(SystemExit) as help_exit:
        cli_biased.build_parser().parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "opes" in help_text
    assert "metad" in help_text
    assert "icf" in help_text


def test_metad_requires_exactly_one_weight_and_thermal_source():
    parser = cli_biased.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "metad", "COLVAR", "--cv-field", "s", "--thermal-energy", "0.025"
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "metad", "COLVAR", "--cv-field", "s",
            "--logweight-field", "logw", "--rbias-field", "metad.rbias",
            "--thermal-energy", "0.025",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "metad", "COLVAR", "--cv-field", "s",
            "--logweight-field", "logw",
        ])


def test_opes_forwards_strict_bias_controls_and_writes_json_safe_outputs(
    tmp_path, monkeypatch, capsys,
):
    calls = []
    monkeypatch.setattr(
        cli_biased,
        "analyze_opes_1d",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _fake_opes_result(),
    )
    output_dir = tmp_path / "opes-output"
    assert cli_biased.main([
        "opes", "COLVAR.0", "COLVAR.1",
        "--cv-field", "path.s",
        "--other-bias-field", "wall_path.bias",
        "--other-bias-field", "wall_relz.bias",
        "--temperature", "300",
        "--energy-unit", "eV",
        "--bias-energy-unit", "kJ/mol",
        "--start-time", "10",
        "--stop-time", "40",
        "--stride", "2",
        "--bins", "25",
        "--range", "0", "16",
        "--cumulative-cutoff", "100",
        "--cumulative-cutoff", "200",
        "--n-blocks", "3",
        "--basin-a", "0", "2",
        "--basin-b", "12", "16",
        "--transition-region", "5", "9",
        "--allow-unlisted-bias-fields",
        "--strict",
        "--output-dir", str(output_dir),
    ]) == 0

    positional, options = calls[0]
    assert positional == (("COLVAR.0", "COLVAR.1"),)
    assert options["other_bias_fields"] == (
        "wall_path.bias", "wall_relz.bias"
    )
    assert options["allow_unlisted_bias_fields"] is True
    assert options["strict"] is True
    assert options["value_range"] == (0.0, 16.0)
    assert options["bias_energy_unit"] == "kJ/mol"
    assert options["start_time"] == pytest.approx(10.0)
    assert options["stop_time"] == pytest.approx(40.0)
    assert options["stride"] == 2
    assert options["cumulative_cutoffs"] == [100.0, 200.0]
    assert options["n_blocks"] == 3
    assert options["basin_a"] == (0.0, 2.0)
    assert options["basin_b"] == (12.0, 16.0)
    assert options["transition_region"] == (5.0, 9.0)
    assert options["regions"] == {
        "basin_a": (0.0, 2.0),
        "basin_b": (12.0, 16.0),
        "transition": (5.0, 9.0),
    }

    assert (output_dir / "opes_pmf_1d.dat").is_file()
    summary = json.loads((output_dir / "opes_summary.json").read_text())
    _assert_no_converged_key(summary)
    assert summary["source_files"] == ["COLVAR.0", "COLVAR.1"]
    assert summary["bias_fields"] == ["opes.bias", "wall.bias"]
    assert summary["records"] == 4
    assert summary["completed_round_trips"] == 1
    assert summary["basin_b_last_qualifying_time"] == pytest.approx(20.0)
    assert summary["blocks_with_both_basins_supported"] == 1
    assert summary["region_transition_kish_ess"] == pytest.approx(1.2)
    assert "converged" not in capsys.readouterr().out.lower()


def test_metad_bounded_raw_opt_in_extras_and_hills_are_forwarded(
    tmp_path, monkeypatch, capsys,
):
    calls = []
    monkeypatch.setattr(
        cli_biased,
        "analyze_metadynamics",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _fake_metad_result(),
    )
    output_dir = tmp_path / "metad-output"
    assert cli_biased.main([
        "metad", "COLVAR",
        "--cv-field", "s",
        "--raw-bias-field", "metad.bias",
        "--allow-quasistatic-bias",
        "--start-time", "100",
        "--stop-time", "700",
        "--stride", "2",
        "--extra-bias-field", "wall1.bias",
        "--extra-bias-field", "wall2.bias",
        "--allow-unlisted-bias-fields",
        "--strict",
        "--hills", "HILLS.0",
        "--hills-file", "HILLS.1",
        "--temperature", "300",
        "--energy-unit", "eV",
        "--range", "-1", "1",
        "--n-blocks", "2",
        "--output-dir", str(output_dir),
    ]) == 0
    positional, options = calls[0]
    assert positional == ("COLVAR",)
    assert options["raw_bias_field"] == "metad.bias"
    assert options["allow_quasistatic_bias"] is True
    assert options["start_time"] == pytest.approx(100.0)
    assert options["stop_time"] == pytest.approx(700.0)
    assert options["stride"] == 2
    assert options["extra_bias_fields"] == ("wall1.bias", "wall2.bias")
    assert options["allow_unlisted_bias_fields"] is True
    assert options["strict"] is True
    assert options["hills_files"] == ("HILLS.0", "HILLS.1")
    assert options["thermal_energy"] == pytest.approx(thermal_energy(300.0))
    assert options["ranges"] == ((-1.0, 1.0),)
    summary = json.loads((output_dir / "metad_summary.json").read_text())
    assert summary["requested_start_time"] == pytest.approx(100.0)
    assert summary["requested_stop_time"] == pytest.approx(700.0)
    assert summary["selected_start_time"] == pytest.approx(100.0)
    assert summary["selected_stop_time"] == pytest.approx(700.0)
    assert summary["stride"] == 2
    assert summary["n_records_before_selection"] == 10
    assert summary["n_records_within_time_bounds"] == 7
    assert summary["n_records_selected"] == 4
    assert summary["explicit_finite_time_range"] is True
    assert summary["quasistatic_segment_explicitly_bounded"] is True
    assert summary["selection_order"] == (
        "restart_deduplication -> time_bounds -> stride"
    )
    _assert_no_converged_key(summary)
    assert "converged" not in capsys.readouterr().out.lower()


def test_raw_metad_bias_without_opt_in_is_a_controlled_error(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli_biased,
        "analyze_metadynamics",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert cli_biased.main([
        "metad", "COLVAR", "--cv-field", "s",
        "--raw-bias-field", "metad.bias", "--thermal-energy", "0.025",
    ]) == 1
    assert calls == []
    assert "allow-quasistatic-bias" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bounds",
    (
        (),
        ("--start-time", "0"),
        ("--stop-time", "100"),
    ),
)
def test_raw_metad_bias_requires_both_finite_time_bounds(
    bounds, monkeypatch, capsys,
):
    calls = []
    monkeypatch.setattr(
        cli_biased,
        "analyze_metadynamics",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert cli_biased.main([
        "metad", "COLVAR", "--cv-field", "s",
        "--raw-bias-field", "metad.bias",
        "--allow-quasistatic-bias",
        "--thermal-energy", "0.025",
        *bounds,
    ]) == 1
    assert calls == []
    error = capsys.readouterr().err
    assert "--start-time" in error
    assert "--stop-time" in error


def test_icf_forwards_aggregation_noise_and_paper_validity_flags(
    tmp_path, monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        cli_biased,
        "reconstruct_pmf_icf",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _fake_icf_result(),
    )
    output_dir = tmp_path / "icf-output"
    assert cli_biased.main([
        "icf", "COLVAR",
        "--cv-field", "path.s",
        "--icf-field", "path_s.icf",
        "--cv-factor", "10",
        "--energy-factor", "0.5",
        "--aggregation-bins", "4",
        "--aggregation-range", "0", "16",
        "--time-block-size", "100",
        "--min-samples-per-observation", "3",
        "--force-error", "0.05",
        "--grid-n", "21",
        "--lengthscale", "0.5",
        "--sigma-f", "1.2",
        "--no-optimize",
        "--no-calibrate",
        "--bias-depends-only-on-modeled-cvs",
        "--quasi-equilibrium",
        "--physical-force-excludes-bias",
        "--metric-correction-included",
        "--require-paper-assumptions",
        "--output-dir", str(output_dir),
    ]) == 0

    positional, options = calls[0]
    assert positional == ("COLVAR",)
    assert options["cv_fields"] == ("path.s",)
    assert options["icf_fields"] == ("path_s.icf",)
    assert options["cv_factors"] == pytest.approx(10.0)
    assert options["energy_factor"] == pytest.approx(0.5)
    assert options["icf_factors"] is None
    assert options["aggregation_bins"] == 4
    assert options["aggregation_ranges"] == ((0.0, 16.0),)
    assert options["force_errors"] == pytest.approx(0.05)
    assert options["optimize_hyperparams"] is False
    assert options["calibrate_uncertainty"] is False
    assert options["require_paper_assumptions"] is True
    assert all(options[name] is True for name in (
        "bias_depends_only_on_modeled_cvs", "quasi_equilibrium",
        "physical_force_excludes_bias", "metric_correction_included",
    ))
    assert (output_dir / "icf_pmf_1d.dat").is_file()
    summary = json.loads((output_dir / "icf_summary.json").read_text())
    _assert_no_converged_key(summary)
    assert summary["cv_factors"] == [10.0]
    assert summary["energy_factor"] == pytest.approx(0.5)
    assert summary["icf_factors"] == [0.05]
    assert summary["reference_source"] == (
        "lowest_prediction_on_observation_support"
    )


def test_icf_cli_requires_paper_assumptions_by_default_and_names_unsafe_opt_out(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        cli_biased,
        "reconstruct_pmf_icf",
        lambda *args, **kwargs: calls.append(kwargs) or _fake_icf_result(),
    )
    base = [
        "icf", "COLVAR", "--cv-field", "s", "--icf-field", "s.icf",
        "--force-error", "0.1",
    ]
    assert cli_biased.main(base) == 0
    assert calls[-1]["require_paper_assumptions"] is True
    assert calls[-1]["allow_uncertainty_downscaling"] is False

    assert cli_biased.main(
        base + [
            "--allow-unverified-paper-assumptions",
            "--allow-uncertainty-downscaling",
            "--strict",
        ]
    ) == 0
    assert calls[-1]["require_paper_assumptions"] is False
    assert calls[-1]["allow_uncertainty_downscaling"] is True
    assert calls[-1]["strict"] is True


def test_subprocess_failures_are_reported_as_exit_code_one(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(2, ["plumed", "sum_hills"])

    monkeypatch.setattr(cli_biased, "analyze_metadynamics", fail)
    assert cli_biased.main([
        "metad", "COLVAR", "--cv-field", "s",
        "--logweight-field", "logw", "--thermal-energy", "0.025",
    ]) == 1
    assert "Error:" in capsys.readouterr().err
