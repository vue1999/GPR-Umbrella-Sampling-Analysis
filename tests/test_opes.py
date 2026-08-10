"""Synthetic T300-compatible tests for the OPES adapter."""

from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella.opes import (
    analyze_opes_1d,
    free_energy_from_opes_bias,
    load_opes_colvar,
    opes_convergence_diagnostics,
    read_opes_state,
)


FIELDS = (
    "time", "HH", "RELZ", "path.s", "path.z", "dz2", "corner",
    "wall_path.bias", "wall_hh_lo.bias", "wall_hh_hi.bias",
    "wall_relz_lo.bias", "wall_relz_hi.bias", "wall_corner.bias",
    "wall_dz.bias", "opes.bias", "opes.rct", "opes.zed", "opes.neff",
    "opes.nker",
)
WALL_FIELDS = FIELDS[7:14]


def _row(time, path_s, walls, opes_bias, rct, zed, neff, nker):
    values = [
        time, 1.0, -2.0, path_s, 0.1, 0.0, 0.0,
        *walls, opes_bias, rct, zed, neff, nker,
    ]
    assert len(values) == len(FIELDS)
    return " ".join(str(value) for value in values)


@pytest.fixture
def synthetic_t300_colvar(tmp_path):
    path = tmp_path / "COLVAR"
    header = "#! FIELDS " + " ".join(FIELDS)
    path.write_text(
        "\n".join([
            header,
            _row(0, 15.5, [0] * 7, -1.5, -1.5, 1.0, 1.0, 0),
            _row(5, 7.0, [0] * 7, -0.5, 0.2, 0.95, 5.0, 1),
            header,
            _row(
                5, 6.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                -0.4, 0.3, 0.94, 5.0, 2,
            ),
            _row(10, 1.0, [0] * 7, -0.2, 0.1, 0.90, 4.0, 2),
            "15 1.0 incomplete",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def test_load_opes_colvar_composes_every_explicit_bias(synthetic_t300_colvar):
    data = load_opes_colvar(
        synthetic_t300_colvar,
        cv_fields="path.s",
        other_bias_fields=WALL_FIELDS,
    )
    np.testing.assert_allclose(data["time"], [0, 5, 10])
    np.testing.assert_allclose(data["cv_values"][:, 0], [15.5, 6.0, 1.0])
    np.testing.assert_allclose(data["total_bias"], [-1.5, 2.4, -0.2])
    assert data["table"]["fields_headers_seen"] == 2
    assert data["table"]["duplicate_records_replaced"] == 1
    assert data["table"]["malformed_records_ignored"] == 1
    assert set(data["diagnostics"]) == {
        "opes.rct", "opes.zed", "opes.neff", "opes.nker"
    }
    assert len(data["available_bias_fields"]) == 8


def test_rct_is_rejected_as_a_reweighting_bias(synthetic_t300_colvar):
    with pytest.raises(ValueError, match="diagnostic only"):
        load_opes_colvar(
            synthetic_t300_colvar,
            cv_fields="path.s",
            other_bias_fields=("opes.rct",),
        )
    with pytest.raises(ValueError, match="diagnostic only"):
        load_opes_colvar(
            synthetic_t300_colvar,
            cv_fields="path.s",
            bias_field="opes.rct",
        )


def test_unlisted_biases_and_duplicate_cvs_are_rejected(synthetic_t300_colvar):
    with pytest.raises(ValueError, match="must be Boolean"):
        load_opes_colvar(
            synthetic_t300_colvar,
            cv_fields="path.s",
            allow_unlisted_bias_fields="false",
        )
    with pytest.raises(ValueError, match="unlisted bias fields"):
        load_opes_colvar(synthetic_t300_colvar, cv_fields="path.s")
    opted_out = load_opes_colvar(
        synthetic_t300_colvar,
        cv_fields="path.s",
        allow_unlisted_bias_fields=True,
    )
    assert set(opted_out["unlisted_bias_fields"]) == set(WALL_FIELDS)
    with pytest.raises(ValueError, match="cv_fields must be unique"):
        load_opes_colvar(
            synthetic_t300_colvar,
            cv_fields=("path.s", "path.s"),
            other_bias_fields=WALL_FIELDS,
        )


def test_opes_diagnostics_report_tail_changes_without_binary_verdict(
    synthetic_t300_colvar,
):
    data = load_opes_colvar(
        synthetic_t300_colvar,
        cv_fields="path.s",
        other_bias_fields=WALL_FIELDS,
    )
    diagnostics = opes_convergence_diagnostics(data, tail_fraction=1.0)
    neff = diagnostics["series"]["opes.neff"]
    assert neff["final"] == pytest.approx(4.0)
    assert neff["decrease_count"] == 1
    assert diagnostics["series"]["opes.nker"]["change_count"] == 1
    assert diagnostics["rct_used_for_reweighting"] is False
    assert "converged" not in diagnostics


def test_analyze_opes_cumulative_terminal_matches_full(synthetic_t300_colvar):
    result = analyze_opes_1d(
        synthetic_t300_colvar,
        cv_field="path.s",
        temperature=300.0,
        other_bias_fields=WALL_FIELDS,
        bins=np.array([0.0, 5.0, 10.0, 17.0]),
        cumulative_cutoffs=[5.0, 10.0],
    )
    assert result["method"] == "direct_total_bias_reweighting"
    assert result["rct_used_for_reweighting"] is False
    terminal = result["cumulative"]["snapshots"][-1]
    np.testing.assert_allclose(
        terminal["pmf"], result["pmf"]["pmf"], equal_nan=True
    )
    np.testing.assert_allclose(
        terminal["local_maximum_weight_fraction"],
        result["pmf"]["local_maximum_weight_fraction"],
    )


def test_analyze_opes_exposes_block_region_and_landmark_evidence(
    synthetic_t300_colvar,
):
    result = analyze_opes_1d(
        synthetic_t300_colvar,
        cv_field="path.s",
        temperature=300.0,
        other_bias_fields=WALL_FIELDS,
        bins=np.array([0.0, 5.0, 10.0, 17.0]),
        cumulative_cutoffs=[5.0, 10.0],
        n_blocks=2,
        regions={"transition": (5.0, 8.0)},
        basin_a=(0.0, 4.0),
        basin_b=(12.0, 16.0),
        transition_region=(5.0, 10.0),
    )

    assert result["blocks"]["n_blocks"] == 2
    assert result["regions"]["regions"]["transition"]["n_records"] == 1
    assert result["landmarks"]["status"] == "ok"
    assert result["landmarks"]["transition_region"]["status"] == "ok"
    assert result["cumulative"]["snapshots"][-1]["landmarks"]["status"] == "ok"
    assert result["convergence_evidence"]["disjoint_block_profiles"] is True
    assert "converged" not in result["convergence_evidence"]


def test_opes_landmark_regions_are_supplied_consistently(synthetic_t300_colvar):
    with pytest.raises(ValueError, match="supplied together"):
        analyze_opes_1d(
            synthetic_t300_colvar,
            cv_field="path.s",
            temperature=300.0,
            other_bias_fields=WALL_FIELDS,
            basin_a=(0.0, 4.0),
        )


def test_saved_state_bias_conversion_and_metadata(tmp_path):
    state = tmp_path / "opes.state"
    state.write_text(
        """#! FIELDS time path.s sigma_path.s height
#! SET action OPES_METAD_state
#! SET biasfactor 10
#! SET adaptive_counter 3000000
1500000 1.0 0.2 0.3
1500000 2.0 0.2 0.4
""",
        encoding="utf-8",
    )
    metadata = read_opes_state(state)
    assert metadata["action"] == "OPES_METAD_state"
    assert metadata["biasfactor"] == pytest.approx(10.0)
    assert metadata["adaptive_counter"] == 3_000_000
    assert metadata["n_kernels"] == 2

    np.testing.assert_allclose(
        free_energy_from_opes_bias([0.0, -0.9, -1.8], 10.0),
        [0.0, 1.0, 2.0],
    )
    np.testing.assert_allclose(
        free_energy_from_opes_bias([0.0, -0.9], np.inf), [0.0, 0.9]
    )
    with pytest.raises(ValueError, match="greater than one"):
        free_energy_from_opes_bias([0.0, -1.0], 1.0)


def test_saved_state_rejects_conflicting_or_kernel_free_metadata(tmp_path):
    conflicting = tmp_path / "conflicting.state"
    conflicting.write_text(
        """#! FIELDS time s sigma_s height
#! SET biasfactor 10
#! SET biasfactor 20
0 1 0.2 0.3
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting biasfactor"):
        read_opes_state(conflicting)

    empty = tmp_path / "empty.state"
    empty.write_text(
        "#! FIELDS time s sigma_s height\n#! SET biasfactor 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no kernels"):
        read_opes_state(empty)

    wrong_action = tmp_path / "wrong-action.state"
    wrong_action.write_text(
        "#! FIELDS time s sigma_s height\n"
        "#! SET action METAD_state\n"
        "#! SET biasfactor 10\n"
        "0 1 0.2 0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="OPES_METAD_state"):
        read_opes_state(wrong_action)


def test_bias_unit_is_independent_from_requested_pmf_unit(tmp_path):
    fields = "#! FIELDS time path.s opes.bias"
    times = np.arange(6, dtype=float)
    positions = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 1.8])
    bias_ev = np.array([0.0, 0.01, 0.02, -0.01, 0.03, 0.015])
    ev_to_kj_per_mol = (
        8.31446261815324e-3 / 8.617333262145e-5
    )

    def write_colvar(path, biases):
        rows = [fields]
        rows.extend(
            f"{time} {position} {bias}"
            for time, position, bias in zip(times, positions, biases)
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    ev_path = write_colvar(tmp_path / "COLVAR.eV", bias_ev)
    kj_path = write_colvar(
        tmp_path / "COLVAR.kJmol", bias_ev * ev_to_kj_per_mol
    )
    common = {
        "cv_field": "path.s",
        "temperature": 300.0,
        "energy_unit": "eV",
        "bins": np.array([0.0, 1.0, 2.0]),
    }
    result_ev = analyze_opes_1d(ev_path, **common)
    result_kj = analyze_opes_1d(
        kj_path, bias_energy_unit="kJ/mol", **common
    )

    np.testing.assert_allclose(
        result_kj["log_weights"], result_ev["log_weights"], rtol=1e-12
    )
    np.testing.assert_allclose(
        result_kj["pmf"]["normalized_weights"],
        result_ev["pmf"]["normalized_weights"],
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result_kj["pmf"]["pmf"], result_ev["pmf"]["pmf"], rtol=1e-12
    )
    assert result_ev["bias_energy_unit"] == "eV"
    assert result_ev["unit_provenance"]["bias_energy_unit_source"] == (
        "output_energy_unit_default"
    )
    assert result_kj["bias_energy_unit"] == "kJ/mol"
    assert result_kj["energy_unit"] == "eV"
    assert result_kj["unit_provenance"] == {
        "pmf_energy_unit": "eV",
        "bias_energy_unit": "kJ/mol",
        "bias_energy_unit_source": "explicit_bias_energy_unit",
        "loader_applied_unit_conversion": False,
    }


def test_opes_selection_is_applied_after_restart_deduplication(
    synthetic_t300_colvar,
):
    data = load_opes_colvar(
        synthetic_t300_colvar,
        cv_fields="path.s",
        other_bias_fields=WALL_FIELDS,
        start_time=0.0,
        stop_time=10.0,
        stride=2,
    )

    np.testing.assert_allclose(data["time"], [0.0, 10.0])
    np.testing.assert_allclose(data["cv_values"][:, 0], [15.5, 1.0])
    np.testing.assert_allclose(data["total_bias"], [-1.5, -0.2])
    assert data["selection"] == {
        "requested_start_time": 0.0,
        "requested_stop_time": 10.0,
        "stride": 2,
        "n_records_before_selection": 3,
        "n_records_within_time_bounds": 3,
        "n_records_selected": 2,
        "selected_start_time": 0.0,
        "selected_stop_time": 10.0,
        "explicit_finite_time_range": True,
        "bounds_inclusive": True,
        "selection_order": (
            "restart_deduplication", "time_bounds", "stride"
        ),
    }

    result = analyze_opes_1d(
        synthetic_t300_colvar,
        cv_field="path.s",
        temperature=300.0,
        other_bias_fields=WALL_FIELDS,
        start_time=0.0,
        stop_time=10.0,
        stride=2,
        bins=np.array([0.0, 5.0, 10.0, 17.0]),
    )
    assert len(result["log_weights"]) == 2
    assert result["opes_data"]["selection"]["n_records_selected"] == 2


def test_custom_opes_time_field_must_be_strictly_increasing(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text(
        "#! FIELDS clock path.s opes.bias\n"
        "0 15 0\n"
        "2 7 -0.1\n"
        "1 1 -0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_opes_colvar(
            path,
            cv_fields="path.s",
            time_field="clock",
        )


def test_analyze_opes_reports_raw_hysteretic_transitions_and_block_support(
    tmp_path,
):
    path = tmp_path / "COLVAR"
    path.write_text(
        "#! FIELDS time path.s opes.bias\n"
        "0 15 0\n"
        "1 7 -0.1\n"
        "2 1 -0.2\n"
        "3 7 -0.1\n"
        "4 14 0\n",
        encoding="utf-8",
    )

    result = analyze_opes_1d(
        path,
        cv_field="path.s",
        temperature=300.0,
        bins=np.array([0.0, 5.0, 10.0, 17.0]),
        n_blocks=2,
        basin_a=(10.0, 16.0),
        basin_b=(0.9, 2.5),
        transition_region=(5.0, 9.0),
    )

    transitions = result["transitions"]
    assert transitions["forward_events"] == 1
    assert transitions["reverse_events"] == 1
    assert transitions["completed_round_trips"] == 1
    assert [event["time"] for event in transitions["events"]] == [2.0, 4.0]
    assert result["convergence_evidence"]["completed_round_trips"] == 1
    block_support = result["blocks"]["landmark_support"]
    assert block_support[0]["basin_b_status"] == "ok"
    assert block_support[1]["basin_b_status"] == "insufficient_support"
    assert block_support[1]["basin_b_supported_bins"] == 0
