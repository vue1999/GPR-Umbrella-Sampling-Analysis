"""Tests for shared direct-reweighting and transition utilities."""

from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella.biased_sampling import (
    block_reweighted_pmf_1d,
    cumulative_reweighted_pmf_1d,
    detect_hysteretic_transitions,
    importance_weight_diagnostics,
    log_reweighting_weights,
    normalize_log_weights,
    pmf_landmarks_1d,
    region_weight_diagnostics,
    reweighted_pmf_1d,
    thermal_energy,
)


def test_log_weight_normalization_is_stable_and_shift_invariant():
    log_weights = np.array([1000.0, 1000.0 - np.log(2.0)])
    weights = normalize_log_weights(log_weights)
    np.testing.assert_allclose(weights, [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(
        normalize_log_weights(log_weights + 1.0e8), weights, rtol=1e-7
    )
    diagnostics = importance_weight_diagnostics(weights)
    assert diagnostics["importance_weight_ess"] == pytest.approx(1.8)
    assert diagnostics["maximum_weight_fraction"] == pytest.approx(2.0 / 3.0)


def test_bias_is_converted_to_log_weights_with_consistent_units():
    kbt = thermal_energy(300.0, "eV")
    np.testing.assert_allclose(
        log_reweighting_weights(
            np.array([0.0, kbt]), temperature=300.0, energy_unit="eV"
        ),
        [0.0, 1.0],
    )
    assert thermal_energy(300.0, "kJ/mol") == pytest.approx(
        300.0 * 8.31446261815324e-3
    )


def test_reweighted_pmf_reports_local_ess_and_weight_leverage():
    values = np.array([0.25, 0.25, 1.25, 1.25])
    log_weights = np.log([1.0, 1.0, 2.0, 2.0])
    result = reweighted_pmf_1d(
        values,
        log_weights,
        bins=np.array([0.0, 1.0, 2.0, 3.0]),
        temperature=300.0,
    )
    kbt = thermal_energy(300.0)
    assert result["pmf"][0] == pytest.approx(kbt * np.log(2.0))
    assert result["pmf"][1] == pytest.approx(0.0)
    assert np.isnan(result["pmf"][2])
    np.testing.assert_allclose(result["local_ess"][:2], [2.0, 2.0])
    np.testing.assert_allclose(
        result["local_maximum_weight_fraction"][:2], [0.5, 0.5]
    )
    assert np.isnan(result["local_maximum_weight_fraction"][2])
    assert not result["support_mask"][2]


def test_reweighted_pmf_preserves_tiny_bins_around_a_dominant_middle_bin():
    """Per-bin log reduction avoids weighted-histogram cancellation."""
    values = np.array([0.25, 0.25, 1.25, 2.25, 2.25])
    log_weights = np.array([-100.0, -101.0, 0.0, -100.0, -102.0])
    result = reweighted_pmf_1d(
        values,
        log_weights,
        bins=np.array([0.0, 1.0, 2.0, 3.0]),
        temperature=300.0,
    )

    assert np.all(result["support_mask"])
    assert np.all(np.isfinite(result["pmf"]))
    assert result["weighted_mass"][0] > 0.0
    assert result["weighted_mass"][2] > 0.0
    kbt = thermal_energy(300.0)
    first_log_mass = -100.0 + np.log1p(np.exp(-1.0))
    last_log_mass = -100.0 + np.log1p(np.exp(-2.0))
    assert result["pmf"][0] == pytest.approx(-kbt * first_log_mass)
    assert result["pmf"][1] == pytest.approx(0.0)
    assert result["pmf"][2] == pytest.approx(-kbt * last_log_mass)

    expected_first_ess = (1.0 + np.exp(-1.0)) ** 2 / (
        1.0 + np.exp(-2.0)
    )
    expected_last_ess = (1.0 + np.exp(-2.0)) ** 2 / (
        1.0 + np.exp(-4.0)
    )
    np.testing.assert_allclose(
        result["local_ess"], [expected_first_ess, 1.0, expected_last_ess]
    )
    np.testing.assert_allclose(
        result["local_maximum_weight_fraction"],
        [1.0 / (1.0 + np.exp(-1.0)), 1.0, 1.0 / (1.0 + np.exp(-2.0))],
    )

    landmarks = pmf_landmarks_1d(
        result,
        basin_a=(0.0, 1.0),
        basin_b=(2.0, 3.0),
        transition_region=(1.0, 2.0),
    )
    assert landmarks["status"] == "ok"
    assert landmarks["basin_a"]["status"] == "ok"
    assert landmarks["basin_b"]["status"] == "ok"
    assert landmarks["population_delta_f_status"] == "ok"


def test_cumulative_terminal_snapshot_matches_full_estimate():
    values = np.array([0.25, 0.25, 1.25, 1.25])
    times = np.arange(4.0)
    log_weights = np.log([1.0, 1.0, 2.0, 2.0])
    edges = np.array([0.0, 1.0, 2.0])
    cumulative = cumulative_reweighted_pmf_1d(
        values,
        times,
        log_weights,
        [1.0, 3.0],
        bins=edges,
        temperature=300.0,
    )
    full = reweighted_pmf_1d(
        values, log_weights, bins=edges, temperature=300.0
    )
    terminal = cumulative["snapshots"][-1]
    assert terminal["status"] == "ok"
    np.testing.assert_allclose(terminal["pmf"], full["pmf"], equal_nan=True)
    np.testing.assert_allclose(terminal["local_ess"], full["local_ess"])


def test_hysteretic_transition_region_retains_the_previous_state():
    state_a = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
    state_b = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=bool)
    result = detect_hysteretic_transitions(
        state_a,
        state_b,
        times=np.arange(8) * 0.5,
        state_names=("adsorbed", "desorbed"),
    )
    assert result["states"][2] == "adsorbed"
    assert result["states"][5] == "desorbed"
    assert result["forward_events"] == 2
    assert result["reverse_events"] == 1
    assert result["completed_round_trips"] == 1
    assert result["initial_state"] == "adsorbed"
    assert result["final_state"] == "desorbed"
    assert result["last_event_time"] == pytest.approx(3.5)
    assert result["basin_evidence"]["adsorbed"] == {
        "n_qualifying_records": 3,
        "first_qualifying_index": 0,
        "last_qualifying_index": 6,
        "first_qualifying_time": 0.0,
        "last_qualifying_time": 3.0,
    }
    assert result["basin_evidence"]["desorbed"][
        "last_qualifying_time"
    ] == pytest.approx(3.5)
    assert [event["time"] for event in result["events"]] == [1.5, 3.0, 3.5]


def test_overlapping_basin_masks_are_rejected():
    with pytest.raises(ValueError, match="must not overlap"):
        detect_hysteretic_transitions([True], [True])


def test_transition_times_must_be_chronological():
    with pytest.raises(ValueError, match="nondecreasing"):
        detect_hysteretic_transitions(
            [True, False], [False, True], times=[1.0, 0.0]
        )


def test_region_weight_diagnostics_report_mass_ess_and_local_leverage():
    values = np.array([0.1, 0.2, 1.1, 1.2, 2.1])
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.0]) * 7.0
    result = region_weight_diagnostics(
        values,
        weights,
        {
            "basin_a": (0.0, 0.5),
            "basin_b": (1.0, 1.5),
            "zero_weight": (2.0, 2.5),
            "unvisited": (3.0, 4.0),
        },
    )

    basin_a = result["regions"]["basin_a"]
    assert basin_a["n_records"] == 2
    assert basin_a["weight_mass"] == pytest.approx(0.3)
    assert basin_a["kish_ess"] == pytest.approx(1.8)
    assert basin_a["maximum_local_leverage"] == pytest.approx(2.0 / 3.0)

    basin_b = result["regions"]["basin_b"]
    assert basin_b["n_records"] == 2
    assert basin_b["weight_mass"] == pytest.approx(0.7)
    assert basin_b["kish_ess"] == pytest.approx(0.7**2 / (0.3**2 + 0.4**2))
    assert basin_b["maximum_local_leverage"] == pytest.approx(4.0 / 7.0)

    zero_weight = result["regions"]["zero_weight"]
    assert zero_weight["status"] == "zero_weight"
    assert zero_weight["n_records"] == 1
    assert zero_weight["kish_ess"] == 0.0
    assert zero_weight["maximum_local_leverage"] is None
    assert result["regions"]["unvisited"]["status"] == "no_records"
    assert result["covered_weight_mass"] == pytest.approx(1.0)


def test_block_pmfs_are_disjoint_and_offset_aligned_on_common_support():
    values = np.array([
        0.25, 0.25, 0.25, 1.25, 1.25, 1.25,
        0.25, 0.25, 1.25, 1.25, 2.25, 2.25,
    ])
    weights = np.array([
        1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3,
        1 / 2, 1 / 2, 1.0, 1.0, 50.0, 50.0,
    ])
    result = block_reweighted_pmf_1d(
        values,
        np.log(weights),
        n_blocks=2,
        bins=np.array([0.0, 1.0, 2.0, 3.0]),
        temperature=300.0,
    )

    first, second = result["blocks"]
    assert (first["start_index"], first["stop_index_exclusive"]) == (0, 6)
    assert (second["start_index"], second["stop_index_exclusive"]) == (6, 12)
    comparison = result["successive_comparisons"][0]
    assert comparison["status"] == "ok"
    assert comparison["common_support_bins"] == 2
    assert abs(comparison["offset_to_previous"]) > 0.01
    assert comparison["offset_aligned_rms"] == pytest.approx(0.0, abs=1e-14)


def test_block_comparison_marks_disjoint_support_as_insufficient():
    result = block_reweighted_pmf_1d(
        np.array([0.25] * 4 + [2.25] * 4),
        np.zeros(8),
        n_blocks=2,
        bins=np.array([0.0, 1.0, 2.0, 3.0]),
        temperature=300.0,
    )
    comparison = result["successive_comparisons"][0]
    assert comparison["status"] == "insufficient_support"
    assert comparison["common_support_bins"] == 0
    assert comparison["offset_aligned_rms"] is None


def _landmark_profile() -> dict:
    return {
        "bin_centers": np.arange(7.0),
        "bin_edges": np.arange(-0.5, 7.5),
        "pmf": np.array([0.1, 0.0, 0.5, 1.5, 0.8, 0.2, 0.3]),
        "support_mask": np.ones(7, dtype=bool),
        "weighted_mass": np.array([0.3, 0.3, 0.05, 0.02, 0.05, 0.15, 0.1]),
        "temperature": 300.0,
        "energy_unit": "eV",
    }


def test_pmf_landmarks_report_full_and_prespecified_barriers_and_population_df():
    profile = _landmark_profile()
    result = pmf_landmarks_1d(
        profile,
        basin_a=(0.0, 1.1),
        basin_b=(4.9, 6.1),
        transition_region=(2.5, 3.5),
    )

    assert result["status"] == "ok"
    assert "converged" not in result
    assert result["basin_a"]["minimum_position"] == pytest.approx(1.0)
    assert result["basin_b"]["minimum_position"] == pytest.approx(5.0)
    assert result["interbasin"]["peak_position"] == pytest.approx(3.0)
    assert result["interbasin"]["barrier_from_a"] == pytest.approx(1.5)
    assert result["interbasin"]["barrier_from_b"] == pytest.approx(1.3)
    assert result["transition_region"]["barrier_from_a"] == pytest.approx(1.5)
    expected_delta_f = -thermal_energy(300.0) * np.log(0.25 / 0.6)
    assert result["population_delta_f_status"] == "ok"
    assert result["population_delta_f"] == pytest.approx(expected_delta_f)


def test_pmf_landmarks_do_not_bridge_an_unsupported_interbasin_gap():
    profile = _landmark_profile()
    profile["pmf"][3] = np.nan
    profile["support_mask"][3] = False
    result = pmf_landmarks_1d(
        profile,
        basin_a=(0.0, 1.1),
        basin_b=(4.9, 6.1),
        transition_region=(2.5, 3.5),
    )

    assert result["status"] == "insufficient_support"
    assert result["interbasin"]["status"] == "insufficient_support"
    assert result["interbasin"]["barrier_from_a"] is None
    assert result["transition_region"]["status"] == "insufficient_support"
    assert any("unsupported" in reason for reason in result["reasons"])


def test_pmf_landmarks_report_missing_basin_support_instead_of_guessing():
    profile = _landmark_profile()
    result = pmf_landmarks_1d(
        profile,
        basin_a=(0.0, 1.1),
        basin_b=(8.0, 9.0),
    )
    assert result["status"] == "insufficient_support"
    assert result["basin_b"]["status"] == "insufficient_support"
    assert result["interbasin"]["peak_position"] is None


def test_pmf_landmark_closed_basins_must_not_touch():
    with pytest.raises(ValueError, match="strictly separated"):
        pmf_landmarks_1d(
            _landmark_profile(), basin_a=(0.0, 2.0), basin_b=(2.0, 4.0)
        )


def test_regions_and_blocks_validate_structural_inputs():
    with pytest.raises(ValueError, match="non-empty mapping"):
        region_weight_diagnostics([0.0], [1.0], {})
    with pytest.raises(ValueError, match="strictly increasing"):
        region_weight_diagnostics([0.0], [1.0], {"bad": (1.0, 1.0)})
    with pytest.raises(ValueError, match="cannot exceed"):
        block_reweighted_pmf_1d(
            [0.0, 1.0],
            [0.0, 0.0],
            n_blocks=3,
            temperature=300.0,
        )
