"""Unified command-line interface for OPES, metadynamics, and ICF analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .biased_sampling import thermal_energy as _thermal_energy
from .icf import reconstruct_pmf_icf
from .metadynamics import analyze_metadynamics
from .opes import analyze_opes_1d


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write a concise 1D profile and JSON scalar summary",
    )


def _add_ranges_argument(
    parser: argparse.ArgumentParser,
    option: str,
    destination: str,
    help_text: str,
) -> None:
    parser.add_argument(
        option,
        dest=destination,
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("MIN", "MAX"),
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the unified biased-sampling argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convergence evidence and free-energy reconstruction for OPES, "
            "metadynamics, and instantaneous collective forces."
        )
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    opes = subparsers.add_parser(
        "opes", help="Direct total-bias reweighting of an OPES trajectory"
    )
    opes.add_argument(
        "colvar", nargs="+", help="One or more consecutive OPES COLVAR files"
    )
    opes.add_argument("--cv-field", required=True)
    opes.add_argument("--bias-field", default="opes.bias")
    opes.add_argument(
        "--other-bias-field",
        action="append",
        default=[],
        help="Additional applied bias field; repeat for every wall/restraint",
    )
    opes.add_argument("--time-field", default="time")
    opes.add_argument(
        "--start-time", type=float, default=None,
        help="Inclusive lower bound in the COLVAR time-field unit",
    )
    opes.add_argument(
        "--stop-time", type=float, default=None,
        help="Inclusive upper bound in the COLVAR time-field unit",
    )
    opes.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth record after applying the time bounds",
    )
    opes.add_argument("--temperature", type=float, required=True)
    opes.add_argument("--energy-unit", default="eV")
    opes.add_argument(
        "--bias-energy-unit", default=None,
        help="Unit of printed bias columns (default: --energy-unit)",
    )
    opes.add_argument("--bins", type=int, default=100)
    opes.add_argument(
        "--range", dest="value_range", type=float, nargs=2,
        metavar=("MIN", "MAX"), default=None,
    )
    opes.add_argument("--min-count", type=int, default=1)
    opes.add_argument(
        "--cumulative-cutoff", type=float, action="append", default=None,
        help="Cumulative trajectory cutoff; repeat in increasing order",
    )
    opes.add_argument(
        "--n-blocks", type=int, default=None,
        help="Number of contiguous disjoint PMF blocks",
    )
    opes.add_argument(
        "--basin-a", type=float, nargs=2, default=None,
        metavar=("MIN", "MAX"), help="First basin range on the analyzed CV",
    )
    opes.add_argument(
        "--basin-b", type=float, nargs=2, default=None,
        metavar=("MIN", "MAX"), help="Second basin range on the analyzed CV",
    )
    opes.add_argument(
        "--transition-region", type=float, nargs=2, default=None,
        metavar=("MIN", "MAX"), help="Prespecified transition range",
    )
    opes.add_argument("--diagnostic-tail-fraction", type=float, default=0.2)
    opes.add_argument(
        "--allow-unlisted-bias-fields",
        action="store_true",
        help="Explicitly allow printed *.bias fields to be omitted from weights",
    )
    opes.add_argument("--strict", action="store_true")
    _add_output_argument(opes)

    metad = subparsers.add_parser(
        "metad", help="Reweighted-density analysis of metadynamics data"
    )
    metad.add_argument("colvar", help="Metadynamics COLVAR file")
    metad.add_argument(
        "--cv-field", action="append", required=True,
        help="Collective-variable field; repeat for multiple CVs",
    )
    source = metad.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--logweight-field", help="Precomputed dimensionless log-weight field"
    )
    source.add_argument(
        "--rbias-field", help="Normalized metadynamics bias-minus-c(t) field"
    )
    source.add_argument(
        "--raw-bias-field",
        help=(
            "Raw bias field; requires --allow-quasistatic-bias plus finite "
            "--start-time and --stop-time bounds"
        ),
    )
    metad.add_argument(
        "--allow-quasistatic-bias",
        action="store_true",
        help=(
            "Assert that the explicitly time-bounded raw-bias segment is "
            "quasi-static"
        ),
    )
    metad.add_argument(
        "--extra-bias-field", action="append", default=[],
        help="Additional static bias energy; repeat as needed",
    )
    metad.add_argument(
        "--allow-unlisted-bias-fields", action="store_true",
        help="Explicitly accept printed *.bias fields not audited as sources",
    )
    metad.add_argument(
        "--strict", action="store_true",
        help="Reject malformed PLUMED records instead of counting/ignoring them",
    )
    metad.add_argument(
        "--hills-file", "--hills", dest="hills_files", action="append",
        default=None, help="HILLS file used for deposition diagnostics; repeatable",
    )
    kbt_source = metad.add_mutually_exclusive_group(required=True)
    kbt_source.add_argument(
        "--thermal-energy", type=float, metavar="KBT",
        help="kBT in the numerical energy unit",
    )
    kbt_source.add_argument(
        "--temperature", type=float,
        help="Temperature used to derive kBT with --energy-unit",
    )
    metad.add_argument("--energy-unit", default="eV")
    metad.add_argument("--bias-energy-factor", type=float, default=1.0)
    metad.add_argument("--time-field", default="time")
    metad.add_argument(
        "--start-time", type=float, default=None,
        help=(
            "Inclusive lower bound in the COLVAR time-field unit, after "
            "restart deduplication"
        ),
    )
    metad.add_argument(
        "--stop-time", type=float, default=None,
        help=(
            "Inclusive upper bound in the COLVAR time-field unit, after "
            "restart deduplication"
        ),
    )
    metad.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth record after applying the time bounds",
    )
    metad.add_argument("--bins", type=int, default=100)
    _add_ranges_argument(
        metad, "--range", "ranges",
        "Histogram range for one CV; repeat once per CV",
    )
    metad.add_argument("--n-blocks", type=int, default=4)
    _add_output_argument(metad)

    icf = subparsers.add_parser(
        "icf", help="Paper-convention instantaneous-force GPR reconstruction"
    )
    icf.add_argument("colvar", help="PLUMED table with CV and ICF fields")
    icf.add_argument(
        "--cv-field", action="append", required=True,
        help="Collective-variable field; repeat once per dimension",
    )
    icf.add_argument(
        "--icf-field", action="append", required=True,
        help="Matching thermodynamic-force field; repeat once per dimension",
    )
    icf.add_argument("--time-field", default="time")
    icf.add_argument("--cv-factor", type=float, action="append", default=None)
    icf.add_argument(
        "--energy-factor", type=float, default=1.0,
        help="Multiplicative input-to-output energy conversion",
    )
    icf.add_argument("--icf-factor", type=float, action="append", default=None)
    icf.add_argument("--start-time", type=float, default=None)
    icf.add_argument("--stop-time", type=float, default=None)
    icf.add_argument("--stride", type=int, default=1)
    icf.add_argument(
        "--aggregation-bins", type=int, nargs="+", default=None,
        metavar="N", help="Enable local ICF aggregation with one bin count per CV",
    )
    _add_ranges_argument(
        icf, "--aggregation-range", "aggregation_ranges",
        "Aggregation range for one CV; repeat once per CV",
    )
    icf.add_argument("--time-block-size", type=int, default=None)
    icf.add_argument("--min-samples-per-observation", type=int, default=2)
    icf.add_argument(
        "--force-error", type=float, default=None,
        help="Scalar ICF standard error; required for unaggregated fitting",
    )
    icf.add_argument("--max-exact-observations", type=int, default=1000)
    icf.add_argument("--grid-n", type=int, nargs="+", default=[200])
    icf.add_argument("--prediction-batch-size", type=int, default=10_000)
    icf.add_argument("--no-optimize", action="store_true")
    icf.add_argument("--lengthscale", type=float, action="append", default=None)
    icf.add_argument("--sigma-f", type=float, default=None)
    icf.add_argument("--no-calibrate", action="store_true")
    for option, destination, help_text in (
        (
            "--bias-depends-only-on-modeled-cvs",
            "bias_depends_only_on_modeled_cvs",
            "All applied biases depend only on the complete modeled CV vector",
        ),
        (
            "--quasi-equilibrium",
            "quasi_equilibrium",
            "The adaptive bias is slow enough for quasi-equilibrium sampling",
        ),
        (
            "--physical-force-excludes-bias",
            "physical_force_excludes_bias",
            "The supplied physical collective force excludes all bias forces",
        ),
        (
            "--metric-correction-included",
            "metric_correction_included",
            "The ICF includes the CV metric/Jacobian correction",
        ),
    ):
        icf.add_argument(
            option,
            dest=destination,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=help_text,
        )
    paper_assumptions = icf.add_mutually_exclusive_group()
    paper_assumptions.add_argument(
        "--require-paper-assumptions",
        dest="require_paper_assumptions",
        action="store_true",
        default=True,
        help="Require every paper-validity condition (default)",
    )
    paper_assumptions.add_argument(
        "--allow-unverified-paper-assumptions",
        dest="require_paper_assumptions",
        action="store_false",
        help="Unsafe opt-out: fit while retaining false/unknown validity flags",
    )
    icf.add_argument(
        "--allow-uncertainty-downscaling", action="store_true",
        help="Allow pointwise LOO to reduce raw GP uncertainty",
    )
    icf.add_argument(
        "--strict", action="store_true",
        help="Reject malformed PLUMED records instead of counting/ignoring them",
    )
    _add_output_argument(icf)
    return parser


def _one_or_tuple(values):
    if values is None:
        return None
    values = tuple(values)
    return values[0] if len(values) == 1 else values


def _ranges(values, dimensions: int, name: str):
    if values is None:
        return None
    if len(values) != dimensions:
        raise ValueError(f"{name} must be supplied exactly once per CV")
    result = tuple(tuple(pair) for pair in values)
    if any(lower >= upper for lower, upper in result):
        raise ValueError(f"Every {name} minimum must be below its maximum")
    return result


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_outputs(
    output_dir: str | Path,
    stem: str,
    profile: np.ndarray,
    header: str,
    summary: dict,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / f"{stem}_pmf_1d.dat"
    summary_path = directory / f"{stem}_summary.json"
    np.savetxt(profile_path, profile, header=header, fmt="%.10g")
    payload = dict(summary)
    payload["profile_file"] = str(profile_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    return profile_path, summary_path


def _print_evidence(title: str, summary: dict, output_paths=None) -> None:
    print(title)
    print("-" * len(title))
    for key, value in summary.items():
        label = key.replace("_", " ")
        if isinstance(value, float):
            rendered = f"{value:.6g}"
        elif value is None:
            rendered = "unavailable"
        else:
            rendered = str(value)
        print(f"{label}: {rendered}")
    if output_paths is not None:
        print(f"profile: {output_paths[0]}")
        print(f"summary: {output_paths[1]}")
    print("Evidence is observable-specific; no binary verdict is emitted.")


def _opes_summary(result: dict) -> dict:
    pmf = result["pmf"]
    data = result["opes_data"]
    table = data["table"]
    selection = data.get("selection", {})
    summary = {
        "method": result.get("method", "direct_total_bias_reweighting"),
        "records": int(
            len(data["time"]) if "time" in data else table["n_records"]
        ),
        "importance_weight_ess": float(pmf["importance_weight_ess"]),
        "maximum_weight_fraction": float(pmf["maximum_weight_fraction"]),
        "supported_bins": int(np.count_nonzero(pmf["support_mask"])),
        "total_bins": int(len(pmf["support_mask"])),
        "source_files": list(table.get("files", ())),
        "bias_fields": list(data.get("bias_fields", ())),
        "available_bias_fields": list(data.get("available_bias_fields", ())),
        "unlisted_bias_fields": list(data.get("unlisted_bias_fields", ())),
        "fields_headers_seen": int(table.get("fields_headers_seen", 0)),
        "duplicate_records_replaced": int(
            table.get("duplicate_records_replaced", 0)
        ),
        "malformed_records_ignored": int(
            table.get("malformed_records_ignored", 0)
        ),
        "rct_used_for_reweighting": bool(result["rct_used_for_reweighting"]),
        "pmf_energy_unit": result.get("energy_unit"),
        "bias_energy_unit": result.get("bias_energy_unit"),
    }
    if selection:
        for name in (
            "requested_start_time", "requested_stop_time",
            "selected_start_time", "selected_stop_time",
            "n_records_before_selection", "n_records_within_time_bounds",
            "n_records_selected", "stride", "explicit_finite_time_range",
            "bounds_inclusive",
        ):
            summary[name] = selection[name]
        summary["selection_order"] = " -> ".join(selection["selection_order"])
    for field, evidence in result["opes_diagnostics"]["series"].items():
        name = field.replace(".", "_")
        summary[f"{name}_final"] = float(evidence["final"])
        summary[f"{name}_tail_drift"] = float(evidence["tail_drift"])
    cumulative = result.get("cumulative")
    if cumulative is not None:
        summary["cumulative_snapshots"] = len(cumulative["snapshots"])
    blocks = result.get("blocks")
    if blocks is not None:
        summary["disjoint_blocks"] = int(blocks["n_blocks"])
        block_rms = [
            comparison["offset_aligned_rms"]
            for comparison in blocks["successive_comparisons"]
            if comparison.get("status") == "ok"
        ]
        summary["maximum_successive_block_shape_rms"] = (
            float(np.max(block_rms)) if block_rms else None
        )
        common_bins = [
            comparison["common_support_bins"]
            for comparison in blocks["successive_comparisons"]
            if comparison.get("status") == "ok"
        ]
        summary["minimum_successive_common_support_bins"] = (
            int(np.min(common_bins)) if common_bins else None
        )
        block_support = blocks.get("landmark_support")
        if block_support is not None:
            summary["blocks_with_basin_a_support"] = sum(
                item.get("basin_a_status") == "ok" for item in block_support
            )
            summary["blocks_with_basin_b_support"] = sum(
                item.get("basin_b_status") == "ok" for item in block_support
            )
            summary["blocks_with_both_basins_supported"] = sum(
                item.get("basin_a_status") == "ok"
                and item.get("basin_b_status") == "ok"
                for item in block_support
            )
            summary["blocks_with_population_delta_f"] = sum(
                item.get("population_delta_f_status") == "ok"
                for item in block_support
            )
    landmarks = result.get("landmarks")
    if landmarks is not None:
        summary["landmark_status"] = landmarks["status"]
        summary["interbasin_barrier_from_a"] = landmarks[
            "interbasin"
        ]["barrier_from_a"]
        summary["population_delta_f"] = landmarks["population_delta_f"]
    regions = result.get("regions")
    if regions is not None:
        for name, evidence in regions["regions"].items():
            prefix = "region_" + name.replace(" ", "_")
            summary[f"{prefix}_status"] = evidence["status"]
            summary[f"{prefix}_records"] = int(evidence["n_records"])
            summary[f"{prefix}_weight_mass"] = float(evidence["weight_mass"])
            summary[f"{prefix}_kish_ess"] = float(evidence["kish_ess"])
            summary[f"{prefix}_maximum_local_leverage"] = evidence[
                "maximum_local_leverage"
            ]
    transitions = result.get("transitions")
    if transitions is not None:
        summary.update({
            "initial_basin_state": transitions["initial_state"],
            "final_basin_state": transitions.get("final_state"),
            "basin_a_to_b_events": int(transitions["forward_events"]),
            "basin_b_to_a_events": int(transitions["reverse_events"]),
            "completed_round_trips": int(transitions["completed_round_trips"]),
            "last_transition_time": transitions.get("last_event_time"),
        })
        for name, evidence in transitions.get("basin_evidence", {}).items():
            prefix = name.replace(" ", "_")
            summary[f"{prefix}_qualifying_records"] = evidence[
                "n_qualifying_records"
            ]
            summary[f"{prefix}_last_qualifying_time"] = evidence[
                "last_qualifying_time"
            ]
    return summary


def _run_opes(args) -> None:
    paths = args.colvar[0] if len(args.colvar) == 1 else tuple(args.colvar)
    regions = {}
    if args.basin_a is not None:
        regions["basin_a"] = tuple(args.basin_a)
    if args.basin_b is not None:
        regions["basin_b"] = tuple(args.basin_b)
    if args.transition_region is not None:
        regions["transition"] = tuple(args.transition_region)
    result = analyze_opes_1d(
        paths,
        cv_field=args.cv_field,
        temperature=args.temperature,
        bias_field=args.bias_field,
        other_bias_fields=tuple(args.other_bias_field),
        time_field=args.time_field,
        start_time=args.start_time,
        stop_time=args.stop_time,
        stride=args.stride,
        energy_unit=args.energy_unit,
        bias_energy_unit=args.bias_energy_unit,
        bins=args.bins,
        value_range=(tuple(args.value_range) if args.value_range else None),
        min_count=args.min_count,
        cumulative_cutoffs=args.cumulative_cutoff,
        n_blocks=args.n_blocks,
        regions=regions or None,
        basin_a=(tuple(args.basin_a) if args.basin_a else None),
        basin_b=(tuple(args.basin_b) if args.basin_b else None),
        transition_region=(
            tuple(args.transition_region) if args.transition_region else None
        ),
        diagnostic_tail_fraction=args.diagnostic_tail_fraction,
        allow_unlisted_bias_fields=args.allow_unlisted_bias_fields,
        strict=args.strict,
    )
    summary = _opes_summary(result)
    output_paths = None
    if args.output_dir is not None:
        pmf = result["pmf"]
        profile = np.column_stack([
            pmf["bin_centers"], pmf["pmf"], pmf["density"],
            pmf["local_ess"], pmf["local_maximum_weight_fraction"],
            pmf["support_mask"].astype(int),
        ])
        output_paths = _write_outputs(
            args.output_dir,
            "opes",
            profile,
            f"{args.cv_field} PMF({args.energy_unit}) density local_ESS "
            "local_max_weight_fraction supported",
            summary,
        )
    _print_evidence("OPES direct-reweighting evidence", summary, output_paths)


def _metad_summary(result: dict, energy_unit: str) -> dict:
    rms = np.asarray(
        result["convergence_evidence"]["successive_block_shape_rms"],
        dtype=float,
    )
    finite_rms = rms[np.isfinite(rms)]
    selection = result["selection"]
    validity = result["validity"]
    summary = {
        "method": "metadynamics_reweighted_density",
        "weight_source": result["weight_source"],
        "records": int(len(result["time"])),
        "global_ess": float(result["global_ess"]),
        "global_ess_fraction": float(result["global_ess_fraction"]),
        "maximum_weight_fraction": float(result["maximum_weight_fraction"]),
        "time_blocks": int(result["convergence_evidence"]["n_time_blocks"]),
        "maximum_successive_block_shape_rms": (
            float(np.max(finite_rms)) if len(finite_rms) else None
        ),
        "energy_unit": energy_unit,
        "normalized_time_dependent_bias": bool(
            validity["normalized_time_dependent_bias"]
        ),
        "quasistatic_bias_assumed": bool(
            validity["quasistatic_bias_assumed"]
        ),
        "quasistatic_segment_explicitly_bounded": bool(
            validity["quasistatic_segment_explicitly_bounded"]
        ),
        "requested_start_time": selection["requested_start_time"],
        "requested_stop_time": selection["requested_stop_time"],
        "stride": int(selection["stride"]),
        "n_records_before_selection": int(
            selection["n_records_before_selection"]
        ),
        "n_records_within_time_bounds": int(
            selection["n_records_within_time_bounds"]
        ),
        "n_records_selected": int(selection["n_records_selected"]),
        "selected_start_time": float(selection["selected_start_time"]),
        "selected_stop_time": float(selection["selected_stop_time"]),
        "explicit_finite_time_range": bool(
            selection["explicit_finite_time_range"]
        ),
        "bounds_inclusive": bool(selection["bounds_inclusive"]),
        "selection_order": " -> ".join(selection["selection_order"]),
        "unlisted_bias_fields": len(result.get("unlisted_bias_fields", ())),
        "bias_energy_factor": float(result.get("bias_energy_factor", 1.0)),
    }
    if "hills_diagnostics" in result:
        summary["hills_files"] = int(result["hills_diagnostics"]["n_files"])
        summary["hills_count"] = int(result["hills_diagnostics"]["n_hills"])
    return summary


def _run_metad(args) -> None:
    if args.raw_bias_field is not None and not args.allow_quasistatic_bias:
        raise ValueError(
            "--raw-bias-field requires the explicit --allow-quasistatic-bias "
            "assumption"
        )
    if args.raw_bias_field is not None and (
        args.start_time is None or args.stop_time is None
    ):
        raise ValueError(
            "--raw-bias-field with --allow-quasistatic-bias requires both "
            "finite --start-time and --stop-time bounds"
        )
    cv_fields = tuple(args.cv_field)
    ranges = _ranges(args.ranges, len(cv_fields), "--range")
    kbt = (
        args.thermal_energy
        if args.thermal_energy is not None
        else _thermal_energy(args.temperature, args.energy_unit)
    )
    if args.output_dir is not None and len(cv_fields) != 1:
        raise ValueError("--output-dir profile writing currently requires one CV")
    result = analyze_metadynamics(
        args.colvar,
        cv_fields=cv_fields,
        thermal_energy=kbt,
        bins=args.bins,
        ranges=ranges,
        n_blocks=args.n_blocks,
        hills_files=(tuple(args.hills_files) if args.hills_files else None),
        time_field=args.time_field,
        logweight_field=args.logweight_field,
        rbias_field=args.rbias_field,
        raw_bias_field=args.raw_bias_field,
        extra_bias_fields=tuple(args.extra_bias_field),
        allow_quasistatic_bias=args.allow_quasistatic_bias,
        allow_unlisted_bias_fields=args.allow_unlisted_bias_fields,
        bias_energy_factor=args.bias_energy_factor,
        start_time=args.start_time,
        stop_time=args.stop_time,
        stride=args.stride,
        strict=args.strict,
    )
    summary = _metad_summary(result, args.energy_unit)
    output_paths = None
    if args.output_dir is not None:
        profile = np.column_stack([
            result["bin_centers"][0], result["free_energy"],
            result["local_ess"], result["local_weight_leverage"],
            result["counts"], result["support_mask"].astype(int),
        ])
        output_paths = _write_outputs(
            args.output_dir,
            "metad",
            profile,
            f"{cv_fields[0]} PMF({args.energy_unit}) local_ESS "
            "local_max_weight_fraction count supported",
            summary,
        )
    _print_evidence("Metadynamics reweighting evidence", summary, output_paths)


def _icf_summary(result: dict) -> dict:
    validity = result["icf_validity"]
    summary = {
        "method": result.get("method", "icf_gradient_gpr"),
        "observations": int(result["n_points"]),
        "dimensions": int(result["n_dimensions"]),
        "prediction_points": int(len(result["prediction_points"])),
        "sigma_f": float(result["sigma_f"]),
        "loo_calibration_factor": float(result["loo_calibration_factor"]),
        "loo_calibration_factor_uncapped": float(
            result.get(
                "loo_calibration_factor_uncapped",
                result["loo_calibration_factor"],
            )
        ),
        "uncertainty_downscaling_allowed": bool(
            result.get("uncertainty_downscaling_allowed", False)
        ),
        "aggregation_kind": result["aggregation"]["kind"],
        "source_file": result.get("source_file"),
        "cv_names": list(result.get("cv_names", ())),
        "icf_fields": list(result.get("icf_fields") or ()),
        "force_convention": result.get("force_convention"),
        "gradient_relation": result.get("gradient_relation"),
        "reference_source": result.get("reference_source"),
        "paper_assumptions_satisfied": validity.get(
            "paper_assumptions_satisfied"
        ),
        "paper_equation_8_applicable": validity.get(
            "paper_equation_8_applicable"
        ),
        "icf_definition_complete": validity.get("icf_definition_complete"),
    }
    if "unit_provenance" in result:
        provenance = result["unit_provenance"]
        summary["cv_factors"] = np.asarray(
            provenance["cv_factors"], dtype=float
        ).tolist()
        summary["energy_factor"] = float(provenance["energy_factor"])
        summary["icf_factors"] = np.asarray(
            provenance["icf_factors"], dtype=float
        ).tolist()
        summary["icf_factors_source"] = provenance["icf_factors_source"]
        summary["force_scaling_relation"] = provenance[
            "force_scaling_relation"
        ]
    for index, value in enumerate(np.atleast_1d(result["lengthscale"])):
        summary[f"lengthscale_{index}"] = float(value)
    return summary


def _run_icf(args) -> None:
    cv_fields = tuple(args.cv_field)
    if len(cv_fields) != len(args.icf_field):
        raise ValueError("--cv-field and --icf-field counts must match")
    aggregation_ranges = _ranges(
        args.aggregation_ranges, len(cv_fields), "--aggregation-range"
    )
    if args.output_dir is not None and len(cv_fields) != 1:
        raise ValueError("--output-dir profile writing currently requires one CV")
    result = reconstruct_pmf_icf(
        args.colvar,
        cv_fields=cv_fields,
        icf_fields=tuple(args.icf_field),
        time_field=args.time_field,
        cv_factors=(1.0 if args.cv_factor is None else _one_or_tuple(args.cv_factor)),
        energy_factor=args.energy_factor,
        icf_factors=_one_or_tuple(args.icf_factor),
        start_time=args.start_time,
        stop_time=args.stop_time,
        stride=args.stride,
        bias_depends_only_on_modeled_cvs=(
            args.bias_depends_only_on_modeled_cvs
        ),
        quasi_equilibrium=args.quasi_equilibrium,
        physical_force_excludes_bias=args.physical_force_excludes_bias,
        metric_correction_included=args.metric_correction_included,
        require_paper_assumptions=args.require_paper_assumptions,
        strict=args.strict,
        aggregation_bins=_one_or_tuple(args.aggregation_bins),
        aggregation_ranges=aggregation_ranges,
        time_block_size=args.time_block_size,
        min_samples_per_observation=args.min_samples_per_observation,
        force_errors=args.force_error,
        max_exact_observations=args.max_exact_observations,
        grid_n=_one_or_tuple(args.grid_n),
        optimize_hyperparams=not args.no_optimize,
        fixed_lengthscale=_one_or_tuple(args.lengthscale),
        fixed_sigma_f=args.sigma_f,
        calibrate_uncertainty=not args.no_calibrate,
        allow_uncertainty_downscaling=args.allow_uncertainty_downscaling,
        prediction_batch_size=args.prediction_batch_size,
    )
    summary = _icf_summary(result)
    output_paths = None
    if args.output_dir is not None:
        profile = np.column_stack([
            result["x_star"], result["pmf"], result["pmf_std"],
            result["pmf_std_raw"], result["pmf_std_calibrated"],
        ])
        output_paths = _write_outputs(
            args.output_dir,
            "icf",
            profile,
            f"{cv_fields[0]} PMF PMF_std PMF_std_raw PMF_std_calibrated",
            summary,
        )
    _print_evidence("ICF/GPR reconstruction evidence", summary, output_paths)


def main(argv=None) -> int:
    """Run one biased-sampling analysis and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.method == "opes":
            _run_opes(args)
        elif args.method == "metad":
            _run_metad(args)
        elif args.method == "icf":
            _run_icf(args)
        else:  # pragma: no cover - argparse guarantees a registered command
            parser.error(f"Unknown method: {args.method}")
    except (ValueError, FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
