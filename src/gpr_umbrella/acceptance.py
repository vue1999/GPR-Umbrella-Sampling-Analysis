"""Independent acceptance checks, never an optimiser-success synonym."""

from __future__ import annotations

import numpy as np


def compare_profiles(profiles, *, tolerance=0.05, uncertainty_target=0.05):
    """Gauge-aligned comparisons on common support, in the input energy unit.

    Absolute tolerances prevent huge uncertainty bands from making an unstable
    analysis pass. This tests fit stability, not physical model adequacy.
    """
    if len(profiles) < 2:
        raise ValueError("Need at least two independently fitted sensitivity profiles")
    lo = max(p["x_star"][0] for p in profiles)
    hi = min(p["x_star"][-1] for p in profiles)
    if lo >= hi:
        raise ValueError("Sensitivity profiles have no common support")
    x = np.linspace(lo, hi, 301)
    aligned = np.array([np.interp(x, p["x_star"], p["pmf_mean"]) for p in profiles])
    aligned -= aligned[:, [0]]
    maximum_spread = float(np.max(np.ptp(aligned, axis=0)))
    max_uncertainty = float(max(np.max(p["pmf_std"]) for p in profiles))
    issues = []
    if maximum_spread > tolerance:
        issues.append("profile_sensitivity_exceeds_tolerance")
    if max_uncertainty > uncertainty_target:
        issues.append("profile_uncertainty_exceeds_target")
    return {
        "quality_issues": issues,
        "maximum_profile_spread": maximum_spread,
        "maximum_profile_sigma": max_uncertainty,
        "profile_tolerance": tolerance,
        "uncertainty_target": uncertainty_target,
        "common_support": [float(lo), float(hi)],
    }


def diagnostic_status(quality_issues):
    """A numerical fit is never declared a validated physical barrier."""
    hard = [
        i
        for i in quality_issues
        if i
        not in ("physical_barrier_basins_not_specified", "sensitivity_suite_required")
    ]
    if hard:
        return "UNRELIABLE"
    if "sensitivity_suite_required" in quality_issues:
        return "PROVISIONAL"
    if "physical_barrier_basins_not_specified" in quality_issues:
        return "FIT_STABLE_BARRIER_UNDEFINED"
    return "PASSED_SPECIFIED_CHECKS"
