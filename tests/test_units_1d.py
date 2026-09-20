"""Unit-rescaling invariance tests for the one-dimensional reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpr_umbrella import reconstruct_pmf_1d


KAPPA = 40.0
ENERGY_SCALE = 96.4853321233  # eV -> kJ/mol
COORDINATE_SCALE = 10.0      # nm -> angstrom


def _harmonic_trajectories() -> tuple[np.ndarray, list[np.ndarray]]:
    """Return compact, deterministic umbrella trajectories in nm."""
    centers = np.linspace(-2.0, 2.0, 9)
    curvature = 1.8
    biased_curvature = KAPPA + curvature
    rng = np.random.default_rng(20260722)
    trajectories = [
        rng.normal(
            loc=KAPPA * center / biased_curvature,
            scale=np.sqrt(1.0 / biased_curvature),
            size=900,
        )
        for center in centers
    ]
    return centers, trajectories


def _write_colvars(
    directory: Path,
    centers: np.ndarray,
    trajectories: list[np.ndarray],
) -> tuple[Path, Path]:
    colvar_dir = directory / "COLVAR"
    colvar_dir.mkdir(parents=True)
    centers_file = directory / "window_centers.txt"
    np.savetxt(centers_file, centers, fmt="%.12f")
    for window, positions in enumerate(trajectories):
        time = np.arange(len(positions), dtype=float) * 0.001
        np.savetxt(
            colvar_dir / f"COLVAR_window_{window}.dat",
            np.column_stack([time, positions]),
            header="#! FIELDS time coordinate",
            comments="",
            fmt="%.12f",
        )
    return colvar_dir, centers_file


@pytest.mark.parametrize("energy,coordinate", [(ENERGY_SCALE, 1.), (1., COORDINATE_SCALE)])
def test_unit_rescaling_preserves_reconstruction(tmp_path, energy, coordinate):
    """Coordinates, gradients, energies and uncertainties carry their own units."""
    centers, trajectories = _harmonic_trajectories()
    fits = []
    for name, e, x in [("base", 1., 1.), ("scaled", energy, coordinate)]:
        directory, center_file = _write_colvars(
            tmp_path / name, centers * x, [q * x for q in trajectories])
        fits.append(reconstruct_pmf_1d(
            colvar_dir=str(directory), centers=str(center_file), kappa=KAPPA * e / x**2,
            cv_unit="nm" if x == 1 else "angstrom",
            energy_unit="eV" if e == 1 else "kJ/mol",
            optimize_hyperparams=True, plot=False, save_outputs=False, verbose=False))
    base, scaled = fits
    assert base["energy_unit"] == "eV" and base["cv_unit"] == "nm"
    assert scaled["energy_unit"] == ("eV" if energy == 1 else "kJ/mol")
    assert scaled["cv_unit"] == ("nm" if coordinate == 1 else "angstrom")
    assert scaled["deriv_unit"] == f'{scaled["energy_unit"]}/{scaled["cv_unit"]}'
    for key in ("x_centers", "x_means", "x_star"):
        np.testing.assert_allclose(scaled[key] / coordinate, base[key],
                                   rtol=0. if coordinate == 1 else 1e-8,
                                   atol=1e-12 if coordinate == 1 else 2e-10)
    groups = [
        (coordinate, ["lengthscale"]),
        (energy, ["sigma_f", "pmf_mean", "pmf_std", "pmf_std_raw", "pmf_f_std", "pmf_f_std_raw"]),
        (energy / coordinate, ["derivatives", "derivative_errors", "deriv_mean", "deriv_std", "deriv_std_raw"]),
        (1., ["loo_z", "uncertainty_calibration_factor"]),
    ]
    for factor, keys in groups:
        for key in keys:
            np.testing.assert_allclose(scaled[key] / factor, base[key], rtol=5e-4, atol=2e-7)
