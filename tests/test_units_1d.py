"""Unit-rescaling invariance tests for the one-dimensional reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def _reconstruct(
    colvar_dir: Path,
    centers_file: Path,
    *,
    kappa: float,
    cv_unit: str,
    energy_unit: str,
) -> dict:
    return reconstruct_pmf_1d(
        colvar_dir=str(colvar_dir),
        kappa=kappa,
        centers=str(centers_file),
        cv_unit=cv_unit,
        energy_unit=energy_unit,
        optimize_hyperparams=True,
        plot=False,
        save_outputs=False,
        verbose=False,
    )


def _assert_close(actual, expected, *, rtol=5e-4, atol=2e-7) -> None:
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_energy_unit_rescaling_preserves_reconstruction(tmp_path: Path) -> None:
    """Changing eV to kJ/mol rescales every energy-valued GP quantity."""
    centers, trajectories = _harmonic_trajectories()
    colvar_dir, centers_file = _write_colvars(
        tmp_path / "energy", centers, trajectories
    )

    ev = _reconstruct(
        colvar_dir,
        centers_file,
        kappa=KAPPA,
        cv_unit="nm",
        energy_unit="eV",
    )
    kj = _reconstruct(
        colvar_dir,
        centers_file,
        kappa=KAPPA * ENERGY_SCALE,
        cv_unit="nm",
        energy_unit="kJ/mol",
    )

    assert ev["energy_unit"] == "eV"
    assert kj["energy_unit"] == "kJ/mol"
    assert kj["deriv_unit"] == "kJ/mol/nm"
    _assert_close(kj["lengthscale"], ev["lengthscale"])
    _assert_close(kj["sigma_f"] / ENERGY_SCALE, ev["sigma_f"])
    _assert_close(kj["x_star"], ev["x_star"], rtol=0.0, atol=1e-12)
    for key in (
        "pmf_mean",
        "pmf_std",
        "pmf_std_raw",
        "pmf_f_std",
        "pmf_f_std_raw",
        "deriv_mean",
        "deriv_std",
        "deriv_std_raw",
        "derivative_errors",
    ):
        _assert_close(kj[key] / ENERGY_SCALE, ev[key])
    _assert_close(kj["loo_z"], ev["loo_z"], atol=2e-7)
    _assert_close(
        kj["uncertainty_calibration_factor"],
        ev["uncertainty_calibration_factor"],
        rtol=5e-4,
    )


def test_cv_unit_rescaling_preserves_reconstruction(tmp_path: Path) -> None:
    """Changing nm to angstrom rescales coordinates and inverse-square kappa."""
    centers, trajectories = _harmonic_trajectories()
    nm_dir, nm_centers = _write_colvars(tmp_path / "nm", centers, trajectories)
    angstrom_dir, angstrom_centers = _write_colvars(
        tmp_path / "angstrom",
        centers * COORDINATE_SCALE,
        [positions * COORDINATE_SCALE for positions in trajectories],
    )

    nm = _reconstruct(
        nm_dir,
        nm_centers,
        kappa=KAPPA,
        cv_unit="nm",
        energy_unit="eV",
    )
    angstrom = _reconstruct(
        angstrom_dir,
        angstrom_centers,
        kappa=KAPPA / COORDINATE_SCALE**2,
        cv_unit="angstrom",
        energy_unit="eV",
    )

    assert nm["cv_unit"] == "nm"
    assert angstrom["cv_unit"] == "angstrom"
    assert angstrom["deriv_unit"] == "eV/angstrom"
    _assert_close(
        angstrom["lengthscale"] / COORDINATE_SCALE,
        nm["lengthscale"],
    )
    _assert_close(angstrom["sigma_f"], nm["sigma_f"])
    for key in ("x_centers", "x_means", "x_star"):
        _assert_close(
            angstrom[key] / COORDINATE_SCALE,
            nm[key],
            rtol=1e-8,
            atol=2e-10,
        )
    for key in (
        "pmf_mean",
        "pmf_std",
        "pmf_std_raw",
        "pmf_f_std",
        "pmf_f_std_raw",
    ):
        _assert_close(angstrom[key], nm[key])
    for key in (
        "derivatives",
        "derivative_errors",
        "deriv_mean",
        "deriv_std",
        "deriv_std_raw",
    ):
        _assert_close(angstrom[key] * COORDINATE_SCALE, nm[key])
    _assert_close(angstrom["loo_z"], nm["loo_z"], atol=2e-7)
    _assert_close(
        angstrom["uncertainty_calibration_factor"],
        nm["uncertainty_calibration_factor"],
        rtol=5e-4,
    )
