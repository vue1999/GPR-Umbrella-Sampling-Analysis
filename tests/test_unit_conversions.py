"""Regression tests for numerical energy-unit conversion in data loaders."""
from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella.integration_1d import load_window_data
from gpr_umbrella.integration_2d import load_plumed_colvar_2d


@pytest.mark.parametrize(
    ("energy_unit", "conversion"),
    [
        ("eV", 1.0 / 96.485),
        ("kJ/mol", 1.0),
        ("kcal/mol", 1.0 / 4.184),
        ("J/mol", 1000.0),
        ("Hartree", 1.0 / 2625.4996394799),
    ],
)
def test_1d_and_2d_loaders_convert_kj_kappa_to_selected_energy_unit(
    tmp_path, energy_unit, conversion,
):
    raw_kappa = 96.485

    # The preprocessed 1D format uses columns time, CV, centre, kappa.
    (tmp_path / "window_0.ui_dat").write_text(
        f"0.0 -0.1 0.0 {raw_kappa}\n"
        f"1.0  0.1 0.0 {raw_kappa}\n"
    )
    one_dimensional = load_window_data(
        str(tmp_path),
        kappa_in_kj_per_mol=True,
        energy_unit=energy_unit,
    )

    # The 2D PLUMED loader reads time, CV0, CV1 plus a separate
    # centre/kappa record c0, c1, kappa0, kappa1.
    (tmp_path / "COLVAR_window_0.dat").write_text(
        "0.0 -0.1 0.2\n"
        "1.0  0.1 0.4\n"
    )
    (tmp_path / "window_centers_kappa_0.txt").write_text(
        f"0.0 0.3 {raw_kappa} {2.0 * raw_kappa}\n"
    )
    two_dimensional = load_plumed_colvar_2d(
        str(tmp_path),
        kappa_in_kj_per_mol=True,
        energy_unit=energy_unit,
    )

    assert one_dimensional["kappa"][0] == pytest.approx(
        raw_kappa * conversion
    )
    np.testing.assert_allclose(
        two_dimensional["kappa"][0],
        raw_kappa * conversion * np.array([1.0, 2.0]),
        rtol=2e-14,
    )
