"""Regression test against the shipped Fe(110) H escape example data.

This test runs the full pipeline on the example data and checks that
the key output metrics are in expected ranges.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from gpr_umbrella_1d import gpr_umbrella_integration

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "fe110_h_escape"


@pytest.mark.skipif(
    not (EXAMPLE_DIR / "COLVAR").is_dir(),
    reason="Example COLVAR data not present",
)
class TestFe110Example:
    @pytest.fixture(autouse=True)
    def run_example(self, tmp_path):
        self.results = gpr_umbrella_integration(
            colvar_dir=str(EXAMPLE_DIR / "COLVAR"),
            kappa=24.305,
            centers=str(EXAMPLE_DIR / "window_centers.txt"),
            cv_unit="nm",
            energy_unit="eV",
            output_dir=str(tmp_path),
            output_prefix="fe110_test",
            plot=False,
            save_outputs=True,
            verbose=False,
        )

    def test_number_of_windows(self):
        assert len(self.results["x_centers"]) == 29

    def test_pmf_has_correct_shape(self):
        assert len(self.results["pmf_mean"]) == 200
        assert len(self.results["pmf_std"]) == 200

    def test_pmf_starts_at_zero(self):
        assert self.results["pmf_mean"][0] == 0.0
        assert self.results["pmf_std"][0] == 0.0

    def test_loo_z_score_std_reasonable(self):
        """After fixes, LOO z-score std should be much better than 12.47."""
        z_std = self.results["loo_z"].std()
        assert z_std < 5.0, f"LOO z-score std = {z_std:.2f}, still too high"

    def test_training_residual_std_reasonable(self):
        """After fixes, training residual std should be closer to 1."""
        r_std = self.results["training_std_residuals"].std()
        assert r_std < 5.0, f"Training residual std = {r_std:.2f}, still too high"

    def test_hyperparams_optimised(self):
        """Sigma_f and lengthscale should be positive and finite."""
        assert 0 < self.results["sigma_f"] < 100
        assert 0 < self.results["lengthscale"] < 20

    def test_output_files_written(self):
        assert self.results["pmf_path"] is not None
        assert os.path.isfile(self.results["pmf_path"])
        assert self.results["deriv_path"] is not None
        assert os.path.isfile(self.results["deriv_path"])
