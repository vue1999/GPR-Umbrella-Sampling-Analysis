"""One end-to-end regression using the shipped scalar-CV example."""
from pathlib import Path

import numpy as np
import pytest

from gpr_umbrella import reconstruct_pmf_1d

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "fe_h_desorption"


@pytest.mark.skipif(not (EXAMPLE_DIR / "COLVAR").is_dir(), reason="Example data not present")
def test_shipped_example_outputs(tmp_path):
    result = reconstruct_pmf_1d(
        colvar_dir=str(EXAMPLE_DIR / "COLVAR"), kappa_dir=str(EXAMPLE_DIR / "window_kappa"),
        cv_unit="nm", energy_unit="eV", output_dir=str(tmp_path),
        output_prefix="example", plot=False, save_outputs=True, verbose=False)
    assert len(result["x_centers"]) == 36
    assert result["pmf_mean"].shape == result["pmf_std"].shape == (200,)
    assert result["pmf_mean"][0] == result["pmf_std"][0] == 0.
    assert result["loo_z"].std() < 5.
    assert result["training_std_residuals"].std() < 5.
    assert 0. < result["sigma_f"] < 100.
    assert 0. < result["lengthscale"] < 20.
    for key, suffix in [("pmf_path", "pmf_1d.dat"), ("deriv_path", "mean_force_1d.dat")]:
        output = Path(result[key])
        assert output.name == f"example_{suffix}" and output.is_file()
        assert np.isfinite(np.loadtxt(output)).all()
