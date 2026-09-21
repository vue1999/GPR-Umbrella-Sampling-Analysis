"""Integration test with synthetic data where the answer is known.

We create a simple harmonic PMF  F(x) = 0.5 * a * x^2  with known
derivative dF/dx = a * x, then run GPR umbrella integration and check
that the recovered PMF and derivatives are close to the ground truth.
"""
import os

import numpy as np
import pytest

from gpr_umbrella_1d.gpr import gpr_umbrella_integration as reconstruct_pmf_1d


def _generate_synthetic_colvar_data(tmpdir: str, n_windows: int = 15):
    """Generate synthetic COLVAR files for a harmonic potential.

    The true PMF is F(x) = 0.5 * a * x^2 with a = 2.0 eV/nm^2,
    so dF/dx = a * x.  Each umbrella window is biased with kappa = 50 eV/nm^2,
    centred at uniformly spaced points.
    """
    a = 2.0  # true curvature, eV/nm^2
    kappa = 50.0  # umbrella force constant, eV/nm^2
    n_samples = 5000

    centers = np.linspace(-2.0, 2.0, n_windows)
    colvar_dir = os.path.join(tmpdir, "COLVAR")
    os.makedirs(colvar_dir, exist_ok=True)

    centers_file = os.path.join(tmpdir, "window_centers.txt")
    np.savetxt(centers_file, centers, fmt="%.6f")

    rng = np.random.default_rng(12345)

    for i, c in enumerate(centers):
        # Effective spring constant: kappa_eff = kappa + a
        # Mean position: x_mean = kappa * c / (kappa + a)
        kappa_eff = kappa + a
        x_mean = kappa * c / kappa_eff
        x_std = np.sqrt(1.0 / kappa_eff)  # units: nm  (kT=1 here conceptually; we treat as eV directly)
        # For testing purposes, we treat kT as absorbed into the units.
        # The derivative will still be kappa * (center - <x>).
        positions = rng.normal(loc=x_mean, scale=x_std, size=n_samples)

        fpath = os.path.join(colvar_dir, f"COLVAR_window_{i}.dat")
        time = np.arange(n_samples, dtype=float) * 0.001
        data = np.column_stack([time, positions])
        header = "#! FIELDS time x"
        np.savetxt(fpath, data, header=header, comments="", fmt="%.6f")

    return colvar_dir, centers_file, kappa, centers, a


@pytest.fixture(scope="module")
def harmonic_fit(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("harmonic"))
    colvars, centers, kappa, _, curvature = _generate_synthetic_colvar_data(root)
    result = reconstruct_pmf_1d(
        colvar_dir=colvars, kappa=kappa, centers=centers, cv_unit="nm", energy_unit="eV",
        output_dir=root, output_prefix="synthetic", plot=True, save_fig=True,
        save_outputs=True, verbose=False, show=False)
    return result, curvature


def test_harmonic_pmf_and_derivative_recovery(harmonic_fit):
    result, curvature = harmonic_fit
    x = result["x_star"]
    truth = .5 * curvature * x**2
    assert np.corrcoef(result["pmf_mean"], truth - truth[0])[0, 1] > .95
    assert np.corrcoef(result["deriv_mean"], curvature * x)[0, 1] > .95
    assert .3 < result["loo_z"].std() < 3.


def test_harmonic_outputs_and_uncertainty(harmonic_fit):
    result, _ = harmonic_fit
    for key, suffix in [("pmf_path", "pmf_1d.dat"), ("deriv_path", "mean_force_1d.dat"),
                        ("figure_path", "diagnostics_1d.png")]:
        assert os.path.isfile(result[key])
        assert os.path.basename(result[key]) == f"synthetic_{suffix}"
    assert np.loadtxt(result["pmf_path"]).shape == (200, 3)
    np.testing.assert_allclose(np.diag(result["pmf_covariance"]), result["pmf_std"]**2, atol=1e-10)
    assert result["uncertainty_calibration_factor"] >= 1.


class TestEdgeCases:
    def test_mismatched_centers_raises(self, tmp_path):
        """Providing wrong number of centres should raise ValueError."""
        tmpdir = str(tmp_path)
        colvar_dir, _, kappa, _, _ = _generate_synthetic_colvar_data(tmpdir, n_windows=10)

        # Write a centres file with wrong count
        bad_centers = os.path.join(tmpdir, "bad_centers.txt")
        np.savetxt(bad_centers, np.linspace(0, 1, 5), fmt="%.4f")

        with pytest.raises(ValueError, match="centres"):
            reconstruct_pmf_1d(
                colvar_dir=colvar_dir,
                kappa=kappa,
                centers=bad_centers,
                plot=False,
                save_outputs=False,
                verbose=False,
            )

    def test_no_data_source_raises(self):
        """Passing neither data_folder nor colvar_dir should raise."""
        with pytest.raises(ValueError, match="Must provide"):
            reconstruct_pmf_1d(plot=False, save_outputs=False, verbose=False)

    def test_both_data_sources_raises(self, tmp_path):
        """Passing both data_folder and colvar_dir should raise."""
        with pytest.raises(ValueError, match="not both"):
            reconstruct_pmf_1d(
                data_folder=str(tmp_path),
                colvar_dir=str(tmp_path),
                plot=False,
                save_outputs=False,
                verbose=False,
            )

    def test_empty_colvar_dir_raises(self, tmp_path):
        """An empty directory should raise ValueError about no files."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with pytest.raises(ValueError, match="No COLVAR"):
            reconstruct_pmf_1d(
                colvar_dir=empty_dir,
                kappa=1.0,
                centers=[0.0],
                plot=False,
                save_outputs=False,
                verbose=False,
            )
