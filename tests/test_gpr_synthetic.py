"""Integration test with synthetic data where the answer is known.

We create a simple harmonic PMF  F(x) = 0.5 * a * x^2  with known
derivative dF/dx = a * x, then run GPR umbrella integration and check
that the recovered PMF and derivatives are close to the ground truth.
"""
import os
import tempfile

import numpy as np
import pytest

from gpr_umbrella_1d.gpr import gpr_umbrella_integration


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


class TestSyntheticHarmonic:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmpdir = str(tmp_path)
        self.colvar_dir, self.centers_file, self.kappa, self.centers, self.a = \
            _generate_synthetic_colvar_data(self.tmpdir)

    def test_pmf_shape_is_parabolic(self):
        """Recovered PMF should approximate 0.5 * a * x^2."""
        results = gpr_umbrella_integration(
            colvar_dir=self.colvar_dir,
            kappa=self.kappa,
            centers=self.centers_file,
            cv_unit="nm",
            energy_unit="eV",
            output_dir=self.tmpdir,
            output_prefix="synthetic",
            plot=False,
            save_outputs=False,
            verbose=False,
        )

        x = results["x_star"]
        pmf = results["pmf_mean"]

        # Shift reference so both are zero at x[0]
        true_pmf = 0.5 * self.a * x**2
        true_pmf_shifted = true_pmf - true_pmf[0]

        # Allow generous tolerance since the synthetic data is noisy
        # and the mock kT assumption is simplified, but the shape
        # should match a parabola.  Check correlation is very high.
        corr = np.corrcoef(pmf, true_pmf_shifted)[0, 1]
        assert corr > 0.95, f"PMF correlation with true parabola = {corr:.3f}"

    def test_derivatives_match_linear(self):
        """Recovered derivatives should approximate a * x."""
        results = gpr_umbrella_integration(
            colvar_dir=self.colvar_dir,
            kappa=self.kappa,
            centers=self.centers_file,
            cv_unit="nm",
            energy_unit="eV",
            output_dir=self.tmpdir,
            output_prefix="synthetic",
            plot=False,
            save_outputs=False,
            verbose=False,
        )

        x = results["x_star"]
        deriv = results["deriv_mean"]
        true_deriv = self.a * x

        corr = np.corrcoef(deriv, true_deriv)[0, 1]
        assert corr > 0.95, f"Derivative correlation with true linear = {corr:.3f}"

    def test_loo_z_scores_reasonable(self):
        """LOO z-score std should be in a reasonable range."""
        results = gpr_umbrella_integration(
            colvar_dir=self.colvar_dir,
            kappa=self.kappa,
            centers=self.centers_file,
            cv_unit="nm",
            energy_unit="eV",
            output_dir=self.tmpdir,
            output_prefix="synthetic",
            plot=False,
            save_outputs=False,
            verbose=False,
        )

        loo_z_std = results["loo_z"].std()
        # A well-calibrated model should have LOO z-score std near 1.
        # Be generous: 0.3 to 3.0
        assert 0.3 < loo_z_std < 3.0, f"LOO z-score std = {loo_z_std:.2f}"

    def test_output_files_created(self):
        """Ensure PMF/derivative/figure files are saved correctly."""
        results = gpr_umbrella_integration(
            colvar_dir=self.colvar_dir,
            kappa=self.kappa,
            centers=self.centers_file,
            cv_unit="nm",
            energy_unit="eV",
            output_dir=self.tmpdir,
            output_prefix="synthetic",
            plot=True,
            save_fig=True,
            save_outputs=True,
            verbose=False,
            show=False,
        )

        assert results["pmf_path"] is not None
        assert results["deriv_path"] is not None
        assert results["figure_path"] is not None
        assert os.path.isfile(results["pmf_path"])
        assert os.path.isfile(results["deriv_path"])
        assert os.path.isfile(results["figure_path"])

        # Check PMF file has correct shape
        pmf_data = np.loadtxt(results["pmf_path"])
        assert pmf_data.shape[1] == 3
        assert pmf_data.shape[0] == 200  # default n_star


class TestEdgeCases:
    def test_mismatched_centers_raises(self, tmp_path):
        """Providing wrong number of centres should raise ValueError."""
        tmpdir = str(tmp_path)
        colvar_dir, _, kappa, _, _ = _generate_synthetic_colvar_data(tmpdir, n_windows=10)

        # Write a centres file with wrong count
        bad_centers = os.path.join(tmpdir, "bad_centers.txt")
        np.savetxt(bad_centers, np.linspace(0, 1, 5), fmt="%.4f")

        with pytest.raises(ValueError, match="centres"):
            gpr_umbrella_integration(
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
            gpr_umbrella_integration(plot=False, save_outputs=False, verbose=False)

    def test_both_data_sources_raises(self, tmp_path):
        """Passing both data_folder and colvar_dir should raise."""
        with pytest.raises(ValueError, match="not both"):
            gpr_umbrella_integration(
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
            gpr_umbrella_integration(
                colvar_dir=empty_dir,
                kappa=1.0,
                centers=[0.0],
                plot=False,
                save_outputs=False,
                verbose=False,
            )
