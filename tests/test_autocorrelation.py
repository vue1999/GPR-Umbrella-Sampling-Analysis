"""Unit tests for the autocorrelation time estimator."""
import numpy as np
import pytest

from gpr_umbrella_1d.gpr import compute_tau_int


class TestComputeTauInt:
    def test_white_noise_returns_near_half(self):
        """White noise has tau_int ~ 0.5."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10_000)
        tau = compute_tau_int(x)
        assert 0.4 <= tau <= 0.6, f"tau_int for white noise should be ~0.5, got {tau}"

    def test_highly_correlated_returns_large(self):
        """A slowly-varying signal should give tau_int >> 1."""
        rng = np.random.default_rng(42)
        # AR(1) process with rho=0.99
        n = 50_000
        x = np.empty(n)
        x[0] = 0.0
        rho = 0.99
        for i in range(1, n):
            x[i] = rho * x[i - 1] + rng.standard_normal()
        tau = compute_tau_int(x, max_lag=5000)
        # Theoretical tau_int ~ 1/(1-rho) / 2 = 50 (roughly)
        assert tau > 10.0, f"Expected large tau_int for correlated signal, got {tau}"

    def test_constant_signal_returns_half(self):
        """A constant signal (zero variance) should return 0.5."""
        x = np.ones(1000)
        tau = compute_tau_int(x)
        assert tau == 0.5

    def test_very_short_signal(self):
        """Very short signals should return 0.5 gracefully."""
        assert compute_tau_int(np.array([1.0])) == 0.5
        assert compute_tau_int(np.array([1.0, 2.0])) == 0.5
        assert compute_tau_int(np.array([])) == 0.5

    def test_minimum_returned_is_half(self):
        """Result should never be less than 0.5."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            x = rng.standard_normal(500)
            tau = compute_tau_int(x)
            assert tau >= 0.5
