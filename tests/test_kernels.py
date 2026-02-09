"""Unit tests for GP kernel functions.

These verify the mathematical correctness of the kernel derivatives by
comparing analytic kernel expressions against finite-difference
approximations of the base SE kernel.
"""
import numpy as np
import pytest

from gpr_umbrella_1d.gpr import k_base, k_f_fprime, k_fprime_fprime


SIGMA_F = 1.5
ELL = 0.8
ATOL = 1e-5  # tolerance for finite-difference comparison


def _finite_diff_dk_dx(x_star, x, sigma_f, ell, h=1e-6):
    """Finite-difference dk/dx  (derivative w.r.t. second argument)."""
    kp = k_base(x_star, x + h, sigma_f, ell)
    km = k_base(x_star, x - h, sigma_f, ell)
    return (kp - km) / (2 * h)


def _finite_diff_d2k_dx1dx2(x1, x2, sigma_f, ell, h=1e-4):
    """Finite-difference d^2k / dx1 dx2.

    Uses a larger step than the first-derivative helper because the
    second derivative via central differences has O(h^2) error that
    is amplified by 1/h^2 in the denominator.
    """
    kpp = k_base(x1 + h, x2 + h, sigma_f, ell)
    kpm = k_base(x1 + h, x2 - h, sigma_f, ell)
    kmp = k_base(x1 - h, x2 + h, sigma_f, ell)
    kmm = k_base(x1 - h, x2 - h, sigma_f, ell)
    return (kpp - kpm - kmp + kmm) / (4 * h * h)


class TestKBase:
    def test_diagonal_equals_sigma_f_squared(self):
        x = np.array([0.0, 1.0, 2.0])
        K = k_base(x, x, SIGMA_F, ELL)
        np.testing.assert_allclose(np.diag(K), SIGMA_F**2)

    def test_symmetric(self):
        x = np.array([0.0, 0.5, 1.5, 3.0])
        K = k_base(x, x, SIGMA_F, ELL)
        np.testing.assert_allclose(K, K.T)

    def test_positive_definite(self):
        x = np.linspace(0, 3, 10)
        K = k_base(x, x, SIGMA_F, ELL)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-10)

    def test_decays_with_distance(self):
        x1 = np.array([0.0])
        x_near = np.array([0.1])
        x_far = np.array([5.0])
        k_near = k_base(x1, x_near, SIGMA_F, ELL)[0, 0]
        k_far = k_base(x1, x_far, SIGMA_F, ELL)[0, 0]
        assert k_near > k_far


class TestKfFprime:
    """Test Cov(f(x*), f'(x)) = dk/dx."""

    def test_matches_finite_difference(self):
        x_star = np.array([0.3, 1.0, 2.5])
        x = np.array([0.5, 1.5, 3.0])
        analytic = k_f_fprime(x_star, x, SIGMA_F, ELL)
        fd = _finite_diff_dk_dx(x_star, x, SIGMA_F, ELL)
        np.testing.assert_allclose(analytic, fd, atol=ATOL)

    def test_zero_on_diagonal(self):
        """dk/dx at x*=x should be zero (derivative of symmetric peak)."""
        x = np.array([1.0, 2.0, 3.0])
        K = k_f_fprime(x, x, SIGMA_F, ELL)
        np.testing.assert_allclose(np.diag(K), 0.0, atol=1e-14)

    def test_antisymmetric_in_arguments(self):
        """k_f_fprime(a, b) = -k_f_fprime(b, a) for SE kernel."""
        a = np.array([0.5, 1.5])
        b = np.array([1.0, 2.0])
        kab = k_f_fprime(a, b, SIGMA_F, ELL)
        kba = k_f_fprime(b, a, SIGMA_F, ELL)
        np.testing.assert_allclose(kab, -kba.T, atol=1e-14)


class TestKfprimeFprime:
    """Test Cov(f'(x1), f'(x2)) = d^2k/dx1 dx2."""

    def test_matches_finite_difference(self):
        x1 = np.array([0.3, 1.0, 2.5])
        x2 = np.array([0.5, 1.5, 3.0])
        analytic = k_fprime_fprime(x1, x2, SIGMA_F, ELL)
        fd = _finite_diff_d2k_dx1dx2(x1, x2, SIGMA_F, ELL)
        np.testing.assert_allclose(analytic, fd, atol=ATOL)

    def test_symmetric(self):
        x = np.array([0.0, 0.5, 1.0, 2.0])
        K = k_fprime_fprime(x, x, SIGMA_F, ELL)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_diagonal_equals_sigma_f_squared_over_ell_squared(self):
        """At x1==x2, d^2k/dx1dx2 = sigma_f^2 / ell^2."""
        x = np.array([0.0, 1.0, 2.0])
        K = k_fprime_fprime(x, x, SIGMA_F, ELL)
        expected = SIGMA_F**2 / ELL**2
        np.testing.assert_allclose(np.diag(K), expected)
