"""Statistical and numerical regression tests for the native noise-aware fit."""
import numpy as np
import pytest

from gpr_umbrella.integration_2d import estimate_mean_covariance, _k_grad_grad, reconstruct_pmf_2d
from gpr_umbrella.fitting_2d import objective, numerical_jitter, training_covariance, fit_hyperparameters
from gpr_umbrella.cli_2d import build_parser


def test_long_blocks_capture_slow_drift_with_fast_local_fluctuations():
    rng = np.random.default_rng(21)
    slow = np.repeat(np.array([-1., 1., -.5, .5, -1.2, 1.2, -.8, .8]), 100)
    q = np.column_stack([slow, .6 * slow]) + rng.normal(0, .02, (800, 2))
    short, _ = estimate_mean_covariance(q, np.ones(2), block_size=5)
    long, size = estimate_mean_covariance(q, np.ones(2), block_size=100)
    assert size == 100 and long[0, 0] > 10 * short[0, 0]
    assert long[0, 1] > 0
    expected = np.cov(q.reshape(8, 100, 2).mean(1), rowvar=False) / 8
    np.testing.assert_allclose(long, expected, rtol=1e-12)
    for bad in [0, -1, 1.5, True, 201]:
        with pytest.raises(ValueError):
            estimate_mean_covariance(q, np.ones(2), block_size=bad)


def test_analytic_noise_objective_gradient_and_unit_scaling():
    rng = np.random.default_rng(13); X = rng.normal(size=(9, 2)); y = rng.normal(size=18)
    N = np.kron(np.eye(9), np.array([[.04, .012], [.012, .02]]))
    p = np.array([1.1, .8, 1.2, .13, .22]); scale = np.array([.7, 1.3])
    value, grad = objective(np.log(p), X, y, N, scale)
    eps = 1e-5
    numeric = np.array([(objective(np.log(p) + np.eye(5)[i] * eps, X, y, N, scale)[0]
                         - objective(np.log(p) - np.eye(5)[i] * eps, X, y, N, scale)[0]) / (2 * eps)
                        for i in range(5)])
    np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=1e-5)
    energy = 96.485; cv = np.array([.1, 3.]); force = energy / cv; obs = np.tile(force, 9)
    pp = np.r_[p[0] * energy, p[1:3] * cv, p[3:] * force]
    vv, gg = objective(np.log(pp), X * cv, y * obs, N * np.outer(obs, obs), scale * force)
    np.testing.assert_allclose(vv, value, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(gg, grad, rtol=1e-9, atol=1e-8)


def test_jitter_does_not_grow_with_signal_or_discrepancy():
    X = np.array([[0., 0.], [.3, .4], [.8, .1]])
    N = np.eye(6) * .04; y = np.arange(6) * .1
    j = numerical_jitter(N, y)
    for sf, eta in [(1., np.array([0., 0.])), (100., np.array([2., 3.]))]:
        K = _k_grad_grad(X, X, sf, np.array([.8, 1.]))
        actual = training_covariance(K, N, y, eta) - K - N - np.diag(np.tile(eta**2, 3))
        np.testing.assert_allclose(np.diag(actual), j, atol=3e-12, rtol=.01)
    scaled = numerical_jitter(N * 4, y * 2)
    np.testing.assert_allclose(scaled, 4 * j)


def test_known_discrepancy_is_recovered_and_cap_is_optional():
    rng = np.random.default_rng(9); X = rng.uniform(-2, 2, (35, 2))
    ell = np.array([1.1, .9]); sf = .7; eta = np.array([.25, .4]); N = np.eye(70) * .002**2
    cov = _k_grad_grad(X, X, sf, ell) + N + np.diag(np.tile(eta**2, len(X)))
    y = np.linalg.cholesky(cov) @ rng.normal(size=70)
    fitted = fit_hyperparameters(X, y, N, np.array([1., 1.]), np.array([6., 6.]),
                                1., ell, sf, True, fit_extra_noise=True)
    assert fitted[3]
    assert fitted[4]['sigma_f_max'] is None
    assert np.all(fitted[2] > .5 * eta) and np.all(fitted[2] < 1.8 * eta)
    capped = fit_hyperparameters(X, y, N, np.ones(2), np.ones(2) * 6,
                                1., ell, None, True, fit_extra_noise=True, sigma_f_max=.2)
    assert capped[0] <= .2 * (1 + 1e-12)
    assert 'sigma_f_upper' in capped[4]['bound_hits']
    with pytest.raises(ValueError, match='exceeds'):
        fit_hyperparameters(X, y, N, np.ones(2), np.ones(2) * 6, 1., ell, sf, True, sigma_f_max=.2)
    with pytest.raises(ValueError, match='requires'):
        fit_hyperparameters(X, y, N, np.ones(2), np.ones(2) * 6, 1., ell, None, False, fit_extra_noise=True)


def test_native_api_retains_separate_uncertainties_and_unit_invariance(tmp_path):
    rng = np.random.default_rng(7); X = rng.uniform(-1., 1., (12, 2)); g = X + rng.normal(0, .12, X.shape)
    fluctuations = rng.normal(0, .01, (12, 160, 2));fluctuations -= fluctuations.mean(1, keepdims=True)
    q = X[:, None, :] + fluctuations; k = np.full_like(X, 10.)
    def run(energy, cv, name):
        positions = [v * cv for v in q]
        data = dict(centers=(X + g / k) * cv, means=X * cv,
                    kappa=k * energy / cv**2, vars=np.array([v.var(0, ddof=1) for v in positions]),
                    n_samples=np.full(12, 160), all_positions=positions)
        return reconstruct_pmf_2d(data=data, covariance_block_size=20,
                                  fit_extra_noise=True, grid_n=(5, 5),
                                  restrict_to_sampled_support=False,
                                  output_dir=str(tmp_path / name), plot=False,
                                  plot_diagnostics=False, verbose=False)
    a = run(1., np.ones(2), 'base');b = run(10., np.array([.1, 2.]), 'scaled')
    assert np.all(a['covariance_batch_sizes'] == 20)
    expected = a['gradient_noise_cov'] + np.diag(np.tile(a['extra_noise']**2, 12))
    np.testing.assert_allclose(a['observation_noise_cov'], expected)
    assert a['optimization']['sigma_f_max'] is None
    np.testing.assert_allclose(b['lengthscale'] / [.1, 2.], a['lengthscale'], rtol=5e-3)
    np.testing.assert_allclose(b['extra_noise'] / [100., 5.], a['extra_noise'], rtol=5e-3, atol=1e-5)
    np.testing.assert_allclose(b['pmf'] / 10., a['pmf'], rtol=5e-3, atol=1e-5)
    assert list((tmp_path/'base').glob('*fit_metadata.json'))


def test_cli_exposes_native_options():
    a = build_parser().parse_args(['--colvar-dir', 'x', '--covariance-block-size', '4000',
                                  '--fit-extra-noise', '--extra-noise-scale', '1', '2', '--sigma-f-max', '3'])
    assert a.covariance_block_size == 4000 and a.fit_extra_noise
    assert a.extra_noise_scale == [1., 2.] and a.sigma_f_max == 3.
    assert build_parser().parse_args(['--colvar-dir', 'x']).sigma_f_max is None
