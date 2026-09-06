import numpy as np
import pytest

from gpr_umbrella.arclength import project_arclength, project_polyline
from gpr_umbrella.robust_1d import (
    barrier_posterior,
    covariance_reference,
    fit_linear_observations,
)


def test_model_averaged_leaveout_matches_explicit_conditioning():
    from scipy.special import logsumexp, ndtr

    from gpr_umbrella.integration_1d import k_base

    nodes = np.linspace(-1.0, 1.0, 7)
    operator = np.diff(np.eye(7), axis=0) / np.diff(nodes)[:, None]
    values = 0.2 * np.sin(np.arange(6.0))
    noise = 0.02**2 * (np.eye(6) + 0.2 * np.ones((6, 6)))
    fit = fit_linear_observations(
        nodes,
        operator,
        values,
        noise,
        nodes,
        lengthscales=[0.4, 0.9],
        amplitudes=[0.2, 0.6],
    )
    for held in (np.array([3]), np.arange(3)):
        rest = np.setdiff1d(np.arange(6), held)
        means = []
        covariances = []
        log_evidence = []
        for ell, amplitude in fit["hyperparameter_table"][:, :2]:
            cov = operator @ k_base(nodes, nodes, amplitude, ell) @ operator.T + noise
            cov += np.eye(6) * np.mean(np.diag(noise)) * 1e-9
            train = cov[np.ix_(rest, rest)]
            cross = cov[np.ix_(held, rest)]
            means.append(cross @ np.linalg.solve(train, values[rest]))
            covariances.append(
                cov[np.ix_(held, held)] - cross @ np.linalg.solve(train, cross.T)
            )
            log_evidence.append(
                -0.5
                * (
                    values[rest] @ np.linalg.solve(train, values[rest])
                    + np.linalg.slogdet(train)[1]
                    + len(rest) * np.log(2 * np.pi)
                )
            )
        means, covariances = np.array(means), np.array(covariances)
        weights = np.exp(np.array(log_evidence) - logsumexp(log_evidence))
        mean = np.sum(weights[:, None] * means, axis=0)
        delta = means - mean
        covariance = np.sum(
            weights[:, None, None]
            * (covariances + delta[:, :, None] * delta[:, None, :]),
            axis=0,
        )
        if len(held) == 1:
            i = held[0]
            np.testing.assert_allclose(
                fit["loo_hyperparameter_weights"][:, i], weights, rtol=1e-8, atol=1e-10
            )
            np.testing.assert_allclose(fit["loo_means"][i], mean[0], rtol=1e-8)
            np.testing.assert_allclose(
                fit["loo_stds"][i] ** 2, covariance[0, 0], rtol=1e-8
            )
            probability = np.sum(
                weights
                * ndtr((values[i] - means[:, 0]) / np.sqrt(covariances[:, 0, 0]))
            )
            np.testing.assert_allclose(ndtr(fit["loo_z"][i]), probability, rtol=1e-8)
        else:
            residual = values[held] - mean
            expected = np.sqrt(
                residual @ np.linalg.solve(covariance, residual) / len(held)
            )
            np.testing.assert_allclose(fit["blocked_cv_rms"][0], expected, rtol=1e-8)


def test_predictive_quantiles_do_not_clip_extreme_failures():
    from gpr_umbrella.robust_1d import _predictive_checks

    result = _predictive_checks(
        np.array([0.0, 40.0, -40.0]), np.array([np.eye(3)]), np.zeros(1)
    )
    np.testing.assert_allclose(result["loo_z"], [0.0, 40.0, -40.0], atol=1e-10)
    np.testing.assert_allclose(result["loo_z"], result["map_loo_z"], atol=1e-10)


def test_endpoint_rays_not_clipped():
    p = project_arclength([[-0.2, 0.1], [1.3, -0.1]], [[0, 0], [1, 0]])
    np.testing.assert_allclose(p["s"], [-0.2, 1.3])
    np.testing.assert_allclose(p["distance"], [0.1, 0.1])


def test_nonmonotonic_cv1_and_reverse():
    vertices = np.array([[0, 0], [1, 1], [0, 2], [-1, 3]])
    q = np.array([[0.5, 0.5], [0.5, 1.5], [-0.5, 2.5]])
    forward = project_arclength(q, vertices)
    reverse = project_arclength(q, vertices[::-1])
    assert np.all(np.diff(forward["s"]) > 0)
    np.testing.assert_allclose(forward["s"] + reverse["s"], 3 * np.sqrt(2))


def test_self_intersection_rejected():
    with pytest.raises(ValueError, match="Self-intersecting"):
        project_arclength([[0, 0]], [[-1, -1], [1, 1], [-1, 1], [1, -1]])


def test_nearby_remote_branches_flagged_not_window_dependent():
    path = [[0, 0], [1, 0], [2, 0], [2, 0.01], [1, 0.01], [0, 0.01]]
    p = project_arclength([[0.3, 0.005]], path, branch_separation=0.5)
    assert p["ambiguous"][0]


def test_duplicate_path_vertices_fail():
    with pytest.raises(ValueError, match="duplicate"):
        project_arclength([[0, 0]], [[0, 0], [0, 0]])


def test_smoothed_reference_fold_cannot_be_sorted_away():
    from gpr_umbrella.projected_inputs import prepare_projected_observations

    path = [[0, 0], [4, 0], [4, 1], [3, 1], [3, 0.1], [2, 0.1], [2, 2], [0, 2]]
    centers = np.array([[0.0, 0.0], [0.0, 2.0]])
    trajectories = [np.tile(c, (800, 1)) for c in centers]
    with pytest.raises(ValueError, match="Nonmonotonic path progress"):
        prepare_projected_observations(
            trajectories,
            centers,
            np.ones_like(centers),
            path,
            kbt=0.03,
            bins=4,
            stride=1,
            block_size=100,
            bootstraps=32,
        )


def test_soft_projection_removes_interior_vertex_point_mass():
    q = np.random.default_rng(123).normal(0, 0.05, (10000, 2))
    path = [[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    hard = project_polyline(q, path)
    soft = project_arclength(q, path, smoothing_width=0.1)
    assert np.mean(np.abs(hard["s"] - 1) < 1e-9) > 0.2
    assert np.mean(np.abs(soft["s"] - 1) < 1e-9) < 0.001
    assert soft["s"].std() > 0.02


def test_soft_straight_line_independent_of_reference_segmentation():
    q = np.c_[np.linspace(-0.3, 1.3, 101), np.full(101, 0.07)]
    a = project_arclength(q, [[0, 0], [1, 0]], smoothing_width=0.03)
    b = project_arclength(
        q, [[0, 0], [0.13, 0], [0.8, 0], [1, 0]], smoothing_width=0.03
    )
    np.testing.assert_allclose(a["s"], q[:, 0], atol=1e-9)
    np.testing.assert_allclose(a["s"], b["s"], atol=1e-9)


def test_adjacent_retracing_rejected():
    with pytest.raises(ValueError, match="retrace"):
        project_arclength([[0.3, 0]], [[0, 0], [1, 0], [0, 0]])


def test_bad_sampling_cannot_pass_by_enormous_uncertainty():
    from gpr_umbrella.acceptance import compare_profiles, diagnostic_status

    a = {
        "x_star": np.arange(5.0),
        "pmf_mean": np.zeros(5),
        "pmf_std": np.full(5, 100.0),
    }
    b = {**a, "pmf_mean": np.arange(5.0)}
    check = compare_profiles([a, b])
    assert len(check["quality_issues"]) == 2
    assert diagnostic_status(check["quality_issues"]) == "UNRELIABLE"


def test_stable_fit_not_automatically_a_physical_barrier():
    from gpr_umbrella.acceptance import diagnostic_status

    assert (
        diagnostic_status(["physical_barrier_basins_not_specified"])
        == "FIT_STABLE_BARRIER_UNDEFINED"
    )


def test_block_reference_scales_with_chosen_block_length():
    from gpr_umbrella.projected_inputs import unweighted_block_support

    positions = [np.full(3200, 0.5)]
    short = unweighted_block_support(positions, [0, 1], block_size=100, stride=5)
    long = unweighted_block_support(positions, [0, 1], block_size=400, stride=5)
    np.testing.assert_allclose(short, [32.0])
    np.testing.assert_allclose(long, [8.0])


def test_disconnected_sampling_does_not_get_a_gp_bridge():
    pytest.importorskip("pymbar")
    from gpr_umbrella.projected_inputs import prepare_projected_observations

    rng = np.random.default_rng(10)
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    trajectories = [rng.normal(c, 0.01, (1000, 2)) for c in centers]
    with pytest.raises(ValueError, match="Disconnected"):
        prepare_projected_observations(
            trajectories,
            centers,
            np.full_like(centers, 30.0),
            centers,
            kbt=0.026,
            bins=8,
            stride=5,
            block_size=100,
            bootstraps=32,
        )


def make_fit(scale_x=1.0, scale_e=1.0):
    nodes = np.linspace(-1.5, 1.5, 25) * scale_x
    op = np.diff(np.eye(len(nodes)), axis=0) / np.diff(nodes)[:, None]
    f = 0.3 * (nodes / scale_x) ** 2 * scale_e
    noise = np.eye(len(nodes) - 1) * (0.025 * scale_e / scale_x) ** 2
    grid = np.linspace(nodes[0], nodes[-1], 81)
    return fit_linear_observations(
        nodes,
        op,
        op @ f,
        noise,
        grid,
        lengthscales=np.geomspace(0.125, 9, 12) * scale_x,
        amplitudes=np.geomspace(0.03, 30, 12) * scale_e,
    )


def test_linear_operator_recovers_known_pmf_and_covariance():
    result = make_fit()
    truth = 0.3 * result["x_star"] ** 2
    truth -= truth[0]
    assert np.max(np.abs(result["pmf_mean"] - truth)) < 0.025
    assert np.linalg.eigvalsh(result["pmf_covariance"]).min() > -1e-8
    assert result["pmf_std"][0] < 1e-7
    assert np.max(np.abs(result["deriv_mean"] - 0.6 * result["x_star"])) < 0.08


def test_units_covariant():
    a, b = make_fit(), make_fit(0.1, 96.485)
    np.testing.assert_allclose(
        b["pmf_mean"] / 96.485, a["pmf_mean"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        b["pmf_std"] / 96.485, a["pmf_std"], rtol=1e-3, atol=1e-5
    )


def test_empty_prediction_grid_fails_cleanly():
    with pytest.raises(ValueError, match="Invalid GP"):
        fit_linear_observations(
            np.arange(5.0), np.diff(np.eye(5), axis=0), np.ones(4), np.eye(4), []
        )


def test_duplicate_hyperparameter_grid_points_fail():
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_linear_observations(
            np.arange(5.0),
            np.diff(np.eye(5), axis=0),
            np.ones(4),
            np.eye(4),
            np.arange(5.0),
            lengthscales=[1.0, 1.0],
        )


def test_barrier_cannot_use_extrapolated_support_or_zero_draws():
    result = make_fit()
    with pytest.raises(ValueError, match="posterior draws"):
        barrier_posterior(result, (-0.3, 0.3), (1.0, 1.5), draws=0)
    result["observed_support"] = [-0.5, 0.5]
    with pytest.raises(ValueError, match="observed support"):
        barrier_posterior(result, (-0.3, 0.3), (1.0, 1.5))


def test_no_sub_resolution_hypers():
    with pytest.raises(ValueError, match="unresolved"):
        fit_linear_observations(
            np.arange(5.0),
            np.diff(np.eye(5), axis=0),
            np.ones(4),
            np.eye(4),
            np.arange(5.0),
            lengthscales=[0.01],
        )


def test_incompatible_rough_observations_cannot_get_a_good_fit_label():
    nodes = np.linspace(0.0, 3.0, 21)
    operator = np.diff(np.eye(len(nodes)), axis=0) / np.diff(nodes)[:, None]
    values = (-1.0) ** np.arange(len(nodes) - 1)
    fit = fit_linear_observations(
        nodes,
        operator,
        values,
        np.eye(len(values)) * 0.001**2,
        np.linspace(0.0, 3.0, 81),
        lengthscales=np.geomspace(0.15, 9.0, 15),
        amplitudes=np.geomspace(0.01, 3.0, 12),
    )
    assert set(fit["quality_issues"]) & {
        "poor_raw_leave_one_out",
        "poor_blocked_cross_validation",
        "lengthscale_prior_boundary",
        "amplitude_prior_boundary",
    }


def test_correlated_barrier_and_explicit_basins():
    result = make_fit()
    barrier = barrier_posterior(result, (-0.3, 0.3), (1.0, 1.5), draws=500)
    assert abs(barrier["median"] - 0.675) < 0.05
    assert barrier["std"] > 0
    with pytest.raises(ValueError, match="non-overlapping"):
        barrier_posterior(result, (-0.3, 0.3), (0, 1))


def test_reference_covariance_gauge_invariant():
    cov = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(
        covariance_reference(cov), covariance_reference(cov + 10)
    )


def test_original_bias_reweighting_recovers_straight_path_marginal(tmp_path):
    pytest.importorskip("pymbar")
    from gpr_umbrella.projected_inputs import prepare_projected_observations

    rng = np.random.default_rng(7)
    centers = np.c_[np.linspace(-1.5, 1.5, 15), np.sin(np.linspace(0, 2, 15)) * 0.2]
    kappa = np.tile([8.0, 0.2], (len(centers), 1))
    curvature = np.array([0.4, 2.0])
    kbt = 0.3
    trajectories = [
        rng.normal(k * c / (k + curvature), np.sqrt(kbt / (k + curvature)), (3200, 2))
        for c, k in zip(centers, kappa)
    ]

    def planned_interruption(message):
        if message == "block bootstrap 16/32":
            raise RuntimeError("planned interruption")

    with pytest.raises(RuntimeError, match="planned interruption"):
        prepare_projected_observations(
            trajectories,
            centers,
            kappa,
            [[-1.5, 0], [1.5, 0]],
            kbt=kbt,
            bins=12,
            stride=4,
            block_size=80,
            bootstraps=32,
            reweight_cache=tmp_path / "cache",
            progress=planned_interruption,
        )
    with np.load(next((tmp_path / "cache").glob("*.npz"))) as checkpoint:
        assert np.all(np.isfinite(checkpoint["bootstrap_f"]), axis=1).sum() == 16
    p = prepare_projected_observations(
        trajectories,
        centers,
        kappa,
        [[-1.5, 0], [1.5, 0]],
        kbt=kbt,
        bins=12,
        stride=4,
        block_size=80,
        bootstraps=32,
        reweight_cache=tmp_path / "cache",
    )
    truth = 0.2 * (p["nodes"] - 1.5) ** 2
    difference = p["histogram_pmf"] - truth
    assert np.ptp(difference) < 0.18
    assert p["overlap_components"] == 1
    assert np.linalg.eigvalsh(p["noise_covariance"]).min() > 0
    # In a separable straight-path system a common normal restraint changes
    # only the free-energy constant, not the longitudinal PMF shape.
    confined = prepare_projected_observations(
        trajectories,
        centers,
        kappa,
        [[-1.5, 0], [1.5, 0]],
        kbt=kbt,
        bins=12,
        stride=4,
        block_size=80,
        bootstraps=32,
        target_normal_kappa=10.0,
        bin_edges=np.r_[np.linspace(-0.25, 0.5, 5), np.linspace(0.5, 3.25, 9)[1:]],
        reweight_cache=tmp_path / "cache",
    )
    assert confined["nodes"][0] < 0 and confined["nodes"][-1] > 3
    confined_truth = 0.2 * (confined["nodes"] - 1.5) ** 2
    assert np.ptp(confined["histogram_pmf"] - confined_truth) < 0.18
    assert p["reweight_cache_hits"] == 16
    assert confined["reweight_cache_hits"] == 32


def test_curved_path_original_2d_bias_known_angular_barrier():
    pytest.importorskip("pymbar")
    from gpr_umbrella.projected_inputs import prepare_projected_observations

    rng = np.random.default_rng(712)
    angles = np.linspace(0.2, np.pi - 0.2, 17)
    centers = np.c_[np.cos(angles), np.sin(angles)]
    kappas = np.full_like(centers, 20.0)
    path_angles = np.linspace(0.2, np.pi - 0.2, 81)
    path = np.c_[np.cos(path_angles), np.sin(path_angles)]
    kbt = 0.2

    def energy(q):
        theta = np.arctan2(q[:, 1], q[:, 0])
        physical = 0.7 * np.sin(theta) + 100 * (np.linalg.norm(q, axis=1) - 1) ** 2
        return np.where(
            (theta > -0.1) & (theta < np.pi + 0.1), physical, np.inf
        ) + 10 * np.sum((q - centers) ** 2, axis=1)

    q = centers.copy()
    u = energy(q)
    samples = []
    for step in range(14000):
        proposal = q + rng.normal(0, 0.05, q.shape)
        trial = energy(proposal)
        accept = np.log(rng.random(len(q))) < -(trial - u) / kbt
        q[accept], u[accept] = proposal[accept], trial[accept]
        if step >= 2000 and step % 4 == 0:
            samples.append(q.copy())
    trajectories = np.transpose(samples, (1, 0, 2))
    p = prepare_projected_observations(
        trajectories,
        centers,
        kappas,
        path,
        kbt=kbt,
        bins=12,
        stride=2,
        block_size=100,
        bootstraps=32,
    )
    truth = 0.7 * np.sin(0.2 + p["nodes"])
    assert np.ptp(p["histogram_pmf"] - truth) < 0.16
    fit = fit_linear_observations(
        p["nodes"],
        p["operator"],
        p["values"],
        p["noise_covariance"],
        np.linspace(p["nodes"][0], p["nodes"][-1], 81),
    )
    truth = 0.7 * np.sin(0.2 + fit["x_star"])
    truth -= truth[0]
    assert np.max(np.abs(fit["pmf_mean"] - truth)) < 0.15
