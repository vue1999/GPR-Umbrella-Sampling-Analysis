"""Prepare correlated 1D free-energy observations from original 2D biases.

MBAR is used only for unbiasing/preparation, not as a replacement GP. The
physical Hamiltonian (including common walls) must be the same in all windows.
Uncertainty is estimated by resampling contiguous blocks within each window;
PyMBAR's independent-sample covariance is deliberately NOT used on MD frames.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.sparse.csgraph import connected_components

from .arclength import project_arclength


def unweighted_block_support(positions, edges, *, block_size, stride):
    """Reference block ESS for the *same* binning and block-length convention."""
    histograms = []
    for pos in positions:
        raw = np.arange(0, len(pos), stride)
        ids = np.searchsorted(edges, np.asarray(pos)[::stride], side="right") - 1
        ids[np.asarray(pos)[::stride] == edges[-1]] = len(edges) - 2
        blocks = [np.flatnonzero(raw // block_size == b) for b in np.unique(raw // block_size)]
        if len(blocks) > 1 and len(blocks[-1]) < .5 * block_size / stride:
            blocks[-2] = np.r_[blocks[-2], blocks[-1]]; blocks.pop()
        histograms.extend([np.bincount(ids[b][(ids[b] >= 0) & (ids[b] < len(edges)-1)], minlength=len(edges)-1) for b in blocks])
    counts = np.array(histograms, dtype=float)
    return np.divide(counts.sum(axis=0)**2, np.sum(counts**2, axis=0),
                     out=np.zeros(len(edges)-1), where=np.sum(counts**2, axis=0) > 0)


def _mbar_weights(reduced_bias, counts, initial=None):
    from pymbar import MBAR
    mbar = MBAR(reduced_bias, counts, initial_f_k=initial,
                relative_tolerance=1e-9, maximum_iterations=10000,
                solver_protocol="robust", verbose=False)
    log_denominator = logsumexp(np.log(counts)[:, None] + mbar.f_k[:, None] - reduced_bias, axis=0)
    log_weights = -log_denominator
    log_weights -= logsumexp(log_weights)
    # Explicit self-consistency verification: do not trust solver messages.
    residual = -logsumexp(-reduced_bias - log_denominator, axis=1) - mbar.f_k
    if np.max(np.abs(residual - residual[0])) > 1e-6:
        raise ValueError("MBAR did not converge to the requested tolerance")
    return mbar, log_weights


def _histogram_free_energy(log_weights, s, edges, kbt):
    ids = np.searchsorted(edges, s, side="right") - 1
    ids[s == edges[-1]] = len(edges) - 2
    log_mass = np.array([logsumexp(log_weights[ids == i]) for i in range(len(edges) - 1)])
    if not np.all(np.isfinite(log_mass)):
        raise ValueError("Unsampled arclength bins: reduce bins or collect missing windows; no GP bridging allowed")
    f = -kbt * (log_mass - np.log(np.diff(edges)))
    return f - f[0], ids


def prepare_projected_observations(trajectories, centers, kappas, vertices, *,
                                   kbt, bins=None, stride=10, block_size=1000,
                                   bootstraps=128, seed=2026,
                                   ambiguity_distance=0.01, target_normal_kappa=0., progress=None):
    """All trajectory rows must already have burn-in removed.

    ``block_size`` and ``stride`` are in original stored-sample units. Thinning
    controls computational cost, not claimed independence. Bootstrap blocks
    are resampled independently for each window and include the remainder.
    Repeat with longer blocks and finer thinning/binning before accepting a
    barrier. No pseudocounts, outlier deletion or fitted noise inflation.

    A nonzero ``target_normal_kappa`` defines a DIFFERENT thermodynamic target:
    a common harmonic restraint on distance from the reference path. All
    original bias energies are still used exactly. This is a path-restrained
    marginal, not an estimate of the fully unrestrained marginal. It can be
    better supported by narrow ribbons, but must be labelled and reported.
    """
    trajectories = [np.asarray(q, float) for q in trajectories]
    centers, kappas, vertices = [np.asarray(a, float) for a in (centers, kappas, vertices)]
    if (len(trajectories) < 2 or centers.shape != (len(trajectories), 2)
            or kappas.shape != centers.shape or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(kappas)) or np.any(kappas <= 0)
            or not np.isfinite(kbt) or kbt <= 0 or stride < 1 or block_size < stride
            or bootstraps < 16):
        raise ValueError("Invalid projected-umbrella inputs or sampling settings")
    if not np.isfinite(target_normal_kappa) or target_normal_kappa < 0:
        raise ValueError("Target normal restraint must be finite and nonnegative")
    if any(q.ndim != 2 or q.shape[1] != 2 or len(q) < block_size * 8 or not np.all(np.isfinite(q)) for q in trajectories):
        raise ValueError("Need finite 2D trajectories with at least eight time blocks per window")
    projected = [project_arclength(q, vertices, ambiguity_distance=ambiguity_distance) for q in trajectories]
    center_projection = project_arclength(centers, vertices, ambiguity_distance=ambiguity_distance)
    path_length = projected[0]["vertex_s"][-1]
    bins = bins or 2 * len(centers)
    if bins < 4:
        raise ValueError("At least four histogram bins required")
    if bootstraps < 2 * bins:
        raise ValueError("Use at least twice as many block bootstraps as histogram bins")
    edges = np.linspace(0., path_length, bins + 1)
    q = np.concatenate([t[::stride] for t in trajectories])
    s = np.concatenate([p["s"][::stride] for p in projected])
    normal = np.concatenate([p["distance"][::stride] for p in projected])
    reduced_target = .5 * target_normal_kappa / kbt * normal**2
    def target_weights(lw, indices=None):
        lw = lw - (reduced_target if indices is None else reduced_target[indices])
        return lw - logsumexp(lw)
    counts = np.array([len(t[::stride]) for t in trajectories])
    reduced_bias = .5 / kbt * np.sum((q[None, :, :] - centers[:, None, :])**2 * kappas[:, None, :], axis=2)
    mbar, log_weights = _mbar_weights(reduced_bias, counts)
    log_weights = target_weights(log_weights)
    overlap = mbar.compute_overlap()
    overlap_scalar = float(np.real_if_close(overlap["scalar"]))
    matrix = overlap["matrix"]
    n_components, labels = connected_components((matrix + matrix.T) / 2 > 1e-3, directed=False)
    if n_components > 1:
        raise ValueError(f"Disconnected window overlap ({n_components} components at threshold 0.001); relative free energies are not supported. Collect bridge windows.")
    f, bin_ids = _histogram_free_energy(log_weights, s, edges, kbt)
    offsets = np.r_[0, np.cumsum(counts)]
    blocks = []
    for i, t in enumerate(trajectories):
        raw_indices = np.arange(0, len(t), stride)
        ids = raw_indices // block_size
        window_blocks = [offsets[i] + np.flatnonzero(ids == b) for b in np.unique(ids)]
        # Merge a short remainder to avoid nearly empty blocks.
        if len(window_blocks[-1]) < .5 * block_size / stride:
            window_blocks[-2] = np.r_[window_blocks[-2], window_blocks[-1]]
            window_blocks.pop()
        blocks.append(window_blocks)
    # Effective number of time blocks supporting each bin (weight concentration).
    log_block_mass = np.array([[logsumexp(log_weights[b][bin_ids[b] == j]) for j in range(bins)]
                              for window in blocks for b in window])
    log_mass = logsumexp(log_block_mass, axis=0)
    block_ess = np.exp(2 * log_mass - logsumexp(2 * log_block_mass, axis=0))
    unweighted_block_ess = unweighted_block_support([p["s"] for p in projected], edges,
                                                   block_size=block_size, stride=stride)
    relative_block_ess = block_ess / unweighted_block_ess
    rng = np.random.default_rng(seed)
    boot_f = []
    failed_bootstraps = 0
    for repeat in range(bootstraps):
        window_indices = [np.concatenate([bs[i] for i in rng.integers(0, len(bs), len(bs))]) for bs in blocks]
        indices = np.concatenate(window_indices)
        try:
            _, lw = _mbar_weights(reduced_bias[:, indices], np.array([len(a) for a in window_indices]), mbar.f_k)
            lw = target_weights(lw, indices)
            bf, _ = _histogram_free_energy(lw, s[indices], edges, kbt)
            boot_f.append(bf)
        except (ValueError, np.linalg.LinAlgError):
            failed_bootstraps += 1
        if progress is not None and (repeat + 1) % 16 == 0:
            progress(f"block bootstrap {repeat + 1}/{bootstraps}")
    if len(boot_f) < max(16, 2 * bins):
        raise ValueError(f"Too many failed/empty-bin block bootstraps ({failed_bootstraps}); sampled support is insufficient")
    nodes = (edges[1:] + edges[:-1]) / 2
    operator = np.diff(np.eye(bins), axis=0) / np.diff(nodes)[:, None]
    bootstrap_values = np.array(boot_f) @ operator.T
    covariance = np.cov(bootstrap_values, rowvar=False)
    # Small, reported covariance shrinkage stabilises finite bootstrap estimates.
    covariance = .98 * covariance + .02 * np.diag(np.diag(covariance))
    issues = []
    if n_components > 1:
        issues.append("disconnected_window_overlap")
    if overlap_scalar < 1e-5:
        issues.append("weak_global_window_overlap")
    # Eight 10-ps blocks cannot have ESS >=10, even for perfect IID sampling.
    # Check both absolute support and loss relative to the same block layout;
    # otherwise increasing block length automatically manufactures a failure.
    if block_ess.min() < 4 or relative_block_ess.min() < .25:
        issues.append("low_effective_block_support")
    if failed_bootstraps:
        issues.append("failed_block_bootstraps")
    projection_summary = []
    for i, p in enumerate(projected):
        fraction = float(np.mean(p["ambiguous"]))
        if fraction > .01:
            issues.append(f"ambiguous_projection_window_{i}")
        projection_summary.append({"window": i, "center_s": float(center_projection["s"][i]),
                                   "mean_s": float(np.mean(p["s"])),
                                   "distance_p95": float(np.quantile(p["distance"], .95)),
                                   "ambiguous_fraction": fraction,
                                   "outside_endpoint_fraction": float(np.mean((p["s"] < 0) | (p["s"] > path_length)))})
    # First/last halves are independent convergence checks, not an error model.
    halves = []
    for half in (0, 1):
        indices = np.concatenate([np.arange(offsets[i], offsets[i + 1])[slice(None, counts[i] // 2) if half == 0 else slice(counts[i] // 2, None)]
                                  for i in range(len(counts))])
        half_counts = np.array([c // 2 if half == 0 else c - c // 2 for c in counts])
        _, lw = _mbar_weights(reduced_bias[:, indices], half_counts, mbar.f_k)
        lw = target_weights(lw, indices)
        try:
            hf, _ = _histogram_free_energy(lw, s[indices], edges, kbt)
            halves.append(hf)
        except ValueError:
            issues.append("half_trajectory_missing_support")
    half_difference = None
    if len(halves) == 2:
        diff = halves[1] - halves[0]
        diff -= np.mean(diff)
        half_difference = float(np.ptp(diff))
        # Compare shape drift with full-run block-bootstrap noise; halves have
        # difference SE about twice full-run SE. Remove the arbitrary gauge.
        centered_boot = np.array(boot_f) - np.mean(boot_f, axis=1)[:, None]
        shape_se = centered_boot.std(axis=0, ddof=1)
        if np.max(np.abs(diff) / np.maximum(2 * shape_se, 1e-12)) > 3:
            issues.append("time_split_nonstationarity")
    return {"nodes": nodes, "operator": operator, "values": operator @ f,
            "noise_covariance": covariance, "histogram_pmf": f,
            "bootstrap_pmf": np.array(boot_f), "bin_edges": edges,
            "block_ess": block_ess, "unweighted_block_ess": unweighted_block_ess,
            "relative_block_ess": relative_block_ess, "overlap_matrix": matrix,
            "overlap_scalar": overlap_scalar, "overlap_components": int(n_components),
            "overlap_component_labels": labels.tolist(), "quality_issues": issues,
            "projection_summary": projection_summary, "all_positions": [p["s"] for p in projected],
            "center_s": center_projection["s"], "counts": counts,
            "half_profiles": halves, "half_profile_difference_range": half_difference,
            "block_size": block_size, "stride": stride, "bootstraps": bootstraps,
            "seed": seed, "covariance_shrinkage": .02, "failed_bootstraps": failed_bootstraps}
