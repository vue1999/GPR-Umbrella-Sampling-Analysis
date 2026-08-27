"""Tests for sampled-support geometry."""
from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella import sampled_support_mask
from gpr_umbrella.support import path_valid_mask, window_anchored_display_policy


def test_support_is_union_of_ellipses_not_a_convex_hull():
    means = np.array([[-2.0, 0.0], [2.0, 0.0]])
    points = np.array([
        [-2.0, 0.0],
        [-1.5, 0.0],
        [-2.0, 1.0],
        [0.0, 0.0],
    ])

    mask = sampled_support_mask(
        points, means, lengthscale=(0.5, 1.0), radius=1.0
    )

    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_support_allows_rank_deficient_window_layouts():
    means = np.array([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    points = np.array([[0.0, 0.0], [0.4, 0.0], [2.0, 0.0]])

    mask = sampled_support_mask(points, means, lengthscale=0.5, radius=1.0)

    np.testing.assert_array_equal(mask, [True, True, False])



def test_default_radius_is_half_a_lengthscale():
    means = np.array([[0.0, 0.0]])
    points = np.array([[0.9, 0.0], [1.1, 0.0], [0.0, 1.9], [0.0, 2.1]])

    mask = sampled_support_mask(points, means, lengthscale=(2.0, 4.0))

    np.testing.assert_array_equal(mask, [True, False, True, False])

@pytest.mark.parametrize("radius", [None, 0.0, -1.0, np.nan, True])
def test_support_radius_must_be_a_finite_positive_number(radius):
    with pytest.raises(ValueError, match="radius"):
        sampled_support_mask(
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            lengthscale=1.0,
            radius=radius,
        )


def test_window_anchored_policy_defines_shared_path_valid_region():
    results = {
        "gx": np.array([0.0, 1.0, 2.0]),
        "gy": np.array([0.0, 1.0]),
        "means": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "pmf": np.array([[0.0, 0.2], [0.8, 1.0], [50.0, 60.0]]),
        "pmf_std_raw": np.full((3, 2), 0.1),
        "pmf_std_calibrated": np.full((3, 2), 0.2),
        "support_mask": np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    }

    policy = window_anchored_display_policy(results)
    valid = path_valid_mask(results, policy)

    assert policy["pmf_limits"] == pytest.approx((-0.25, 1.25))
    assert valid[0, 0]
    assert valid[1, 1]
    assert not valid[2, 0]  # supported but red/out of the trusted PMF range
    assert not valid[2, 1]  # geometrically unsupported and non-finite


def test_window_anchored_policy_is_invariant_to_additive_pmf_shift():
    base = {
        "gx": np.array([0.0, 1.0]),
        "gy": np.array([0.0, 1.0]),
        "means": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "pmf": np.array([[0.0, 0.4], [0.6, 1.0]]),
        "pmf_std_raw": np.full((2, 2), 0.1),
        "pmf_std_calibrated": np.full((2, 2), 0.2),
        "support_mask": np.ones((2, 2), dtype=bool),
    }
    shifted = dict(base, pmf=base["pmf"] + 123.0)

    first = window_anchored_display_policy(base)
    second = window_anchored_display_policy(shifted)
    np.testing.assert_allclose(first["pmf"], second["pmf"])
    np.testing.assert_allclose(first["pmf_limits"], second["pmf_limits"])
