"""Tests for sampled-support geometry."""
from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella import sampled_support_mask


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

    mask = sampled_support_mask(points, means, lengthscale=0.5)

    np.testing.assert_array_equal(mask, [True, True, False])


@pytest.mark.parametrize("radius", [None, 0.0, -1.0, np.nan, True])
def test_support_radius_must_be_a_finite_positive_number(radius):
    with pytest.raises(ValueError, match="radius"):
        sampled_support_mask(
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            lengthscale=1.0,
            radius=radius,
        )
