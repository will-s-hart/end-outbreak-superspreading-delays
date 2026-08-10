"""Extinction roots for the branching models behind RST."""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import branching_process


def test_subcritical_and_critical_processes_become_extinct_certainly():
    R = np.array([0.0, 0.7, 1.0])
    np.testing.assert_array_equal(
        branching_process.negative_binomial_extinction_probability(R, 0.18), np.ones(3)
    )
    np.testing.assert_array_equal(branching_process.poisson_extinction_probability(R), np.ones(3))


def test_supercritical_negative_binomial_roots_satisfy_the_pgf_fixed_point():
    R = np.array([1.000001, 1.2, 2.0, 4.0])
    k = np.array([0.05, 0.18, 0.7, 2.0])
    q = branching_process.negative_binomial_extinction_probability(R, k)
    np.testing.assert_allclose(q, (1.0 + R * (1.0 - q) / k) ** (-k), rtol=2e-13)
    assert np.all((q > 0.0) & (q < 1.0))


def test_supercritical_poisson_roots_satisfy_the_pgf_fixed_point():
    R = np.array([1.000001, 1.2, 2.0, 4.0])
    q = branching_process.poisson_extinction_probability(R)
    np.testing.assert_allclose(q, np.exp(R * (q - 1.0)), rtol=2e-13)
    assert np.all((q > 0.0) & (q < 1.0))


def test_extinction_decreases_with_R_and_increases_with_overdispersion():
    q_by_R = branching_process.negative_binomial_extinction_probability(
        np.array([1.1, 1.5, 2.0, 3.0]), 0.18
    )
    assert np.all(np.diff(q_by_R) < 0.0)

    # Larger k means less heterogeneity and hence a smaller extinction probability at fixed mean.
    q_by_k = branching_process.negative_binomial_extinction_probability(
        2.0, np.array([0.05, 0.18, 0.7, 4.0])
    )
    assert np.all(np.diff(q_by_k) < 0.0)


def test_negative_binomial_converges_to_the_poisson_root():
    R = np.array([1.1, 1.5, 2.0, 3.0])
    negative_binomial = branching_process.negative_binomial_extinction_probability(R, 1e9)
    poisson = branching_process.poisson_extinction_probability(R)
    np.testing.assert_allclose(negative_binomial, poisson, rtol=2e-8, atol=2e-10)


def test_inputs_broadcast_and_every_root_stays_in_bounds():
    R = np.array([[0.8], [1.5], [3.0]])
    k = np.array([[0.1, 0.5, 2.0, 10.0]])
    q = branching_process.negative_binomial_extinction_probability(R, k)
    assert q.shape == (3, 4)
    assert np.all((q >= 0.0) & (q <= 1.0))
    np.testing.assert_array_equal(q[0], np.ones(4))


@pytest.mark.parametrize(
    ("R", "k", "match"),
    [([-0.1], [0.2], "R"), ([1.2], [0.0], "k"), ([1.2], [np.nan], "k")],
)
def test_invalid_inputs_are_refused(R, k, match):
    with pytest.raises(ValueError, match=match):
        branching_process.negative_binomial_extinction_probability(R, k)
