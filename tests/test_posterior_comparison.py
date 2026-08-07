"""Tests for the posterior summaries Fig. 2 quotes.

Property-based wherever the property exists: the overlap of two log-normals has a closed form
on the log scale, ``P(X_a > X_b)`` and ``P(X_b > X_a)`` must sum to one, and both measures must
be symmetric under swapping the arguments in the way each is symmetric.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from end_of_outbreak import posterior_comparison as pc


def lognormal_draws(*, median: float, sigma_log: float, size: int, seed: int) -> np.ndarray:
    """Draws from a log-normal, as a ``k`` posterior on this project's parameterisation."""
    rng = np.random.default_rng(seed)
    return median * np.exp(sigma_log * rng.standard_normal(size))


# ---------------------------------------------------------------------------------------
# summarise_posterior
# ---------------------------------------------------------------------------------------


def test_summary_recovers_the_generating_lognormal() -> None:
    """Median, mean and interval match the distribution the draws came from."""
    median, sigma_log = 0.18, 0.35
    summary = pc.summarise_posterior(
        lognormal_draws(median=median, sigma_log=sigma_log, size=200_000, seed=0)
    )
    assert summary.n_draws == 200_000
    assert summary.median == pytest.approx(median, rel=0.01)
    assert summary.mean == pytest.approx(median * np.exp(0.5 * sigma_log**2), rel=0.01)
    for bound, probability in ((summary.interval_lower, 0.025), (summary.interval_upper, 0.975)):
        expected = median * np.exp(sigma_log * scipy.stats.norm.ppf(probability))
        assert bound == pytest.approx(expected, rel=0.02)


def test_summary_interval_level_is_configurable_and_nested() -> None:
    """A tighter credible level gives a strictly narrower interval."""
    draws = lognormal_draws(median=1.0, sigma_log=0.5, size=20_000, seed=1)
    wide = pc.summarise_posterior(draws, interval_probability=0.95)
    narrow = pc.summarise_posterior(draws, interval_probability=0.5)
    assert narrow.interval_probability == 0.5
    assert wide.interval_lower < narrow.interval_lower < narrow.interval_upper < wide.interval_upper


def test_summary_rejects_non_positive_draws() -> None:
    """The summaries are log-scale, so a non-positive draw is an error rather than a NaN."""
    with pytest.raises(ValueError, match="strictly positive"):
        pc.summarise_posterior(np.array([0.1, 0.0, 0.3]))


# ---------------------------------------------------------------------------------------
# posterior_overlap
# ---------------------------------------------------------------------------------------


def test_overlap_of_a_posterior_with_itself_is_one() -> None:
    draws = lognormal_draws(median=0.18, sigma_log=0.4, size=5_000, seed=2)
    assert pc.posterior_overlap(draws, draws) == pytest.approx(1.0, abs=0.01)


def test_overlap_matches_the_closed_form_for_two_lognormals() -> None:
    """Equal log-scale spreads: the overlap is ``2 Φ(−|Δμ| / 2σ)``.

    The kernel density estimate widens each density slightly, so the measured overlap sits a
    little above the analytic value; the tolerance allows for that and no more.
    """
    sigma_log = 0.4
    for separation in (0.0, 0.5, 1.5):
        first = lognormal_draws(median=1.0, sigma_log=sigma_log, size=40_000, seed=3)
        second = lognormal_draws(
            median=float(np.exp(separation)), sigma_log=sigma_log, size=40_000, seed=4
        )
        expected = 2.0 * scipy.stats.norm.cdf(-0.5 * separation / sigma_log)
        assert pc.posterior_overlap(first, second) == pytest.approx(expected, abs=0.03)


def test_overlap_is_symmetric_and_grid_independent() -> None:
    """Neither the argument order nor the grid resolution may change the answer."""
    first = lognormal_draws(median=0.1, sigma_log=0.2, size=8_000, seed=5)
    second = lognormal_draws(median=0.5, sigma_log=0.9, size=8_000, seed=6)
    forward = pc.posterior_overlap(first, second)
    assert pc.posterior_overlap(second, first) == pytest.approx(forward, abs=1e-6)
    assert pc.posterior_overlap(first, second, grid_size=8192) == pytest.approx(forward, abs=1e-3)


def test_overlap_vanishes_for_well_separated_posteriors() -> None:
    first = lognormal_draws(median=0.01, sigma_log=0.1, size=5_000, seed=7)
    second = lognormal_draws(median=10.0, sigma_log=0.1, size=5_000, seed=8)
    assert pc.posterior_overlap(first, second) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------
# probability_greater
# ---------------------------------------------------------------------------------------


def test_probability_greater_is_a_half_against_an_identical_posterior() -> None:
    """Ties split evenly, so a posterior compared with itself gives exactly one half."""
    draws = lognormal_draws(median=0.18, sigma_log=0.4, size=2_000, seed=9)
    assert pc.probability_greater(draws, draws) == pytest.approx(0.5, abs=1e-12)


def test_probability_greater_complements_the_reversed_comparison() -> None:
    first = lognormal_draws(median=0.2, sigma_log=0.5, size=4_000, seed=10)
    second = lognormal_draws(median=0.3, sigma_log=0.5, size=4_000, seed=11)
    forward = pc.probability_greater(first, second)
    assert forward + pc.probability_greater(second, first) == pytest.approx(1.0, abs=1e-12)


def test_probability_greater_matches_the_analytic_lognormal_value() -> None:
    """``P(X_a > X_b) = Φ((μ_a − μ_b) / √(σ_a² + σ_b²))`` for independent log-normals."""
    median_a, sigma_a = 0.6, 0.5
    median_b, sigma_b = 0.3, 0.8
    measured = pc.probability_greater(
        lognormal_draws(median=median_a, sigma_log=sigma_a, size=60_000, seed=12),
        lognormal_draws(median=median_b, sigma_log=sigma_b, size=60_000, seed=13),
    )
    expected = scipy.stats.norm.cdf(
        (np.log(median_a) - np.log(median_b)) / np.hypot(sigma_a, sigma_b)
    )
    assert measured == pytest.approx(float(expected), abs=0.01)


# ---------------------------------------------------------------------------------------
# compare_posteriors
# ---------------------------------------------------------------------------------------


def test_comparison_of_a_posterior_with_itself_is_the_null_divergence() -> None:
    draws = lognormal_draws(median=0.18, sigma_log=0.4, size=5_000, seed=14)
    divergence = pc.compare_posteriors(draws, draws)
    assert divergence.median_ratio == pytest.approx(1.0)
    assert divergence.overlap == pytest.approx(1.0, abs=0.01)
    assert divergence.probability_greater == pytest.approx(0.5, abs=1e-12)


def test_comparison_reverses_consistently() -> None:
    """Swapping the arguments inverts the ratio, complements the probability, keeps the overlap."""
    first = lognormal_draws(median=0.6, sigma_log=0.3, size=6_000, seed=15)
    second = lognormal_draws(median=0.15, sigma_log=0.6, size=6_000, seed=16)
    forward = pc.compare_posteriors(first, second)
    reverse = pc.compare_posteriors(second, first)
    assert forward.median_ratio * reverse.median_ratio == pytest.approx(1.0)
    assert forward.probability_greater + reverse.probability_greater == pytest.approx(1.0)
    assert forward.overlap == pytest.approx(reverse.overlap, abs=1e-6)


def test_comparison_is_json_writable() -> None:
    """The results files the report quotes are JSON, so the dataclasses must flatten cleanly."""
    draws = lognormal_draws(median=0.18, sigma_log=0.4, size=1_000, seed=17)
    assert set(pc.compare_posteriors(draws, draws).as_dict()) == {
        "median_ratio",
        "overlap",
        "probability_greater",
    }
    assert all(
        isinstance(value, float | int) for value in pc.summarise_posterior(draws).as_dict().values()
    )
