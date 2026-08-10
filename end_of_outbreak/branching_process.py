"""Branching-process extinction probabilities used by sustained-transmission risk.

The reset future has a constant reproduction number, so the generation timing changes only
*when* descendants appear, not whether their Galton--Watson family eventually becomes extinct.
For the SSE/SSI mechanisms one case has negative-binomial offspring with mean ``R`` and
dispersion ``k``; Cori is the Poisson limit.

The production risk path evaluates these roots for millions of posterior draws.  The solver is
therefore vectorised rather than calling a scalar root finder per draw.  It bisects the
non-trivial survival probability ``p = 1 - q`` after dividing the log fixed-point equation by
``p``; that division removes the trivial root at ``p = 0`` and remains stable near criticality.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

_BISECTION_ITERATIONS = 80
"""Enough bracket halvings to put the root below double-precision resolution."""


def negative_binomial_extinction_probability(R: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
    """Smallest extinction root for ``NB(mean=R, dispersion=k)`` offspring.

    Inputs broadcast.  Subcritical and critical entries return one exactly; supercritical
    entries solve

    ``q = (1 + R (1 - q) / k) ** (-k)``.
    """
    R_values, k_values = np.broadcast_arrays(
        np.asarray(R, dtype=np.float64), np.asarray(k, dtype=np.float64)
    )
    _validate_reproduction_number(R_values)
    if not np.isfinite(k_values).all() or (k_values <= 0.0).any():
        raise ValueError("k must contain finite positive values")

    q = np.ones(R_values.shape, dtype=np.float64)
    supercritical = R_values > 1.0
    if not supercritical.any():
        return q

    R_active = R_values[supercritical]
    k_active = k_values[supercritical]
    survival = _bisect_survival_probability(
        lambda p: np.log1p(-p) / p + k_active * np.log1p(R_active * p / k_active) / p,
        size=R_active.size,
    )
    # Re-evaluate q from the pgf rather than subtracting two nearly equal numbers.  This keeps
    # both near-critical roots and very small extinction probabilities representable.
    q[supercritical] = np.exp(-k_active * np.log1p(R_active * survival / k_active))
    return q


def poisson_extinction_probability(R: ArrayLike) -> NDArray[np.float64]:
    """Smallest extinction root for ``Poisson(R)`` offspring, broadcasting over ``R``."""
    R_values = np.asarray(R, dtype=np.float64)
    _validate_reproduction_number(R_values)
    q = np.ones(R_values.shape, dtype=np.float64)
    supercritical = R_values > 1.0
    if not supercritical.any():
        return q

    R_active = R_values[supercritical]
    survival = _bisect_survival_probability(
        lambda p: np.log1p(-p) / p + R_active,
        size=R_active.size,
    )
    q[supercritical] = np.exp(-R_active * survival)
    return q


def simulate_galton_watson_survival(
    seed_counts: NDArray[np.int64],
    *,
    R: float,
    k: float | None,
    survival_threshold: int,
    rng: np.random.Generator,
    max_generations: int = 10_000,
) -> NDArray[np.bool_]:
    """Resolve seeded Galton--Watson replicates as extinct or effectively surviving.

    This is the independent simulation route used by the on-demand RST validation, not by the
    report pipeline. Once a generation reaches ``survival_threshold`` it is labelled surviving;
    the probability that all of those lineages later die is at most ``q**survival_threshold``.
    Negative-binomial offspring pool over the active generation by closure.
    """
    active = np.asarray(seed_counts, dtype=np.int64).copy()
    if active.ndim != 1 or (active < 0).any():
        raise ValueError("seed_counts must be a one-dimensional non-negative integer array")
    if not np.isfinite(R) or R < 0.0:
        raise ValueError("R must be finite and non-negative")
    if k is not None and (not np.isfinite(k) or k <= 0.0):
        raise ValueError("k must be finite and positive")
    if survival_threshold < 1:
        raise ValueError("survival_threshold must be positive")

    survived = active >= survival_threshold
    unresolved = (active > 0) & ~survived
    for _ in range(max_generations):
        if not unresolved.any():
            return survived
        parents = active[unresolved]
        if k is None:
            offspring = rng.poisson(R * parents)
        else:
            offspring = rng.negative_binomial(k * parents, k / (k + R))
        active[unresolved] = offspring
        newly_survived = active >= survival_threshold
        survived |= newly_survived
        unresolved = (active > 0) & ~survived
    raise RuntimeError(
        f"{int(unresolved.sum())} branching replicates remained unresolved after "
        f"{max_generations} generations"
    )


def _bisect_survival_probability(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]], *, size: int
) -> NDArray[np.float64]:
    """Positive root of a vector function whose sign is positive at zero and negative at one."""
    lower = np.zeros(size, dtype=np.float64)
    upper = np.ones(size, dtype=np.float64)
    for _ in range(_BISECTION_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        value = function(midpoint)
        lower = np.where(value > 0.0, midpoint, lower)
        upper = np.where(value > 0.0, upper, midpoint)
    return (lower + upper) / 2.0


def _validate_reproduction_number(R: NDArray[np.float64]) -> None:
    if not np.isfinite(R).all() or (R < 0.0).any():
        raise ValueError("R must contain finite non-negative values")
