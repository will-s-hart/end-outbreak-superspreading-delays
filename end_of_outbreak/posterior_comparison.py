"""Summarising one positive scalar's posterior, and measuring how far two of them differ.

Analysis 2 estimates ``k`` under a prior that is deliberately identical across the three naive
models (§6.2), which turns the three ``k`` posteriors into a direct measurement of aim 2: the
same symbol means an individual-level offspring dispersion in SSE/SSI and a day-level incidence
dispersion in DLO (§5.4), so the data should pull it to different places. "Different places"
has to be a number rather than an impression of three curves on a panel, and this module is
where those numbers come from.

Three complementary quantities, because no single one of them says enough:

``median_ratio``
    Where the two posteriors sit relative to each other, on the scale ``k`` is read on. Says
    nothing about width, so on its own it cannot distinguish two sharp posteriors an order of
    magnitude apart from two vague ones.
``overlap``
    ``∫ min(p_a, p_b)`` — the probability mass the two posteriors share, 1 when they coincide
    and 0 when they are disjoint. Scale-free, symmetric, and unlike a Kullback–Leibler
    divergence it stays finite and interpretable when the two barely meet.
``probability_greater``
    ``P(k_a > k_b)`` for independent draws from the two posteriors. Reads directly as "how
    confident is the pair of fits that this model's ``k`` is the larger one".

Densities are estimated on the **log** scale, as :func:`scripts.utils.positive_density` draws
them: ``k`` is positive and right-skewed, and a Gaussian kernel on the raw draws would put mass
below zero. The overlap is unaffected by the choice — ``∫ min(p_a, p_b) dx = ∫ min(q_a, q_b) du``
for ``u = log x``, because both densities pick up the same Jacobian — so it measures the
posteriors themselves, not the coordinates they were estimated in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.stats
from numpy.typing import NDArray

DEFAULT_INTERVAL = 0.95
"""Equal-tailed credible level reported alongside each posterior median."""

DENSITY_GRID_SIZE = 2048
"""Points in the log-scale grid the overlap integral is evaluated on."""


@dataclass(frozen=True)
class PosteriorSummary:
    """Location and spread of one positive scalar's posterior."""

    median: float
    mean: float
    interval_lower: float
    interval_upper: float
    interval_probability: float
    n_draws: int

    def as_dict(self) -> dict[str, Any]:
        """A JSON-writable view, for the results files the report quotes."""
        return asdict(self)


@dataclass(frozen=True)
class PosteriorDivergence:
    """How far apart two posteriors for the same quantity are.

    "Same quantity" is doing real work here: the whole point of Fig. 2 is that ``k`` is the
    *same symbol* in DLO and in SSE/SSI while being a different quantity in each mechanism, so
    a divergence between the two posteriors is evidence about that, not a modelling error.
    """

    median_ratio: float
    """Ratio of the two posterior medians, first over second."""

    overlap: float
    """``∫ min(p_a, p_b)``: 1 if the posteriors coincide, 0 if they are disjoint."""

    probability_greater: float
    """``P(k_a > k_b)`` for independent draws from the two posteriors."""

    def as_dict(self) -> dict[str, Any]:
        """A JSON-writable view, for the results files the report quotes."""
        return asdict(self)


def summarise_posterior(
    draws: NDArray[np.float64], *, interval_probability: float = DEFAULT_INTERVAL
) -> PosteriorSummary:
    """Median, mean and an equal-tailed credible interval for a set of draws."""
    values = _positive_draws(draws)
    if not 0.0 < interval_probability < 1.0:
        raise ValueError(
            f"interval_probability must lie strictly in (0, 1), got {interval_probability}"
        )
    tail = 0.5 * (1.0 - interval_probability)
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return PosteriorSummary(
        median=float(np.median(values)),
        mean=float(values.mean()),
        interval_lower=float(lower),
        interval_upper=float(upper),
        interval_probability=float(interval_probability),
        n_draws=int(values.size),
    )


def compare_posteriors(
    draws_a: NDArray[np.float64],
    draws_b: NDArray[np.float64],
    *,
    grid_size: int = DENSITY_GRID_SIZE,
) -> PosteriorDivergence:
    """The three divergence measures for two independent posteriors of a positive scalar."""
    first = _positive_draws(draws_a)
    second = _positive_draws(draws_b)
    return PosteriorDivergence(
        median_ratio=float(np.median(first) / np.median(second)),
        overlap=posterior_overlap(first, second, grid_size=grid_size),
        probability_greater=probability_greater(first, second),
    )


def posterior_overlap(
    draws_a: NDArray[np.float64],
    draws_b: NDArray[np.float64],
    *,
    grid_size: int = DENSITY_GRID_SIZE,
) -> float:
    """``∫ min(p_a, p_b)``, the mass two posteriors share, in ``[0, 1]``.

    Evaluated on a shared log-scale grid wide enough to hold both kernel density estimates out
    to where they have decayed: the integrand is a minimum, so it is the *narrower* density
    that determines the answer and a grid clipped to one posterior's support would report an
    overlap that depends on which posterior was passed first.
    """
    first = np.log(_positive_draws(draws_a))
    second = np.log(_positive_draws(draws_b))
    kernels = [scipy.stats.gaussian_kde(sample) for sample in (first, second)]
    # Three bandwidths beyond the outermost draw, so each density is grid-supported out to
    # where its own kernel has effectively decayed. In one dimension the kernel covariance is
    # the squared bandwidth, in the units the samples are given in — here log k.
    padding = 3.0 * max(
        float(np.sqrt(np.asarray(kernel.covariance).reshape(-1)[0])) for kernel in kernels
    )
    grid = np.linspace(
        min(first.min(), second.min()) - padding,
        max(first.max(), second.max()) + padding,
        grid_size,
    )
    densities = [np.asarray(kernel(grid)) for kernel in kernels]
    return float(np.clip(np.trapezoid(np.minimum(*densities), grid), 0.0, 1.0))


def probability_greater(draws_a: NDArray[np.float64], draws_b: NDArray[np.float64]) -> float:
    """``P(X_a > X_b)`` for independent draws, evaluated exactly over every pair.

    Independence is the right assumption and not an approximation: the two posteriors come from
    separate fits of separate models to the same data, so nothing couples a draw of one to a
    draw of the other. Ties split evenly, which matters only for discrete-valued inputs.
    """
    first = _positive_draws(draws_a)
    second = np.sort(_positive_draws(draws_b))
    strictly_below = np.searchsorted(second, first, side="left")
    at_or_below = np.searchsorted(second, first, side="right")
    return float(np.mean(0.5 * (strictly_below + at_or_below)) / second.size)


def _positive_draws(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flattened draws, checked for the positivity the log-scale estimates require."""
    values = np.asarray(draws, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"need at least two draws to summarise, got {values.size}")
    if not np.all(values > 0.0):
        raise ValueError("draws must be strictly positive; these summaries are on the log scale")
    return values
