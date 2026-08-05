"""Discrete delay distributions: serial interval, incubation period and TOST.

Three delay distributions drive the models:

``w_s`` (``s >= 1``)
    The onset-to-onset **serial interval**, used directly by the infection-anchored models
    DLO, SSE and SSI (which treat onset dates as infection dates).
``f_inc,a`` (``a >= 1``)
    The **incubation period**, mapping an infection forward to the resulting symptom onset.
``f_tost,s`` (``s >= 0``)
    **Time from an infector's own symptom onset to transmission.**

The onset-anchored models SSE-SO and SSI-SO use the latter two, and are tied to the same
onset-to-onset serial interval as the naive models by the variance budget

    mean(f_tost) + mean(f_inc) = mean(w),
    var(f_tost)  + var(f_inc)  = var(w),

so that the five-model comparison isolates onset-anchoring rather than confounding it with a
different assumed serial interval. ``check_delay_budget`` enforces this and rejects any
incubation estimate whose variance exceeds the serial-interval variance.

Lag-0 rules
-----------
``f_inc`` must have **no mass at lag 0**: an infection cannot produce an onset the same day.
This is what makes the onset-anchored recursion well ordered, since ``D_t`` then depends only
on ``E_{<t}`` and hence only on ``D_{<t}``. ``f_tost`` **may** place mass at lag 0 — a case can
transmit on its own onset day. At most one of the two may be supported at lag 0.

Indexing convention
-------------------
Weight arrays are stored densely from their **first supported lag**, which differs between
distributions:

===================  ================  =================================
Array                First lag         Interpretation
===================  ================  =================================
``w``                1                 ``w[i] == w_{i+1}``
``f_inc``            1                 ``f_inc[i] == f_inc,{i+1}``
``f_tost``           0                 ``f_tost[i] == f_tost,i``
===================  ================  =================================

``SERIAL_INTERVAL_FIRST_LAG``, ``INCUBATION_FIRST_LAG`` and ``TOST_FIRST_LAG`` name these so
that callers need not hard-code them.

Discretisation follows Cori et al. (2013), web appendix 11 (https://doi.org/10.1093/aje/kwt133),
which is also equation (2) of Thompson et al. (2024): the mass at integer lag ``s`` is
``∫ (1 - |s - u|) g(u) du`` over ``u in [s - 1, s + 1]``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.stats
from numpy.typing import NDArray

SERIAL_INTERVAL_FIRST_LAG = 1
"""``w`` is supported on ``s >= 1``: no same-day onset-to-onset transmission."""

INCUBATION_FIRST_LAG = 1
"""``f_inc`` is supported on ``a >= 1``; see the lag-0 rules above."""

TOST_FIRST_LAG = 0
"""``f_tost`` may place mass at lag 0: a case can transmit on its own onset day."""

DEFAULT_MAX_LAG = 110
"""Largest lag with its own probability mass. Residual tail mass is folded into it.

Set to the length of the analysis window (days 0–110), so that no transmission is truncated
*within* the window: the renewal sum on the final day, ``Σ_{s=1}^{110} w_s I_{110-s}``, reaches
all the way back to the index case. The generic EVD serial interval carries about 0.1% of its
mass beyond lag 60, which is small but not negligible against a risk that falls below 0.01.
"""


# ---------------------------------------------------------------------------------------
# Continuous gamma delays
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GammaDelay:
    """A continuous gamma delay parameterised by its mean and standard deviation."""

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if not self.mean > 0:
            raise ValueError(f"mean must be positive, got {self.mean}")
        if not self.sd > 0:
            raise ValueError(f"sd must be positive, got {self.sd}")

    @property
    def variance(self) -> float:
        """``sd ** 2``."""
        return self.sd**2

    @property
    def shape(self) -> float:
        """Gamma shape parameter ``mean ** 2 / variance``."""
        return self.mean**2 / self.variance

    @property
    def scale(self) -> float:
        """Gamma scale parameter ``variance / mean``."""
        return self.variance / self.mean

    def frozen(self) -> scipy.stats.rv_continuous:
        """The corresponding frozen ``scipy.stats.gamma``."""
        return scipy.stats.gamma(a=self.shape, scale=self.scale)


# --- published estimates -----------------------------------------------------------------

GENERIC_EVD_SERIAL_INTERVAL = GammaDelay(mean=15.3, sd=9.3)
"""Generic EVD serial interval; the project default for all five models.

Van Kerkhove et al. (2015), Sci Data 2:150019 (https://doi.org/10.1038/sdata.2015.19), as
used in the main analyses of Thompson et al. (2024) and shown discretised in their Fig. S1.

Chosen over the outbreak-specific estimate because its variance budget of 86.49 d² leaves
room to decompose the onset-to-onset interval into a published incubation period and a
plausible TOST distribution; the outbreak-specific SD of 6.08 d does not.
"""

OUTBREAK_SPECIFIC_SERIAL_INTERVAL = GammaDelay(mean=19.46, sd=6.08)
"""Serial interval fitted to 41 contact-traced pairs from the 2018 Équateur outbreak.

Thompson et al. (2024) supplementary Fig. S3C. Retained for the sensitivity analysis: it is
the estimate the authors argued should be preferred when outbreak-specific contact-tracing
data exist, and its lighter tail materially lowers the risk after a long case-free period.
It is **not** admissible for the onset-anchored models alongside the incubation estimate
below (its variance is smaller), which is why it is not the project default.
"""

WHO_ERT_INCUBATION_PERIOD = GammaDelay(mean=11.4, sd=8.1)
"""EVD incubation period; the project default.

WHO Ebola Response Team (2014), N Engl J Med 371:1481–1495
(https://doi.org/10.1056/NEJMoa1411100), gamma distributed with mean 11.4 d and SD 8.1 d.

Against the generic serial interval this leaves a residual TOST with mean 3.9 d and variance
20.88 d² (SD 4.57 d) — a plausible onset-to-transmission profile for EVD, where
pre-symptomatic transmission is negligible.
"""

DEFAULT_SERIAL_INTERVAL = GENERIC_EVD_SERIAL_INTERVAL
DEFAULT_INCUBATION_PERIOD = WHO_ERT_INCUBATION_PERIOD


# ---------------------------------------------------------------------------------------
# Cori discretisation
# ---------------------------------------------------------------------------------------


def discretise(
    distribution: scipy.stats.rv_continuous,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    first_lag: int = 1,
) -> NDArray[np.float64]:
    """Discretise a continuous delay by the Cori et al. method.

    Parameters
    ----------
    distribution
        A frozen continuous ``scipy.stats`` distribution, so that ``.pdf(u)`` works.
    max_lag
        Largest lag with its own probability mass; residual tail mass is folded into it, so
        the returned weights always sum to exactly 1.
    first_lag
        0 to keep the lag-0 bin, 1 to fold it into lag 1 and start the array at lag 1.

    Returns
    -------
    Weights ``p`` with ``p[i]`` the mass at lag ``first_lag + i``, summing to 1.
    """
    if first_lag not in (0, 1):
        raise ValueError(f"first_lag must be 0 or 1, got {first_lag}")
    if max_lag < first_lag:
        raise ValueError(f"max_lag must be at least first_lag ({first_lag}), got {max_lag}")

    def integrand(lag: float, u: float) -> float:
        return (1.0 - abs(lag - u)) * distribution.pdf(u)

    weights = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        # A gamma with shape < 1 has an integrable singularity at 0; start just inside the
        # support rather than at it.
        lower = float(lag) - 1.0 if lag > 0 else 1e-12
        weights[lag] = scipy.integrate.quad(
            functools.partial(integrand, float(lag)), lower, float(lag) + 1.0, limit=200
        )[0]

    if first_lag == 1:
        weights[1] += weights[0]
        weights = weights[1:]

    # Fold the truncated tail into the final bin so the weights are a proper pmf.
    weights[-1] += 1.0 - weights.sum()
    return weights


def discretise_gamma(
    delay: GammaDelay,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    first_lag: int = 1,
) -> NDArray[np.float64]:
    """Discretise a :class:`GammaDelay` by the Cori et al. method."""
    return discretise(delay.frozen(), max_lag=max_lag, first_lag=first_lag)


# --- named constructors, each enforcing its own lag-0 rule -------------------------------


def serial_interval_weights(
    delay: GammaDelay = DEFAULT_SERIAL_INTERVAL,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
) -> NDArray[np.float64]:
    """Discrete onset-to-onset serial interval ``w``, supported on lags ``>= 1``."""
    return discretise_gamma(delay, max_lag=max_lag, first_lag=SERIAL_INTERVAL_FIRST_LAG)


def incubation_weights(
    delay: GammaDelay = DEFAULT_INCUBATION_PERIOD,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
) -> NDArray[np.float64]:
    """Discrete incubation period ``f_inc``, supported on lags ``>= 1`` (no lag-0 mass)."""
    return discretise_gamma(delay, max_lag=max_lag, first_lag=INCUBATION_FIRST_LAG)


def tost_weights(
    delay: GammaDelay,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
) -> NDArray[np.float64]:
    """Discrete onset-to-transmission delay ``f_tost``, supported on lags ``>= 0``."""
    return discretise_gamma(delay, max_lag=max_lag, first_lag=TOST_FIRST_LAG)


def cumulative(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative serial interval ``F_r`` for ``r = 0, 1, ...``, with ``F_0 = 0``.

    Returns an array of length ``len(w) + 1``; ``F[r] = sum(w[:r])`` is the probability that
    a case's transmission to a given secondary case has already happened by ``r`` days after
    it appeared. Used to build the pooled remaining-transmission weight ``Λ(t)``.
    """
    return np.concatenate(([0.0], np.cumsum(w)))


# ---------------------------------------------------------------------------------------
# The onset-anchored variance budget
# ---------------------------------------------------------------------------------------


def check_delay_budget(
    *,
    serial_interval: GammaDelay = DEFAULT_SERIAL_INTERVAL,
    incubation: GammaDelay = DEFAULT_INCUBATION_PERIOD,
) -> GammaDelay:
    """Residual TOST delay implied by mean and variance additivity. Raises if inadmissible.

    For an infector with onset on day ``u``, transmission happens on day ``u + S`` with
    ``S ~ f_tost`` and the infectee's onset on day ``u + S + A`` with ``A ~ f_inc``. The
    onset-to-onset serial interval is therefore ``S + A``, so matching a target serial
    interval requires the mean and variance of ``f_tost`` to be the residuals.

    Raises
    ------
    ValueError
        If the residual mean or variance is non-positive — i.e. the incubation estimate is
        inadmissible against this serial interval, and a different one must be used.
    """
    residual_mean = serial_interval.mean - incubation.mean
    residual_variance = serial_interval.variance - incubation.variance

    if residual_mean <= 0:
        raise ValueError(
            f"incubation mean {incubation.mean} leaves no room in the serial-interval mean "
            f"{serial_interval.mean}: residual TOST mean would be {residual_mean:.4g}. "
            "Choose an incubation estimate with a smaller mean."
        )
    if residual_variance <= 0:
        raise ValueError(
            f"incubation variance {incubation.variance:.4g} exceeds the serial-interval "
            f"variance {serial_interval.variance:.4g}: residual TOST variance would be "
            f"{residual_variance:.4g}. This incubation estimate is inadmissible against "
            "this serial interval; choose another (or a serial interval with a larger SD)."
        )

    return GammaDelay(mean=residual_mean, sd=float(np.sqrt(residual_variance)))


@dataclass(frozen=True)
class OnsetAnchoredDelays:
    """The discretised delay triple driving the onset-anchored models.

    ``serial_interval`` is carried alongside ``tost``/``incubation`` because the naive and
    onset-anchored models must be driven by the *same* onset-to-onset interval for the
    comparison to isolate onset-anchoring.
    """

    serial_interval: NDArray[np.float64]
    """``w``, from lag 1."""

    tost: NDArray[np.float64]
    """``f_tost``, from lag 0."""

    incubation: NDArray[np.float64]
    """``f_inc``, from lag 1."""

    tost_delay: GammaDelay
    """The continuous residual TOST gamma the discrete ``tost`` came from."""

    incubation_delay: GammaDelay
    """The continuous incubation gamma the discrete ``incubation`` came from."""

    def implied_serial_interval(self) -> NDArray[np.float64]:
        """``f_tost * f_inc``: the onset-to-onset interval the discrete delays imply."""
        return convolve_onset_to_onset(self.tost, self.incubation)

    def serial_interval_discrepancy(self) -> float:
        """Total-variation distance between the implied and target serial intervals.

        Non-zero only because discretising then convolving is not the same as convolving
        then discretising: each Cori triangular kernel adds ~1/6 d² of variance, so the
        implied interval is very slightly more diffuse than the target.
        """
        return total_variation_distance(self.implied_serial_interval(), self.serial_interval)


def build_onset_anchored_delays(
    *,
    serial_interval: GammaDelay = DEFAULT_SERIAL_INTERVAL,
    incubation: GammaDelay = DEFAULT_INCUBATION_PERIOD,
    max_lag: int = DEFAULT_MAX_LAG,
    tolerance: float | None = 0.02,
) -> OnsetAnchoredDelays:
    """Build ``(w, f_tost, f_inc)`` for a target serial interval and incubation estimate.

    The TOST is the gamma carrying the residual mean and variance
    (:func:`check_delay_budget`); all three are then discretised under their own lag-0 rule
    and the convolution ``f_tost * f_inc`` is checked against the target ``w``.

    Parameters
    ----------
    tolerance
        Maximum acceptable total-variation distance between ``f_tost * f_inc`` and the
        discretised target ``w``. Pass ``None`` to skip the check.
    """
    tost_delay = check_delay_budget(serial_interval=serial_interval, incubation=incubation)
    delays = OnsetAnchoredDelays(
        serial_interval=serial_interval_weights(serial_interval, max_lag=max_lag),
        tost=tost_weights(tost_delay, max_lag=max_lag),
        incubation=incubation_weights(incubation, max_lag=max_lag),
        tost_delay=tost_delay,
        incubation_delay=incubation,
    )

    if tolerance is not None:
        discrepancy = delays.serial_interval_discrepancy()
        if discrepancy > tolerance:
            raise ValueError(
                f"discretised f_tost * f_inc differs from the target serial interval by "
                f"{discrepancy:.4g} in total variation, above the tolerance {tolerance:.4g}. "
                "Increase max_lag or check the delay estimates."
            )
    return delays


# ---------------------------------------------------------------------------------------
# Utilities for combining and checking discrete delays
# ---------------------------------------------------------------------------------------


def convolve_onset_to_onset(
    f_tost: NDArray[np.float64], f_inc: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distribution of ``S + A`` for ``S ~ f_tost`` (lag >= 0) and ``A ~ f_inc`` (lag >= 1).

    The result is supported on lags ``>= 1``, i.e. in the same indexing convention as ``w``.
    """
    return np.convolve(f_tost, f_inc)


def delay_moments(weights: NDArray[np.float64], *, first_lag: int) -> tuple[float, float]:
    """``(mean, variance)`` of a discrete delay stored from ``first_lag``."""
    lags = np.arange(first_lag, first_lag + weights.size, dtype=np.float64)
    mean = float(np.sum(lags * weights))
    variance = float(np.sum((lags - mean) ** 2 * weights))
    return mean, variance


def total_variation_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Total-variation distance between two pmfs sharing a first lag, zero-padded to match."""
    length = max(p.size, q.size)
    p_padded = np.zeros(length)
    q_padded = np.zeros(length)
    p_padded[: p.size] = p
    q_padded[: q.size] = q
    return 0.5 * float(np.abs(p_padded - q_padded).sum())
