"""The reporting layer: under-reporting, onset-to-report delays, and the observed node.

Every model in this project specifies one per-day count density for the **true** onsets —
``Poisson(μ_t)`` for SSI/SSE-SO/SSI-SO/Cori/Cori-SO, ``NegativeBinomial(μ_t, α_t)`` for
DLO/SSE. Reporting sits on top of that, and is the same for all seven:

``c_t ~ thinned CountDist_t(θ, D_{<t}, latents)``  — the reported marginal

``U_t | c_t, θ ~ hidden CountDist_t``  — the unreported cases, with ``D_t = c_t + U_t``

so the whole of the reported/unreported logic lives here and no builder repeats it. What a
builder supplies is a ``moments`` callable mapping a true-count series to that model's
``(mean, dispersion)`` — the expression it already needs for its own observation node.

Why the latent is the unreported count
--------------------------------------
The self-referential part is that ``μ_t`` depends on the true counts, which are latent. Carrying
the **unreported** cases ``U = D − c`` leaves their support as the unconstrained non-negative
integers and writes the renewal density as that variable's own ``logp``; no ``pm.Potential`` is
needed. For a Poisson count law the split is the familiar identity

``Poisson(D; μ)·Binom(c; D, π) == Poisson(c; πμ)·Poisson(U; (1−π)μ)``

For ``D ~ NB(μ, α)``, thinning gives ``c ~ NB(πμ, α)`` and
``U | c ~ NB((1−π)μ(α+c)/(α+πμ), α+c)``. Thus one layer covers all seven models exactly; the
negative-binomial pieces are conditionally rather than marginally independent. Both identities
are pinned in ``tests/test_reporting.py``.

The "as of" day
---------------
``π_t`` combines a constant reporting probability with right-truncation by the onset-to-report
delay: an onset that happened yesterday has had one day in which to be reported. That makes it
a function of how far each onset day sits from the day the snapshot was taken, so the as-of day
is an **explicit argument** here rather than being read off the length of the series.

That is deliberate. ``end-of-outbreak-vbd`` — following Thompson et al. — conditions on the
record *strictly before* its calculation day, and its ``_reporting_prob_vec`` accordingly takes
the as-of day to be ``len(incidence) - 1``. This project conditions on the record *through and
including* day ``t`` (``counts[:t+1]``; see the ``γ(t) = Λ(t−1)`` shift recorded in
``docs/risk.md``). Porting that length-based rule would silently move every reporting
probability one day, so it is not ported: callers pass ``as_of_day`` themselves, and under
``refit_risk`` it is the conditioning day.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import NDArray

from end_of_outbreak.delay_distributions import (
    DEFAULT_MAX_LAG,
    GammaDelay,
    discretise_gamma,
)

UNREPORTED_INCIDENCE_VARIABLE = "unreported_incidence"
"""Name of the free latent count block, present only when reporting is incomplete."""

TRUE_INCIDENCE_VARIABLE = "true_incidence"
"""Name of the deterministic true-count block, present only when reporting is incomplete."""

LATENT_DAY_DIMENSION = "latent_day"
"""Coordinate for the latent true counts: days ``1 ... T``, day 0 being the fixed index."""


@dataclass(frozen=True)
class ReportingModel:
    """How true onsets become reported onsets.

    Parameters
    ----------
    probability
        Probability that a case is *ever* reported, in ``[0, 1]``. ``1.0`` with no delay means
        complete reporting, and every builder then produces exactly the model it produced
        before this module existed.
    delay
        Onset-to-report delay. Supported from lag 0 — a case may be reported on its own onset
        day — so it is discretised with ``first_lag=0``, unlike the incubation period.
        ``None`` means reports are instantaneous and only under-reporting applies.
    max_lag
        Largest delay lag with its own probability mass; residual tail mass is folded into it.
    """

    probability: float = 1.0
    delay: GammaDelay | None = None
    max_lag: int = DEFAULT_MAX_LAG

    def __post_init__(self) -> None:
        probability = float(self.probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"reporting probability must be finite and in [0, 1], got {self.probability}"
            )
        if self.max_lag < 0:
            raise ValueError(f"max_lag must be non-negative, got {self.max_lag}")

    @property
    def is_complete(self) -> bool:
        """Whether every case is reported on its onset day, so no latent block is needed."""
        return self.probability == 1.0 and self.delay is None

    def probability_by_day(self, n_days: int, *, as_of_day: int) -> NDArray[np.float64]:
        """``π_t`` for onset days ``0 ... n_days - 1``, as seen on ``as_of_day``.

        Without a delay this is the constant :attr:`probability`. With one it is
        ``probability · P(delay <= as_of_day - t)``, so onset days close to the as-of day are
        truncated toward zero (nowcasting) while old ones plateau at :attr:`probability`.
        Onset days after the as-of day cannot have been reported at all.
        """
        if n_days < 1:
            raise ValueError(f"n_days must be positive, got {n_days}")
        if self.delay is None:
            return np.full(n_days, float(self.probability))
        # cumsum of a lag-0 pmf is P(delay <= d) directly, indexed from d = 0.
        reported_by = np.cumsum(discretise_gamma(self.delay, max_lag=self.max_lag, first_lag=0))
        available = as_of_day - np.arange(n_days, dtype=np.int64)
        truncated = self.probability * reported_by[np.clip(available, 0, reported_by.size - 1)]
        return np.where(available < 0, 0.0, truncated)

    @classmethod
    def from_config(cls, block: dict[str, Any] | None) -> ReportingModel:
        """Build from a ``reporting:`` block, or complete reporting when there is none."""
        if block is None:
            return COMPLETE_REPORTING
        delay_block = block.get("delay")
        delay = (
            None
            if delay_block is None
            else GammaDelay(mean=float(delay_block["mean"]), sd=float(delay_block["sd"]))
        )
        return cls(
            probability=float(block["probability"]),
            delay=delay,
            max_lag=int(block.get("max_lag", DEFAULT_MAX_LAG)),
        )


COMPLETE_REPORTING = ReportingModel()
"""Every case reported on its onset day: the assumption every analysis made before this."""


CountMoments = tuple[Any, Any | None]
"""``(mean, dispersion)`` of a model's true-count density over its likelihood days.

``dispersion`` is ``None`` for the Poisson-observed models and the negative binomial's
``alpha`` otherwise.
"""

MomentsCallable = Callable[..., CountMoments]
"""``moments(totals, *parameters) -> CountMoments``.

The parameters are passed in rather than closed over because PyMC refuses a ``CustomDist``
whose ``logp`` graph reaches a random variable it was not given: *"Random variables detected in
the logp graph ... when CustomDist logp ... reference nonlocal variables"*. Inside the ``logp``
they arrive as the distribution's own copies, and using those is what keeps the graph valid.
"""


def latent_coords(
    reported: NDArray[np.int64], reporting: ReportingModel
) -> dict[str, NDArray[np.int64]]:
    """Model coordinates the reporting layer needs, to merge into a builder's ``coords``.

    Empty under complete reporting, so a builder can splice it in unconditionally.
    """
    if reporting.is_complete:
        return {}
    return {LATENT_DAY_DIMENSION: np.arange(1, len(reported), dtype=np.int64)}


def build_observation(
    reported: NDArray[np.int64],
    *,
    reporting: ReportingModel,
    days: NDArray[np.int64],
    as_of_day: int,
    name: str,
    dims: str,
    moments: MomentsCallable,
    parameters: Sequence[Any] = (),
) -> Any:
    """Declare a model's observation, with or without a reporting layer, and return the totals.

    This is the single place any observed node is created. Under complete reporting it is
    ``moments`` evaluated at the data and the resulting ``pm.Poisson``/``pm.NegativeBinomial``
    — exactly what the builders did inline before. Under incomplete reporting the unreported
    cases become a scalar-support ``pm.CustomDist`` and the observed node is the corresponding
    thinned marginal. Both use ``moments`` evaluated at ``reported + unreported``.

    Parameters
    ----------
    reported
        The observed series ``c_0, ..., c_T``. Day 0 is the fixed index (``D_0 = c_0``).
    reporting
        The reporting model. :attr:`ReportingModel.is_complete` selects the branch.
    days
        Likelihood days, indexing ``reported``.
    as_of_day
        Day the snapshot was taken from, for the delay's right-truncation. See the module
        docstring: this is *not* inferred from ``len(reported)``.
    name, dims
        Name and model coordinate of the observed node, so the caller keeps ownership of both.
    moments
        ``moments(totals, *parameters)``, mapping a true-count series — a NumPy array under
        complete reporting, a symbolic vector otherwise — to that model's
        ``(mean, dispersion)`` over ``days``.
    parameters
        Model variables ``moments`` depends on. Passed through to :class:`pm.CustomDist` and
        handed back to ``moments``; see :data:`MomentsCallable` for why they cannot simply be
        closed over.

    Returns
    -------
    The true-count series: ``reported`` itself under complete reporting, otherwise a symbolic
    vector of length ``len(reported)`` that the caller can reuse (for instance to record the
    latent block it implies).
    """
    reported = np.asarray(reported, dtype=np.int64)
    if reporting.is_complete:
        mean, dispersion = moments(reported, *parameters)
        _observe(name, mean=mean, dispersion=dispersion, observed=reported[days], dims=dims)
        return reported

    probability = reporting.probability_by_day(reported.size, as_of_day=as_of_day)
    impossible_days = days[(probability[days] == 0.0) & (reported[days] > 0)]
    if impossible_days.size:
        raise ValueError(
            f"days {impossible_days.tolist()} carry reported cases but have an "
            "effective reporting probability of zero; check the delay and the as-of day"
        )

    expected_days = np.arange(1, reported.size, dtype=np.int64)
    if not np.array_equal(days, expected_days):
        raise ValueError(
            "incomplete reporting must evaluate every day after the fixed index case; "
            f"got days {days.tolist()}"
        )

    index_case = pt.as_tensor_variable(reported[:1].astype(np.float64))
    reported_after_index = pt.as_tensor_variable(reported[1:].astype(np.float64))

    def _totals(unreported: Any) -> Any:
        return pt.concatenate([index_case, reported_after_index + unreported])

    def _logp(value: Any, *args: Any) -> Any:
        totals = _totals(value)
        mean, dispersion = moments(totals, *args)
        hidden_mean, hidden_dispersion = _hidden_moments(
            reported[days], mean=mean, dispersion=dispersion, probability=probability[days]
        )
        return _count_logp(value, mean=hidden_mean, dispersion=hidden_dispersion)

    def _dist(*args: Any) -> Any:
        # Initial values and prior-predictive draws only: `_logp` defines the density, so this
        # cannot move the posterior. The final positional argument is PyMC's symbolic `size`.
        del args
        return pm.Poisson.dist(
            mu=_initial_unreported(reported, probability), shape=reported.size - 1
        )

    unreported_rv = pm.CustomDist(
        UNREPORTED_INCIDENCE_VARIABLE,
        *parameters,
        logp=_logp,
        dist=_dist,
        dtype="int64",
        dims=LATENT_DAY_DIMENSION,
        # Scalar support with a vector batch is load-bearing: PyMC can then use its ordinary
        # coordinate-wise Metropolis sweep instead of treating the block like a multinomial.
        signature=_scalar_signature(parameters),
        initval=_initial_unreported(reported, probability),
    )
    totals = _totals(unreported_rv)
    pm.Deterministic(
        TRUE_INCIDENCE_VARIABLE,
        totals[1:],
        dims=LATENT_DAY_DIMENSION,
    )
    mean, dispersion = moments(totals, *parameters)
    _observe_thinned(
        name,
        mean=mean,
        dispersion=dispersion,
        probability=probability[days],
        observed=reported[days],
        dims=dims,
    )
    return totals


def thin(
    counts: NDArray[np.int64],
    *,
    reporting: ReportingModel,
    as_of_day: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Reported counts drawn from true counts by independent per-case thinning.

    The generative counterpart of :func:`build_observation`, and deliberately *not* part of the
    simulators in :mod:`end_of_outbreak.forward_simulation`: reporting is conditionally
    independent of the whole transmission recursion given the true counts, so it composes onto
    a finished simulation rather than living inside one. That is what lets the
    likelihood-versus-simulation checks reuse the simulators unchanged.

    Day 0 is returned untouched, matching the ``D_0 = c_0`` convention the models are built on.
    A leading replicate axis is carried through.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim == 0:
        raise ValueError("counts must have at least one axis, indexed by day")
    probability = reporting.probability_by_day(counts.shape[-1], as_of_day=as_of_day)
    reported = rng.binomial(counts, probability)
    reported[..., 0] = counts[..., 0]
    return np.asarray(reported, dtype=np.int64)


def _scalar_signature(parameters: Sequence[Any]) -> str:
    """A scalar-support ``CustomDist`` signature for unrelated model parameters.

    Without one, PyMC tries to broadcast every distribution parameter against the variable's
    own shape, and a latent block indexed by cohort day is not the same length as the counts
    indexed by day. Naming each parameter's axes separately says they are unrelated.
    """
    inputs = ",".join(
        "()" if getattr(parameter, "ndim", 0) == 0 else f"(p{index})"
        for index, parameter in enumerate(parameters)
    )
    return f"{inputs}->()"


def _safe_positive(value: Any) -> Any:
    """A positive stand-in used only in an inactive branch of an exact zero-mean logp."""
    return pt.switch(pt.gt(value, 0.0), value, 1.0)


def _count_logp(value: Any, *, mean: Any, dispersion: Any | None) -> Any:
    """Exact Poisson/NB logp, including the point mass obtained when ``mean == 0``."""
    distribution = (
        pm.Poisson.dist(mu=_safe_positive(mean))
        if dispersion is None
        else pm.NegativeBinomial.dist(mu=_safe_positive(mean), alpha=_safe_positive(dispersion))
    )
    ordinary = pm.logp(distribution, value)
    zero_mean = pt.switch(pt.eq(value, 0), 0.0, -np.inf)
    return pt.switch(pt.eq(mean, 0.0), zero_mean, ordinary)


def _hidden_moments(
    reported: Any, *, mean: Any, dispersion: Any | None, probability: Any
) -> CountMoments:
    """Moments of unreported counts conditional on the reported count."""
    hidden_fraction = 1.0 - probability
    if dispersion is None:
        return hidden_fraction * mean, None
    hidden_dispersion = dispersion + reported
    denominator = dispersion + probability * mean
    hidden_mean = hidden_fraction * mean * hidden_dispersion / _safe_positive(denominator)
    return hidden_mean, hidden_dispersion


def _observe_thinned(
    name: str,
    *,
    mean: Any,
    dispersion: Any | None,
    probability: Any,
    observed: NDArray[np.int64],
    dims: str,
) -> None:
    """Observed marginal after thinning, with an exact structurally-zero mean."""
    thinned_mean = probability * mean
    if dispersion is None:

        def logp(value: Any, distribution_mean: Any) -> Any:
            return _count_logp(value, mean=distribution_mean, dispersion=None)

        pm.CustomDist(name, thinned_mean, logp=logp, observed=observed, dims=dims)
        return

    def logp(value: Any, distribution_mean: Any, alpha: Any) -> Any:
        return _count_logp(value, mean=distribution_mean, dispersion=alpha)

    pm.CustomDist(name, thinned_mean, dispersion, logp=logp, observed=observed, dims=dims)


def _observe(
    name: str, *, mean: Any, dispersion: Any | None, observed: NDArray[np.int64], dims: str
) -> None:
    """The observed node of a completely reported model."""
    if dispersion is None:
        pm.Poisson(name, mu=mean, observed=observed, dims=dims)
        return
    pm.NegativeBinomial(name, mu=mean, alpha=dispersion, observed=observed, dims=dims)


def _initial_unreported(
    reported: NDArray[np.int64], probability: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Finite starting value for the unreported counts over days ``1 ... T``."""
    ratio = np.divide(
        1.0 - probability[1:],
        probability[1:],
        out=np.zeros(reported.size - 1, dtype=np.float64),
        where=probability[1:] > 0.0,
    )
    return np.round(reported[1:] * ratio).astype(np.int64)
