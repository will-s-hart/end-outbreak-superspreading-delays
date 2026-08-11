"""The reporting layer: under-reporting, onset-to-report delays, and the observed node.

Every model in this project specifies one per-day count density for the **true** onsets —
``Poisson(μ_t)`` for SSI/SSE-SO/SSI-SO/Cori/Cori-SO, ``NegativeBinomial(μ_t, α_t)`` for
DLO/SSE. Reporting sits on top of that, and is the same for all seven:

``D_t ~ CountDist_t(θ, D_{<t}, latents)``  — the model's own likelihood, at the true counts

``c_t | D_t ~ Binomial(D_t, π_t)``  — this module

so the whole of the reported/unreported logic lives here and no builder repeats it. What a
builder supplies is a ``moments`` callable mapping a true-count series to that model's
``(mean, dispersion)`` — the expression it already needs for its own observation node.

Why the latent is the totals
----------------------------
The self-referential part is that ``μ_t`` depends on the true counts, which are latent. Writing
the latent as the **totals** ``D`` rather than as the unreported cases ``U = D − c`` makes that
density a function of the latent itself, so it is expressible as that latent's own ``logp`` and
needs no ``pm.Potential``. This is the ``likelihood="binomial"`` formulation of
``end-of-outbreak-vbd``'s ``underreporting_sandbox/models.py``; the ``"poisson"`` formulation
there, and ``endoutbreakvbd._inference_models._build_underreporting_model``, carry ``U``
instead and split the Poisson by thinning. The two are the same model:

``Poisson(D; μ)·Binom(c; D, π) == Poisson(c; πμ)·Poisson(U; (1−π)μ)``

and ``tests/test_reporting.py`` pins that identity. The totals form is the one taken here
because it is also the only one that covers the negative-binomial models, whose thinned parts
are not independent.

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

TRUE_INCIDENCE_VARIABLE = "true_incidence"
"""Name of the latent true-count block, present only when reporting is incomplete."""

LATENT_DAY_DIMENSION = "latent_day"
"""Coordinate for the latent true counts: days ``1 ... T``, day 0 being the fixed index."""

MEAN_FLOOR = 1e-12
"""Added to every count mean so that a structurally zero ``mu`` stays strictly positive.

Small enough to leave the likelihood otherwise untouched, and needed because the latent
totals can drive a day's mean to exactly zero, which ``Poisson``/``NegativeBinomial`` reject.
Mirrors ``_POISSON_MU_FLOOR`` in ``endoutbreakvbd._inference_models``.
"""

PROBABILITY_FLOOR = 1e-6
"""Lower clip on ``π_t``. Keeps the binomial well posed on days truncated to (almost) zero."""


@dataclass(frozen=True)
class ReportingModel:
    """How true onsets become reported onsets.

    Parameters
    ----------
    probability
        Probability that a case is *ever* reported, in ``(0, 1]``. ``1.0`` with no delay means
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
        if not np.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError(
                f"reporting probability must be finite and in (0, 1], got {self.probability}"
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
    — exactly what the builders did inline before. Under incomplete reporting the true counts
    become a latent ``pm.CustomDist`` whose ``logp`` *is* ``moments`` evaluated at itself, and
    the observed node becomes the reporting binomial.

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

    probability = np.clip(
        reporting.probability_by_day(reported.size, as_of_day=as_of_day),
        PROBABILITY_FLOOR,
        1.0,
    )
    unreportable = (probability <= PROBABILITY_FLOOR) & (reported > 0)
    if unreportable.any():
        raise ValueError(
            f"days {np.flatnonzero(unreportable).tolist()} carry reported cases but have an "
            "effective reporting probability of zero; check the delay and the as-of day"
        )

    index_case = pt.as_tensor_variable(reported[:1].astype(np.float64))

    def _logp(value: Any, *args: Any) -> Any:
        totals = pt.concatenate([index_case, value])
        mean, dispersion = moments(totals, *args)
        distribution = (
            pm.Poisson.dist(mu=mean + MEAN_FLOOR)
            if dispersion is None
            else pm.NegativeBinomial.dist(mu=mean + MEAN_FLOOR, alpha=dispersion)
        )
        return pm.logp(distribution, totals[days])

    def _dist(*args: Any) -> Any:
        # Initial values and prior-predictive draws only: `_logp` defines the density, so this
        # cannot move the posterior. The mean is the reported count scaled up by the reporting
        # probability, which is finite and of the right order.
        return pm.Poisson.dist(mu=_initial_totals(reported, probability), shape=reported.size - 1)

    totals_rv = pm.CustomDist(
        TRUE_INCIDENCE_VARIABLE,
        *parameters,
        logp=_logp,
        dist=_dist,
        dtype="int64",
        dims=LATENT_DAY_DIMENSION,
        signature=_signature(parameters),
        # A binomial with n < observed has zero density, so a default initial point can start
        # every chain at -inf. Start at the scaled-up reported counts, which cannot.
        initval=_initial_totals(reported, probability),
    )
    totals = pt.concatenate([index_case, totals_rv])
    pm.Binomial(
        name,
        n=totals[days],
        p=probability[days],
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


def _signature(parameters: Sequence[Any]) -> str:
    """A ``CustomDist`` signature for parameters that do not share the output's shape.

    Without one, PyMC tries to broadcast every distribution parameter against the variable's
    own shape, and a latent block indexed by cohort day is not the same length as the counts
    indexed by day. Naming each parameter's axes separately says they are unrelated.
    """
    inputs = ",".join(
        "()" if getattr(parameter, "ndim", 0) == 0 else f"(p{index})"
        for index, parameter in enumerate(parameters)
    )
    return f"{inputs}->(t)"


def _observe(
    name: str, *, mean: Any, dispersion: Any | None, observed: NDArray[np.int64], dims: str
) -> None:
    """The observed node of a completely reported model."""
    if dispersion is None:
        pm.Poisson(name, mu=mean, observed=observed, dims=dims)
        return
    pm.NegativeBinomial(name, mu=mean, alpha=dispersion, observed=observed, dims=dims)


def _initial_totals(
    reported: NDArray[np.int64], probability: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Starting value for the latent totals over days ``1 ... T``: at least what was reported."""
    scaled = np.round(reported[1:] / probability[1:])
    return np.maximum(scaled, reported[1:]).astype(np.int64)
