"""PyMC model builders for all five models, plus the two Poisson limits.

Infection-anchored ("naive") models
-----------------------------------
DLO, SSE and SSI read the observed counts as infections, ``I_t := C_t`` — the practice this
project sets out to quantify. They share the renewal structure
``Λ_t = Σ_{s>=1} w_s ·(driving series)_{t-s}`` and differ only in where the excess variance
enters:

``dlo``
    ``I_t ~ NB(mean = R_t Λ_t, disp = k)`` with ``Λ_t`` built from the observed counts. The
    dispersion is constant across days and does **not** scale with the force of infection —
    the defining feature, and the reason the same numerical ``k`` means something quite
    different here than in SSE/SSI. The negative-binomial variant of Supplementary Analysis 1
    of Thompson et al. (2024).
``sse``
    ``I_t ~ NB(mean = R_t Λ_t, disp = k Λ_t)``. The dispersion scales with the force of
    infection, which is what makes ``k`` an individual-level offspring dispersion. Closed form
    despite the overdispersion acting below the day level, so no latent block.
``ssi``
    ``Y_t | I_t ~ Gamma(k I_t, k)`` and ``I_t ~ Poisson(R_t Σ_s w_s Y_{t-s})``, with one
    latent per day carrying cases.

Onset-anchored models
---------------------
SSE-SO and SSI-SO read the counts as onsets, ``D_t := C_t``, anchor transmission to the
infector's own onset through ``f_tost``, and map infections forward to onsets through
``f_inc``. Writing ``E_t`` for the *expected* number of infections on day ``t``:

``sse_so``
    ``λ̃_t ~ Gamma(k Σ_s f_tost,s D_{t-s}, k)``, ``E_t = R_t λ̃_t``,
    ``D_t ~ Poisson(Σ_a f_inc,a E_{t-a})``.
``ssi_so``
    ``Y_t | D_t ~ Gamma(k D_t, k)``, ``E_t = R_t Σ_s f_tost,s Y_{t-s}``,
    ``D_t ~ Poisson(Σ_a f_inc,a E_{t-a})``.

Both are the inference-convenient forms of §3.5; §4 shows they define the same joint law for
the observed onsets as the natural forms that generate explicit infection counts. Because
``f_inc`` has no mass at lag 0, ``D_t`` depends only on ``E_{<t}`` and hence only on
``D_{<t}``, so the recursion is well ordered even though ``f_tost`` may place mass at lag 0.

``cori`` and ``cori_so`` are the ``k → ∞`` Poisson limits. They are not compared models; they
exist as the targets of the collapse checks and, for ``cori_so``, as the §4.1 warm-up.

Conventions shared by every builder
-----------------------------------
Day 0 is an initial condition, so the likelihood runs over days ``1, ..., T`` (§3.1). ``R``
switches from ``R_pre`` to ``R_post`` on ``switch_day`` in the model's **own** time index
(§5.7) — for the naive models that indexes the day a case appears, for the onset-anchored ones
the day a transmission occurs, and the resulting ~11-day asymmetry is a reportable result.
Each of ``R_pre``, ``R_post`` and ``k`` is either a fixed ``float`` — no random variable, which
is what the fixed-``k`` analyses and the fixed-``θ`` cross-checks need — or a
:class:`~end_of_outbreak.model_specifications.LogNormalPrior` to estimate.

Latent blocks are built through :mod:`end_of_outbreak.latent_parameterisations`, which owns the
Stage-3 strategy choice. The parameterisation is a **required** argument: there is no silent
default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import NDArray

from end_of_outbreak import latent_parameterisations as lp
from end_of_outbreak import renewal
from end_of_outbreak.delay_distributions import (
    INCUBATION_FIRST_LAG,
    SERIAL_INTERVAL_FIRST_LAG,
    TOST_FIRST_LAG,
    OnsetAnchoredDelays,
)
from end_of_outbreak.model_specifications import (
    SSE_SO,
    LogNormalPrior,
    ModelSpecification,
    specification_of,
)

OBSERVED_VARIABLE = "incidence"
"""Name of the observed node in every model, so downstream code can find it."""

INFECTIVITY_VARIABLE = "Y"
"""Name of the SSI / SSI-SO latent block: the aggregate infectivity of a day's cohort."""

TRANSMISSIBILITY_VARIABLE = "lambda_tilde"
"""Name of the SSE-SO latent block: the day's realised transmission-event intensity."""

COHORT_DAY_DIMENSION = "cohort_day"
"""Coordinate for a per-cohort latent block, over the sampled positions only."""

TRANSMISSION_DAY_DIMENSION = "transmission_day"
"""Coordinate for the SSE-SO latent block, over the sampled positions only."""

LIKELIHOOD_DAY_DIMENSION = "likelihood_day"
"""Coordinate for the observations actually evaluated (§3.1: day 0 is an initial condition)."""


# ---------------------------------------------------------------------------------------
# Building blocks shared by every model
# ---------------------------------------------------------------------------------------


def _parameter(name: str, value: float | LogNormalPrior) -> Any:
    """A fixed constant or a log-normal random variable, depending on what was passed."""
    if isinstance(value, LogNormalPrior):
        kwargs: Any = value.pymc_kwargs()
        return pm.LogNormal(name, **kwargs)
    numeric = float(value)
    if not numeric > 0:
        raise ValueError(f"{name} must be positive when fixed, got {numeric}")
    return numeric


def _reproduction_number(
    R_pre: Any, R_post: Any, *, days: NDArray[np.int64], n_days: int, switch_day: int
) -> Any:
    """``R_t`` on each of ``days``, under the §5.7 switch convention."""
    period = renewal.switch_index(n_days, switch_day)[days]
    return pt.stack([R_pre, R_post])[period]


def _validate_counts(counts: NDArray[np.int64]) -> None:
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("counts must be a one-dimensional series of at least two days")
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    if counts[0] < 1:
        raise ValueError(
            "day 0 must carry at least one case: it is the initial condition of every model "
            "(§3.1, the sole-index-case assumption)"
        )


def _validate_weights(weights: NDArray[np.float64], label: str) -> None:
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"{label} must be a normalised probability vector")


def _validate_dispersion(
    specification: ModelSpecification, k: float | LogNormalPrior | None
) -> None:
    if specification.has_dispersion and k is None:
        raise ValueError(f"model {specification.name!r} needs a dispersion parameter k")
    if not specification.has_dispersion and k is not None:
        raise ValueError(
            f"model {specification.name!r} has no dispersion parameter; it is the k → ∞ limit"
        )


def _require_parameterisation(
    specification: ModelSpecification,
    parameterisation: str | lp.LatentParameterisation | None,
) -> lp.LatentParameterisation | None:
    """Resolve the latent strategy, insisting on one whenever the model has latents."""
    if specification.latent_variable is None:
        return None
    if parameterisation is None:
        raise ValueError(
            f"model {specification.name!r} has a latent block, so latent_parameterisation must "
            "be given explicitly — see config/config.yaml and §6.3 of the implementation plan"
        )
    return lp.parameterisation_of(parameterisation)


def _couples_to_observations(
    influence: NDArray[np.float64], counts: NDArray[np.int64], days: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Which latents appear in some ``log μ_j`` term, i.e. can reach a positive-count day.

    ``influence[j, u]`` is the non-negative coefficient of latent ``u`` in ``μ_j``. Latents
    that reach only zero-count days enter the likelihood solely through ``exp(−Σ_j μ_j)``, are
    therefore conditionally independent of everything else, and can be integrated out exactly
    (see :mod:`end_of_outbreak.latent_parameterisations`).
    """
    positive_days = days[counts[days] > 0]
    if positive_days.size == 0:
        return np.zeros(influence.shape[1], dtype=bool)
    return np.asarray((influence[positive_days] > 0.0).any(axis=0), dtype=bool)


def _observed_days(
    influence: NDArray[np.float64],
    days: NDArray[np.int64],
    classification: lp.LatentClassification,
) -> NDArray[np.int64]:
    """The subset of ``days`` that still has a *sampled* latent driving it.

    A day reached only by marginalised or dropped latents has ``μ_j = 0`` in the graph. Its
    count is necessarily zero — a positive count would keep its latents coupled, hence sampled
    — so its Poisson term is identically zero and is better left out than evaluated at
    ``mu = 0``. Its ``−μ_j`` contribution is not lost: the marginalisation potential integrates
    over the whole of ``days``.
    """
    driven = influence[:, classification.sampled].sum(axis=1) > 0.0
    return days[driven[days]]


# ---------------------------------------------------------------------------------------
# Infection-anchored builders
# ---------------------------------------------------------------------------------------


def build_naive_model(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    serial_interval: NDArray[np.float64],
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None = None,
    latent_parameterisation: str | lp.LatentParameterisation | None = None,
    negligible_latent_threshold: float = 0.0,
) -> pm.Model:
    """Build the PyMC model for one of ``dlo``, ``sse``, ``ssi`` or ``cori``.

    Parameters
    ----------
    model
        Model name or :class:`~end_of_outbreak.model_specifications.ModelSpecification`. Must
        be infection-anchored; see :func:`build_onset_anchored_model` for the others.
    counts
        Observed daily counts ``C_0, ..., C_T``, read as infections. ``C_0 >= 1``.
    serial_interval
        Discrete serial interval ``w``, stored from lag 1.
    switch_day
        Day index on which ``R`` changes from ``R_pre`` to ``R_post``.
    R_pre, R_post, k
        Each either a fixed positive ``float`` or a
        :class:`~end_of_outbreak.model_specifications.LogNormalPrior`. ``k`` must be ``None``
        for ``cori`` and given otherwise.
    latent_parameterisation
        Required for ``ssi``; ignored by the others.
    negligible_latent_threshold
        Latents whose cohort size falls below this are dropped. Meaningless for SSI, whose
        scales are case counts, and kept only for signature symmetry.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise ValueError(
            f"model {specification.name!r} is onset-anchored; use build_onset_anchored_model"
        )
    counts = np.asarray(counts, dtype=np.int64)
    serial_interval = np.asarray(serial_interval, dtype=np.float64)
    _validate_counts(counts)
    _validate_weights(serial_interval, "serial_interval")
    _validate_dispersion(specification, k)
    parameterisation = _require_parameterisation(specification, latent_parameterisation)

    n_days = counts.size
    cohort_days = np.flatnonzero(counts > 0).astype(np.int64)

    if parameterisation is None:
        # The force of infection is a known constant: the driving series is the data itself.
        force_of_infection = renewal.delay_weighted_sum(
            counts.astype(np.float64), serial_interval, first_lag=SERIAL_INTERVAL_FIRST_LAG
        )
        days = renewal.likelihood_days(force_of_infection, counts)
        return _build_closed_form_naive_model(
            specification,
            counts,
            force_of_infection=force_of_infection,
            days=days,
            n_days=n_days,
            switch_day=switch_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
        )

    # SSI: the driving series is latent, so the renewal operator stays symbolic. Its row sums
    # say which days can be driven at all, which is what selects the likelihood days.
    design = renewal.delay_design_matrix(
        n_days, serial_interval, first_lag=SERIAL_INTERVAL_FIRST_LAG, source_days=cohort_days
    )
    days = renewal.likelihood_days(design.sum(axis=1), counts)
    scale = counts[cohort_days].astype(np.float64)
    classification = lp.classify_latents(
        scale=scale,
        couples_to_observations=_couples_to_observations(design, counts, days),
        parameterisation=parameterisation,
        negligible_threshold=negligible_latent_threshold,
    )
    _check_positive_days_stay_reachable(counts, days, design, classification)
    observed_days = _observed_days(design, days, classification)

    coords = {
        LIKELIHOOD_DAY_DIMENSION: observed_days,
        COHORT_DAY_DIMENSION: cohort_days[classification.sampled],
    }
    with pm.Model(coords=coords) as built:
        R_pre_value = _parameter("R_pre", R_pre)
        R_post_value = _parameter("R_post", R_post)
        assert k is not None  # guaranteed by _validate_dispersion for a latent model
        k_value = _parameter("k", k)
        # Coefficient of Y_u in Σ_j μ_j over every likelihood day, needed to integrate out the
        # latents the data cannot resolve. It runs over `days`, not the reduced `observed_days`.
        coupling = pt.dot(
            _reproduction_number(
                R_pre_value, R_post_value, days=days, n_days=n_days, switch_day=switch_day
            ),
            design[days],
        )
        Y = lp.build_gamma_latent_block(
            INFECTIVITY_VARIABLE,
            k=k_value,
            scale=scale,
            coupling=coupling,
            classification=classification,
            parameterisation=parameterisation,
            dims=COHORT_DAY_DIMENSION,
        )
        R_by_observed_day = _reproduction_number(
            R_pre_value, R_post_value, days=observed_days, n_days=n_days, switch_day=switch_day
        )
        _poisson_observation(
            counts,
            days=observed_days,
            mean_incidence=R_by_observed_day * pt.dot(design[observed_days], Y),
        )
    return built


def _build_closed_form_naive_model(
    specification: ModelSpecification,
    counts: NDArray[np.int64],
    *,
    force_of_infection: NDArray[np.float64],
    days: NDArray[np.int64],
    n_days: int,
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None,
) -> pm.Model:
    """DLO, SSE and Cori: no latents, so the whole likelihood is available in closed form."""
    coords = {LIKELIHOOD_DAY_DIMENSION: days}
    with pm.Model(coords=coords) as built:
        R_pre_value = _parameter("R_pre", R_pre)
        R_post_value = _parameter("R_post", R_post)
        k_value = None if k is None else _parameter("k", k)
        R_by_day = _reproduction_number(
            R_pre_value, R_post_value, days=days, n_days=n_days, switch_day=switch_day
        )
        mean_incidence = R_by_day * force_of_infection[days]

        if k_value is None:  # cori, the k → ∞ limit
            _poisson_observation(counts, days=days, mean_incidence=mean_incidence)
        else:
            # DLO holds the dispersion constant across days; SSE scales it with the force of
            # infection. That one difference is the whole of §5.4.
            dispersion = (
                k_value if specification.name == "dlo" else k_value * force_of_infection[days]
            )
            pm.NegativeBinomial(
                OBSERVED_VARIABLE,
                mu=mean_incidence,
                alpha=dispersion,
                observed=counts[days],
                dims=LIKELIHOOD_DAY_DIMENSION,
            )
    return built


# ---------------------------------------------------------------------------------------
# Onset-anchored builders
# ---------------------------------------------------------------------------------------


def build_onset_anchored_model(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None = None,
    latent_parameterisation: str | lp.LatentParameterisation | None = None,
    negligible_latent_threshold: float = 0.0,
) -> pm.Model:
    """Build the PyMC model for one of ``sse_so``, ``ssi_so`` or ``cori_so``.

    Parameters
    ----------
    model
        Model name or :class:`~end_of_outbreak.model_specifications.ModelSpecification`. Must
        be onset-anchored.
    counts
        Observed daily counts ``C_0, ..., C_T``, read as symptom onsets. ``C_0 >= 1``.
    delays
        The ``(w, f_tost, f_inc)`` triple from
        :func:`end_of_outbreak.delay_distributions.build_onset_anchored_delays`. Only
        ``f_tost`` and ``f_inc`` are used; the serial interval rides along because it is what
        ties these models to the naive ones.
    switch_day
        Day on which ``R`` changes — here the day a **transmission** occurs, not the day the
        resulting case appears (§5.7).
    R_pre, R_post, k
        As in :func:`build_naive_model`. ``k`` must be ``None`` for ``cori_so``.
    latent_parameterisation
        Required for ``sse_so`` and ``ssi_so``.
    negligible_latent_threshold
        Threshold on the SSE-SO latent scale ``λ_t`` below which the latent is dropped and
        ``E_t`` set to 0, at a cost bounded by ``max(R) · Σ_dropped λ_t`` (§6.3). ``0.0``
        drops nothing.
    """
    specification = specification_of(model)
    if specification.anchoring != "onsets":
        raise ValueError(
            f"model {specification.name!r} is infection-anchored; use build_naive_model"
        )
    counts = np.asarray(counts, dtype=np.int64)
    _validate_counts(counts)
    _validate_weights(delays.tost, "f_tost")
    _validate_weights(delays.incubation, "f_inc")
    _validate_dispersion(specification, k)
    parameterisation = _require_parameterisation(specification, latent_parameterisation)

    n_days = counts.size
    # E_t for t = n_days - 1 could only produce onsets after the window, so it is unidentified
    # and left out; f_inc has no lag-0 mass, which is what makes the recursion well ordered.
    transmission_days = np.arange(n_days - 1, dtype=np.int64)
    incubation_design = renewal.delay_design_matrix(
        n_days, delays.incubation, first_lag=INCUBATION_FIRST_LAG, source_days=transmission_days
    )
    tost_sum = latent_scale_by_day(SSE_SO, counts, delays=delays)[transmission_days]

    if specification.name == "ssi_so":
        return _build_ssi_so(
            counts,
            delays=delays,
            transmission_days=transmission_days,
            incubation_design=incubation_design,
            switch_day=switch_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            parameterisation=parameterisation,
            negligible_latent_threshold=negligible_latent_threshold,
        )
    return _build_sse_so_or_cori_so(
        specification,
        counts,
        transmission_days=transmission_days,
        incubation_design=incubation_design,
        tost_sum=tost_sum,
        switch_day=switch_day,
        R_pre=R_pre,
        R_post=R_post,
        k=k,
        parameterisation=parameterisation,
        negligible_latent_threshold=negligible_latent_threshold,
    )


def _build_sse_so_or_cori_so(
    specification: ModelSpecification,
    counts: NDArray[np.int64],
    *,
    transmission_days: NDArray[np.int64],
    incubation_design: NDArray[np.float64],
    tost_sum: NDArray[np.float64],
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None,
    parameterisation: lp.LatentParameterisation | None,
    negligible_latent_threshold: float,
) -> pm.Model:
    """``E_t = R_t λ̃_t`` (SSE-SO) or ``E_t = R_t λ_t`` (Cori-SO), then onsets by ``f_inc``."""
    n_days = counts.size
    days = renewal.likelihood_days(incubation_design @ tost_sum, counts)
    # Fraction of day-t infections whose onset falls inside the window: the coefficient that
    # turns a latent's mean into its contribution to Σ_j μ_j.
    reach = incubation_design[days].sum(axis=0)

    if parameterisation is None:
        coords = {LIKELIHOOD_DAY_DIMENSION: days}
        with pm.Model(coords=coords) as built:
            R_by_transmission_day = _reproduction_number(
                _parameter("R_pre", R_pre),
                _parameter("R_post", R_post),
                days=transmission_days,
                n_days=n_days,
                switch_day=switch_day,
            )
            expected_infections = R_by_transmission_day * tost_sum
            _poisson_observation(
                counts,
                days=days,
                mean_incidence=pt.dot(incubation_design[days], expected_infections),
            )
        return built

    classification = lp.classify_latents(
        scale=tost_sum,
        couples_to_observations=_couples_to_observations(incubation_design, counts, days),
        parameterisation=parameterisation,
        negligible_threshold=negligible_latent_threshold,
    )
    _check_positive_days_stay_reachable(counts, days, incubation_design, classification)
    observed_days = _observed_days(incubation_design, days, classification)

    coords = {
        LIKELIHOOD_DAY_DIMENSION: observed_days,
        TRANSMISSION_DAY_DIMENSION: transmission_days[classification.sampled],
    }
    with pm.Model(coords=coords) as built:
        assert k is not None  # guaranteed by _validate_dispersion for a latent model
        k_value = _parameter("k", k)
        R_by_transmission_day = _reproduction_number(
            _parameter("R_pre", R_pre),
            _parameter("R_post", R_post),
            days=transmission_days,
            n_days=n_days,
            switch_day=switch_day,
        )
        lambda_tilde = lp.build_gamma_latent_block(
            TRANSMISSIBILITY_VARIABLE,
            k=k_value,
            scale=tost_sum,
            coupling=R_by_transmission_day * reach,
            classification=classification,
            parameterisation=parameterisation,
            dims=TRANSMISSION_DAY_DIMENSION,
        )
        expected_infections = R_by_transmission_day * lambda_tilde
        _poisson_observation(
            counts,
            days=observed_days,
            mean_incidence=pt.dot(incubation_design[observed_days], expected_infections),
        )
    return built


def _build_ssi_so(
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    transmission_days: NDArray[np.int64],
    incubation_design: NDArray[np.float64],
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None,
    parameterisation: lp.LatentParameterisation | None,
    negligible_latent_threshold: float,
) -> pm.Model:
    """``Y_t | D_t ~ Gamma(k D_t, k)``, spread forward by ``f_tost`` and then by ``f_inc``."""
    assert parameterisation is not None  # ssi_so always has latents
    n_days = counts.size
    cohort_days = np.flatnonzero(counts > 0).astype(np.int64)
    tost_design = renewal.delay_design_matrix(
        n_days, delays.tost, first_lag=TOST_FIRST_LAG, source_days=cohort_days
    )[transmission_days]

    # Coefficient of Y_u in μ_j, up to the reproduction number: two delay hops in series.
    influence = incubation_design @ tost_design
    days = renewal.likelihood_days(influence.sum(axis=1), counts)
    reach = incubation_design[days].sum(axis=0)
    scale = counts[cohort_days].astype(np.float64)
    classification = lp.classify_latents(
        scale=scale,
        couples_to_observations=_couples_to_observations(influence, counts, days),
        parameterisation=parameterisation,
        negligible_threshold=negligible_latent_threshold,
    )
    _check_positive_days_stay_reachable(counts, days, influence, classification)
    observed_days = _observed_days(influence, days, classification)

    coords = {
        LIKELIHOOD_DAY_DIMENSION: observed_days,
        COHORT_DAY_DIMENSION: cohort_days[classification.sampled],
    }
    with pm.Model(coords=coords) as built:
        assert k is not None
        k_value = _parameter("k", k)
        R_by_transmission_day = _reproduction_number(
            _parameter("R_pre", R_pre),
            _parameter("R_post", R_post),
            days=transmission_days,
            n_days=n_days,
            switch_day=switch_day,
        )
        Y = lp.build_gamma_latent_block(
            INFECTIVITY_VARIABLE,
            k=k_value,
            scale=scale,
            coupling=pt.dot(R_by_transmission_day * reach, tost_design),
            classification=classification,
            parameterisation=parameterisation,
            dims=COHORT_DAY_DIMENSION,
        )
        expected_infections = R_by_transmission_day * pt.dot(tost_design, Y)
        _poisson_observation(
            counts,
            days=observed_days,
            mean_incidence=pt.dot(incubation_design[observed_days], expected_infections),
        )
    return built


def _check_positive_days_stay_reachable(
    counts: NDArray[np.int64],
    days: NDArray[np.int64],
    influence: NDArray[np.float64],
    classification: lp.LatentClassification,
) -> None:
    """Dropping latents must never make an observed case impossible."""
    if classification.dropped.size == 0:
        return
    positive_days = days[counts[days] > 0]
    remaining = influence[np.ix_(positive_days, classification.sampled)].sum(axis=1)
    orphaned = positive_days[remaining <= 0.0]
    if orphaned.size > 0:
        raise ValueError(
            f"negligible_latent_threshold leaves days {orphaned.tolist()} with cases but no "
            "surviving latent to explain them; lower the threshold"
        )


def _poisson_observation(
    counts: NDArray[np.int64], *, days: NDArray[np.int64], mean_incidence: Any
) -> None:
    """The observation node shared by every Poisson-observed model."""
    pm.Poisson(
        OBSERVED_VARIABLE,
        mu=mean_incidence,
        observed=counts[days],
        dims=LIKELIHOOD_DAY_DIMENSION,
    )


# ---------------------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------------------


def build_model(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None = None,
    latent_parameterisation: str | lp.LatentParameterisation | None = None,
    negligible_latent_threshold: float = 0.0,
) -> pm.Model:
    """Build any model from the delay triple, dispatching on its anchoring.

    The naive models take only ``delays.serial_interval``, the onset-anchored ones only
    ``delays.tost`` and ``delays.incubation`` — but all five must be driven by the *same*
    onset-to-onset interval for the comparison to isolate onset-anchoring, so the pipeline
    passes the whole triple and lets each builder take what it needs.
    """
    specification = specification_of(model)
    common: dict[str, Any] = {
        "switch_day": switch_day,
        "R_pre": R_pre,
        "R_post": R_post,
        "k": k,
        "latent_parameterisation": latent_parameterisation,
        "negligible_latent_threshold": negligible_latent_threshold,
    }
    if specification.anchoring == "infections":
        return build_naive_model(
            specification, counts, serial_interval=delays.serial_interval, **common
        )
    return build_onset_anchored_model(specification, counts, delays=delays, **common)


def latent_scale_by_day(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
) -> NDArray[np.float64]:
    """The Gamma shape multiplier ``scale_u`` of each day's latent, by day index.

    Every latent block in the project is ``Gamma(k · scale_u, k)``; what differs is what
    ``scale_u`` is. For the individual-level models it is the size of the day's onset cohort;
    for SSE-SO it is the TOST-weighted onset sum ``λ_u = Σ_s f_tost,s D_{u-s}``.

    Exposed because callers outside the builder need it — to compute an initial point for the
    sampled block (:func:`end_of_outbreak.latent_parameterisations.suggested_initial_values`),
    or to rebuild a marginalised latent's exact conditional posterior. Index it with
    :func:`model_days` to get the values for the latents a built model actually samples.
    """
    specification = specification_of(model)
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block")
    counts = np.asarray(counts, dtype=np.int64)
    if specification.name == "sse_so":
        return renewal.delay_weighted_sum(
            counts.astype(np.float64), delays.tost, first_lag=TOST_FIRST_LAG
        )
    return counts.astype(np.float64)


def latent_dimension(model: str | ModelSpecification) -> str:
    """Which coordinate a model's latent block is indexed by."""
    specification = specification_of(model)
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block")
    return TRANSMISSION_DAY_DIMENSION if specification.name == "sse_so" else COHORT_DAY_DIMENSION


def model_days(model: pm.Model, dimension: str) -> NDArray[np.int64]:
    """The day indices behind one of a built model's coordinates.

    ``model_days(model, COHORT_DAY_DIMENSION)`` gives the days carrying a sampled latent, in
    the order the latent vector uses — which is what downstream code needs to align posterior
    draws back onto the calendar. Latents that were marginalised or dropped do not appear.
    """
    coordinate = model.coords.get(dimension)
    if coordinate is None:
        raise KeyError(f"model has no coordinate {dimension!r}")
    return np.asarray(coordinate, dtype=np.int64)


# ---------------------------------------------------------------------------------------
# Evaluating a model's joint density at given values
# ---------------------------------------------------------------------------------------


def _compile_named_logp(model: pm.Model, *, terms: list[Any] | None) -> Callable[..., float]:
    """Compile some of a model's log-density terms as a function of named natural values."""
    logp = (
        model.compile_logp(jacobian=False)
        if terms is None
        else model.compile_logp(vars=terms, jacobian=False)
    )
    forward: dict[str, tuple[str, Callable[[NDArray[np.float64]], Any]]] = {}
    for rv in model.free_RVs:
        value_variable = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv)
        if transform is None:
            forward[rv.name] = (value_variable.name, lambda value: value)
        else:
            placeholder = rv.type()
            forward[rv.name] = (
                value_variable.name,
                pytensor.function(
                    [placeholder],
                    transform.forward(placeholder, *rv.owner.inputs),
                    on_unused_input="ignore",
                ),
            )

    def evaluate(**values: Any) -> float:
        unknown = sorted(set(values) - set(forward))
        if unknown:
            raise ValueError(f"model has no free variables named {unknown}")
        missing = sorted(set(forward) - set(values))
        if missing:
            raise ValueError(f"no values supplied for free variables {missing}")
        point = {
            forward[name][0]: forward[name][1](np.asarray(value, dtype=np.float64))
            for name, value in values.items()
        }
        return float(logp(point))

    return evaluate


def compile_joint_logp(model: pm.Model) -> Callable[..., float]:
    """Compile ``log p(observed counts, latents, parameters)`` as a function of named values.

    The returned callable takes one keyword argument per free random variable, named as in
    the model (``R_pre``, ``k``, ``Y``, ...), with values on their **natural** scale — no
    log transforms, no Jacobian. That makes it directly comparable with a density written out
    by hand, which is how the tests check the likelihoods, and is the form the Stage-6
    evidence estimators consume.

    Under a reparameterised latent block the free variable is the reparameterised one
    (``Y_uniform``, ``Y_unit_mean``); see
    :func:`end_of_outbreak.latent_parameterisations.free_variable_name`.

    Compiling is the expensive part, so build the callable once and call it many times.
    """
    return _compile_named_logp(model, terms=None)


def compile_observation_logp(model: pm.Model) -> Callable[..., float]:
    """Compile ``log p(observed counts | latents, parameters)`` alone — no prior terms.

    Same calling convention as :func:`compile_joint_logp`. Isolating the observation term is
    what makes two models with *different latent blocks* comparable: the joint densities differ
    in how many prior factors they carry, so only the likelihood can be held against a common
    yardstick. That is exactly what the negligible-latent threshold has to be judged on.
    """
    return _compile_named_logp(model, terms=[model[OBSERVED_VARIABLE]])


def joint_logp(model: pm.Model, **values: Any) -> float:
    """One-shot :func:`compile_joint_logp`. Compiles on every call — use it sparingly."""
    return compile_joint_logp(model)(**values)
