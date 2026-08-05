"""PyMC model builders for the infection-anchored ("naive") models.

DLO, SSE and SSI all read the observed counts as infections, ``I_t := C_t`` — the practice
this project sets out to quantify. They share the renewal structure

    Λ_t = Σ_{s>=1} w_s · (driving series)_{t-s},

and differ only in *where the excess variance enters*:

``dlo``
    ``I_t ~ NB(mean = R_t Λ_t, disp = k)`` with ``Λ_t`` built from the observed counts. The
    dispersion is constant across days and does **not** scale with the force of infection —
    the defining feature, and the reason the same numerical ``k`` means something quite
    different here than in SSE/SSI. This is the negative-binomial variant of Supplementary
    Analysis 1 of Thompson et al. (2024).
``sse``
    ``I_t ~ NB(mean = R_t Λ_t, disp = k Λ_t)``. The dispersion scales with the force of
    infection, which is what makes ``k`` an individual-level offspring dispersion. Closed
    form despite the overdispersion acting below the day level, so no latent block.
``ssi``
    ``Y_t | I_t ~ Gamma(k I_t, k)`` and ``I_t ~ Poisson(R_t Σ_s w_s Y_{t-s})``. One latent
    per day with ``I_t > 0`` (31 for the Équateur series, day 0 included).

``cori`` — ``I_t ~ Poisson(R_t Λ_t)`` — is built here too. It is not one of the compared
models; it is the ``k → ∞`` limit of both SSE and SSI and exists as the target of that check.

Day 0 is an initial condition in every model, so the likelihood runs over days ``1, ..., T``
(§3.1). ``R`` switches from ``R_pre`` to ``R_post`` on ``switch_day`` in the model's own time
index (§5.7); the convention itself lives in :func:`end_of_outbreak.renewal.switch_index`.

Each of ``R_pre``, ``R_post`` and ``k`` may be passed either as a ``float`` — fixed, no random
variable, which is what the fixed-``k`` analyses and the particle-filter cross-checks need —
or as a :class:`~end_of_outbreak.model_specifications.LogNormalPrior`, in which case it is
estimated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import NDArray

from end_of_outbreak import renewal
from end_of_outbreak.delay_distributions import SERIAL_INTERVAL_FIRST_LAG
from end_of_outbreak.model_specifications import (
    LogNormalPrior,
    ModelSpecification,
    specification_of,
)

OBSERVED_VARIABLE = "incidence"
"""Name of the observed node in every naive model, so downstream code can find it."""

INFECTIVITY_VARIABLE = "Y"
"""Name of the SSI latent block: the aggregate infectivity of each day's cohort."""

COHORT_DAY_DIMENSION = "cohort_day"
"""Coordinate for the SSI latents: the days carrying a non-zero cohort, day 0 included."""

LIKELIHOOD_DAY_DIMENSION = "likelihood_day"
"""Coordinate for the observations actually evaluated (§3.1: day 0 is an initial condition)."""


# ---------------------------------------------------------------------------------------
# Building blocks
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


def _infectivity_latents(k: Any, cohort_counts: NDArray[np.int64]) -> Any:
    """The SSI latent block ``Y_t | I_t ~ Gamma(shape = k I_t, rate = k)``.

    Centred, which is the Stage-3 benchmark's baseline. When Stage 3 lands
    ``latent_parameterisations.py`` this is the single call site to redirect: the alternatives
    of §6.3 (mean-1 rescaling, inverse-CDF, partial marginalisation) all produce a variable
    with this same law, differing only in the coordinates the sampler works in.
    """
    kwargs: Any = {"alpha": k * cohort_counts.astype(np.float64), "beta": k}
    return pm.Gamma(INFECTIVITY_VARIABLE, **kwargs, dims=COHORT_DAY_DIMENSION)


def _validate_inputs(
    specification: ModelSpecification,
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    k: float | LogNormalPrior | None,
) -> None:
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("counts must be a one-dimensional series of at least two days")
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    if counts[0] < 1:
        raise ValueError(
            "day 0 must carry at least one case: it is the initial condition of every model "
            "(§3.1, the sole-index-case assumption)"
        )
    if serial_interval.ndim != 1 or serial_interval.size == 0:
        raise ValueError("serial_interval must be a non-empty one-dimensional array")
    if (serial_interval < 0).any() or not np.isclose(serial_interval.sum(), 1.0):
        raise ValueError("serial_interval must be a normalised probability vector")
    if specification.has_dispersion and k is None:
        raise ValueError(f"model {specification.name!r} needs a dispersion parameter k")
    if not specification.has_dispersion and k is not None:
        raise ValueError(
            f"model {specification.name!r} has no dispersion parameter; it is the k → ∞ limit"
        )


# ---------------------------------------------------------------------------------------
# The builder
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
) -> pm.Model:
    """Build the PyMC model for one of ``dlo``, ``sse``, ``ssi`` or ``cori``.

    Parameters
    ----------
    model
        Model name or :class:`~end_of_outbreak.model_specifications.ModelSpecification`. Must
        be infection-anchored; the onset-anchored builders arrive with Stage 8.
    counts
        Observed daily counts ``C_0, ..., C_T``, read as infections. ``C_0 >= 1``.
    serial_interval
        Discrete serial interval ``w``, stored from lag 1 (see
        :mod:`end_of_outbreak.delay_distributions`).
    switch_day
        Day index on which ``R`` changes from ``R_pre`` to ``R_post``.
    R_pre, R_post, k
        Each either a fixed positive ``float`` or a
        :class:`~end_of_outbreak.model_specifications.LogNormalPrior` to estimate. ``k`` must
        be ``None`` for ``cori`` and given otherwise.

    Returns
    -------
    A :class:`pymc.Model` whose observed node is named
    :data:`OBSERVED_VARIABLE` and whose SSI latent block, when present, is named
    :data:`INFECTIVITY_VARIABLE` over the :data:`COHORT_DAY_DIMENSION` coordinate.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise NotImplementedError(
            f"model {specification.name!r} is onset-anchored; build_naive_model handles the "
            "infection-anchored models only (the onset-anchored builders land in Stage 8)"
        )
    counts = np.asarray(counts, dtype=np.int64)
    serial_interval = np.asarray(serial_interval, dtype=np.float64)
    _validate_inputs(specification, counts, serial_interval, k)

    n_days = counts.size
    cohort_days = np.flatnonzero(counts > 0).astype(np.int64)

    if specification.latent_variable is None:
        # The force of infection is a known constant: the driving series is the data itself.
        force_of_infection = renewal.delay_weighted_sum(
            counts.astype(np.float64), serial_interval, first_lag=SERIAL_INTERVAL_FIRST_LAG
        )
        design = None
    else:
        # The driving series is latent, so the operator has to stay symbolic. Its row sums
        # tell us which days can be driven at all, which is what selects the likelihood days.
        design = renewal.delay_design_matrix(
            n_days,
            serial_interval,
            first_lag=SERIAL_INTERVAL_FIRST_LAG,
            source_days=cohort_days,
        )
        force_of_infection = design.sum(axis=1)

    days = renewal.likelihood_days(force_of_infection, counts)
    coords = {
        LIKELIHOOD_DAY_DIMENSION: days,
        COHORT_DAY_DIMENSION: cohort_days,
    }

    with pm.Model(coords=coords) as built:
        R_pre_value = _parameter("R_pre", R_pre)
        R_post_value = _parameter("R_post", R_post)
        k_value = None if k is None else _parameter("k", k)
        R_by_day = _reproduction_number(
            R_pre_value, R_post_value, days=days, n_days=n_days, switch_day=switch_day
        )
        observed = counts[days]

        if specification.name == "ssi":
            assert design is not None  # narrows the type; guaranteed by latent_variable
            Y = _infectivity_latents(k_value, counts[cohort_days])
            mean_incidence = R_by_day * pt.dot(design[days], Y)
            pm.Poisson(
                OBSERVED_VARIABLE,
                mu=mean_incidence,
                observed=observed,
                dims=LIKELIHOOD_DAY_DIMENSION,
            )
        elif k_value is None:  # cori, the k → ∞ limit
            pm.Poisson(
                OBSERVED_VARIABLE,
                mu=R_by_day * force_of_infection[days],
                observed=observed,
                dims=LIKELIHOOD_DAY_DIMENSION,
            )
        else:
            # DLO holds the dispersion constant across days; SSE scales it with the force of
            # infection. That one difference is the whole of §5.4.
            dispersion = (
                k_value if specification.name == "dlo" else k_value * force_of_infection[days]
            )
            pm.NegativeBinomial(
                OBSERVED_VARIABLE,
                mu=R_by_day * force_of_infection[days],
                alpha=dispersion,
                observed=observed,
                dims=LIKELIHOOD_DAY_DIMENSION,
            )

    return built


def model_days(model: pm.Model, dimension: str) -> NDArray[np.int64]:
    """The day indices behind one of a built model's coordinates.

    ``model_days(model, COHORT_DAY_DIMENSION)`` gives the days carrying an SSI latent, in the
    order the ``Y`` vector uses — which is what downstream code needs to align posterior draws
    of ``Y`` back onto the calendar.
    """
    coordinate = model.coords.get(dimension)
    if coordinate is None:
        raise KeyError(f"model has no coordinate {dimension!r}")
    return np.asarray(coordinate, dtype=np.int64)


# ---------------------------------------------------------------------------------------
# Evaluating a model's joint density at given values
# ---------------------------------------------------------------------------------------


def compile_joint_logp(model: pm.Model) -> Callable[..., float]:
    """Compile ``log p(observed counts, latents, parameters)`` as a function of named values.

    The returned callable takes one keyword argument per free random variable, named as in
    the model (``R_pre``, ``k``, ``Y``, ...), with values on their **natural** scale — no
    log transforms, no Jacobian. That makes it directly comparable with a density written out
    by hand, which is how the tests check the likelihoods, and is the form the Stage-6
    evidence estimators consume.

    Compiling is the expensive part, so build the callable once and call it many times.
    """
    logp = model.compile_logp(jacobian=False)
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

    def joint_logp(**values: Any) -> float:
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

    return joint_logp


def joint_logp(model: pm.Model, **values: Any) -> float:
    """One-shot :func:`compile_joint_logp`. Compiles on every call — use it sparingly."""
    return compile_joint_logp(model)(**values)
