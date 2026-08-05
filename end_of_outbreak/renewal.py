"""Shared renewal-equation machinery: delay convolutions and the ``R`` switch.

Every model in the project drives day ``t`` from a delay-weighted sum over earlier days,

    Λ_t = Σ_s weights[s - first_lag] · values_{t-s},

differing only in what ``values`` and ``weights`` are:

===========  ==============================  ==========================  ===========
Model        ``values``                      ``weights``                 ``first_lag``
===========  ==============================  ==========================  ===========
DLO, SSE     observed counts ``I``           serial interval ``w``       1
SSI          latent infectivities ``Y``      serial interval ``w``       1
SSE-SO       observed onsets ``D``           ``f_tost``                  0
SSI-SO       latent infectivities ``Y``      ``f_tost``                  0
all SO       expected infections ``E``       ``f_inc``                   1
===========  ==============================  ==========================  ===========

Two equivalent evaluations are provided, because the models need different ones:

:func:`delay_weighted_sum`
    A dense convolution. Use it when ``values`` is *observed data*, so the whole sum is a
    known constant — DLO and SSE.
:func:`delay_design_matrix`
    The same operator as an explicit matrix ``M``, so that ``Λ = M @ values``. Use it when
    ``values`` is a *latent vector*, so the sum must be expressed symbolically — SSI, and the
    onset-anchored models of Stage 8. Restricting its columns to the days that actually carry
    a latent keeps the matrix narrow (31 columns rather than 111 for SSI).

The two agree exactly; :mod:`tests.test_renewal` checks that they do.

The ``R`` switch lives here too (:func:`switch_index`), so the convention of §5.7 of the
implementation plan — ``R`` changes from ``R_pre`` to ``R_post`` on the ERT arrival day, in
each model's *own* time index — is written down in exactly one place.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def delay_weighted_sum(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    first_lag: int,
    n_days: int | None = None,
) -> NDArray[np.float64]:
    """``Λ_t = Σ_s weights[s - first_lag] · values_{t-s}`` for every day of the window.

    Parameters
    ----------
    values
        The driving series, indexed by day from day 0.
    weights
        Delay weights stored densely from ``first_lag`` (see
        :mod:`end_of_outbreak.delay_distributions`).
    first_lag
        Lag of ``weights[0]``. ``1`` for the serial interval and the incubation period,
        ``0`` for the TOST — a case may transmit on its own onset day.
    n_days
        Length of the returned array; defaults to ``len(values)``. Pass a larger value to
        project past the end of ``values`` (the trailing days are driven purely by the tail
        of ``weights``).

    Returns
    -------
    ``Λ`` of length ``n_days``. Lags reaching before day 0 contribute nothing, so ``Λ_t``
    for small ``t`` is a partial sum — that is the sole-index-case initial condition of §3.1,
    not a truncation artefact.
    """
    if first_lag < 0:
        raise ValueError(f"first_lag must be non-negative, got {first_lag}")
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("values and weights must both be one-dimensional")

    length = values.size if n_days is None else n_days
    convolution = np.convolve(values, weights)
    padded = np.concatenate((np.zeros(first_lag, dtype=np.float64), convolution))
    result = np.zeros(length, dtype=np.float64)
    result[: min(length, padded.size)] = padded[:length]
    return result


def delay_design_matrix(
    n_days: int,
    weights: NDArray[np.float64],
    *,
    first_lag: int,
    source_days: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """The operator of :func:`delay_weighted_sum` as a matrix, for latent driving series.

    ``M[t, j] = weights[t - source_days[j] - first_lag]`` where that index falls inside the
    support, and 0 elsewhere — so ``M @ values[source_days] == delay_weighted_sum(...)``
    whenever ``values`` vanishes off ``source_days``.

    Parameters
    ----------
    n_days
        Number of rows, i.e. days in the analysis window.
    weights, first_lag
        As in :func:`delay_weighted_sum`.
    source_days
        Days carrying a driving value, i.e. the columns to keep. Defaults to every day.
        For SSI these are the days with ``I_t > 0``, the only days with a latent ``Y_t``.

    Returns
    -------
    ``(n_days, len(source_days))`` matrix.
    """
    if first_lag < 0:
        raise ValueError(f"first_lag must be non-negative, got {first_lag}")
    if n_days < 0:
        raise ValueError(f"n_days must be non-negative, got {n_days}")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    columns = (
        np.arange(n_days, dtype=np.int64)
        if source_days is None
        else np.asarray(source_days, dtype=np.int64)
    )

    lags = np.arange(n_days, dtype=np.int64)[:, None] - columns[None, :] - first_lag
    inside = (lags >= 0) & (lags < weights.size)
    return np.where(inside, weights[np.clip(lags, 0, weights.size - 1)], 0.0)


def switch_index(n_days: int, switch_day: int) -> NDArray[np.int64]:
    """Index into ``(R_pre, R_post)`` for each day: 0 before ``switch_day``, 1 from it on.

    Intended as ``R_by_day = np.stack([R_pre, R_post])[switch_index(...)]`` (or its PyTensor
    equivalent), so that the §5.7 convention lives in one place. ``R`` switches on the ERT
    arrival day in *each model's own* time index: for the naive models that indexes the day a
    case appears, for the onset-anchored models the day a transmission occurs. The resulting
    ~11-day (mean incubation period) asymmetry between the two is a deliberate, reportable
    result.
    """
    if n_days < 0:
        raise ValueError(f"n_days must be non-negative, got {n_days}")
    return (np.arange(n_days, dtype=np.int64) >= switch_day).astype(np.int64)


def reproduction_number_by_day(
    R_pre: float, R_post: float, *, n_days: int, switch_day: int
) -> NDArray[np.float64]:
    """``R_t`` for every day of the window, under the :func:`switch_index` convention."""
    return np.asarray([R_pre, R_post], dtype=np.float64)[switch_index(n_days, switch_day)]


def likelihood_days(
    force_of_infection: NDArray[np.float64],
    counts: NDArray[np.int64],
    *,
    first_day: int = 1,
) -> NDArray[np.int64]:
    """Days that contribute a non-degenerate term to the likelihood.

    A day whose force of infection is zero has an expected count of zero under every model
    here, so its observation is a point mass at 0 and contributes nothing. Such days are
    dropped rather than evaluated, which keeps ``NegativeBinomial(alpha=0)`` and
    ``Poisson(mu=0)`` out of the graph.

    Day 0 is an initial condition in every model (§3.1) and so is excluded by ``first_day``.

    Raises
    ------
    ValueError
        If a day with zero force of infection carries a positive count — the observation is
        then impossible under the model, and silently dropping it would hide that.
    """
    force_of_infection = np.asarray(force_of_infection, dtype=np.float64)
    counts = np.asarray(counts)
    if force_of_infection.size != counts.size:
        raise ValueError("force_of_infection and counts must have the same length")

    days = np.arange(counts.size, dtype=np.int64)
    driven = force_of_infection > 0.0
    impossible = (~driven) & (counts > 0) & (days >= first_day)
    if impossible.any():
        raise ValueError(
            f"days {days[impossible].tolist()} carry positive counts but zero force of "
            "infection, which is impossible under the model; check the initial condition "
            "and the serial-interval support"
        )
    return days[driven & (days >= first_day)]
