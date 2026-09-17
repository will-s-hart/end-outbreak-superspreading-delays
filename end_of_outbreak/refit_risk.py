"""RAC and RAT by refitting the model to every conditioning day — the gold standard.

The estimand is real-time (:mod:`end_of_outbreak.risk_of_additional_cases`): RAC(t) conditions
on the record through day ``t`` and nothing after it. So the honest estimator fits the model to
``counts[:t + 1]`` and evaluates the closed form at day ``t``, once per day of the curve.

Why the whole fit and not just the state
----------------------------------------
It is tempting to keep one full-record parameter posterior and re-derive only the latents per
day, which is what :mod:`end_of_outbreak.filtered_risk` does. That is a different quantity, not a
cheaper route to this one: ``R`` and the latent infectivities are correlated in the posterior, so
conditioning the latents on less data while leaving ``R`` conditioned on all of it is not the
day-``t`` posterior of either. The comparison between the two is worth making — it is the
approximation the project used to rely on — but it has to be reported as a comparison.

What this module does *not* own
-------------------------------
Any mathematics. It loops days, calls :func:`end_of_outbreak.fitting.fit_model` on a truncated
series, and hands the draws to
:func:`end_of_outbreak.risk_of_additional_cases.risk_log_probabilities`. The closed forms are
unchanged by real-time conditioning; only the posterior the state is drawn from moves. Two
consequences worth knowing:

- ``Λ(t)``, ``W(t)`` and ``M(t)`` computed from ``counts[:t + 1]`` equal those computed from the
  whole series, because each sums only over days ``u ≤ t``. Truncating the input is therefore
  exactly the change of conditioning and nothing else.
- SSE-SO's boundary latent moves with the window. On a day-``t`` fit it sits on day ``t``, and
  its conditional law is its prior — which is right, because no observation constrains it.
  :func:`~end_of_outbreak.risk_of_additional_cases.unsampled_latent_conditional` already handles
  it wherever the window ends.

A hundred and ten fits per model is a hundred and ten chances for one of them to go quietly
wrong, so every day's sampler diagnostics come back with the curve
(:class:`DayDiagnostics`) and the caller is expected to act on them.

Where the time goes
-------------------
This is the most expensive step in the project by a wide margin, and the days are independent
by construction — day ``t``'s fit sees ``counts[:t + 1]`` and a seed of its own, and nothing
else. So it is also the natural place to spend cores: ``n_jobs`` runs the days across worker
processes (:mod:`end_of_outbreak.parallel`), which is why
:func:`end_of_outbreak.fitting.fit_model` leaves the chains of a single fit sequential. The
curve does not depend on ``n_jobs`` — the per-day seeds were fixed before any of them ran.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from end_of_outbreak import fitting, parallel, pymc_models, reporting
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.latent_parameterisations import LatentParameterisation
from end_of_outbreak.model_specifications import (
    LogNormalPrior,
    ModelSpecification,
    specification_of,
)

FIRST_CONDITIONING_DAY = 1
"""Day 0 is an initial condition in every model (§3.1), so it contributes no likelihood term.

A "fit to the record through day 0" is therefore the prior, and the curve starts at day 1. RAC(0)
is one to within rounding on any series this project fits, so nothing is lost.
"""

type CountSnapshots = tuple[NDArray[np.int64], ...]
"""Historical reported-count snapshots, each running from day 0 through its as-of day."""


@dataclass(frozen=True)
class DayDiagnostics:
    """How one conditioning day's fit behaved."""

    day: int
    divergences: int
    max_r_hat: float
    min_ess_bulk: float
    seconds: float

    def is_suspect(self, *, max_r_hat: float, divergence_fraction: float, n_draws: int) -> bool:
        """Whether this day's fit fails the acceptance thresholds."""
        return (
            not np.isfinite(self.max_r_hat)
            or self.max_r_hat > max_r_hat
            or self.divergences > divergence_fraction * n_draws
        )


@dataclass(frozen=True)
class DayResult:
    """One conditioning day's contribution to the curve, as a worker hands it back.

    Small on purpose. The fit itself stays in the worker that produced it: what the curve needs
    is one day of per-draw log-probabilities and the diagnostics, and shipping a whole
    ``DataTree`` back per day would cost more than the fit.
    """

    day: int
    estimate: rac.DailyRiskEstimate
    diagnostics: DayDiagnostics


@dataclass(frozen=True)
class RefitRiskResult:
    """The curve and the evidence that the fits behind it converged."""

    estimate: rac.DailyRiskEstimate
    diagnostics: list[DayDiagnostics]

    def suspect_days(
        self, *, max_r_hat: float = 1.02, divergence_fraction: float = 0.01
    ) -> list[DayDiagnostics]:
        """Days whose fit should be looked at before the curve is believed."""
        n_draws = self.estimate.n_draws
        return [
            day
            for day in self.diagnostics
            if day.is_suspect(
                max_r_hat=max_r_hat, divergence_fraction=divergence_fraction, n_draws=n_draws
            )
        ]


def conditioning_days(n_days: int, *, first_day: int = FIRST_CONDITIONING_DAY) -> NDArray[np.int64]:
    """Days a refit curve is evaluated on: ``first_day`` through the last day of the window."""
    if not FIRST_CONDITIONING_DAY <= first_day < n_days:
        raise ValueError(
            f"first_day must lie in {FIRST_CONDITIONING_DAY}..{n_days - 1}, got {first_day}; "
            "day 0 is an initial condition and carries no likelihood term"
        )
    return np.arange(first_day, n_days, dtype=np.int64)


def risk_by_refitting(
    model: str | ModelSpecification,
    counts: NDArray[np.int64] | CountSnapshots,
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None = None,
    latent_parameterisation: str | LatentParameterisation | None = None,
    negligible_latent_threshold: float = 0.0,
    reporting_model: reporting.ReportingModel | None = None,
    days: Sequence[int] | NDArray[np.int64] | None = None,
    sampler: fitting.SamplerSettings | None = None,
    seed_for_day: Callable[[int], int | None] | None = None,
    final_day_fit: xr.DataTree | None = None,
    on_day: Callable[[DayDiagnostics], None] | None = None,
    n_jobs: int = 1,
) -> RefitRiskResult:
    """RAC and RAT over the conditioning days, one fit per day.

    Parameters
    ----------
    model, counts, delays, switch_day, R_pre, R_post, k, latent_parameterisation,
    negligible_latent_threshold, reporting_model
        As for :func:`end_of_outbreak.fitting.fit_model`, which each day's fit is handed.
        ``counts`` may instead be a tuple of historical reported-count snapshots. Each starts
        on day 0 and ends on its own as-of day; lengths must increase strictly but may skip days,
        and the curve is evaluated exactly on those days. This form is required for reporting
        delays because a final series cannot reconstruct what had been reported earlier.
    days
        Conditioning days; :func:`conditioning_days` over the whole window by default. Rejected
        with a snapshot tuple, whose lengths define the days unambiguously.
    sampler
        NUTS settings, applied to every day. ``seed_for_day`` overrides the seed.
    seed_for_day
        Per-day sampler seed. A single seed reused across days would correlate the curve's
        Monte-Carlo error from day to day, which is exactly what
        :func:`~end_of_outbreak.risk_of_additional_cases.monte_carlo_standard_error` is there to
        measure.
    final_day_fit
        A fit already run on the whole window, reused for the last conditioning day instead of
        repeating it. The pipeline always has one, because the evidence and dispersion steps need
        it, and reusing it also guarantees that the end of the curve and those summaries describe
        the same posterior.
    on_day
        Called with each day's diagnostics as it completes, for progress reporting. Under
        ``n_jobs > 1`` the days complete out of order, so it is called out of order too.
    n_jobs
        Worker processes to spread the days over; one, meaning this process, by default. The
        curve is the same either way — every day carries its own seed — so this is a throughput
        setting and never a modelling one. See :mod:`end_of_outbreak.parallel`.
    """
    specification = specification_of(model)
    resolved_reporting = (
        reporting.COMPLETE_REPORTING if reporting_model is None else reporting_model
    )
    snapshots = _resolve_snapshots(counts, days=days, reporting_model=resolved_reporting)
    selected = np.asarray([day for day, _ in snapshots], dtype=np.int64)
    settings = fitting.SamplerSettings() if sampler is None else sampler

    # The last day of the window is the whole-record fit, which the pipeline already holds. It
    # is evaluated here rather than in a worker: shipping a `DataTree` into a worker process
    # just to skip the sampling it stands in for would cost more than it saves.
    final_snapshot_day = (
        int(np.asarray(counts[-1]).size - 1)
        if isinstance(counts, tuple)
        else int(np.asarray(counts).size - 1)
    )
    reused_day = (
        final_snapshot_day
        if final_day_fit is not None and final_snapshot_day in set(selected.tolist())
        else None
    )
    reused_snapshot = (
        None
        if reused_day is None
        else next(window for day, window in snapshots if day == reused_day)
    )
    if final_day_fit is not None and reused_day is not None:
        assert reused_snapshot is not None
        _validate_reused_fit(
            final_day_fit,
            specification=specification,
            snapshot=reused_snapshot,
            day=reused_day,
            switch_day=switch_day,
            reporting_model=resolved_reporting,
        )

    def contribution(
        day: int, window: NDArray[np.int64], idata: xr.DataTree, started: float
    ) -> DayResult:
        """One day's per-draw log-probabilities and diagnostics, given its fit."""
        # Every parameter the analysis fixed is a constant in the graph and so is nowhere in
        # the draws. The fit records what it was held at; read it back rather than assuming it
        # was estimated, or `posterior_state` cannot find it at all.
        fixed_R_pre, fixed_R_post = fitting.fitted_reproduction_numbers(idata)
        state = rac.posterior_state(
            specification,
            idata,
            window,
            delays=delays,
            switch_day=switch_day,
            latent_parameterisation=fitting.fitted_parameterisation(idata),
            fixed_R_pre=fixed_R_pre,
            fixed_R_post=fixed_R_post,
            fixed_k=fitting.fitted_dispersion(idata),
            negligible_latent_threshold=negligible_latent_threshold,
            reporting_model=fitting.fitted_reporting(idata),
        )
        return DayResult(
            day=day,
            estimate=rac.risk_log_probabilities(
                specification,
                state,
                counts=window,
                delays=delays,
                switch_day=switch_day,
                days=np.array([day], dtype=np.int64),
            ),
            diagnostics=summarise_fit(idata, day=day, seconds=time.perf_counter() - started),
        )

    def evaluate(task: tuple[int, NDArray[np.int64]]) -> DayResult:
        """Fit the window ending on ``day`` and take its contribution to the curve.

        This is what crosses into a worker process, so it must not close over ``final_day_fit``:
        a closure is pickled once per task, and the whole-record fit is tens of megabytes that
        no worker ever reads. Hence the separate ``reuse`` path below.
        """
        day, window = task
        started = time.perf_counter()
        idata = fitting.fit_model(
            specification,
            window,
            delays=delays,
            switch_day=switch_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=latent_parameterisation,
            negligible_latent_threshold=negligible_latent_threshold,
            reporting_model=resolved_reporting,
            # The window includes its conditioning day, so day t is the day it was observed on.
            # Stated here rather than left to fit_model's default, because this is the one place
            # that knows the two coincide.
            as_of_day=day,
            sampler=(
                settings if seed_for_day is None else replace_seed(settings, seed_for_day(day))
            ),
        )
        return contribution(day, window, idata, started)

    snapshots_to_fit = [(day, window) for day, window in snapshots if day != reused_day]
    results = parallel.map_fits(
        evaluate,
        snapshots_to_fit,
        n_jobs=n_jobs,
        description=f"{specification.name} conditioning days",
        on_result=None if on_day is None else (lambda result: on_day(result.diagnostics)),
    )
    if final_day_fit is not None and reused_day in {int(day) for day in selected}:
        assert reused_snapshot is not None
        reused = contribution(int(reused_day), reused_snapshot, final_day_fit, time.perf_counter())
        if on_day is not None:
            on_day(reused.diagnostics)
        results.append(reused)

    # `map_fits` returns results as they finish, so the curve has to be put back in day order
    # before anything reads it as a time series.
    results.sort(key=lambda result: result.day)
    return RefitRiskResult(
        estimate=rac.DailyRiskEstimate.concatenate([result.estimate for result in results]),
        diagnostics=[result.diagnostics for result in results],
    )


def _resolve_snapshots(
    counts: NDArray[np.int64] | CountSnapshots,
    *,
    days: Sequence[int] | NDArray[np.int64] | None,
    reporting_model: reporting.ReportingModel,
) -> list[tuple[int, NDArray[np.int64]]]:
    """Normalise a final series or sparse historical snapshots into ``(day, series)`` tasks."""
    if isinstance(counts, tuple):
        if days is not None:
            raise ValueError(
                "days cannot be supplied with historical snapshots; lengths define them"
            )
        if not counts:
            raise ValueError("at least one historical count snapshot is required")
        snapshots = [_validated_snapshot(snapshot) for snapshot in counts]
        lengths = np.asarray([snapshot.size for snapshot in snapshots], dtype=np.int64)
        if np.any(np.diff(lengths) <= 0):
            raise ValueError("historical snapshot lengths must increase strictly")
        return [(int(snapshot.size - 1), snapshot) for snapshot in snapshots]

    series = _validated_snapshot(counts)
    if reporting_model.delay is not None:
        raise ValueError(
            "a real-time curve with a reporting delay requires a tuple of historical count "
            "snapshots; one final series cannot reconstruct earlier reporting states"
        )
    selected = conditioning_days(series.size) if days is None else np.asarray(days, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("no conditioning days to evaluate")
    if selected.min() < FIRST_CONDITIONING_DAY or selected.max() >= series.size:
        raise ValueError(
            f"conditioning days must lie in {FIRST_CONDITIONING_DAY}..{series.size - 1}"
        )
    return [(int(day), series[: int(day) + 1]) for day in selected]


def _validated_snapshot(snapshot: Any) -> NDArray[np.int64]:
    """One reported-count snapshot, preserving corrections between separate snapshots."""
    values = np.asarray(snapshot)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("each count snapshot must be one-dimensional with at least two days")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("count snapshots must contain integers")
    counts = values.astype(np.int64, copy=False)
    if counts[0] < 1 or np.any(counts < 0):
        raise ValueError("each count snapshot must start with a case and be non-negative")
    return counts


def _validate_reused_fit(
    idata: xr.DataTree,
    *,
    specification: ModelSpecification,
    snapshot: NDArray[np.int64],
    day: int,
    switch_day: int,
    reporting_model: reporting.ReportingModel,
) -> None:
    """Ensure a supplied final-day fit describes exactly the snapshot it would replace."""
    fitted_model = str(idata.attrs.get(fitting.MODEL_ATTRIBUTE, ""))
    if fitted_model != specification.name:
        raise ValueError(
            f"final_day_fit records model {fitted_model!r}, expected {specification.name!r}"
        )
    if fitting.fitted_reporting(idata) != reporting_model:
        raise ValueError("final_day_fit's reporting assumption does not match this curve")
    fitted_switch = fitting.fitted_switch_day(idata)
    if fitted_switch is not None and fitted_switch != switch_day:
        raise ValueError(
            f"final_day_fit switched R on day {fitted_switch}, but this curve switches on day "
            f"{switch_day}. The two describe different models, and reusing the fit would put "
            "one convention's posterior on the other's last conditioning day"
        )
    if fitting.fitted_as_of_day(idata, snapshot) != day:
        raise ValueError("final_day_fit's as-of day does not match the final snapshot")
    if pymc_models.OBSERVED_VARIABLE not in idata.observed_data:
        raise ValueError("final_day_fit does not contain its reported-count observations")
    fitted_counts = fitting.fitted_reported_counts(idata)
    if fitted_counts is not None:
        matches = np.array_equal(fitted_counts, snapshot)
    else:
        observed_variable = idata.observed_data[pymc_models.OBSERVED_VARIABLE]
        fitted_days = np.asarray(
            observed_variable.coords[pymc_models.LIKELIHOOD_DAY_DIMENSION], dtype=np.int64
        )
        observed = np.asarray(observed_variable, dtype=np.int64)
        matches = np.array_equal(observed, snapshot[fitted_days])
    if not matches:
        raise ValueError("final_day_fit's reported counts do not match the final snapshot")


def replace_seed(settings: fitting.SamplerSettings, seed: int | None) -> fitting.SamplerSettings:
    """The same sampler settings under a different seed."""
    return fitting.SamplerSettings(
        draws=settings.draws,
        tune=settings.tune,
        chains=settings.chains,
        target_accept=settings.target_accept,
        seed=seed,
    )


def summarise_fit(idata: Any, *, day: int, seconds: float) -> DayDiagnostics:
    """Convergence diagnostics for one day's fit, over every variable it sampled.

    The worst ``R̂`` and the smallest bulk ESS across the whole posterior, not a chosen
    parameter: the latent block is where a badly conditioned day would show up first, and it is
    the part nobody looks at.
    """
    divergences = 0
    if "sample_stats" in idata and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    if not idata.posterior.data_vars:
        # A fit with every parameter fixed and no latent block sampled nothing, so there is no
        # convergence to report. `az.rhat` raises on an empty posterior rather than returning
        # an empty result, so this is a guard and not merely a shortcut. `nan` compares false
        # against every threshold, which is what lets the acceptance gate pass it unremarked.
        return DayDiagnostics(
            day=int(day),
            divergences=divergences,
            max_r_hat=float("nan"),
            min_ess_bulk=float("nan"),
            seconds=float(seconds),
        )
    return DayDiagnostics(
        day=int(day),
        divergences=divergences,
        max_r_hat=_worst(az.rhat(idata.posterior), np.nanmax),
        min_ess_bulk=_worst(az.ess(idata.posterior, method="bulk"), np.nanmin),
        seconds=float(seconds),
    )


def _worst(summary: xr.Dataset, reduce: Any) -> float:
    """The extreme value of a diagnostic across every variable and coordinate."""
    values = [np.asarray(summary[name], dtype=np.float64).ravel() for name in summary.data_vars]
    if not values:
        return float("nan")
    return float(reduce(np.concatenate(values)))
