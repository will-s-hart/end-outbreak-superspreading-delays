"""Every number the report quotes, collected from ``results/`` into LaTeX macros.

The report is prose, and prose cannot be re-run. Left to itself a ``.tex`` file accumulates
numbers that were true of the fits that existed when the sentence was written, and nothing
downstream ever notices when a refit moves one of them — which is the same failure mode
``scripts/utils.py`` guards the figures against, in the one place where it would be least
visible. So the report quotes **no literal number of its own**: it writes
``\\resultnum{onsetmodelsestimatedk.sseso.rac05.day}`` and this script writes what that expands
to, from the files the tier-2 rules produced.

Two consequences worth keeping:

- **A key that does not exist is a compile error**, not a blank. ``\\resultnum`` is defined in
  ``report/report.tex`` to raise a LaTeX error for an unknown key, so a renamed model or a
  deleted analysis stops the build instead of quietly printing nothing.
- **This is a tier-2 script, not a tier-3 one.** It reads posteriors and summarises them, so it
  writes to ``results/`` and the ``report`` rule (tier 3) reads only what it wrote. Restyling
  the report's prose never re-runs a fit; refitting does re-run this.

What it does *not* do is compute anything the pipeline does not already compute. Posterior
summaries come from :mod:`end_of_outbreak.posterior_comparison`, exactly as
``dispersion_posteriors.json`` does; threshold crossings come from
:meth:`~end_of_outbreak.risk_curves.RiskCurve.first_day_below`, exactly as the
markers on the figures do. The three cross-analysis quantities it forms — the crossing shift
between an onset-anchored model and its naive counterpart, the log-evidence gain from fixing
``k`` to estimating it, and the largest RAC/RAT gap — are subtractions of numbers already in
``results/``, done here so that no sentence in the report has to do arithmetic. RST receives the
same reference-day, final-day, threshold-crossing and Monte-Carlo-error keys wherever its column
is present; DLO receives none.

Keys are ``.``-separated and carry no underscores, since they are expanded through
``\\csname``: an analysis or model name is slugged by dropping everything but its letters and
digits (``onset_models_estimated_k`` → ``onsetmodelsestimatedk``, ``sse_so`` → ``sseso``).
The mapping is mechanical rather than a table, so it cannot fall out of date.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from end_of_outbreak import (
    configuration,
    delay_distributions,
    fitting,
    outbreak_data,
    pymc_models,
    risk_curves,
)
from end_of_outbreak import posterior_comparison as pc
from end_of_outbreak.model_specifications import LogNormalPrior, specification_of

RISK_COLUMN = "risk_of_additional_cases"
TRANSMISSION_RISK_COLUMN = "risk_of_additional_transmission"
SUSTAINED_RISK_COLUMN = "risk_of_sustained_transmission"
STANDARD_ERROR_COLUMN = "monte_carlo_standard_error"

THRESHOLDS: tuple[float, ...] = (0.05, 0.01)
"""The two RAC levels the end-of-outbreak literature declares on, as the figures draw them."""

REPRODUCTION_NUMBERS: tuple[str, ...] = ("R_pre", "R_post")
DISPERSION = "k"

UNDETERMINED = "undetermined"
"""A difference between two crossings when neither curve settles inside the window.

A curve still above a threshold on the last day has not settled *yet*, which bounds its crossing
from below and nothing more — so a lone unsettled curve is written as a bound (see
:func:`crossing_day_text`), and a difference involving one as a one-sided bound. Two unsettled
curves bound nothing about the difference between them, and saying so is the only honest value.
"""

COMBINED_EVIDENCE_FILE = "combined_model_evidence.json"
"""Probabilities over an analysis's models and those it borrows; ``run_combined_evidence.py``."""

REFERENCE_DAY = 90
"""One mid-descent day the report compares every model on.

The crossings say when each curve settles; they do not say how far apart the curves are while
they are still falling, and on this series the models are ordered differently at different
heights. Day 90 is the day §5.4 of the implementation plan works its SSE/SSI/DLO decomposition
on, so quoting the fitted curves there keeps the report's arithmetic comparable with it.
"""


# ---------------------------------------------------------------------------------------
# The macro file
# ---------------------------------------------------------------------------------------


class NumberFile:
    """Accumulates ``\\defresultnum{key}{value}`` lines, refusing to define a key twice."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        if key in self._entries:
            raise ValueError(f"{key!r} defined twice, with {self._entries[key]!r} and {value!r}")
        self._entries[key] = value

    def __len__(self) -> int:
        return len(self._entries)

    def render(self, *, sources: list[str]) -> str:
        """The file to write, sorted by key so a diff shows what moved."""
        provenance = "\n".join(f"%   {source}" for source in sources)
        body = "\n".join(
            f"\\defresultnum{{{key}}}{{{self._entries[key]}}}" for key in sorted(self._entries)
        )
        return (
            "% Generated by scripts/run_report_numbers.py -- do not edit.\n"
            "%\n"
            "% Every number report/report.tex quotes, keyed for \\resultnum. Read from:\n"
            f"{provenance}\n"
            "%\n"
            f"% {len(self._entries)} entries.\n"
            f"{body}\n"
        )


def slug(name: str) -> str:
    """An analysis or model name as it appears in a key: letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def repository_path(path: Path) -> str:
    """``path`` relative to the repository root where it is inside it, so the header commits well.

    The generated file is committed, so an absolute path in it would differ between machines and
    show up as a spurious diff on every regeneration.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(configuration.REPO_ROOT))
    except ValueError:
        return str(resolved)


# --- formatting ------------------------------------------------------------------------
#
# Rounding is a presentation decision and so it belongs here rather than in the report, where
# the same quantity could end up quoted to three decimal places in one section and two in
# another. One helper per kind of quantity, used everywhere that quantity appears.


def fixed(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def trimmed(value: float, decimals: int) -> str:
    """``fixed``, with trailing zeros dropped: 11.40 becomes 11.4 and 4.57 stays 4.57.

    Used for the delay distributions, which are quoted in running prose as ``a mean of 11.4
    days`` rather than in a column of a table. A fixed number of decimals would either write
    ``11.40`` there or lose the second digit of the TOST standard deviation.
    """
    return fixed(value, decimals).rstrip("0").rstrip(".")


def interval(summary: pc.PosteriorSummary, decimals: int) -> str:
    """``median (lower--upper)``, the form the figures' legends use."""
    return (
        f"{summary.median:.{decimals}f} "
        f"({summary.interval_lower:.{decimals}f}--{summary.interval_upper:.{decimals}f})"
    )


def date_text(date: datetime.date) -> str:
    """``5 April 2018`` — no leading zero, since these are read as prose."""
    return f"{date.day} {date:%B} {date.year}"


def date_of_day(data: outbreak_data.OutbreakData, day: int) -> datetime.date:
    """The calendar date of a day index, including one just past the end of the window."""
    return data.date_of(0) + datetime.timedelta(days=int(day))


# --- threshold crossings -----------------------------------------------------------------
#
# A crossing that has not happened by the last day of the window is not "never": the curve is
# still falling, and all the window says is that it settles later than that. So it is written as
# the bound it is -- a day of at least one past the end, a date of "not before" the day after --
# and a difference between two crossings inherits a one-sided bound from it.


def crossing_day_text(day: int | None, *, last_day: int) -> str:
    """A crossing day, or ``≥ last_day + 1`` for a curve that had not settled by the end."""
    return str(day) if day is not None else math_mode(f"\\geq {last_day + 1}")


def crossing_date_text(day: int | None, data: outbreak_data.OutbreakData, *, last_day: int) -> str:
    """A crossing date, or ``not before`` the day after the window for an unsettled curve."""
    if day is not None:
        return date_text(date_of_day(data, day))
    return f"not before {date_text(date_of_day(data, last_day + 1))}"


def crossing_difference(later: int | None, earlier: int | None, *, last_day: int) -> str:
    """``later - earlier`` in days, where ``None`` is a curve still above on ``last_day``.

    An unsettled curve crosses on day ``last_day + 1`` at the earliest, which turns the
    difference into a bound on one side: at least so many days if ``later`` is the unsettled
    one, at most so many if ``earlier`` is. If both are, nothing is bounded.
    """
    if later is not None and earlier is not None:
        return str(later - earlier)
    if later is None and earlier is not None:
        return math_mode(f"\\geq {last_day + 1 - earlier}")
    if later is not None and earlier is None:
        return math_mode(f"\\leq {later - (last_day + 1)}")
    return UNDETERMINED


def small_probability(probability: float) -> str:
    """A posterior model probability, rendered so that "essentially zero" stays readable.

    Matches :func:`scripts.utils._probability_text`, which labels the same numbers on the pie:
    the report and the panel must not disagree about how small "small" is.
    """
    if probability >= 0.01:
        return f"{probability:.2f}"
    if probability == 0.0:
        return "\\ensuremath{<10^{-300}}"
    exponent = int(np.floor(np.log10(probability)))
    mantissa = probability / 10.0**exponent
    return f"\\ensuremath{{{mantissa:.0f}\\times 10^{{{exponent}}}}}"


def signed(value: float, decimals: int) -> str:
    """A difference, with its sign always shown; ``+0`` and ``-0`` are both written ``0``."""
    rounded = round(value, decimals)
    if rounded == 0:
        return math_mode(f"{0:.{decimals}f}")
    return math_mode(f"{rounded:+.{decimals}f}")


def math_mode(text: str) -> str:
    """Wrap a value that carries a sign, so that TeX sets a minus rather than a hyphen.

    A bare ``-97.1`` in running text renders as a hyphen, which is visibly too short and reads
    as a dash. Every quantity that can be negative goes through here.
    """
    return f"\\ensuremath{{{text}}}"


# ---------------------------------------------------------------------------------------
# Reading what the tier-2 rules wrote
# ---------------------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Load a tier-2 summary, failing loudly rather than letting the report print a blank."""
    if not path.exists():
        raise FileNotFoundError(
            f"no results file at {path}. The report quotes numbers from it and there is "
            "deliberately no fallback — run the pipeline (`pixi run pipeline`) first."
        )
    with open(path) as handle:
        return json.load(handle)


def read_risk_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"no RAC curve at {path}; run the `rac` rule that writes it")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def read_diagnostics(path: Path) -> pd.DataFrame:
    """One model's per-conditioning-day sampler diagnostics."""
    if not path.exists():
        raise FileNotFoundError(
            f"no sampler diagnostics at {path}; the `rac` rule writes them beside the curve"
        )
    return pd.read_csv(path)


def open_posterior(path: Path) -> xr.DataTree:
    """One fit's draws.

    Opened with xarray directly rather than through :mod:`end_of_outbreak.fitting`, for the
    reason ``scripts/utils.py`` gives: a fit is an ``xarray.DataTree`` and nothing here needs
    the model-building machinery that importing ``fitting`` would pull in.
    """
    if not path.exists():
        raise FileNotFoundError(f"no fit at {path}; run the `fit` rule that writes it")
    return xr.open_datatree(str(path), engine="h5netcdf").load()


def draws_of(posterior: xr.DataTree, name: str) -> NDArray[np.float64]:
    """One scalar parameter's draws, chains flattened."""
    return np.asarray(posterior.posterior.data_vars[name]).reshape(-1)


# ---------------------------------------------------------------------------------------
# The sections of the file
# ---------------------------------------------------------------------------------------


def add_setting(numbers: NumberFile, data: outbreak_data.OutbreakData) -> None:
    """The outbreak itself: totals, key dates and the analysis window."""
    onsets = data.onsets
    last_onset_day = int(np.flatnonzero(onsets > 0)[-1])
    numbers.set("data.totalcases", str(data.total_cases))
    numbers.set("data.days", str(data.n_days))
    numbers.set("data.preertcases", str(int(onsets[data.pre_ert_mask].sum())))
    numbers.set("data.postertcases", str(int(onsets[data.post_ert_mask].sum())))
    numbers.set("data.onsetdays", str(int((onsets > 0).sum())))
    numbers.set("data.maxdailyonsets", str(int(onsets.max())))
    numbers.set("data.firstday", "0")
    numbers.set("data.firstdate", date_text(data.date_of(0)))
    numbers.set("data.ertarrivalday", str(data.ert_arrival_day))
    numbers.set("data.ertarrivaldate", date_text(data.date_of(data.ert_arrival_day)))
    numbers.set("data.lastonsetday", str(last_onset_day))
    numbers.set("data.lastonsetdate", date_text(data.date_of(last_onset_day)))
    numbers.set("data.withdrawalday", str(data.ert_withdrawal_day))
    numbers.set("data.withdrawaldate", date_text(data.date_of(data.ert_withdrawal_day)))
    # The window runs past the withdrawal, so its end is a number of its own. It is also the
    # number of conditioning days, and so of fits, behind every curve that starts on day 1.
    numbers.set("data.lastday", str(data.last_day))
    numbers.set("data.lastdate", date_text(data.date_of(data.last_day)))
    numbers.set("data.extradays", str(data.last_day - data.ert_withdrawal_day))
    numbers.set("data.referenceday", str(REFERENCE_DAY))
    numbers.set("data.referencedate", date_text(data.date_of(REFERENCE_DAY)))


def add_convergence_settings(numbers: NumberFile, config: dict[str, Any]) -> None:
    """Configured gates quoted by the methods, distinct from measured diagnostics."""
    convergence = config["rac"]["convergence"]
    numbers.set("convergence.maxrhat", fixed(float(convergence["max_r_hat"]), 2))


def add_delays(numbers: NumberFile, config: dict[str, Any]) -> None:
    """The delay triple, and the variance budget that admits it.

    Rebuilt from the config rather than quoted from it, so that the residual TOST and the
    fidelity of the decomposition are the ones the models were actually driven by.
    """
    shared = config["shared"]
    delays = configuration.onset_anchored_delays_from_config(config)
    alternative = configuration.gamma_delay_from_config(shared["serial_interval_outbreak_specific"])
    for prefix, delay in (
        ("delays.serialinterval", configuration.gamma_delay_from_config(shared["serial_interval"])),
        ("delays.incubation", delays.incubation_delay),
        ("delays.tost", delays.tost_delay),
        ("delays.alternativeserialinterval", alternative),
    ):
        numbers.set(f"{prefix}.mean", trimmed(delay.mean, 2))
        numbers.set(f"{prefix}.sd", trimmed(delay.sd, 2))
        numbers.set(f"{prefix}.variance", trimmed(delay.variance, 2))
    numbers.set("delays.maxlag", str(int(shared["max_lag"])))
    numbers.set("delays.discrepancy", f"{delays.serial_interval_discrepancy():.5f}")
    numbers.set("delays.tolerance", fixed(float(shared["serial_interval_tolerance"]), 2))


def add_priors(numbers: NumberFile, config: dict[str, Any], analyses: list[str]) -> None:
    """The shared priors. Identical across models by design (§6.2), so defined once."""
    priors = config["shared"]["priors"]
    R_prior = LogNormalPrior.from_config(priors["R_pre"])
    numbers.set("priors.R.median", fixed(R_prior.median, 1))
    numbers.set("priors.R.quantilelower", fixed(float(R_prior.frozen().ppf(0.025)), 1))
    numbers.set("priors.R.quantileupper", fixed(float(R_prior.frozen().ppf(0.975)), 1))

    # One `k` prior for the whole report, so every analysis that estimates `k` must use the same
    # one. The report states it once, and two analyses quietly disagreeing would leave one of
    # them described by numbers that are not its own. Analyses whose models have no `k` at all
    # give no `k_prior` and are not asked for one.
    k_blocks = {
        analysis: config["analyses"][analysis]["k_prior"]
        for analysis in analyses
        if config["analyses"][analysis].get("k_prior") is not None
    }
    k_priors = {analysis: LogNormalPrior.from_config(block) for analysis, block in k_blocks.items()}
    if len(set(k_priors.values())) > 1:
        listed = "; ".join(f"{analysis}: {prior}" for analysis, prior in k_priors.items())
        raise ValueError(
            "the analyses that estimate k do not share one prior, but the report states it once "
            f"as `priors.k.*`: {listed}"
        )
    if k_priors:
        k_prior = next(iter(k_priors.values()))
        # Trimmed to three decimals rather than fixed at two: the interval runs a decade either
        # side of 0.18, and two decimals would print its lower end, 0.018, as 0.02.
        numbers.set("priors.k.median", trimmed(k_prior.median, 3))
        numbers.set("priors.k.quantilelower", trimmed(float(k_prior.frozen().ppf(0.025)), 3))
        numbers.set("priors.k.quantileupper", trimmed(float(k_prior.frozen().ppf(0.975)), 3))


def add_sampler(
    numbers: NumberFile, block: dict[str, Any], prefix: str, *, data: outbreak_data.OutbreakData
) -> None:
    """What the sampler was asked for, so the methods can state it per analysis."""
    sampler = block["sampler"]
    numbers.set(f"{prefix}.draws", str(int(sampler["draws"])))
    numbers.set(f"{prefix}.tune", str(int(sampler["tune"])))
    numbers.set(f"{prefix}.chains", str(int(sampler["chains"])))
    numbers.set(f"{prefix}.targetaccept", fixed(float(sampler["target_accept"]), 2))
    fixed_k = block.get("fixed_k")
    numbers.set(f"{prefix}.fixedk", "estimated" if fixed_k is None else fixed(float(fixed_k), 2))
    # The same for the reproduction numbers, so that an analysis which holds one constant has
    # its value on the page rather than nowhere -- `add_parameters` cannot summarise a
    # posterior that does not exist. Both read "estimated" for every analysis that fits them.
    for name in REPRODUCTION_NUMBERS:
        held = block.get(f"fixed_{name}")
        numbers.set(
            f"{prefix}.fixed{slug(name)}", "estimated" if held is None else fixed(float(held), 2)
        )
    # The day R actually switched on: the ERT arrival day unless the analysis moved it, which
    # is the whole content of a shifted-switch or no-switch variant. Resolved here rather than
    # left as an override, so the macro reads the same whether or not one was given -- and
    # through the same function the fits used, so it cannot disagree with them. An analysis
    # with no switch reads "none" rather than a day past the end of the window.
    switch_day = configuration.switch_day_from_config(
        block, ert_arrival_day=data.ert_arrival_day, n_days=data.n_days
    )
    numbers.set(f"{prefix}.switchday", "none" if switch_day >= data.n_days else str(switch_day))
    # The number of models is the denominator of the uniform prior over models (§6.2), which the
    # posterior model probabilities are not interpretable without.
    numbers.set(f"{prefix}.nmodels", str(len(block["models"])))
    # The assumed reporting probability, as a percentage, for the sweep of Stage 10. An input
    # rather than an estimate -- these data do not identify it -- but the report may not write
    # it as a literal, and an analysis that assumes nothing still has to say so.
    probability = float((block.get("reporting") or {}).get("probability", 1.0))
    numbers.set(f"{prefix}.reportingpercent", f"{round(probability * 100)}")


def add_parameters(numbers: NumberFile, posterior: xr.DataTree, prefix: str) -> None:
    """Posterior medians and 95% credible intervals for the scalars the report compares.

    Summarised with the same function ``dispersion_posteriors.json`` uses, so the ``k`` line the
    report quotes and the one the figure's legend carries are the same computation.
    """
    fitted_R_pre, fitted_R_post = fitting.fitted_reproduction_numbers(posterior)
    held_at = {
        "R_pre": fitted_R_pre,
        "R_post": fitted_R_post,
        DISPERSION: fitting.fitted_dispersion(posterior),
    }
    # The Poisson limits have no `k` to be fixed or estimated, so its absence from their draws
    # is the model rather than a broken input. The fit records which model it is.
    model = posterior.attrs.get(fitting.MODEL_ATTRIBUTE)
    names = [*REPRODUCTION_NUMBERS]
    if model is None or specification_of(str(model)).has_dispersion:
        names.append(DISPERSION)
    for name in names:
        if name not in posterior.posterior.data_vars:
            # A parameter the analysis fixed is a constant in the graph and so is nowhere in
            # the draws; `add_sampler` has already written the value it was held at. Any other
            # reason for its absence is a broken input, and silently emitting no macro is the
            # failure the report's scheme exists to prevent -- so ask the fit which it is.
            if held_at[name] is None:
                raise KeyError(
                    f"{prefix}: {name!r} is absent from the posterior, and the fit records no "
                    f"value it was fixed at. Either the fit is not the one this analysis "
                    f"describes, or it was written by a code version that did not record "
                    f"fixed parameters. There is deliberately no fallback: a quietly missing "
                    f"macro looks exactly like one that was never needed."
                )
            continue
        summary = pc.summarise_posterior(draws_of(posterior, name))
        decimals = 3 if name == DISPERSION else 2
        key = f"{prefix}.{slug(name)}"
        numbers.set(f"{key}.median", fixed(summary.median, decimals))
        numbers.set(f"{key}.lower", fixed(summary.interval_lower, decimals))
        numbers.set(f"{key}.upper", fixed(summary.interval_upper, decimals))
        numbers.set(f"{key}.interval", interval(summary, decimals))


def add_latent_block(
    numbers: NumberFile,
    posterior: xr.DataTree,
    model: str,
    data: outbreak_data.OutbreakData,
    delays: delay_distributions.OnsetAnchoredDelays,
    prefix: str,
) -> None:
    """How many latents the model carries, and how many survived exact marginalisation.

    The total comes from the block layout the builders use; the sampled count is the width of
    the block in the draws. The difference is what the conjugate marginalisation removed, which
    is the methods claim this pins.

    The layout has to be derived the way the builder derived it, which under incomplete
    reporting is from ``layout_counts`` rather than from the data: a day reporting nothing may
    still have had a case, so every day carries a latent. Reading it off the reported counts
    instead gives a total *smaller* than the block actually sampled, and a negative number of
    marginalised latents.
    """
    specification = specification_of(model)
    if specification.latent_variable is None:
        return
    # Both the reporting model and the switch day come from the fit rather than from the data
    # or the config: the layout has to be derived exactly as the builder derived it, and an
    # analysis may have moved the switch.
    switch_day = fitting.fitted_switch_day(posterior)
    structure = pymc_models.latent_block_structure(
        model,
        pymc_models.layout_counts(data.onsets, fitting.fitted_reporting(posterior)),
        delays=delays,
        switch_day=data.ert_arrival_day if switch_day is None else switch_day,
    )
    sampled = int(posterior.posterior.sizes[structure.dimension])
    numbers.set(f"{prefix}.latents.total", str(int(structure.days.size)))
    numbers.set(f"{prefix}.latents.sampled", str(sampled))
    numbers.set(f"{prefix}.latents.marginalised", str(int(structure.days.size) - sampled))


def add_risk_curve(
    numbers: NumberFile,
    frame: pd.DataFrame,
    data: outbreak_data.OutbreakData,
    prefix: str,
    *,
    reference_day: int = REFERENCE_DAY,
) -> None:
    """Threshold crossings, final/reference risks, and the Monte-Carlo error.

    RAT is written only for onset-anchored models, while RST is written for every model with an
    individual branching-process interpretation. Their keys therefore follow the columns in
    the result file rather than being inferred from the model name.
    """
    if reference_day not in set(np.asarray(frame["day"], dtype=np.int64).tolist()):
        raise ValueError(
            f"the reference day {reference_day} is not in this curve, which runs from "
            f"{int(frame['day'].min())} to {int(frame['day'].max())}. It is the one day the "
            "report compares every model on, so it has to be inside the analysis window."
        )
    columns = [(RISK_COLUMN, "rac", STANDARD_ERROR_COLUMN)]
    if TRANSMISSION_RISK_COLUMN in frame:
        columns.append((TRANSMISSION_RISK_COLUMN, "rat", "transmission_monte_carlo_standard_error"))
    if SUSTAINED_RISK_COLUMN in frame:
        columns.append(
            (
                SUSTAINED_RISK_COLUMN,
                "rst",
                "sustained_transmission_monte_carlo_standard_error",
            )
        )
    days = np.asarray(frame["day"], dtype=np.int64)
    last_day = int(days[-1])
    withdrawal_day = data.ert_withdrawal_day
    if withdrawal_day not in set(days.tolist()):
        raise ValueError(
            f"the ERT's withdrawal, day {withdrawal_day}, is not in this curve, which runs from "
            f"{int(days[0])} to {last_day}. The report quotes every model's risk on that day."
        )
    # Two Monte-Carlo error summaries, because the honest maximum and the useful one differ. The
    # error is largest while the curve is near 1 and every draw is contributing, which is a
    # region no comparison is made in; the decision-relevant figure is the largest error over the
    # case-free tail, where the thresholds are crossed and two curves are read against each other.
    late = days >= int(np.flatnonzero(np.asarray(data.onsets) > 0)[-1])
    for column, name, error_column in columns:
        curve = risk_curves.RiskCurve(
            days=days, risk=np.asarray(frame[column], dtype=np.float64), n_draws=0
        )
        for threshold in THRESHOLDS:
            day = curve.first_day_below(threshold)
            key = f"{prefix}.{name}{round(threshold * 100):02d}"
            numbers.set(f"{key}.day", crossing_day_text(day, last_day=last_day))
            numbers.set(f"{key}.date", crossing_date_text(day, data, last_day=last_day))
        numbers.set(f"{prefix}.{name}.final", fixed(float(curve.risk[-1]), 3))
        # The risk on the day the ERT actually left. Once the same as `.final`; now twenty days
        # before it, and the number to quote wherever the sentence is about the real decision.
        numbers.set(
            f"{prefix}.{name}.withdrawal",
            fixed(float(curve.risk[days == withdrawal_day][0]), 3),
        )
        numbers.set(
            f"{prefix}.{name}.reference",
            fixed(float(curve.risk[days == reference_day][0]), 3),
        )
        numbers.set(f"{prefix}.{name}.mcse", f"{float(frame[error_column].max()):.4f}")
        numbers.set(
            f"{prefix}.{name}.mcselate", f"{float(frame.loc[late, error_column].max()):.4f}"
        )

    if TRANSMISSION_RISK_COLUMN in frame:
        gap = np.asarray(frame[RISK_COLUMN] - frame[TRANSMISSION_RISK_COLUMN], dtype=np.float64)
        widest = int(np.argmax(gap))
        numbers.set(f"{prefix}.gap.max", fixed(float(gap[widest]), 3))
        numbers.set(f"{prefix}.gap.day", str(int(days[widest])))
        numbers.set(f"{prefix}.gap.date", date_text(data.date_of(int(days[widest]))))
        # How long the pipeline holds a case-based declaration back after transmission risk has
        # already settled: the same threshold, read off the two curves, differenced here so the
        # report states it rather than working it out.
        for threshold in THRESHOLDS:
            label = f"{round(threshold * 100):02d}"
            crossings = [
                risk_curves.RiskCurve(
                    days=days, risk=np.asarray(frame[column], dtype=np.float64), n_draws=0
                ).first_day_below(threshold)
                for column in (RISK_COLUMN, TRANSMISSION_RISK_COLUMN)
            ]
            case_day, transmission_day = crossings
            numbers.set(
                f"{prefix}.gap.shift{label}",
                crossing_difference(case_day, transmission_day, last_day=last_day),
            )


def add_evidence(numbers: NumberFile, evidence: dict[str, Any], analysis: str) -> None:
    """Log marginal likelihoods and the posterior model probabilities they imply."""
    prefix = slug(analysis)
    for model in evidence["models"]:
        key = f"{prefix}.{slug(model)}"
        numbers.set(
            f"{key}.logevidence", math_mode(fixed(float(evidence["log_evidence"][model]), 1))
        )
        numbers.set(
            f"{key}.logevidencese",
            f"{float(evidence['log_evidence_standard_error'][model]):.3f}",
        )
        numbers.set(
            f"{key}.probability",
            small_probability(float(evidence["posterior_model_probability"][model])),
        )


def add_dispersion(numbers: NumberFile, dispersion: dict[str, Any], analysis: str) -> None:
    """Every pairwise divergence between the ``k`` posteriors, as the file records them."""
    prefix = slug(analysis)
    for pair, divergence in dispersion["divergence"].items():
        first, second = pair.split("_vs_")
        key = f"{prefix}.{slug(first)}-vs-{slug(second)}"
        numbers.set(f"{key}.medianratio", fixed(float(divergence["median_ratio"]), 2))
        numbers.set(f"{key}.overlap", fixed(float(divergence["overlap"]), 3))
        numbers.set(f"{key}.probabilitygreater", fixed(float(divergence["probability_greater"]), 3))


@dataclass(frozen=True)
class Diagnostics:
    """The worst convergence figures over a set of fits.

    The worst case is the only summary worth quoting: "every fit converged" is a claim about the
    maximum ``R̂`` and the minimum effective sample size, not about their averages. Aggregating
    the same way twice — per analysis and over all of them — is what lets the report make the
    global claim from a global number rather than from one analysis's.

    The set is every fit behind every curve, not just the fits to the complete record: the
    estimand conditions on the record through each day, so each curve rests on one fit per
    conditioning day and a claim that leaves them out would cover a small fraction of the study.
    """

    fits: int
    """Every conditioning-day fit behind these curves, including any that sampled nothing."""

    sampled_fits: int
    """Those of them with a variable to sample, which is what ``R̂`` and the ESS describe.

    The two differ wherever a fit had nothing to sample: ``no_switch_fixed_R``'s SSE fixes every
    parameter and has no latent block, so all of its fits are point masses, and the first days of
    the latent models' curves have an empty latent block. Such a fit has no convergence to
    report — ``R̂`` and the ESS are nan in its row — and averaging it in as though it had is
    how a nan reached a macro the report quotes.
    """

    max_rhat: float
    min_bulk_ess: float
    divergences: int

    def merged_with(self, other: Diagnostics) -> Diagnostics:
        return Diagnostics(
            fits=self.fits + other.fits,
            sampled_fits=self.sampled_fits + other.sampled_fits,
            max_rhat=max(self.max_rhat, other.max_rhat),
            min_bulk_ess=min(self.min_bulk_ess, other.min_bulk_ess),
            divergences=self.divergences + other.divergences,
        )


def measure_diagnostics(tables: dict[str, pd.DataFrame]) -> Diagnostics:
    """Convergence over every conditioning-day fit of every model in one analysis.

    Read from the per-day tables the ``rac`` step writes rather than recomputed, so the numbers
    the report quotes are the ones the pipeline's own acceptance gate applied. Those cover every
    sampled variable, latent block included — which is a stronger claim than the reported
    scalars alone, and the one the gate enforces.

    Fits that sampled nothing are counted and then set aside: they have no ``R̂`` and no ESS, so
    including their nan rows would make the maximum and the minimum nan too, exactly as the gate
    itself had to learn (``refit_risk.DayDiagnostics``). A nan that reaches the macro file is
    worse than an error, because ``\resultnum`` would print it: the crossing macros show a
    missing value loudly, but a nan reads as a number. A *sampled* fit whose ``R̂`` is nan is a
    different thing — a diagnostic that could not be computed — and raises here rather than
    being dropped.
    """
    fits = sampled_fits = divergences = 0
    rhats: list[float] = []
    ess: list[float] = []
    for model, table in tables.items():
        fits += len(table)
        divergences += int(table["divergences"].sum())
        sampled = table[sampled_mask(table)]
        sampled_fits += len(sampled)
        unusable = sampled[sampled["max_r_hat"].isna() | sampled["min_ess_bulk"].isna()]
        if not unusable.empty:
            days = ", ".join(str(int(day)) for day in unusable["day"])
            raise ValueError(
                f"{model}: days {days} sampled variables but have no R-hat or ESS. A "
                "diagnostic that could not be computed is not a diagnostic that passed."
            )
        if not sampled.empty:
            rhats.append(float(sampled["max_r_hat"].max()))
            ess.append(float(sampled["min_ess_bulk"].min()))
    if not rhats:
        models = ", ".join(tables)
        raise ValueError(
            f"no fit of {models} sampled any variable, so there is no convergence to summarise"
        )
    return Diagnostics(
        fits=fits,
        sampled_fits=sampled_fits,
        max_rhat=max(rhats),
        min_bulk_ess=min(ess),
        divergences=divergences,
    )


def sampled_mask(table: pd.DataFrame) -> pd.Series:
    """Which rows of a diagnostics table describe a fit that had something to sample.

    ``n_sampled_variables`` is the column that says so, and is what the acceptance gate reads.
    Older tables, written before it existed, are judged by whether an ``R̂`` was recorded.
    """
    if "n_sampled_variables" in table:
        return table["n_sampled_variables"] > 0
    return table["max_r_hat"].notna()


def add_diagnostics(numbers: NumberFile, diagnostics: Diagnostics, prefix: str) -> None:
    """Write one set of convergence figures under ``<prefix>.diagnostics``."""
    numbers.set(f"{prefix}.diagnostics.fits", str(diagnostics.fits))
    # The two counts are equal for every analysis whose models all sample something, and a
    # sentence claiming convergence "over all N fits" should quote whichever it means.
    numbers.set(f"{prefix}.diagnostics.sampledfits", str(diagnostics.sampled_fits))
    # Three decimals, not two: the acceptance gate is close to one, and "1.00" would not let a
    # reader tell a comfortable pass from a marginal one.
    numbers.set(f"{prefix}.diagnostics.maxrhat", fixed(diagnostics.max_rhat, 3))
    numbers.set(f"{prefix}.diagnostics.minbulkess", str(int(np.floor(diagnostics.min_bulk_ess))))
    numbers.set(f"{prefix}.diagnostics.divergences", str(diagnostics.divergences))


# --- the cross-analysis subtractions -----------------------------------------------------


ONSET_PAIRS: tuple[tuple[str, str], ...] = (
    ("sse_so", "sse"),
    ("ssi_so", "ssi"),
    ("cori_so", "cori"),
)
"""Each onset-anchored model with the naive model of the same mechanism.

The Poisson pair is here for the analysis that removes superspreading: the same subtraction,
with no dispersion mechanism for the anchoring to interact with.
"""


def add_onset_shifts(numbers: NumberFile, analysis: str, curves: dict[str, pd.DataFrame]) -> None:
    """How far onset anchoring moves each threshold crossing, in days.

    The headline of Stage 9, and a subtraction rather than a new quantity: it pairs each
    onset-anchored model with the naive model of the same mechanism, which is the only
    comparison in the analysis that holds the mechanism fixed and varies the anchoring.
    """
    add_crossing_shifts(numbers, slug(analysis), ONSET_PAIRS, curves)


SUPERSPREADING_PAIRS: tuple[tuple[str, str], ...] = (("ssi", "cori"), ("ssi_so", "cori_so"))
"""Each individual-level superspreading model with its Poisson limit, anchoring held fixed.

The complement of :data:`ONSET_PAIRS`: there the mechanism is held and the anchoring varies,
here the anchoring is held and superspreading is switched on.
"""


def add_crossing_shifts(
    numbers: NumberFile,
    prefix: str,
    pairs: tuple[tuple[str, str], ...],
    curves: dict[str, pd.DataFrame],
) -> None:
    """``<prefix>.<first>-vs-<second>.shiftNN``: how many days earlier ``first`` settles.

    Pairs with a model missing from ``curves`` are skipped, since every analysis is offered
    every pair and most hold only some of the models.
    """
    for first, second in pairs:
        if first not in curves or second not in curves:
            continue
        key = f"{prefix}.{slug(first)}-vs-{slug(second)}"
        for threshold in THRESHOLDS:
            first_day, second_day = (
                _crossing(curves[model], threshold) for model in (first, second)
            )
            numbers.set(
                f"{key}.shift{round(threshold * 100):02d}",
                crossing_difference(
                    second_day,
                    first_day,
                    last_day=_shared_last_day(curves[first], curves[second]),
                ),
            )


def add_combined_evidence(
    numbers: NumberFile,
    combined: dict[str, Any],
    analysis: str,
    curves: dict[str, pd.DataFrame],
) -> None:
    """An analysis set beside models it borrows: probabilities over all of them, and the shifts.

    Keyed under ``<analysis>.combined`` so that they cannot be mistaken for the probabilities
    the analysis's own evidence file normalises over its own models alone. ``curves`` holds the
    borrowed models' curves as well as the analysis's own, which is what lets superspreading's
    shift be read off with the anchoring held fixed.
    """
    prefix = f"{slug(analysis)}.combined"
    for model in combined["models"]:
        key = f"{prefix}.{slug(model)}"
        numbers.set(
            f"{key}.logevidence", math_mode(fixed(float(combined["log_evidence"][model]), 1))
        )
        numbers.set(
            f"{key}.probability",
            small_probability(float(combined["posterior_model_probability"][model])),
        )
    add_crossing_shifts(numbers, prefix, SUPERSPREADING_PAIRS, curves)


def _shared_last_day(*frames: pd.DataFrame) -> int:
    """The last conditioning day of curves that are being differenced, which must agree."""
    last_days = {int(frame["day"].max()) for frame in frames}
    if len(last_days) != 1:
        raise ValueError(f"curves compared day by day must share a window; they end on {last_days}")
    return last_days.pop()


def _crossing(frame: pd.DataFrame, threshold: float) -> int | None:
    curve = risk_curves.RiskCurve(
        days=np.asarray(frame["day"], dtype=np.int64),
        risk=np.asarray(frame[RISK_COLUMN], dtype=np.float64),
        n_draws=0,
    )
    return curve.first_day_below(threshold)


def add_evidence_gains(
    numbers: NumberFile, results_root: Path, analyses: list[str], config: dict[str, Any]
) -> None:
    """What letting ``k`` move is worth, in nats, for each model that is fitted both ways.

    Analysis 2 against Analysis 1 and Analysis 4 against Analysis 3: the same models, the same
    data, the same priors on ``R``, differing only in whether ``k`` is held at the literature
    value. A large gain says the literature value was not this model's.
    """
    for analysis in analyses:
        if config["analyses"][analysis].get("fixed_k") is not None:
            continue
        counterpart = analysis.replace("estimated_k", "fixed_k")
        if counterpart == analysis or counterpart not in analyses:
            continue
        estimated = read_json(results_root / analysis / "model_evidence.json")["log_evidence"]
        held = read_json(results_root / counterpart / "model_evidence.json")["log_evidence"]
        for model in estimated:
            if model in held:
                gain = float(estimated[model]) - float(held[model])
                # Two decimals rather than one: the gains span 7.4 nats down to hundredths, and
                # the small ones are the interesting end — a model whose evidence does not move
                # when `k` is released is one the literature value already fitted.
                numbers.set(f"{slug(analysis)}.{slug(model)}.evidencegain", signed(gain, 2))


# ---------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------


def collect(
    *,
    analyses: list[str],
    config: dict[str, Any],
    data_path: Path,
    results_root: Path,
    variant_analyses: list[str] | None = None,
    rac_only_analyses: list[str] | None = None,
) -> tuple[NumberFile, list[str]]:
    """Every macro the report can use, and the files they came from.

    ``analyses`` are the core four, whose convergence the report summarises as "the core".
    ``variant_analyses`` are collected in exactly the same way — evidence included, dispersion
    wherever `k` is estimated — but their convergence is summarised separately, so that a claim
    about the four main analyses cannot quietly absorb the supplementary ones.

    ``rac_only_analyses`` are analyses that produce risk curves but none of the tier-2
    summaries — the under-reporting sweeps have no model evidence and no dispersion posterior,
    because they compare a model with itself under a different assumption. They contribute
    their crossings and their convergence, and are asked for nothing else.
    """
    variants = list(variant_analyses or [])
    rac_only_list = list(rac_only_analyses or [])
    numbers = NumberFile()
    data = outbreak_data.load_onset_data(data_path)
    delays = configuration.onset_anchored_delays_from_config(config)
    sources = [repository_path(data_path)]

    add_setting(numbers, data)
    add_convergence_settings(numbers, config)
    add_delays(numbers, config)
    add_priors(numbers, config, [*analyses, *variants])
    buckets: dict[str, Diagnostics] = {}
    overall: Diagnostics | None = None

    for analysis in [*analyses, *variants, *rac_only_list]:
        rac_only = analysis in rac_only_list
        bucket = "reporting" if rac_only else "variants" if analysis in variants else "core"
        block = configuration.analysis_config(config, analysis)
        models = list(block["models"])
        directory = results_root / analysis
        add_sampler(numbers, block, slug(analysis), data=data)

        posteriors = {
            model: open_posterior(directory / f"{model}_posterior.nc") for model in models
        }
        curves = {model: read_risk_curve(directory / f"{model}_rac.csv") for model in models}
        for model in models:
            prefix = f"{slug(analysis)}.{slug(model)}"
            add_parameters(numbers, posteriors[model], prefix)
            add_latent_block(numbers, posteriors[model], model, data, delays, prefix)
            add_risk_curve(numbers, curves[model], data, prefix)
        diagnostics = measure_diagnostics(
            {
                model: read_diagnostics(directory / f"{model}_rac_diagnostics.csv")
                for model in models
            }
        )
        add_diagnostics(numbers, diagnostics, slug(analysis))
        overall = diagnostics if overall is None else overall.merged_with(diagnostics)
        buckets[bucket] = (
            diagnostics if bucket not in buckets else buckets[bucket].merged_with(diagnostics)
        )
        add_onset_shifts(numbers, analysis, curves)

        if not rac_only:
            add_evidence(numbers, read_json(directory / "model_evidence.json"), analysis)
            # The same test as the Snakefile's `estimates_dispersion`: no `fixed_k` alone is also
            # true of the Poisson limits, which have no `k` and so no summary of one.
            if block.get("fixed_k") is None and block.get("k_prior") is not None:
                add_dispersion(
                    numbers,
                    read_json(directory / "dispersion_posteriors.json"),
                    analysis,
                )
            borrowed = block.get("compared_with")
            if borrowed is not None:
                other = results_root / str(borrowed["analysis"])
                add_combined_evidence(
                    numbers,
                    read_json(directory / COMBINED_EVIDENCE_FILE),
                    analysis,
                    curves
                    | {
                        model: read_risk_curve(other / f"{model}_rac.csv")
                        for model in borrowed["models"]
                    },
                )
        sources.extend(
            sorted(
                repository_path(path)
                for path in _analysis_files(directory, models, rac_only=rac_only)
            )
        )

    for bucket, diagnostics in buckets.items():
        add_diagnostics(numbers, diagnostics, bucket)
    # Keep an explicitly all-analysis summary as well: the core, variant and reporting claims
    # should not accidentally borrow it, but a genuine study-wide statement can.
    if overall is not None:
        add_diagnostics(numbers, overall, "overall")
    add_evidence_gains(numbers, results_root, analyses, config)
    return numbers, sources


def _analysis_files(
    directory: Path, models: list[str], *, rac_only: bool = False
) -> Iterator[Path]:
    for model in models:
        yield directory / f"{model}_posterior.nc"
        yield directory / f"{model}_rac.csv"
        yield directory / f"{model}_rac_diagnostics.csv"
    if rac_only:
        return
    yield directory / "model_evidence.json"
    for optional in ("dispersion_posteriors.json", COMBINED_EVIDENCE_FILE):
        if (directory / optional).exists():
            yield directory / optional


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    numbers, sources = collect(
        analyses=args.analyses,
        variant_analyses=args.variant_analyses,
        rac_only_analyses=args.rac_only_analyses,
        config=configuration.load_config(args.config),
        data_path=args.data,
        results_root=args.results_root,
    )
    output = configuration.ensure_parent(args.output)
    output.write_text(numbers.render(sources=sources))
    print(f"wrote {len(numbers)} numbers to {output}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument(
        "--analyses",
        nargs="+",
        required=True,
        help="the core analyses, in report order; the Snakefile passes CORE_ANALYSES",
    )
    parser.add_argument(
        "--variant-analyses",
        nargs="*",
        default=[],
        help="the supplementary variants, collected alike but summarised apart",
    )
    parser.add_argument(
        "--rac-only-analyses",
        nargs="*",
        default=[],
        help=(
            "analyses that produce risk curves but no model evidence or dispersion summary; "
            "the Snakefile passes RAC_ONLY_ANALYSES"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
