"""The panels the report's analysis figures are assembled from.

Figs. 1–4 are the same four kinds of panel in different arrangements: posterior densities for a
scalar parameter, the posterior model probabilities, and the RAC curves over the conditioning
days. Each figure script chooses the arrangement, the titles and the letters; the drawing lives
here so that a change to how a RAC panel looks lands on every figure at once and the two naive
analyses cannot drift apart visually while claiming to be comparable.

The split from :mod:`utils` is by altitude rather than by subject. ``utils`` holds the house
style, the loading of what the tier-2 rules wrote, and the drawing primitives — a density, a
pie, an incidence backdrop — and computes nothing at all. This module composes those primitives
into the labelled panels the report actually quotes, which is why it is the one place under
``scripts/`` that touches :mod:`end_of_outbreak.risk_curves`: "the day a curve
settles below a threshold" is a definition the report states, so it must come from
:meth:`~end_of_outbreak.risk_curves.RiskCurve.first_day_below` rather than being
reimplemented beside the panel.
"""

from __future__ import annotations

from typing import Any

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import utils
from matplotlib.axes import Axes
from numpy.typing import NDArray

from end_of_outbreak import outbreak_data, risk_curves
from end_of_outbreak.model_specifications import LogNormalPrior

SETTLING_THRESHOLD = 0.05
"""The threshold whose crossing is marked on a RAC panel; the report quotes 0.01 as well."""


def parameter_posterior_panel(
    ax: Axes,
    draws: dict[str, NDArray[np.float64]],
    *,
    prior: LogNormalPrior,
    xlabel: str,
    title: str,
    letter: str | None = None,
    labels: dict[str, str] | None = None,
    legend: bool = False,
) -> None:
    """One posterior density per model for a single scalar, over the prior they share.

    The prior is not decoration. The posterior model probabilities compare the models *as
    specified, priors included* (§6.5), so a reader needs to see how much of each posterior is
    prior — most of all for ``k``, where the prior is deliberately fairly informative and the
    question is whether the data move each model away from it in different directions.
    """
    utils.plot_parameter_posteriors(ax, draws, prior=prior, xlabel=xlabel, labels=labels)
    ax.set_title(title)
    if letter is not None:
        utils.panel_label(ax, letter)
    if legend:
        ax.legend(loc="upper right")


def model_probability_panel(
    ax: Axes, evidence: dict[str, Any], models: list[str], *, letter: str
) -> None:
    """The posterior model probabilities, straight from the evidence file."""
    utils.plot_model_probability_pie(
        ax, {model: float(evidence["posterior_model_probability"][model]) for model in models}
    )
    ax.set_title("Posterior model probability")
    utils.panel_label(ax, letter)


def risk_curve_panel(
    ax: Axes,
    curves: dict[str, pd.DataFrame],
    data: outbreak_data.OutbreakData,
    *,
    letter: str,
    first_day: int | None = None,
) -> None:
    """RAC over the conditioning days, with the onset series behind it.

    The marker on each curve is the day it settles below :data:`SETTLING_THRESHOLD` — the date
    an end-of-outbreak declaration would be made on — which is the summary the report compares
    across models and across analyses.

    ``first_day`` trims the *view*, not the curves: every model is still drawn over the whole
    window and the settling markers are unaffected, so a trimmed panel cannot show a different
    crossing date from an untrimmed one. It exists because the interesting part of the curve is
    its descent, and on a figure where this panel shares its row the opening weeks — where every
    model has already reached 1 and stays there — compress that descent into the right-hand
    third. From the ERT's arrival to the last observed case there is nothing to see.

    An untrimmed panel does show something the old retrospective estimand could not: the risk
    *climbing* over the first days of the outbreak, from well under 1 on day 1. That is
    real-time conditioning — early on the record is one case and a few days of silence, so the
    posterior is prior-dominated — and the day-to-day roughness there is genuine too, since
    each day is its own fit on its own data.
    """
    utils.plot_incidence(ax, data.dates, data.onsets)
    utils.mark_thresholds(ax)
    for model, frame in curves.items():
        ax.plot(
            frame["date"],
            frame[utils.RISK_COLUMN],
            color=utils.model_colour(model),
            label=utils.model_label(model),
            zorder=3,
        )
        settled = settling_day(frame)
        if settled is not None:
            ax.plot(
                mdates.date2num(data.date_of(settled)),
                float(frame.loc[frame["day"] == settled, utils.RISK_COLUMN].iloc[0]),
                marker="o",
                markersize=3.5,
                color=utils.model_colour(model),
                zorder=4,
            )
    utils.mark_intervention_dates(
        ax,
        arrival=data.date_of(data.ert_arrival_day),
        withdrawal=data.date_of(data.ert_withdrawal_day),
    )
    utils.date_axis(ax)
    ax.set_xlim(data.dates[first_day or 0], data.dates[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("conditioning day $t$ (2018)")
    ax.set_ylabel("risk of additional cases")
    ax.set_title("Risk of at least one further case after day $t$")
    # Mid-left: the curves sit at ~1 from the opening weeks until the last observed case, and
    # the onset bars are confined to the bottom, so this is the one large region that stays
    # empty on both the trimmed and the untrimmed view.
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.55))
    utils.panel_label(ax, letter)


def rac_rat_panel(
    ax: Axes,
    curves: dict[str, pd.DataFrame],
    data: outbreak_data.OutbreakData,
    *,
    title: str,
    letter: str,
    first_day: int | None = None,
) -> None:
    """RAC and RAT for the onset-anchored models."""
    utils.plot_incidence(ax, data.dates, data.onsets)
    utils.mark_thresholds(ax)
    for model, frame in curves.items():
        if utils.TRANSMISSION_RISK_COLUMN not in frame:
            raise ValueError(
                f"the {model} RAC file has no {utils.TRANSMISSION_RISK_COLUMN!r} column; "
                "RAT is written only by the onset-anchored RAC tier"
            )
        colour = utils.model_colour(model)
        ax.plot(
            frame["date"],
            frame[utils.RISK_COLUMN],
            color=colour,
            linestyle="--",
            label=f"{utils.model_label(model)} RAC",
            zorder=3,
        )
        ax.plot(
            frame["date"],
            frame[utils.TRANSMISSION_RISK_COLUMN],
            color=colour,
            label=f"{utils.model_label(model)} RAT",
            zorder=4,
        )
    utils.mark_intervention_dates(
        ax,
        arrival=data.date_of(data.ert_arrival_day),
        withdrawal=data.date_of(data.ert_withdrawal_day),
    )
    utils.date_axis(ax)
    ax.set_xlim(data.dates[first_day or 0], data.dates[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("conditioning day $t$ (2018)")
    ax.set_ylabel("posterior risk")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.55), ncols=2)
    utils.panel_label(ax, letter)


def settling_day(frame: pd.DataFrame, threshold: float = SETTLING_THRESHOLD) -> int | None:
    """The first day a curve falls below ``threshold`` and stays there.

    Delegated to :class:`~end_of_outbreak.risk_curves.RiskCurve` rather than
    reimplemented here, because "settles below" is a definition the report quotes and it must not
    be able to drift between the marker on the panel and the number in the text.
    """
    curve = risk_curves.RiskCurve(
        days=np.asarray(frame["day"], dtype=np.int64),
        risk=np.asarray(frame[utils.RISK_COLUMN], dtype=np.float64),
        n_draws=0,
    )
    return curve.first_day_below(threshold)


def credible_interval_labels(
    summaries: dict[str, dict[str, Any]], *, precision: int = 2
) -> dict[str, str]:
    """Legend labels carrying each model's posterior median and credible interval.

    Reads the numbers the ``dispersion`` rule wrote rather than recomputing them from the draws:
    the median and interval the panel shows and the ones the report quotes have to be the same
    numbers, and a figure that summarises its own inputs is exactly how they come apart.
    """
    return {
        model: (
            f"{utils.model_label(model)}  {summary['median']:.{precision}f} "
            f"({summary['interval_lower']:.{precision}f}–{summary['interval_upper']:.{precision}f})"
        )
        for model, summary in summaries.items()
    }
