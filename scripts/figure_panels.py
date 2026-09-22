"""The panels the report's analysis figures are assembled from.

The per-analysis figures are the same few kinds of panel in different arrangements: posterior
densities for a scalar parameter, the posterior model probabilities, and the RAC curves over the
conditioning days. Each figure script chooses the arrangement, the titles and the letters; the
drawing lives here so that a change to how a RAC panel looks lands on every figure at once and
the two naive analyses cannot drift apart visually while claiming to be comparable.

The split from :mod:`utils` is by altitude rather than by subject. ``utils`` holds the house
style, the loading of what the tier-2 rules wrote, and the drawing primitives — a density, a
pie, an incidence backdrop — and computes nothing at all. This module composes those primitives
into the labelled panels the report actually quotes, which is why it is the one place under
``scripts/`` that touches :mod:`end_of_outbreak.risk_curves`: "the day a curve
settles below a threshold" is a definition the report states, so it must come from
:meth:`~end_of_outbreak.risk_curves.RiskCurve.first_day_below` rather than being
reimplemented beside the panel.

The risk-metric panels reuse the same backdrop and annotation rules but distinguish RAC, RAT
and RST by linestyle while preserving the established model colours.
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import utils
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from end_of_outbreak import outbreak_data, risk_curves
from end_of_outbreak.model_specifications import LogNormalPrior, specification_of

SETTLING_THRESHOLD = 0.05
"""The threshold whose crossing is marked on a RAC panel; the report quotes 0.01 as well."""

type RiskMetric = Literal["rac", "rat", "rst"]
"""One of the three risk estimands a curve file can carry."""

METRIC_COLUMNS: dict[RiskMetric, str] = {
    "rac": utils.RISK_COLUMN,
    "rat": utils.TRANSMISSION_RISK_COLUMN,
    "rst": utils.SUSTAINED_RISK_COLUMN,
}
METRIC_LABELS: dict[RiskMetric, str] = {"rac": "RAC", "rat": "RAT", "rst": "RST"}
METRIC_LINESTYLES: dict[RiskMetric, str] = {"rac": "-", "rat": "--", "rst": ":"}
"""Where more than one estimand shares a panel, linestyle is the estimand and colour the model.

RAC is solid because it is the headline, and solid is what every single-estimand RAC panel
draws: a reader moving between figures keeps one line for it. An infection-anchored model's
combined "RAC/RAT" curve follows RAC and is solid too."""


def parameter_posterior_panel(
    ax: Axes,
    draws: dict[str, NDArray[np.float64]],
    *,
    prior: LogNormalPrior,
    xlabel: str,
    title: str,
    letter: str | None = None,
    labels: dict[str, str] | None = None,
    colours: dict[str, str] | None = None,
    legend: bool = False,
    log_scale: bool = False,
) -> None:
    """One posterior density per model for a single scalar, over the prior they share.

    The prior is not decoration. The posterior model probabilities compare the models *as
    specified, priors included* (§6.5), so a reader needs to see how much of each posterior is
    prior — most of all for ``k``, where the prior is deliberately fairly informative and the
    question is whether the data move each model away from it in different directions.

    ``colours`` overrides the palette for this figure only; see :func:`utils.model_colour`.

    ``log_scale`` is for a parameter whose posteriors span decades; see
    :func:`utils.plot_parameter_posteriors`.
    """
    utils.plot_parameter_posteriors(
        ax, draws, prior=prior, xlabel=xlabel, labels=labels, colours=colours, log_scale=log_scale
    )
    ax.set_title(title)
    if letter is not None:
        utils.panel_label(ax, letter)
    if legend:
        ax.legend(loc="upper right")


def model_probability_panel(
    ax: Axes,
    evidence: dict[str, Any],
    models: list[str],
    *,
    letter: str,
    colours: dict[str, str] | None = None,
) -> None:
    """The posterior model probabilities, straight from the evidence file.

    ``colours`` overrides the palette for this figure only; see :func:`utils.model_colour`.
    """
    utils.plot_model_probability_pie(
        ax,
        {model: float(evidence["posterior_model_probability"][model]) for model in models},
        colours=colours,
    )
    ax.set_title("Posterior model probability")
    utils.panel_label(ax, letter)


def risk_curve_panel(
    ax: Axes,
    curves: dict[str, pd.DataFrame],
    data: outbreak_data.OutbreakData,
    *,
    letter: str | None = None,
    first_day: int | None = None,
    linestyles: dict[str, str] | None = None,
    colours: dict[str, str] | None = None,
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

    ``linestyles`` overrides a model's line, and exists for a figure where two curves coincide
    to within the line width — which is a result, but reads on the page as a curve that failed
    to draw. Colour alone then cannot separate them. Every model is solid by default, so the
    analysis figures are unaffected. ``letter`` may be ``None`` where the figure is one panel
    and a label would be noise.

    ``colours`` overrides the palette for this figure only; see :func:`utils.model_colour`.
    """
    resolved_linestyles = linestyles or {}
    utils.plot_incidence(ax, data.dates, data.onsets)
    utils.mark_thresholds(ax)
    for model, frame in curves.items():
        colour = utils.model_colour(model, colours)
        ax.plot(
            frame["date"],
            frame[utils.RISK_COLUMN],
            color=colour,
            linestyle=resolved_linestyles.get(model, "-"),
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
                color=colour,
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
    if letter is not None:
        utils.panel_label(ax, letter)


def risk_metrics_panel(
    ax: Axes,
    curves: dict[str, pd.DataFrame],
    data: outbreak_data.OutbreakData,
    *,
    title: str,
    letter: str,
    metrics: tuple[RiskMetric, ...] = ("rac", "rat", "rst"),
    first_day: int | None = None,
    metric_linestyles: dict[RiskMetric, str] | None = None,
) -> None:
    """The chosen risk estimands, for any mix of infection- and onset-anchored models.

    Colour identifies the model throughout; linestyle identifies the estimand. Both anchoring
    conventions share the panel, so the naive/onset comparison is read within one estimand
    rather than across panels.

    Under infection anchoring a case *is* a transmission event, so RAC and RAT are one quantity
    and the curve file carries only the first. Asked for both, such a model is drawn once and
    labelled for both — and its missing second line is itself the point that the two separate
    only under onset anchoring. The convention is read off the model's specification rather than
    inferred from the columns present, so a file that has lost a column fails instead of quietly
    drawing one line fewer.

    ``metric_linestyles`` overrides the default estimand linestyles, for a panel showing a single
    estimand, where linestyle has nothing left to distinguish. It is keyed by estimand, unlike
    :func:`risk_curve_panel`'s ``linestyles``, which is keyed by model.
    """
    resolved_linestyles = METRIC_LINESTYLES | (metric_linestyles or {})
    utils.plot_incidence(ax, data.dates, data.onsets)
    utils.mark_thresholds(ax)
    for model, frame in curves.items():
        for style, column_metric, label in _drawn_metrics(model, metrics):
            column = METRIC_COLUMNS[column_metric]
            if column not in frame:
                raise ValueError(
                    f"the {model} curve file has no {column!r} column, which "
                    f"{METRIC_LABELS[style]} needs; regenerate it with the current risk rule"
                )
            ax.plot(
                frame["date"],
                frame[column],
                color=utils.model_colour(model),
                linestyle=resolved_linestyles[style],
                label=f"{utils.model_label(model)} {label}",
                zorder=3 + list(METRIC_COLUMNS).index(style),
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
    # Lower left, one column: with both anchorings in one panel the onset-anchored curves start
    # their descent weeks earlier, straight through the mid-left region the RAC panel uses. Below
    # the plateau and above the onset bars is the one region every curve here leaves empty.
    ax.legend(loc="lower left", bbox_to_anchor=(0.015, 0.15))
    utils.panel_label(ax, letter)


def _drawn_metrics(
    model: str, metrics: tuple[RiskMetric, ...]
) -> list[tuple[RiskMetric, RiskMetric, str]]:
    """What to draw for one model: the estimand setting the style, the column, and the label.

    Onset-anchored models draw every requested estimand from its own column. An
    infection-anchored model's RAT is its RAC column, so with both requested it contributes one
    curve labelled ``RAC/RAT``, and asked for RAT alone it draws that same column in RAT's style.
    """
    if specification_of(model).anchoring == "onsets":
        return [(metric, metric, METRIC_LABELS[metric]) for metric in metrics]
    both = "rac" in metrics and "rat" in metrics
    drawn: list[tuple[RiskMetric, RiskMetric, str]] = []
    for metric in metrics:
        if metric == "rac":
            drawn.append(("rac", "rac", "RAC/RAT" if both else "RAC"))
        elif metric == "rat" and not both:
            drawn.append(("rat", "rac", "RAT"))
        elif metric == "rst":
            drawn.append(("rst", "rst", "RST"))
    return drawn


def reporting_comparison_panel(
    ax: Axes,
    curves: dict[float, dict[str, pd.DataFrame]],
    data: outbreak_data.OutbreakData,
    *,
    title: str,
    letter: str | None = None,
    first_day: int | None = None,
) -> None:
    """RAC against conditioning day, for each model at each assumed reporting probability.

    The same convention as :func:`risk_metrics_panel`, with the reporting probability in the
    estimand's place: colour identifies the model, linestyle the variant. So the vertical gap
    between two lines of one colour is what the assumption costs.

    ``curves`` is keyed by reporting probability and then by model. The settling markers are the
    same ones every RAC panel carries, so the crossings can be compared by eye across the sweep.
    """
    utils.plot_incidence(ax, data.dates, data.onsets)
    utils.mark_thresholds(ax)
    for order, (probability, by_model) in enumerate(sorted(curves.items(), reverse=True)):
        for model, frame in by_model.items():
            colour = utils.model_colour(model)
            ax.plot(
                frame["date"],
                frame[utils.RISK_COLUMN],
                color=colour,
                linestyle=utils.reporting_linestyle(probability),
                label=f"{utils.model_label(model)}, {probability:.0%} reported",
                zorder=3 + order,
            )
            day = settling_day(frame)
            if day is not None:
                ax.plot(
                    [data.date_of(day)],
                    [float(frame.loc[frame["day"] == day, utils.RISK_COLUMN].iloc[0])],
                    marker="o",
                    markersize=3.5,
                    color=colour,
                    zorder=6,
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
    ax.set_title(title)
    models = list(next(iter(curves.values())).keys())
    model_key = ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=utils.model_colour(model),
                label=utils.model_label(model),
            )
            for model in models
        ],
        title="Model",
        loc="center left",
        bbox_to_anchor=(0.015, 0.62),
    )
    ax.add_artist(model_key)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                linestyle=utils.reporting_linestyle(probability),
                label=f"{probability:.0%} reported",
            )
            for probability in sorted(curves, reverse=True)
        ],
        title="Reporting",
        loc="center left",
        bbox_to_anchor=(0.015, 0.32),
    )
    if letter is not None:
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
