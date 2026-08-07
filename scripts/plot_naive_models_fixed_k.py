"""Figure 1 — the three naive models with ``k`` held at 0.18.

Four panels, reading left to right and then down::

    A  pre-ERT reproduction number    B  post-ERT reproduction number    C  model probability
    D  risk of additional cases over time, with the onset series behind it

Panel D is the result the paper is about. All three models see the same onsets, the same serial
interval and the same ``k``, and they disagree about the risk of a further case by a factor of
several — because a dispersion calibrated on *individual* offspring counts does almost no work
once it is applied to a whole day's aggregate incidence (§5.4). The tail of the outbreak is the
regime that makes this worst: the remaining force of infection is thin and spread over the whole
serial interval, which is exactly where DLO's day-level ``k`` washes out and its risk climbs back
towards the Poisson answer.

Panel C is neither decoration nor a mechanism comparison. It reports the evidence for the models
**as specified, priors included** (§6.5) — and one of these specifications applies an
individual-level ``k`` to a day-level mechanism deliberately, as the demonstration of a mistake
made in the literature. The caption must say so. Fig. 2, where ``k`` is estimated under a common
prior, is the closer thing to a comparison of mechanisms.

Loads only what the tier-2 rules wrote, so a restyle costs seconds rather than an MCMC run::

    python scripts/plot_naive_models_fixed_k.py --results-dir results/naive_models_fixed_k \\
        --output-pdf figures/naive_models_fixed_k/naive_models_fixed_k.pdf \\
        --output-png figures/naive_models_fixed_k/naive_models_fixed_k.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import xarray as xr
from matplotlib.figure import Figure

from end_of_outbreak import configuration, outbreak_data
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.model_specifications import LogNormalPrior

ANALYSIS = "naive_models_fixed_k"

SETTLING_THRESHOLD = 0.05
"""The threshold whose crossing is marked on panel D; the report quotes 0.01 as well."""


def build_figure(
    *,
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    evidence: dict[str, Any],
    data: outbreak_data.OutbreakData,
    R_pre_prior: LogNormalPrior,
    R_post_prior: LogNormalPrior,
    fixed_k: float,
) -> Figure:
    """Assemble the four panels into one figure."""
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 6.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.6, wspace=0.35)
    models = list(curves)

    _plot_reproduction_numbers(
        figure.add_subplot(grid[0, 0]),
        {model: utils.posterior_draws(posteriors[model], "R_pre") for model in models},
        prior=R_pre_prior,
        xlabel="$R_{\\mathrm{pre}}$",
        title="Before ERT arrival",
        letter="A",
        legend=True,
    )
    _plot_reproduction_numbers(
        figure.add_subplot(grid[0, 1]),
        {model: utils.posterior_draws(posteriors[model], "R_post") for model in models},
        prior=R_post_prior,
        xlabel="$R_{\\mathrm{post}}$",
        title="After ERT arrival",
        letter="B",
        legend=False,
    )
    _plot_model_probabilities(figure.add_subplot(grid[0, 2]), evidence, models)
    _plot_risk_curves(figure.add_subplot(grid[1, :]), curves, data)

    figure.suptitle(
        f"Équateur 2018: naive (onset-as-infection) models, $k$ fixed at {fixed_k:g}",
        fontsize=9.5,
    )
    return figure


def _plot_reproduction_numbers(
    ax: Any,
    draws: dict[str, np.ndarray],
    *,
    prior: LogNormalPrior,
    xlabel: str,
    title: str,
    letter: str,
    legend: bool,
) -> None:
    """Panels A and B: one posterior density per model, over the shared prior."""
    utils.plot_parameter_posteriors(ax, draws, prior=prior, xlabel=xlabel)
    ax.set_title(title)
    utils.panel_label(ax, letter)
    if legend:
        ax.legend(loc="upper right")


def _plot_model_probabilities(ax: Any, evidence: dict[str, Any], models: list[str]) -> None:
    """Panel C: the posterior model probabilities, straight from the evidence file."""
    probabilities = {
        model: float(evidence["posterior_model_probability"][model]) for model in models
    }
    utils.plot_model_probability_pie(ax, probabilities)
    ax.set_title("Posterior model probability")
    utils.panel_label(ax, "C")


def _plot_risk_curves(
    ax: Any, curves: dict[str, pd.DataFrame], data: outbreak_data.OutbreakData
) -> None:
    """Panel D: RAC over the conditioning days, with the onsets behind it."""
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
                data.date_of(settled),
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
    ax.set_xlim(data.dates[0], data.dates[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("conditioning day $t$ (2018)")
    ax.set_ylabel("risk of additional cases")
    ax.set_title("Risk of at least one further case after day $t$")
    # Mid-left: the curves sit flat at ~1 until the last observed case and the onset bars are
    # confined to the bottom, so this is the one large region of the panel that stays empty.
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.55))
    utils.panel_label(ax, "D")


def settling_day(frame: pd.DataFrame, threshold: float = SETTLING_THRESHOLD) -> int | None:
    """The first day a curve falls below ``threshold`` and stays there.

    Delegated to :class:`~end_of_outbreak.risk_of_additional_cases.RiskCurve` rather than
    reimplemented here, because "settles below" is a definition the report quotes and it must not
    be able to drift between the marker on the panel and the number in the text.
    """
    curve = rac.RiskCurve(
        days=np.asarray(frame["day"], dtype=np.int64),
        risk=np.asarray(frame[utils.RISK_COLUMN], dtype=np.float64),
        n_draws=0,
    )
    return curve.first_day_below(threshold)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, ANALYSIS)
    data = outbreak_data.load_onset_data(args.data)
    models = list(analysis["models"])
    priors = analysis["shared"]["priors"]

    figure = build_figure(
        curves=utils.read_risk_curves(args.results_dir, models),
        posteriors=utils.read_posteriors(args.results_dir, models),
        evidence=utils.read_model_evidence(args.results_dir),
        data=data,
        R_pre_prior=LogNormalPrior.from_config(priors["R_pre"]),
        R_post_prior=LogNormalPrior.from_config(priors["R_post"]),
        fixed_k=float(analysis["fixed_k"]),
    )
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=configuration.REPO_ROOT / "results" / ANALYSIS,
        help="where the tier-2 rules wrote this analysis's RAC curves and evidence",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
