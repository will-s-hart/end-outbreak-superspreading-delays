"""The three naive models with ``k`` estimated.

Five panels, reading left to right and then down::

    A  pre-ERT reproduction number   B  post-ERT reproduction number   C  dispersion $k$
    D  model probability             E  risk of additional cases over time

**What differs from the fixed-``k`` figure, and why.** Estimating ``k`` adds a third parameter
posterior to show, so this figure carries five panels where that one carries four: the three
parameters keep the top row, and the model-probability pie drops to the bottom row beside the
RAC curves. Panel E also starts at the ERT's arrival rather than at the first onset. From there
to well past the last observed case all three curves sit at 1, so on a panel that now shares its
row the opening weeks would compress the descent — the part the figure is about — into the
right-hand third. The curves themselves are drawn over the whole window either way; only the
view is trimmed, so the crossing markers cannot differ between a trimmed panel and an untrimmed
one.

Panel C is the new result of this analysis. The three models are given the *same* prior on ``k``
(§6.2) and land in different places, which is aim 2 measured rather than demonstrated. Each
posterior's median and 95% credible interval appear in the legend, read from the
``dispersion_posteriors.json`` the tier-2 ``dispersion`` rule wrote; that file also holds every
pairwise divergence — median ratio, posterior overlap, ``P(k_a > k_b)`` — which is where the
report's quantification of the disagreement comes from. Nothing here recomputes them.

**The line the split falls along.** DLO (0.38) and SSE (0.50) are barely distinguishable from
each other and sit three times above SSI (0.14), which is the only one compatible with the
literature 0.18. The division is *day-level against individual-level*: DLO and SSE both attach
their excess variance to a day, and the series being fitted is onsets, so the incubation period
has already smoothed away much of the day-to-day variation their ``k`` is estimated from. SSI's
variance is attached to individuals, which the same convolution merely regroups. The
onset-anchored estimated-``k`` figure tests that reading, where SSE-SO models the
incubation explicitly.

Panel D is the closer thing to a comparison of mechanisms that the fixed-``k`` figure's
panel C explicitly is not — with ``k`` estimated under a common prior, the models are
no longer being charged for a dispersion value that was never theirs. It still reports
the models *as specified, priors included* (§6.5).

Loads only what the tier-2 rules wrote, so a restyle costs seconds rather than an MCMC run::

    python scripts/plot_naive_models_estimated_k.py \\
        --results-dir results/naive_models_estimated_k \\
        --output-pdf figures/naive_models_estimated_k/naive_models_estimated_k.pdf \\
        --output-png figures/naive_models_estimated_k/naive_models_estimated_k.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import figure_panels
import matplotlib.pyplot as plt
import pandas as pd
import utils
import xarray as xr
from matplotlib.figure import Figure

from end_of_outbreak import configuration, outbreak_data
from end_of_outbreak.model_specifications import LogNormalPrior

ANALYSIS = "naive_models_estimated_k"


def build_figure(
    *,
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    evidence: dict[str, Any],
    dispersion: dict[str, Any],
    data: outbreak_data.OutbreakData,
    R_pre_prior: LogNormalPrior,
    R_post_prior: LogNormalPrior,
    k_prior: LogNormalPrior,
) -> Figure:
    """Assemble the five panels into one figure.

    A grid per row rather than one shared grid: the top row wants three equal columns and the
    bottom row an uneven two, since the pie reads at a quarter of the width and the RAC curves
    want the rest. Nesting keeps each row's spacing its own, and keeps the top row's at
    the fixed-``k`` figure's value so the two figures' parameter panels sit at the same
    size.
    """
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 6.4))
    rows = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)
    top = rows[0].subgridspec(1, 3, wspace=0.35)
    bottom = rows[1].subgridspec(1, 2, width_ratios=[1.15, 3.0], wspace=0.22)
    models = list(curves)

    figure_panels.parameter_posterior_panel(
        figure.add_subplot(top[0]),
        {model: utils.posterior_draws(posteriors[model], "R_pre") for model in models},
        prior=R_pre_prior,
        xlabel="$R_{\\mathrm{pre}}$",
        title="Before ERT arrival",
        letter="A",
        legend=True,
    )
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(top[1]),
        {model: utils.posterior_draws(posteriors[model], "R_post") for model in models},
        prior=R_post_prior,
        xlabel="$R_{\\mathrm{post}}$",
        title="After ERT arrival",
        letter="B",
    )
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(top[2]),
        {model: utils.posterior_draws(posteriors[model], "k") for model in models},
        prior=k_prior,
        xlabel="$k$",
        title="Dispersion, under a shared prior",
        letter="C",
        labels=figure_panels.credible_interval_labels(dispersion["posterior"]),
        legend=True,
    )
    figure_panels.model_probability_panel(
        figure.add_subplot(bottom[0]), evidence, models, letter="D"
    )
    figure_panels.risk_curve_panel(
        figure.add_subplot(bottom[1]),
        curves,
        data,
        letter="E",
        first_day=data.ert_arrival_day,
    )

    figure.suptitle("Équateur 2018: naive (onset-as-infection) models, $k$ estimated", fontsize=9.5)
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, ANALYSIS)
    models = list(analysis["models"])
    priors = analysis["shared"]["priors"]

    figure = build_figure(
        curves=utils.read_risk_curves(args.results_dir, models),
        posteriors=utils.read_posteriors(args.results_dir, models),
        evidence=utils.read_model_evidence(args.results_dir),
        dispersion=utils.read_dispersion_summary(args.results_dir),
        data=outbreak_data.load_onset_data(args.data),
        R_pre_prior=LogNormalPrior.from_config(priors["R_pre"]),
        R_post_prior=LogNormalPrior.from_config(priors["R_post"]),
        k_prior=LogNormalPrior.from_config(analysis["k_prior"]),
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
        help="where the tier-2 rules wrote this analysis's RAC curves, evidence and k summaries",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
