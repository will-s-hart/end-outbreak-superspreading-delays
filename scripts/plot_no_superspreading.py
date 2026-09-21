"""Exploratory: onset anchoring compared with superspreading removed.

``cori`` against ``cori_so``, the ``k → ∞`` Poisson limits of the two mechanisms. The report's
figures measure the naive/onset difference in models that also carry superspreading; this one
takes superspreading away, so what is left between the two curves is the anchoring convention
alone.

One analysis, so the analysis is hard-coded rather than taken as a flag. This is deliberately
*not* an extra branch in ``plot_onset_models_fixed_k.py``: that script reads
``analysis["fixed_k"]`` for its suptitle, which this analysis has no value for, and putting
exploratory branching on a report figure's dependency list is what the bespoke figure rules
exist to avoid.

The RAC panel is :func:`figure_panels.risk_curve_panel` rather than ``risk_metrics_panel``,
because the settling-day markers are the result here: the gap between them is the naive/onset
asymmetry with superspreading removed, and it is the one quantity this figure exists to show.
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

ANALYSIS = "onset_models_no_superspreading"

# `utils.MODEL_COLOURS` gives cori and cori_so neutral greys, which is right everywhere else:
# in the report's figures the Poisson limits would be reference lines behind the compared
# models. Here they *are* the comparison, and two greys would read as one model drawn twice.
# So the naive/onset convention the rest of the project uses is borrowed rather than invented:
# the naive model takes SSE's blue and the onset-anchored one takes SSE-SO's pink, which is the
# pairing a reader of the other figures already has in their eye. The palette itself is
# untouched — this is a local override, passed to each panel (`utils.model_colour`).
MODEL_COLOURS = {"cori": "#0072B2", "cori_so": "#CC79A7"}


def build_figure(
    *,
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    evidence: dict[str, Any],
    data: outbreak_data.OutbreakData,
    R_pre_prior: LogNormalPrior,
    R_post_prior: LogNormalPrior,
) -> Figure:
    """The fixed-``k`` four-panel layout, with the ``k`` that is not there left out.

    Two free scalars per model, so the arrangement is the one the fixed-``k`` figures use: the
    two ``R`` posteriors and the model probabilities across the top, the RAC curves along the
    bottom. There is deliberately no ``k`` panel and no dispersion input — the absence is the
    point of the analysis, not an omission from the figure.
    """
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 6.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.6, wspace=0.35)
    models = list(curves)
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, 0]),
        {model: utils.posterior_draws(posteriors[model], "R_pre") for model in models},
        prior=R_pre_prior,
        xlabel="$R_{\\mathrm{pre}}$",
        title="Before ERT arrival",
        letter="A",
        colours=MODEL_COLOURS,
        legend=True,
    )
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, 1]),
        {model: utils.posterior_draws(posteriors[model], "R_post") for model in models},
        prior=R_post_prior,
        xlabel="$R_{\\mathrm{post}}$",
        title="After ERT arrival",
        letter="B",
        colours=MODEL_COLOURS,
    )
    figure_panels.model_probability_panel(
        figure.add_subplot(grid[0, 2]), evidence, models, letter="C", colours=MODEL_COLOURS
    )
    # `first_day` left unset: the whole window, so that the climb over the opening weeks and
    # the two descents are both visible, and the two settling markers can be read off directly.
    figure_panels.risk_curve_panel(
        figure.add_subplot(grid[1, :]), curves, data, letter="D", colours=MODEL_COLOURS
    )
    figure.suptitle(
        "Équateur 2018: onset anchoring compared with no superspreading — the $k → ∞$ "
        "Poisson limits",
        fontsize=9.5,
    )
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
        data=outbreak_data.load_onset_data(args.data),
        R_pre_prior=LogNormalPrior.from_config(priors["R_pre"]),
        R_post_prior=LogNormalPrior.from_config(priors["R_post"]),
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
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
