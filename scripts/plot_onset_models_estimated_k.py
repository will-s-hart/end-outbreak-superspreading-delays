"""Naive and onset-anchored SSE/SSI with ``k`` estimated.

The five-panel layout of the naive estimated-``k`` figure, so the two read side by side. The
``k`` panel is on a log axis: under the shared prior, a decade wide on each side of its median,
the posteriors run from about 0.01 to 5, and on a linear axis every one near zero is a spike at
the edge with the prior invisible behind it — and whether a posterior has moved off that prior
is the comparison the panel exists to show.
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

ANALYSIS = "onset_models_estimated_k"


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
    """Assemble it in the same five-panel layout as the naive estimated-``k`` figure."""
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
        log_scale=True,
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
    figure.suptitle("Équateur 2018: onset anchoring compared, $k$ estimated", fontsize=9.5)
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
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
