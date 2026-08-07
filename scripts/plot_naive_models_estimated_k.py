"""Figure 2 — the three naive models with ``k`` estimated, and its supplementary panel.

Four panels, reading left to right and then down::

    A  pre-ERT reproduction number    B  dispersion $k$    C  model probability
    D  risk of additional cases over time, with the onset series behind it

**The panel swap against Fig. 1, and why this way round.** Fig. 1 spends its second panel on the
post-ERT reproduction number; here that slot goes to the ``k`` posteriors, and ``R_post`` becomes
the standalone supplementary figure this script also draws. The three models are given the *same*
prior on ``k`` (§6.2) and land in different places, which is aim 2 measured rather than
demonstrated — so panel B is the new result of this analysis. ``R_post`` is the panel that loses
least by being displaced: it enters no headline quantity (the RAC resets ``R`` to ``R_pre``,
§5.1), it barely separates the models, and it barely moves between the two analyses. ``R_pre``
cannot go, because it is what panel D projects forward with; panel C cannot go either, since with
``k`` estimated under a common prior the model probabilities become the closer thing to a
comparison of mechanisms that Fig. 1C explicitly is not.

Panel B carries each posterior's median and 95% credible interval in its legend, read from the
``dispersion_posteriors.json`` the tier-2 ``dispersion`` rule wrote. That file also holds every
pairwise divergence — median ratio, posterior overlap, ``P(k_a > k_b)`` — which is where the
report's quantification of the disagreement comes from. Nothing here recomputes them.

Two invocations, one per figure::

    python scripts/plot_naive_models_estimated_k.py --figure main \\
        --output-pdf figures/naive_models_estimated_k/naive_models_estimated_k.pdf \\
        --output-png figures/naive_models_estimated_k/naive_models_estimated_k.png
    python scripts/plot_naive_models_estimated_k.py --figure supplementary \\
        --output-pdf figures/.../naive_models_estimated_k_supplementary.pdf \\
        --output-png figures/.../naive_models_estimated_k_supplementary.png
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

MAIN_FIGURE = "main"
SUPPLEMENTARY_FIGURE = "supplementary"
FIGURES = (MAIN_FIGURE, SUPPLEMENTARY_FIGURE)


def build_main_figure(
    *,
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    evidence: dict[str, Any],
    dispersion: dict[str, Any],
    data: outbreak_data.OutbreakData,
    R_pre_prior: LogNormalPrior,
    k_prior: LogNormalPrior,
) -> Figure:
    """Assemble the four panels of Fig. 2."""
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
        legend=True,
    )
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, 1]),
        {model: utils.posterior_draws(posteriors[model], "k") for model in models},
        prior=k_prior,
        xlabel="$k$",
        title="Dispersion, under a shared prior",
        letter="B",
        labels=figure_panels.credible_interval_labels(dispersion["posterior"]),
        legend=True,
    )
    figure_panels.model_probability_panel(
        figure.add_subplot(grid[0, 2]), evidence, models, letter="C"
    )
    figure_panels.risk_curve_panel(figure.add_subplot(grid[1, :]), curves, data, letter="D")

    figure.suptitle("Équateur 2018: naive (onset-as-infection) models, $k$ estimated", fontsize=9.5)
    return figure


def build_supplementary_figure(
    *, posteriors: dict[str, xr.DataTree], R_post_prior: LogNormalPrior
) -> Figure:
    """The post-ERT reproduction number, displaced from Fig. 2 by the ``k`` panel.

    Kept as a figure of its own rather than dropped: ``R_post`` is what the ERT's arrival is
    supposed to have changed, so a reader is entitled to see that the three mechanisms agree
    about it — and that agreement is itself the reason the panel could be displaced.
    """
    utils.apply_house_style()
    figure = plt.figure(figsize=(4.0, 3.2))
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(1, 1, 1),
        {model: utils.posterior_draws(idata, "R_post") for model, idata in posteriors.items()},
        prior=R_post_prior,
        xlabel="$R_{\\mathrm{post}}$",
        title="After ERT arrival, $k$ estimated",
        legend=True,
    )
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, ANALYSIS)
    models = list(analysis["models"])
    priors = analysis["shared"]["priors"]
    posteriors = utils.read_posteriors(args.results_dir, models)

    if args.figure == MAIN_FIGURE:
        figure = build_main_figure(
            curves=utils.read_risk_curves(args.results_dir, models),
            posteriors=posteriors,
            evidence=utils.read_model_evidence(args.results_dir),
            dispersion=utils.read_dispersion_summary(args.results_dir),
            data=outbreak_data.load_onset_data(args.data),
            R_pre_prior=LogNormalPrior.from_config(priors["R_pre"]),
            k_prior=LogNormalPrior.from_config(analysis["k_prior"]),
        )
    else:
        figure = build_supplementary_figure(
            posteriors=posteriors,
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
        help="where the tier-2 rules wrote this analysis's RAC curves, evidence and k summaries",
    )
    parser.add_argument(
        "--figure",
        choices=FIGURES,
        default=MAIN_FIGURE,
        help="which of this analysis's two figures to draw",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
