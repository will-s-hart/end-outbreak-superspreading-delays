"""RAC for the four onset-comparison models with the R switchpoint removed.

Two supplementary analyses share this script, because they differ only in whether there is a
parameter posterior to draw. Both remove the switch, so neither carries the ``R_pre``/``R_post``
pair the report's fixed-``k`` figure devotes two panels to. Both draw the posterior model
probabilities, in the pie the other analysis figures use.

- ``no_switch_fixed_R`` fixes ``R`` as well as ``k``, so the figure is the pie beside the RAC
  panel. SSE has no free variable at all there, and its evidence is its likelihood, exactly.
- ``no_switch_single_R`` estimates one ``R`` per model, which gets a panel beside the pie and
  above the curves. It is read off ``R_pre``: with no switch that *is* the single reproduction
  number, and ``R_post`` reaches no day and merely returns its prior.

Both titles state that the reset to ``R_pre`` is a no-op here, so that no reader takes these
curves for the reset predictive the report's analyses report.
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

FIXED_R = "no_switch_fixed_R"
SINGLE_R = "no_switch_single_R"
ANALYSES = (FIXED_R, SINGLE_R)

# Removing the switchpoint is what these figures are for, and it brings the onset-anchored
# curves onto the naive ones — SSI and SSI-SO to within the line width. That coincidence is the
# result, but four solid lines would show it as a curve that failed to draw, so the two
# onset-anchored models are dash-dotted. Not dashed: the report's other figures keep every
# model solid and use dashes for RAT, and these panels draw RAC alone.
ONSET_ANCHORED_LINESTYLE = {"sse_so": "-.", "ssi_so": "-."}


def build_fixed_R_figure(
    curves: dict[str, pd.DataFrame],
    evidence: dict[str, Any],
    data: outbreak_data.OutbreakData,
    *,
    fixed_k: float,
    fixed_R: float,
) -> Figure:
    """The model probabilities beside the curves: every parameter is fixed, so no posterior.

    The row is the estimated-``k`` figures' bottom row, at the same width ratio, so the pie and
    the curves read at the sizes a reader has already seen them at.
    """
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 4.2))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 3.0], wspace=0.22)
    figure_panels.model_probability_panel(
        figure.add_subplot(grid[0, 0]), evidence, list(curves), letter="A"
    )
    figure_panels.risk_curve_panel(
        figure.add_subplot(grid[0, 1]),
        curves,
        data,
        letter="B",
        linestyles=ONSET_ANCHORED_LINESTYLE,
    )
    figure.suptitle(
        f"Équateur 2018: no $R$ switchpoint, $R$ fixed at {fixed_R:g} and $k$ at {fixed_k:g} "
        "— the reset is a no-op, so this is a forward predictive",
        fontsize=9.5,
    )
    return figure


def build_single_R_figure(
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    evidence: dict[str, Any],
    data: outbreak_data.OutbreakData,
    *,
    R_prior: LogNormalPrior,
    fixed_k: float,
) -> Figure:
    """One estimated ``R`` and the model probabilities, above the RAC curves they produced.

    The fixed-``k`` figures' grid with one parameter panel instead of two: the posterior takes
    the two columns ``R_pre`` and ``R_post`` would, and the pie keeps its column.
    """
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 6.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.6, wspace=0.35)
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, :2]),
        # `R_pre` is the single R: the switch sits past the end of the window.
        {model: utils.posterior_draws(posteriors[model], "R_pre") for model in curves},
        prior=R_prior,
        xlabel="$R$",
        title="One reproduction number for the whole outbreak",
        letter="A",
        legend=True,
    )
    figure_panels.model_probability_panel(
        figure.add_subplot(grid[0, 2]), evidence, list(curves), letter="B"
    )
    figure_panels.risk_curve_panel(
        figure.add_subplot(grid[1, :]),
        curves,
        data,
        letter="C",
        linestyles=ONSET_ANCHORED_LINESTYLE,
    )
    figure.suptitle(
        f"Équateur 2018: no $R$ switchpoint, a single estimated $R$ and $k$ fixed at "
        f"{fixed_k:g} — the reset is a no-op, so this is a forward predictive",
        fontsize=9.5,
    )
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, args.analysis)
    models = list(analysis["models"])
    data = outbreak_data.load_onset_data(args.data)
    curves = utils.read_risk_curves(args.results_dir, models)
    evidence = utils.read_model_evidence(args.results_dir)
    fixed_k = float(analysis["fixed_k"])

    if args.analysis == FIXED_R:
        figure = build_fixed_R_figure(
            curves, evidence, data, fixed_k=fixed_k, fixed_R=float(analysis["fixed_R_pre"])
        )
    else:
        figure = build_single_R_figure(
            curves,
            utils.read_posteriors(args.results_dir, models),
            evidence,
            data,
            R_prior=LogNormalPrior.from_config(analysis["shared"]["priors"]["R_pre"]),
            fixed_k=fixed_k,
        )
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--analysis", choices=ANALYSES, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
