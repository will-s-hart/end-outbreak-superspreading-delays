"""RAC under incomplete reporting — naive and onset-anchored models at 100%, 80% and 60%.

Four panels, reading left to right and then down::

    A  naive models, RAC by reporting        B  onset-anchored models, RAC by reporting
    C  all four, RAC and RAT at 80%          D  all four, RAC and RAT at 60%

The question is whether the naive/onset gap survives the reporting assumption. The top row asks
it across the sweep, one anchoring convention per panel, with colour the model and linestyle the
assumed reporting probability. The bottom row asks it at each under-reported rung, with both
conventions in one panel, and there linestyle is the estimand instead — the convention of the
sustained-transmission figure. Each panel's legend says which it is using. The view starts at the
ERT's arrival, which is also where the under-reported curves start being computed.

The 100% curves are Analysis 3's. At a reporting probability of one the reporting layer is the
identity and the model *is* that analysis's, so reusing its committed curves makes the baseline
of this figure the paper's headline result rather than a Monte-Carlo-different rerun of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import figure_panels
import matplotlib.pyplot as plt
import pandas as pd
import utils
from matplotlib.figure import Figure

from end_of_outbreak import configuration, outbreak_data

NAIVE_MODELS = ["sse", "ssi"]
ONSET_MODELS = ["sse_so", "ssi_so"]
MODELS = NAIVE_MODELS + ONSET_MODELS

ANALYSIS_OF_PROBABILITY: dict[float, str] = {
    1.0: "onset_models_fixed_k",
    0.8: "underreporting_80",
    0.6: "underreporting_60",
}
"""Where each rung of the sweep's curves live. Mirrors ``UNDERREPORTING_ANALYSES`` in the
``Snakefile``, which is what puts them on this rule's input list."""

ESTIMAND_PANEL_PROBABILITIES = (0.8, 0.6)
"""The under-reported rungs that get a RAC/RAT panel of their own, in panel order."""


def build_figure(
    *, curves: dict[float, dict[str, pd.DataFrame]], data: outbreak_data.OutbreakData
) -> Figure:
    """Assemble the figure from curves already loaded. Pure assembly: no I/O, no computation."""
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.6), sharex=True, sharey=True)
    for ax, models, title, letter in (
        (axes[0, 0], NAIVE_MODELS, "Infection-anchored models, by assumed reporting", "A"),
        (axes[0, 1], ONSET_MODELS, "Onset-anchored models, by assumed reporting", "B"),
    ):
        figure_panels.reporting_comparison_panel(
            ax,
            {
                probability: {model: by_model[model] for model in models}
                for probability, by_model in curves.items()
            },
            data,
            title=title,
            letter=letter,
            first_day=data.ert_arrival_day,
        )
    for ax, probability, letter in zip(
        axes[1], ESTIMAND_PANEL_PROBABILITIES, ("C", "D"), strict=True
    ):
        figure_panels.risk_metrics_panel(
            ax,
            {model: curves[probability][model] for model in MODELS},
            data,
            title=f"Cases and transmission, {probability:.0%} reported",
            letter=letter,
            metrics=("rac", "rat"),
            first_day=data.ert_arrival_day,
        )
    # `figure.axes`, not `axes`: each panel's onset backdrop is a twin with its label on the right,
    # and only the right-hand column's should survive, or it sits in the gap between the panels.
    for ax in figure.axes:
        ax.label_outer()
    figure.subplots_adjust(wspace=0.12, hspace=0.18)
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    data = outbreak_data.load_onset_data(args.data)
    curves = {
        probability: utils.read_risk_curves(args.results_root / analysis, MODELS)
        for probability, analysis in ANALYSIS_OF_PROBABILITY.items()
    }
    utils.apply_house_style()
    figure = build_figure(curves=curves, data=data)
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
