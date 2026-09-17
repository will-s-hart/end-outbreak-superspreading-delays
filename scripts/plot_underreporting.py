"""RAC under incomplete reporting — SSE-SO and SSI-SO at 100%, 80% and 60%.

One panel, six curves: colour is the transmission mechanism and linestyle the assumed reporting
probability, following the convention of the sustained-transmission figure. The view starts at
the ERT's arrival, which is also where the two under-reported curves start being computed.

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

MODELS = ["sse_so", "ssi_so"]

ANALYSIS_OF_PROBABILITY: dict[float, str] = {
    1.0: "onset_models_fixed_k",
    0.8: "underreporting_80",
    0.6: "underreporting_60",
}
"""Where each rung of the sweep's curves live. Mirrors ``UNDERREPORTING_ANALYSES`` in the
``Snakefile``, which is what puts them on this rule's input list."""


def build_figure(
    *, curves: dict[float, dict[str, pd.DataFrame]], data: outbreak_data.OutbreakData
) -> Figure:
    """Assemble the figure from curves already loaded. Pure assembly: no I/O, no computation."""
    figure = plt.figure(figsize=(6.6, 4.3))
    figure_panels.reporting_comparison_panel(
        figure.add_subplot(1, 1, 1),
        curves,
        data,
        title="Risk of at least one further case after day $t$, by assumed reporting",
        first_day=data.ert_arrival_day,
    )
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
