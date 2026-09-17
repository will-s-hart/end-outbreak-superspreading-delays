"""RAC, RAT and RST for the infection- and onset-anchored branching models.

Two panels, split by estimand rather than by anchoring, with all four models in each::

    A  RAC and RAT                 B  RST

The comparison the figure is for is naive against onset-anchored, and splitting by estimand
puts both conventions side by side in each panel, so the reader sees the gap between them hold
whichever risk is asked about. That RAC and RAT separate only under onset anchoring is still
visible — the infection-anchored models have one line in panel A where the onset-anchored ones
have two — but as the secondary point it is.

Panel B draws RST solid. With one estimand in the panel, linestyle has nothing to distinguish,
and a dotted line is the harder read.

One script, two figures: ``--analysis`` picks fixed or estimated ``k``.
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

ANALYSES = ("onset_models_fixed_k", "onset_models_estimated_k")
MODELS = ["sse", "ssi", "sse_so", "ssi_so"]


def build_figure(*, curves: dict[str, pd.DataFrame], data: outbreak_data.OutbreakData) -> Figure:
    """Assemble the figure from curves already loaded. Pure assembly: no I/O, no computation."""
    utils.apply_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharex=True, sharey=True)
    figure_panels.risk_metrics_panel(
        axes[0],
        curves,
        data,
        title="Cases and transmission",
        letter="A",
        metrics=("rac", "rat"),
        first_day=data.ert_arrival_day,
    )
    figure_panels.risk_metrics_panel(
        axes[1],
        curves,
        data,
        title="Sustained transmission",
        letter="B",
        metrics=("rst",),
        first_day=data.ert_arrival_day,
        metric_linestyles={"rst": "-"},
    )
    figure.subplots_adjust(wspace=0.24)
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    figure = build_figure(
        curves=utils.read_risk_curves(args.results_root / args.analysis, MODELS),
        data=outbreak_data.load_onset_data(args.data),
    )
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument("--analysis", choices=ANALYSES, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
