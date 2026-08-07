"""Supplementary figure — delay distributions used by the two anchoring conventions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from end_of_outbreak import configuration


def _last_visible_lag(
    weights: NDArray[np.float64], *, first_lag: int, cumulative_mass: float = 0.999
) -> int:
    """Last plotted lag, including ``cumulative_mass`` and one day of visual padding."""
    quantile_index = int(np.searchsorted(np.cumsum(weights), cumulative_mass))
    return first_lag + min(quantile_index + 1, weights.size - 1)


def _single_delay_panel(
    ax: Axes,
    lags: NDArray[np.int64],
    weights: NDArray[np.float64],
    *,
    colour: str,
    title: str,
    letter: str,
) -> None:
    ax.bar(lags, weights, width=0.86, color=colour, linewidth=0)
    ax.set_xlim(0, _last_visible_lag(weights, first_lag=int(lags[0])))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("delay (days)")
    ax.set_ylabel("probability mass")
    ax.set_title(title)
    utils.panel_label(ax, letter)


def _distribution(frame: pd.DataFrame, name: str) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    subset = frame.loc[frame["distribution"] == name]
    if subset.empty:
        raise ValueError(f"delay-distribution file has no {name!r} rows")
    return (
        np.asarray(subset["lag"], dtype=np.int64),
        np.asarray(subset["probability"], dtype=np.float64),
    )


def build_figure(frame: pd.DataFrame) -> Figure:
    """Assemble the three delay panels from the exact distributions used by the models."""
    utils.apply_house_style()
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))

    incubation_lags, incubation = _distribution(frame, "incubation")
    _single_delay_panel(
        axes[0],
        incubation_lags,
        incubation,
        colour="#0072B2",
        title="Incubation period",
        letter="A",
    )
    tost_lags, tost = _distribution(frame, "tost")
    _single_delay_panel(
        axes[1],
        tost_lags,
        tost,
        colour="#D55E00",
        title="Onset to transmission",
        letter="B",
    )

    target_lags, serial_interval = _distribution(frame, "serial_interval")
    implied_lags, implied_serial_interval = _distribution(frame, "implied_serial_interval")
    target_bars = axes[2].bar(
        target_lags,
        serial_interval,
        width=0.86,
        color="0.72",
        linewidth=0,
    )
    (implied_line,) = axes[2].plot(
        implied_lags,
        implied_serial_interval,
        color="#CC79A7",
        linestyle=(0, (3, 1.5)),
        linewidth=1.5,
    )
    axes[2].set_xlim(
        0,
        max(
            _last_visible_lag(
                serial_interval,
                first_lag=int(target_lags[0]),
            ),
            _last_visible_lag(
                implied_serial_interval,
                first_lag=int(implied_lags[0]),
            ),
        ),
    )
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlabel("delay (days)")
    axes[2].set_ylabel("probability mass")
    axes[2].set_title("Onset-to-onset serial interval")
    axes[2].legend(
        [target_bars, implied_line],
        ["infection-anchored input", "implied by A + B"],
        loc="upper right",
    )
    utils.panel_label(axes[2], "C")

    figure.subplots_adjust(wspace=0.4)
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    figure = build_figure(pd.read_csv(args.delays))
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delays",
        type=Path,
        default=configuration.REPO_ROOT / "results" / "delay_distributions.csv",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
