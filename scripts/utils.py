"""Presentation-only helpers shared by the plotting scripts.

Nothing here computes a result. Every number a figure draws was written to ``results/`` by a
tier-2 rule; this module loads those files, decides what things look like, and gets a figure
onto disk. That split is what lets a restyle re-run in seconds without touching MCMC, so
resist the temptation to put a calculation here — if a figure needs a quantity that is not in
``results/``, the run script should be writing it out.

Two import decisions are deliberate:

- **No ``fitting`` import**, and so no PyMC. A plotting script would otherwise pay several
  seconds of model-building import cost to call what is a thin wrapper around
  :func:`xarray.open_datatree`. The posteriors are read directly here — a fit is an
  ``xarray.DataTree`` (``arviz.InferenceData`` is a deprecated alias for it in arviz 1.x), and
  a figure needs nothing from it but the draws of two scalars.
- **``risk_of_additional_cases`` is imported by ``figure_panels``, not by this module.** A RAC
  panel needs :class:`~end_of_outbreak.risk_of_additional_cases.RiskCurve.first_day_below` —
  the rule for when a curve has settled below a threshold — and that rule must live in exactly
  one place. It is the one tier-2 module the ``figure`` rule depends on, which is why it
  appears in ``PLOT_CORE`` in the ``Snakefile``. Keeping it out of here keeps this module free
  of PyMC too, since that is what the RAC calculators import.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from end_of_outbreak.model_specifications import LogNormalPrior, specification_of

matplotlib.use("Agg")

MODEL_COLOURS: dict[str, str] = {
    "dlo": "#D55E00",
    "sse": "#0072B2",
    "ssi": "#009E73",
    "sse_so": "#CC79A7",
    "ssi_so": "#E69F00",
    # cori and cori_so are validation targets rather than compared models; they get a neutral
    # grey on the rare figure that shows the Poisson limit for reference.
    "cori": "#7F7F7F",
    "cori_so": "#BFBFBF",
}
"""Okabe–Ito colours, one per model, fixed across every figure in the report."""

DECISION_THRESHOLDS: tuple[float, ...] = (0.05, 0.01)
"""The two RAC levels the end-of-outbreak literature declares on; drawn on every RAC panel."""

type ReportingLinestyle = Literal["-", "--", ":"]

REPORTING_LINESTYLES: dict[float, ReportingLinestyle] = {1.0: "-", 0.8: "--", 0.6: ":"}
"""Linestyle per assumed reporting probability, solid for complete reporting.

The same solid/dashed/dotted ladder ``risk_metrics_panel`` uses for the three risk estimands,
so a reader carries one convention between the two figures: colour is always the transmission
mechanism, linestyle always the variant being swept.
"""

RISK_COLUMN = "risk_of_additional_cases"
TRANSMISSION_RISK_COLUMN = "risk_of_additional_transmission"
SUSTAINED_RISK_COLUMN = "risk_of_sustained_transmission"
STANDARD_ERROR_COLUMN = "monte_carlo_standard_error"


def apply_house_style() -> None:
    """Set the rcParams every figure in the report shares."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.4,
        }
    )


def model_label(model: str) -> str:
    """Display name for a model, from its specification rather than a local table."""
    return specification_of(model).label


def model_colour(model: str) -> str:
    """Fixed colour for a model."""
    return MODEL_COLOURS[model]


def reporting_linestyle(probability: float) -> ReportingLinestyle:
    """Linestyle for an assumed reporting probability, raising on one with no style.

    No fallback, for the reason the readers of ``model_evidence`` get none: a sweep silently
    drawn in one linestyle looks exactly like a sweep that found no difference.
    """
    for level, style in REPORTING_LINESTYLES.items():
        if np.isclose(probability, level):
            return style
    known = ", ".join(f"{level:.0%}" for level in REPORTING_LINESTYLES)
    raise KeyError(f"no linestyle for {probability:.0%} reporting; utils defines {known}")


def panel_label(ax: Axes, letter: str) -> None:
    """Put a bold panel letter above the top-left corner of an axis."""
    ax.text(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def date_axis(ax: Axes) -> None:
    """Month ticks with abbreviated names, for a panel whose x-axis is calendar dates."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)


# ---------------------------------------------------------------------------------------
# Loading what the tier-2 rules wrote
# ---------------------------------------------------------------------------------------


def risk_curve_path(results_dir: str | Path, model: str) -> Path:
    return Path(results_dir) / f"{model}_rac.csv"


def posterior_path(results_dir: str | Path, model: str) -> Path:
    return Path(results_dir) / f"{model}_posterior.nc"


def evidence_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "model_evidence.json"


def dispersion_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "dispersion_posteriors.json"


def read_risk_curves(results_dir: str | Path, models: list[str]) -> dict[str, pd.DataFrame]:
    """Load one analysis's RAC curves, keyed by model, with ``date`` parsed."""
    curves = {}
    for model in models:
        frame = pd.read_csv(risk_curve_path(results_dir, model))
        frame["date"] = pd.to_datetime(frame["date"])
        curves[model] = frame
    return curves


def read_posteriors(results_dir: str | Path, models: list[str]) -> dict[str, xr.DataTree]:
    """Load one analysis's fits, keyed by model."""
    return {
        model: xr.open_datatree(str(posterior_path(results_dir, model)), engine="h5netcdf").load()
        for model in models
    }


def read_model_evidence(results_dir: str | Path) -> dict[str, Any]:
    """Load the evidence file, failing loudly and usefully when it is not there.

    The posterior model probabilities in panel C come from here and from nowhere else. A
    plotting script must never fall back to a placeholder: a pie chart of made-up numbers is
    indistinguishable from a real one on the page.
    """
    path = evidence_path(results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no model evidence at {path}. Panel C reports posterior model probabilities, "
            "which are computed by the `evidence` rule from end_of_outbreak.model_evidence — "
            f"run `snakemake {path}` (or the analysis's run script with the `evidence` "
            "subcommand) first. There is deliberately no fallback: a fabricated pie chart "
            "looks exactly like a real one."
        )
    with open(path) as handle:
        return json.load(handle)


def read_dispersion_summary(results_dir: str | Path) -> dict[str, Any]:
    """Load the ``k`` posterior summaries and their pairwise divergences.

    Same no-fallback rule as :func:`read_model_evidence`, for the same reason: the medians and
    credible intervals a ``k`` panel puts in its legend are quoted in the report, so they must
    be the ones the ``dispersion`` rule computed and not a summary the figure invented.
    """
    path = dispersion_path(results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no dispersion summary at {path}. The k panel reports posterior medians and "
            "credible intervals, which are computed by the `dispersion` rule from "
            f"end_of_outbreak.posterior_comparison — run `snakemake {path}` (or the analysis's "
            "run script with the `dispersion` subcommand) first."
        )
    with open(path) as handle:
        return json.load(handle)


def posterior_draws(idata: xr.DataTree, name: str) -> NDArray[np.float64]:
    """One scalar parameter's draws, chains flattened."""
    posterior = idata.posterior
    if name not in posterior.data_vars:
        available = ", ".join(sorted(posterior.data_vars))
        raise ValueError(f"the posterior has no variable {name!r}; it holds {available}")
    return np.asarray(posterior.data_vars[name]).reshape(-1)


# ---------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------


def positive_density(
    samples: NDArray[np.float64], *, grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Kernel density of a strictly positive quantity, estimated on the log scale.

    ``R_pre``, ``R_post`` and ``k`` are all positive and the posteriors are right-skewed, so a
    Gaussian kernel applied to the raw draws puts visible mass below zero and flattens the
    mode. Estimating in ``log`` and transforming back with the ``1/x`` Jacobian keeps the
    support right and costs nothing.
    """
    kernel = scipy.stats.gaussian_kde(np.log(samples))
    return np.asarray(kernel(np.log(grid))) / grid


def plot_parameter_posteriors(
    ax: Axes,
    draws: dict[str, NDArray[np.float64]],
    *,
    prior: LogNormalPrior | None = None,
    xlabel: str,
    labels: dict[str, str] | None = None,
    n_grid: int = 400,
) -> None:
    """Overlaid posterior densities for one parameter, one line per model.

    The prior is drawn too when given. It is not decoration: the posterior model probabilities
    of the model-probability panel compare the models *as specified*, priors included (§6.5), so
    a reader needs to see how much of each posterior is prior.

    ``labels`` overrides the legend entry for a model, which is how an estimated-``k``
    figure's ``k`` panel carries each posterior's median and credible interval. Those
    numbers come from the file the ``dispersion`` rule wrote; this module summarises
    nothing itself.
    """
    # A weakly identified R_pre has a long right tail — the 99.9th percentile of the Équateur
    # posterior is near 9 — and drawing out to it squashes the region the reader is comparing.
    # The curves are visually flat well before the 99th, so that is where the panel stops.
    stacked = np.concatenate(list(draws.values()))
    lower = float(np.quantile(stacked, 0.002)) * 0.8
    upper = float(np.quantile(stacked, 0.99)) * 1.1
    grid = np.linspace(max(lower, 1e-6), upper, n_grid)

    if prior is not None:
        ax.plot(
            grid,
            prior.frozen().pdf(grid),
            color="0.55",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label="prior",
            zorder=1,
        )
    for model, samples in draws.items():
        ax.plot(
            grid,
            positive_density(samples, grid=grid),
            color=model_colour(model),
            label=(labels or {}).get(model, model_label(model)),
            zorder=2,
        )
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("posterior density")


def plot_model_probability_pie(
    ax: Axes, probabilities: dict[str, float], *, minimum_label: float = 0.01
) -> None:
    """Posterior model probabilities as a pie, with the small slices labelled honestly.

    A slice below ``minimum_label`` is invisible on a pie, so its value goes in the legend
    rather than being silently dropped — the whole point of the model-probability panel is
    that one model is overwhelmingly favoured, and the size of "overwhelmingly" is the result.
    """
    models = list(probabilities)
    values = [probabilities[model] for model in models]
    wedges = ax.pie(
        values,
        colors=[model_colour(model) for model in models],
        radius=0.95,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )[0]
    labels = [
        f"{model_label(model)}  {_probability_text(probabilities[model], minimum_label)}"
        for model in models
    ]
    ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        handlelength=1.0,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    # An equal aspect shrinks the axes box to a square, and matplotlib centres the result in its
    # grid cell. Anchoring north instead keeps the title and the panel letter level with the
    # panels beside it, which is otherwise wrong wherever the cell is taller than it is wide.
    ax.set_aspect("equal")
    ax.set_anchor("N")


def _probability_text(probability: float, minimum_label: float) -> str:
    """A probability rendered so that "essentially zero" is still readable."""
    if probability >= minimum_label:
        return f"{probability:.2f}"
    if probability == 0.0:
        return "< 1e-300"
    return f"{probability:.0e}"


def plot_incidence(ax: Axes, dates: pd.DatetimeIndex, onsets: NDArray[np.int64]) -> Axes:
    """Daily onsets as background bars on a right-hand axis, returning that axis."""
    background = ax.twinx()
    background.bar(dates, onsets, width=1.0, color="0.82", zorder=0)
    background.set_ylim(0, max(int(onsets.max()) * 4, 1))
    background.set_yticks(range(0, int(onsets.max()) + 1, 2))
    background.set_ylabel("daily onsets", color="0.45")
    background.tick_params(axis="y", colors="0.45")
    background.spines["right"].set_visible(True)
    background.spines["right"].set_color("0.7")
    background.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    return background


def mark_intervention_dates(
    ax: Axes, *, arrival: datetime.date, withdrawal: datetime.date, label_y: float = 0.97
) -> None:
    """Vertical rules at the ERT's arrival and withdrawal, annotated.

    The withdrawal label goes to the *left* of its rule: withdrawal is the last day of the
    analysis window, so a right-hand label would hang off the axes.
    """
    for date, text, side in (
        (arrival, "ERT arrives", "left"),
        (withdrawal, "ERT withdraws", "right"),
    ):
        ax.axvline(
            mdates.date2num(date), color="0.35", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1
        )
        ax.annotate(
            text,
            xy=(mdates.date2num(date), label_y),
            xycoords=("data", "axes fraction"),
            xytext=(3 if side == "left" else -3, 0),
            textcoords="offset points",
            fontsize=7,
            color="0.35",
            va="top",
            ha=side,
        )


def mark_thresholds(ax: Axes, thresholds: tuple[float, ...] = DECISION_THRESHOLDS) -> None:
    """Horizontal rules at the decision thresholds, labelled at the left margin.

    The thresholds are 0.05 and 0.01 on an axis that runs to 1, so their labels sit four percent
    of the panel apart — about one line of type. They are staggered horizontally rather than
    vertically: moving one below its rule would push it under the axis, and nudging either off
    its own value would misreport where the rule is.
    """
    for index, threshold in enumerate(thresholds):
        ax.axhline(threshold, color="0.6", linestyle=":", linewidth=0.9, zorder=1)
        ax.annotate(
            f"{threshold:g}",
            xy=(0.004 + 0.035 * index, threshold),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=7,
            color="0.5",
            va="bottom",
            ha="left",
        )


def save_figure(figure: Figure, *, pdf: str | Path, png: str | Path) -> None:
    """Write both formats, creating the directories as needed."""
    for path in (pdf, png):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
