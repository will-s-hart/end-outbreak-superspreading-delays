"""The 2018 Équateur (DRC) Ebola outbreak: loading, padding and day indexing.

The observed series is a daily count of cases **by symptom-onset date**. The three
infection-anchored ("naive") models treat those counts as infection counts, ``I_t := C_t``;
the two onset-anchored models treat them as onsets, ``D_t := C_t``. This module is
deliberately agnostic between the two readings — it exposes the counts and the calendar, and
each model decides what they mean.

Day 0 is the date of the first observed onset (5 April 2018) and is an initial condition in
every model; likelihoods run over days ``1, ..., 110``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# --- key dates ---------------------------------------------------------------------------
# Reported in Supplementary Analysis 2 of Thompson et al. (2024), Nat Commun 15:5667.

FIRST_ONSET_DATE = datetime.date(2018, 4, 5)
"""Symptom-onset date of the index case; day 0 of every model's time index."""

ERT_ARRIVAL_DATE = datetime.date(2018, 5, 8)
"""Arrival of the Ebola Response Team; the day on which ``R`` switches from pre to post."""

LAST_OBSERVED_ONSET_DATE = datetime.date(2018, 6, 2)
"""Onset date of the final observed case."""

ERT_WITHDRAWAL_DATE = datetime.date(2018, 7, 24)
"""Actual withdrawal of the Ebola Response Team; the last day of the analysis window."""

TOTAL_CASES = 54
"""Total number of cases in the outbreak, used as a load-time consistency check."""

DEFAULT_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "equateur_2018_onsets.csv"
"""Committed, already-padded onset series (see ``data/README.md``)."""


def day_index_of(date: datetime.date, *, origin: datetime.date = FIRST_ONSET_DATE) -> int:
    """Day index of ``date`` relative to ``origin`` (default: the first onset date)."""
    return (date - origin).days


@dataclass(frozen=True)
class OutbreakData:
    """A daily onset series on a contiguous calendar, with the ERT day indices attached.

    Attributes
    ----------
    dates
        Contiguous daily ``DatetimeIndex``; ``dates[0]`` is day 0.
    onsets
        Case counts aligned with ``dates``.
    ert_arrival_day
        Day index on which ``R`` switches from ``R_pre`` to ``R_post`` (§5.7 of the plan:
        the switch happens on this day in *each model's own* time index).
    ert_withdrawal_day
        Day index of the final row, i.e. the end of the analysis window.
    """

    dates: pd.DatetimeIndex
    onsets: NDArray[np.int64]
    ert_arrival_day: int
    ert_withdrawal_day: int

    def __post_init__(self) -> None:
        if self.onsets.ndim != 1:
            raise ValueError("onsets must be one-dimensional")
        if len(self.dates) != self.onsets.size:
            raise ValueError("dates and onsets must have the same length")
        if not 0 <= self.ert_arrival_day < self.onsets.size:
            raise ValueError("ert_arrival_day lies outside the series")
        if self.ert_withdrawal_day != self.onsets.size - 1:
            raise ValueError("ert_withdrawal_day must index the final row of the series")

    # --- shape and totals ---

    @property
    def n_days(self) -> int:
        """Number of days in the analysis window (111 for the full Équateur series)."""
        return self.onsets.size

    @property
    def total_cases(self) -> int:
        """Total number of cases over the analysis window."""
        return int(self.onsets.sum())

    @property
    def day_index(self) -> NDArray[np.int64]:
        """``[0, 1, ..., n_days - 1]``."""
        return np.arange(self.n_days, dtype=np.int64)

    # --- pre/post-ERT split ---

    @property
    def pre_ert_mask(self) -> NDArray[np.bool_]:
        """True on days before the ERT arrived (days 0–32 for the Équateur series)."""
        return self.day_index < self.ert_arrival_day

    @property
    def post_ert_mask(self) -> NDArray[np.bool_]:
        """True from the day the ERT arrived onwards (days 33–110)."""
        return ~self.pre_ert_mask

    @property
    def reproduction_number_period(self) -> NDArray[np.int64]:
        """Index into ``(R_pre, R_post)`` for each day: 0 before ERT arrival, 1 after.

        Intended for use as ``R[reproduction_number_period]`` inside a model, so that the
        switch convention lives in one place.
        """
        return self.post_ert_mask.astype(np.int64)

    def date_of(self, day: int) -> datetime.date:
        """Calendar date of a day index."""
        return self.dates[day].date()

    def as_frame(self) -> pd.DataFrame:
        """The series as a tidy frame with ``date``, ``day`` and ``onsets`` columns."""
        return pd.DataFrame({"date": self.dates, "day": self.day_index, "onsets": self.onsets})


def pad_to(
    frame: pd.DataFrame,
    *,
    end_date: datetime.date,
    date_column: str = "date",
    count_column: str = "onsets",
) -> pd.DataFrame:
    """Reindex ``frame`` onto a contiguous daily calendar ending at ``end_date``.

    Missing days — including the long case-free tail between the last observed onset and the
    ERT withdrawal — are filled with zero counts. Idempotent: a frame that already covers the
    window is returned unchanged (up to sorting).
    """
    dates = pd.to_datetime(frame[date_column])
    end = pd.Timestamp(end_date)
    if dates.max() > end:
        raise ValueError(f"series extends past the padding target {end_date.isoformat()}")

    calendar = pd.date_range(start=dates.min(), end=end, freq="D")
    padded = (
        frame.assign(**{date_column: dates})
        .set_index(date_column)
        .reindex(calendar, fill_value=0)
        .rename_axis(date_column)
        .reset_index()
    )
    padded[count_column] = padded[count_column].astype(np.int64)
    return padded


def load_onset_data(
    path: str | Path = DEFAULT_DATA_FILE,
    *,
    end_date: datetime.date = ERT_WITHDRAWAL_DATE,
    ert_arrival_date: datetime.date = ERT_ARRIVAL_DATE,
    expected_total: int | None = TOTAL_CASES,
) -> OutbreakData:
    """Load the daily onset series, padding it out to ``end_date`` with zero counts.

    The committed CSV is already padded, so the padding step is normally a no-op; it is
    applied anyway so that the raw (unpadded) series loads correctly too.

    Parameters
    ----------
    path
        CSV with a ``date`` column and a count column named ``onsets`` or ``incidence``.
    end_date
        Last day of the analysis window; defaults to the actual ERT withdrawal date.
    ert_arrival_date
        Date on which ``R`` switches from ``R_pre`` to ``R_post``.
    expected_total
        If given, the total case count is checked against it and a mismatch raises. Pass
        ``None`` to skip (e.g. when loading a synthetic or subset series).
    """
    frame = pd.read_csv(path)
    if "onsets" not in frame.columns:
        if "incidence" not in frame.columns:
            raise ValueError("CSV must have an 'onsets' or 'incidence' count column")
        frame = frame.rename(columns={"incidence": "onsets"})
    frame = frame.loc[:, ["date", "onsets"]].sort_values("date")

    padded = pad_to(frame, end_date=end_date)
    dates = pd.DatetimeIndex(padded["date"])
    onsets = padded["onsets"].to_numpy(dtype=np.int64)

    if (onsets < 0).any():
        raise ValueError("onset counts must be non-negative")
    if expected_total is not None and int(onsets.sum()) != expected_total:
        raise ValueError(f"expected {expected_total} cases in total, found {int(onsets.sum())}")

    origin = dates[0].date()
    return OutbreakData(
        dates=dates,
        onsets=onsets,
        ert_arrival_day=day_index_of(ert_arrival_date, origin=origin),
        ert_withdrawal_day=day_index_of(end_date, origin=origin),
    )
