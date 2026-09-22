"""Tests for loading, padding and day-indexing the Équateur 2018 onset series."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from end_of_outbreak import outbreak_data


@pytest.fixture(scope="module")
def data() -> outbreak_data.OutbreakData:
    return outbreak_data.load_onset_data()


# --- the committed series ----------------------------------------------------------------


def test_analysis_window_is_131_days_totalling_54_cases(data):
    assert data.n_days == 131
    assert data.last_day == 130
    assert data.total_cases == outbreak_data.TOTAL_CASES == 54


def test_window_runs_from_first_onset_to_twenty_days_after_ert_withdrawal(data):
    assert data.date_of(0) == outbreak_data.FIRST_ONSET_DATE == datetime.date(2018, 4, 5)
    assert data.date_of(-1) == outbreak_data.ANALYSIS_END_DATE == datetime.date(2018, 8, 13)
    assert data.last_day - data.ert_withdrawal_day == 20


def test_calendar_is_contiguous_and_daily(data):
    assert (np.diff(data.dates.values).astype("timedelta64[D]") == np.timedelta64(1, "D")).all()


def test_ert_day_indices(data):
    assert data.ert_arrival_day == 33
    assert data.date_of(data.ert_arrival_day) == outbreak_data.ERT_ARRIVAL_DATE
    assert data.ert_withdrawal_day == 110
    assert data.date_of(data.ert_withdrawal_day) == outbreak_data.ERT_WITHDRAWAL_DATE


def test_pre_and_post_ert_case_totals(data):
    assert int(data.onsets[data.pre_ert_mask].sum()) == 28
    assert int(data.onsets[data.post_ert_mask].sum()) == 26
    assert data.pre_ert_mask.sum() == 33
    assert data.post_ert_mask.sum() == 98


def test_index_case_on_day_zero(data):
    assert data.onsets[0] == 1


def test_last_observed_onset_is_followed_by_72_zero_days(data):
    last_onset_day = int(np.flatnonzero(data.onsets)[-1])
    assert data.date_of(last_onset_day) == outbreak_data.LAST_OBSERVED_ONSET_DATE
    assert last_onset_day == 58
    trailing = data.onsets[last_onset_day + 1 :]
    assert trailing.size == 72
    assert not trailing.any()


def test_the_committed_file_stops_at_the_withdrawal_and_the_loader_pads_the_rest(data):
    """So that the window the validation studies ran on can still be loaded exactly."""
    committed = pd.read_csv(outbreak_data.DEFAULT_DATA_FILE)
    assert committed["date"].iloc[-1] == outbreak_data.ERT_WITHDRAWAL_DATE.isoformat()
    historical = outbreak_data.load_onset_data(end_date=outbreak_data.ERT_WITHDRAWAL_DATE)
    assert historical.n_days == 111
    assert historical.ert_withdrawal_day == historical.last_day == 110
    np.testing.assert_array_equal(historical.onsets, data.onsets[: historical.n_days])


def test_thirty_one_days_carry_a_case(data):
    # The number of latent infectivities Y_t the SSI model must sample.
    assert int(np.count_nonzero(data.onsets)) == 31


# --- the R-switch convention -------------------------------------------------------------


def test_reproduction_number_period_switches_on_ert_arrival(data):
    period = data.reproduction_number_period
    assert period[data.ert_arrival_day - 1] == 0
    assert period[data.ert_arrival_day] == 1
    assert set(np.unique(period)) == {0, 1}


# --- padding -----------------------------------------------------------------------------


def test_padding_fills_missing_days_with_zeros():
    frame = pd.DataFrame({"date": ["2018-04-05", "2018-04-08"], "onsets": [1, 2]})
    padded = outbreak_data.pad_to(frame, end_date=datetime.date(2018, 4, 10))
    assert len(padded) == 6
    assert padded["onsets"].tolist() == [1, 0, 0, 2, 0, 0]


def test_padding_is_idempotent():
    frame = pd.DataFrame({"date": ["2018-04-05", "2018-04-06"], "onsets": [1, 2]})
    once = outbreak_data.pad_to(frame, end_date=datetime.date(2018, 4, 8))
    twice = outbreak_data.pad_to(once, end_date=datetime.date(2018, 4, 8))
    pd.testing.assert_frame_equal(once, twice)


def test_padding_rejects_a_series_extending_past_the_target():
    frame = pd.DataFrame({"date": ["2018-04-05", "2018-04-30"], "onsets": [1, 2]})
    with pytest.raises(ValueError, match="past the padding target"):
        outbreak_data.pad_to(frame, end_date=datetime.date(2018, 4, 10))


def test_loader_pads_an_unpadded_series(tmp_path):
    # The raw starter series stops at the last observed onset; loading it must reproduce the
    # committed padded series exactly.
    committed = outbreak_data.load_onset_data()
    truncated = committed.as_frame().iloc[: committed.ert_arrival_day + 26]
    path = tmp_path / "truncated.csv"
    truncated.loc[:, ["date", "onsets"]].to_csv(path, index=False)

    reloaded = outbreak_data.load_onset_data(path, expected_total=int(truncated["onsets"].sum()))
    assert reloaded.n_days == committed.n_days
    np.testing.assert_array_equal(reloaded.onsets[: len(truncated)], truncated["onsets"].to_numpy())
    assert not reloaded.onsets[len(truncated) :].any()


# --- validation --------------------------------------------------------------------------


def test_loader_accepts_an_incidence_column(tmp_path):
    path = tmp_path / "incidence.csv"
    pd.DataFrame({"date": ["2018-04-05"], "incidence": [54]}).to_csv(path, index=False)
    assert outbreak_data.load_onset_data(path).total_cases == 54


def test_loader_rejects_an_unexpected_total(tmp_path):
    path = tmp_path / "wrong_total.csv"
    pd.DataFrame({"date": ["2018-04-05"], "onsets": [3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="expected 54 cases"):
        outbreak_data.load_onset_data(path)


def test_loader_rejects_negative_counts(tmp_path):
    path = tmp_path / "negative.csv"
    pd.DataFrame({"date": ["2018-04-05", "2018-04-06"], "onsets": [55, -1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="non-negative"):
        outbreak_data.load_onset_data(path)


def test_loader_rejects_a_missing_count_column(tmp_path):
    path = tmp_path / "no_counts.csv"
    pd.DataFrame({"date": ["2018-04-05"], "cases": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="count column"):
        outbreak_data.load_onset_data(path)


def test_withdrawal_day_must_lie_inside_the_window():
    with pytest.raises(ValueError, match="between the ERT's arrival and the end"):
        outbreak_data.OutbreakData(
            dates=pd.DatetimeIndex(pd.date_range("2018-04-05", periods=5, freq="D")),
            onsets=np.ones(5, dtype=np.int64),
            ert_arrival_day=2,
            ert_withdrawal_day=110,
        )
