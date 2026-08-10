"""The risk command writes exactly the metrics each model defines."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from end_of_outbreak import outbreak_data
from end_of_outbreak import risk_of_additional_cases as rac

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_risk_script() -> ModuleType:
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "run_risk_curves.py"
    spec = importlib.util.spec_from_file_location("_scripts_run_risk_curves", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


risk_script = load_risk_script()


def synthetic_data() -> outbreak_data.OutbreakData:
    return outbreak_data.OutbreakData(
        dates=pd.date_range("2018-04-05", periods=3, freq="D"),
        onsets=np.array([1, 0, 0]),
        ert_arrival_day=1,
        ert_withdrawal_day=2,
    )


def estimate(*, onset: bool, sustained: bool) -> rac.DailyRiskEstimate:
    cases = np.log(np.array([[0.4, 0.6], [0.6, 0.8]]))
    transmission = np.log(np.array([[0.5, 0.7], [0.7, 0.9]])) if onset else cases
    no_sustained = np.log(np.array([[0.8, 0.9], [0.9, 0.95]])) if sustained else None
    return rac.DailyRiskEstimate(
        days=np.array([1, 2]),
        log_no_further_cases=cases,
        log_no_further_transmission=transmission,
        log_no_sustained_transmission=no_sustained,
        n_chains=2,
    )


def test_dlo_csv_has_neither_rat_nor_rst_columns():
    columns = risk_script.risk_columns(
        estimate(onset=False, sustained=False), model="dlo", data=synthetic_data()
    )
    assert "risk_of_additional_transmission" not in columns
    assert "risk_of_sustained_transmission" not in columns


def test_infection_anchored_branching_csv_has_rst_but_no_separate_rat():
    columns = risk_script.risk_columns(
        estimate(onset=False, sustained=True), model="sse", data=synthetic_data()
    )
    assert "risk_of_additional_transmission" not in columns
    assert "risk_of_sustained_transmission" in columns


def test_onset_anchored_csv_has_all_three_ordered_risks():
    columns = risk_script.risk_columns(
        estimate(onset=True, sustained=True), model="sse_so", data=synthetic_data()
    )
    rac_values = np.asarray(columns["risk_of_additional_cases"])
    rat_values = np.asarray(columns["risk_of_additional_transmission"])
    rst_values = np.asarray(columns["risk_of_sustained_transmission"])
    assert np.all(rst_values <= rat_values)
    assert np.all(rat_values <= rac_values)
