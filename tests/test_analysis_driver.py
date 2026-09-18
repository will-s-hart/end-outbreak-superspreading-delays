"""How an analysis block in the config becomes the arguments a fit is run with.

`AnalysisSetting` is the only bridge between `config/config.yaml` and the builders, and two of
its resolvers decide things no draw records: whether a parameter is fixed or estimated, and
which day ``R`` switches on. Getting either wrong produces a perfectly plausible fit of the
wrong model, so they are pinned here against the committed config rather than a synthetic one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from end_of_outbreak import configuration, outbreak_data
from end_of_outbreak.model_specifications import LogNormalPrior

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_analysis_driver() -> ModuleType:
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analysis_driver.py"
    spec = importlib.util.spec_from_file_location("_scripts_analysis_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis_driver = load_analysis_driver()


@pytest.fixture(scope="module")
def setting_of():
    def load(analysis: str):
        return analysis_driver.AnalysisSetting.load(
            analysis,
            config_path=configuration.DEFAULT_CONFIG_FILE,
            data_path=outbreak_data.DEFAULT_DATA_FILE,
        )

    return load


# --- what the committed config asks of the sampler ---------------------------------------


def test_the_reporting_sweeps_are_thinned_and_the_report_analyses_are_not(setting_of):
    """Thinning is confined to the analyses that compute no model evidence.

    Where a Bayes factor is computed, the joint draws feed it, and on this data the two
    onset-anchored models are separated by about 2.4 standard errors -- there is no precision
    to give away. The sweeps compute none, which is what makes their fits thinnable.
    """
    from end_of_outbreak import fitting

    for analysis in ("underreporting_60", "underreporting_80"):
        sampler = fitting.SamplerSettings.from_config(setting_of(analysis).block["sampler"])
        assert sampler.thin == 5, analysis
        assert not sampler.reuses_final_day_fit, analysis
    for analysis in (
        "naive_models_fixed_k",
        "naive_models_estimated_k",
        "onset_models_fixed_k",
        "onset_models_estimated_k",
    ):
        sampler = fitting.SamplerSettings.from_config(setting_of(analysis).block["sampler"])
        assert sampler.thin == 1, analysis
        assert sampler.reuses_final_day_fit, analysis


# --- fixed or estimated -----------------------------------------------------------------


def test_an_analysis_that_does_not_fix_R_gets_the_shared_priors(setting_of):
    R_pre, R_post = setting_of("onset_models_fixed_k").reproduction_numbers()
    assert isinstance(R_pre, LogNormalPrior)
    assert isinstance(R_post, LogNormalPrior)


def test_a_fixed_R_reaches_the_builders_as_a_float(setting_of):
    """Which is what makes it a constant in the graph rather than a random variable."""
    assert setting_of("no_switch_fixed_R").reproduction_numbers() == (0.95, 0.95)


def test_the_priors_stay_available_even_where_R_is_fixed(setting_of):
    """The figures draw the prior behind a posterior, so the two accessors are not the same."""
    R_pre_prior, _ = setting_of("no_switch_fixed_R").reproduction_number_priors()
    assert isinstance(R_pre_prior, LogNormalPrior)


# --- the switch day ---------------------------------------------------------------------


def test_the_switch_defaults_to_the_ert_arrival_day(setting_of):
    setting = setting_of("onset_models_fixed_k")
    assert setting.switch_day() == setting.data.ert_arrival_day == 33


@pytest.mark.parametrize("analysis", ["no_switch_fixed_R", "no_switch_single_R"])
def test_the_no_switchpoint_variants_move_the_switch_past_the_window(analysis, setting_of):
    """Which is what leaves ``R_pre`` in force on every day, in each model's own time index."""
    setting = setting_of(analysis)
    assert setting.switch_day() >= setting.data.n_days
