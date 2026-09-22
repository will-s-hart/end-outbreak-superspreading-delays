"""Model probabilities over an analysis's own models and the ones it borrows.

The superspreading/anchoring figure's pie normalises over the Poisson limits and Analysis 3's
SSI and SSI-SO together. That is only a comparison if the two analyses fitted their models under
the same assumptions, so the refusals are pinned as carefully as the arithmetic.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from end_of_outbreak import configuration

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


combined_evidence = load_script("run_combined_evidence")

ANALYSIS = "onset_models_no_superspreading"


def evidence_file(analysis: str, log_evidence: dict[str, float]) -> dict:
    """An evidence file of the shape the `evidence` rule writes."""
    return {
        "analysis": analysis,
        "models": list(log_evidence),
        "estimator": "bridge_sampling",
        "log_evidence": log_evidence,
        "log_evidence_standard_error": dict.fromkeys(log_evidence, 0.01),
        "n_draws": dict.fromkeys(log_evidence, 12000),
        "posterior_model_probability": {},
    }


@pytest.fixture
def config() -> dict:
    return configuration.load_config()


@pytest.fixture
def evidences() -> tuple[dict, dict]:
    own = evidence_file(ANALYSIS, {"cori": -91.8, "cori_so": -82.6})
    compared = evidence_file(
        "onset_models_fixed_k", {"sse": -106.8, "ssi": -84.8, "sse_so": -81.8, "ssi_so": -81.8}
    )
    return own, compared


def test_the_committed_config_borrows_analysis_3s_ssi_and_ssi_so(config):
    borrowed = config["analyses"][ANALYSIS]["compared_with"]
    assert borrowed == {"analysis": "onset_models_fixed_k", "models": ["ssi", "ssi_so"]}


def test_the_four_are_normalised_together_in_order_own_models_first(config, evidences):
    combined = combined_evidence.combine(config, ANALYSIS, *evidences)
    assert combined["models"] == ["cori", "cori_so", "ssi", "ssi_so"]
    assert combined["sources"]["ssi"] == "onset_models_fixed_k"
    probabilities = combined["posterior_model_probability"]
    assert math.fsum(probabilities.values()) == pytest.approx(1.0)
    # The Bayes factor between two models is untouched by which others share the normalisation.
    assert math.log(probabilities["ssi_so"] / probabilities["cori_so"]) == pytest.approx(0.8)
    assert combined["log_evidence"]["ssi"] == -84.8


def test_analyses_fitted_under_different_assumptions_are_refused(config, evidences):
    """A moved switch changes what every evidence is of; renormalising them would mislead."""
    altered = copy.deepcopy(config)
    altered["analyses"]["onset_models_fixed_k"]["switch_day"] = 40
    with pytest.raises(ValueError, match="differ in switch_day"):
        combined_evidence.combine(altered, ANALYSIS, *evidences)


def test_a_borrowed_model_the_other_analysis_did_not_fit_is_refused(config, evidences):
    own, compared = evidences
    del compared["log_evidence"]["ssi_so"]
    compared["models"].remove("ssi_so")
    with pytest.raises(ValueError, match="no evidence for ssi_so"):
        combined_evidence.combine(config, ANALYSIS, own, compared)


def test_an_evidence_file_for_the_wrong_analysis_is_refused(config, evidences):
    own, compared = evidences
    compared["analysis"] = "onset_models_estimated_k"
    with pytest.raises(ValueError, match="was written for onset_models_estimated_k"):
        combined_evidence.combine(config, ANALYSIS, own, compared)
