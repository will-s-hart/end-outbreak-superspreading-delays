"""Tests for the model registry, the shared priors and the parameter container."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from end_of_outbreak import model_specifications as ms

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

# --- LogNormalPrior ----------------------------------------------------------------------


def test_prior_reproduces_its_median_and_quantile():
    prior = ms.LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)
    frozen = prior.frozen()
    assert frozen.median() == pytest.approx(1.0)
    assert frozen.ppf(0.025) == pytest.approx(0.2)


def test_the_k_prior_covers_the_configured_interval():
    # The project's k prior: median 0.18 with a 95% interval of roughly (0.09, 0.36).
    prior = ms.LogNormalPrior.from_config(
        {"distribution": "lognormal", "median": 0.18, "quantile_025": 0.09, "quantile_975": 0.36}
    )
    frozen = prior.frozen()
    assert frozen.ppf(0.025) == pytest.approx(0.09)
    assert frozen.ppf(0.975) == pytest.approx(0.36)


def test_prior_pymc_kwargs_match_the_scipy_view():
    prior = ms.LogNormalPrior.from_median_and_quantile(
        median=0.18, quantile=0.09, probability=0.025
    )
    kwargs = prior.pymc_kwargs()
    assert kwargs["mu"] == pytest.approx(np.log(0.18))
    assert kwargs["sigma"] == pytest.approx(prior.sigma_log)


def test_prior_accepts_either_quantile_alone():
    from_lower = ms.LogNormalPrior.from_config(
        {"median": 0.18, "quantile_025": 0.09},
    )
    from_upper = ms.LogNormalPrior.from_config(
        {"median": 0.18, "quantile_975": 0.36},
    )
    assert from_lower.sigma_log == pytest.approx(from_upper.sigma_log)


def test_prior_rejects_quantiles_that_no_lognormal_can_satisfy():
    # A log-normal is symmetric in log space, so the two quantiles are not free.
    with pytest.raises(ValueError, match="symmetric in log space"):
        ms.LogNormalPrior.from_config({"median": 0.18, "quantile_025": 0.09, "quantile_975": 0.9})


def test_prior_needs_at_least_one_quantile():
    with pytest.raises(ValueError, match="quantile"):
        ms.LogNormalPrior.from_config({"median": 1.0})


def test_prior_rejects_an_unsupported_family():
    with pytest.raises(ValueError, match="lognormal"):
        ms.LogNormalPrior.from_config({"distribution": "gamma", "median": 1.0})


@pytest.mark.parametrize(("median", "sigma_log"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
def test_prior_rejects_non_positive_parameters(median, sigma_log):
    with pytest.raises(ValueError):
        ms.LogNormalPrior(median=median, sigma_log=sigma_log)


def test_the_configured_priors_load():
    # config.yaml is the single source of truth for the priors; check it stays parseable and
    # that R_pre and R_post really are given the same prior (§6.2).
    config = yaml.safe_load(CONFIG_FILE.read_text())
    R_pre = ms.LogNormalPrior.from_config(config["shared"]["priors"]["R_pre"])
    R_post = ms.LogNormalPrior.from_config(config["shared"]["priors"]["R_post"])
    assert R_pre == R_post
    assert R_pre.frozen().ppf(0.025) == pytest.approx(0.2)

    for name in ("naive_models_estimated_k", "onset_models_estimated_k"):
        k_prior = ms.LogNormalPrior.from_config(config["analyses"][name]["k_prior"])
        assert k_prior.median == pytest.approx(0.18)


# --- the registry ------------------------------------------------------------------------


def test_every_registered_model_resolves_by_name():
    for name, specification in ms.MODEL_SPECIFICATIONS.items():
        assert ms.specification_of(name) is specification
        assert ms.specification_of(specification) is specification


def test_unknown_model_names_list_the_known_ones():
    with pytest.raises(KeyError, match="unknown model"):
        ms.specification_of("sse_so_typo")


def test_the_naive_models_are_the_three_infection_anchored_ones():
    assert tuple(m.name for m in ms.NAIVE_MODELS) == ("dlo", "sse", "ssi")
    assert all(m.anchoring == "infections" for m in ms.NAIVE_MODELS)


def test_only_ssi_carries_latents_among_the_naive_models():
    assert {m.name for m in ms.NAIVE_MODELS if m.has_latents} == {"ssi"}
    assert ms.SSI.latent_variable == "Y"


def test_the_overdispersion_levels_are_the_axis_being_compared():
    assert ms.DLO.overdispersion_level == "day"
    assert ms.SSE.overdispersion_level == "event"
    assert ms.SSI.overdispersion_level == "individual"
    assert {m.overdispersion_level for m in ms.NAIVE_MODELS} == {"day", "event", "individual"}


def test_cori_is_the_dispersion_free_limit_and_is_not_a_compared_model():
    assert ms.CORI.overdispersion_level == "none"
    assert not ms.CORI.has_dispersion
    assert ms.CORI not in ms.NAIVE_MODELS


def test_cori_so_is_the_onset_anchored_limit_with_neither_dispersion_nor_latents():
    """Both absences matter to `onset_models_no_superspreading`.

    No dispersion is why that analysis's block gives neither `fixed_k` nor `k_prior`; no latent
    block is why it needs no `latent_parameterisation` of its own and why its ~220 fits are a
    laptop job. `cori_so` is the only onset-anchored model with no latent block, so nothing
    else in the project pins this.
    """
    assert ms.CORI_SO.overdispersion_level == "none"
    assert not ms.CORI_SO.has_dispersion
    assert not ms.CORI_SO.has_latents
    assert ms.CORI_SO.anchoring == "onsets"
    assert ms.CORI_SO not in ms.ONSET_ANCHORED_MODELS


# --- TransmissionParameters --------------------------------------------------------------


def test_parameters_require_positive_values():
    with pytest.raises(ValueError, match="R_pre and R_post"):
        ms.TransmissionParameters(R_pre=0.0, R_post=1.0)
    with pytest.raises(ValueError, match="k must be positive"):
        ms.TransmissionParameters(R_pre=1.0, R_post=1.0, k=-0.1)


def test_require_k_reports_which_model_needed_it():
    parameters = ms.TransmissionParameters(R_pre=1.0, R_post=0.5)
    assert parameters.k is None
    with pytest.raises(ValueError, match="'sse'"):
        parameters.require_k(ms.SSE)
    assert ms.TransmissionParameters(R_pre=1.0, R_post=0.5, k=0.3).require_k(ms.SSE) == 0.3
