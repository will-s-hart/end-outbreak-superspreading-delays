"""Stage-6 validation of the marginal likelihoods, on the real Équateur series.

``tests/test_model_evidence.py`` runs the same three comparisons on eight-day synthetic
histories, which is what keeps ``pixi run test`` quick. This script runs them where they
actually matter — 111 days, 31 latents, the posteriors Fig. 1 is drawn from — and writes down
the numbers::

    python validation/run_evidence_validation.py --checks all

``quadrature``
    **The acceptance criterion of §6.5.** At fixed ``k`` neither DLO nor SSE has latents, so
    ``log p(D | model)`` is a two-dimensional integral and can be done deterministically on a
    Simpson grid. Bridge sampling is then held against an answer rather than against another
    sampler: the grid shares only the integrand with it, and none of the proposal fitting, the
    weighting or the recursion. **Fails if** the two differ by more than a few times the
    reported standard error, or if the grid's boundary density is not far below its peak (in
    which case the box missed mass and the "exact" answer is not exact).

``estimators``
    **Three routes to SSI's evidence.** SSI has 31 latents, so there is no quadrature to check
    it against and the estimators have to check each other. Bridge sampling, importance sampling
    from a moment-matched independent normal, and naive prior Monte Carlo share no proposal and
    no weighting scheme. **Fails if** they disagree by more than their own error bars — which is
    the honest criterion, since prior Monte Carlo on a real series is legitimate but wide, and a
    fixed tolerance would be a test of the tolerance.

``marginalisation``
    **Exactly integrating latents out must not move the evidence.** ``marginalised_inverse_cdf``
    removes one of SSI's 31 latents in closed form and carries it as a ``pm.Potential``; plain
    ``inverse_cdf`` samples all 31. The normalising constant is the same integral either way, so
    a discrepancy means the potential is missing part of the factor — a normalising constant,
    say — and every Bayes factor in the report is wrong by it. **Fails if** the two differ by
    more than their combined Monte-Carlo error. This is the evidence-side counterpart of the
    matched-pair check that Stage 4 ran on the RAC curves.

Writes one CSV per check to ``validation/results/``; ``validation/results/evidence_validation.md``
is the written summary that goes with them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from end_of_outbreak import configuration, fitting, model_evidence, outbreak_data, pymc_models
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import LogNormalPrior

DEFAULT_OUTPUT_DIR = configuration.REPO_ROOT / "validation" / "results"
FIXED_K_ANALYSIS = "naive_models_fixed_k"
CLOSED_FORM_MODELS = ("dlo", "sse")


def _setting(analysis: dict[str, Any]) -> tuple[outbreak_data.OutbreakData, OnsetAnchoredDelays]:
    data = outbreak_data.load_onset_data(analysis["shared"]["data_file"])
    return data, configuration.onset_anchored_delays_from_config({"shared": analysis["shared"]})


def _priors(analysis: dict[str, Any]) -> tuple[LogNormalPrior, LogNormalPrior]:
    block = analysis["shared"]["priors"]
    return LogNormalPrior.from_config(block["R_pre"]), LogNormalPrior.from_config(block["R_post"])


# ---------------------------------------------------------------------------------------
# 1. Bridge sampling against deterministic quadrature
# ---------------------------------------------------------------------------------------


def check_quadrature(
    analysis: dict[str, Any], *, sampler: fitting.SamplerSettings, seed: int, n_points: int
) -> pd.DataFrame:
    """DLO and SSE: the whole integral done on a grid, and by bridge sampling."""
    data, delays = _setting(analysis)
    R_pre, R_post = _priors(analysis)
    k = float(analysis["fixed_k"])
    rows = []

    for model in CLOSED_FORM_MODELS:
        started = time.perf_counter()
        idata = fitting.fit_model(
            model,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            sampler=sampler,
        )
        built = pymc_models.build_model(
            model,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
        )
        target = model_evidence.UnconstrainedTarget(built)
        points = target.posterior_points(idata)
        exact = model_evidence.log_evidence_by_quadrature(
            target, centre=points.mean(axis=0), scale=points.std(axis=0), n_points=n_points
        )
        estimate = model_evidence.log_evidence(
            model,
            idata,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "model": model,
                "quadrature_log_evidence": exact.log_evidence,
                "quadrature_log_boundary_ratio": exact.log_boundary_ratio,
                "quadrature_points_per_axis": exact.n_points,
                "bridge_log_evidence": estimate.log_evidence,
                "bridge_standard_error": estimate.standard_error,
                "difference": estimate.log_evidence - exact.log_evidence,
                "difference_in_standard_errors": (
                    abs(estimate.log_evidence - exact.log_evidence) / estimate.standard_error
                ),
                "seconds": time.perf_counter() - started,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# 2. The three estimators on real-series SSI
# ---------------------------------------------------------------------------------------


def check_estimators(
    analysis: dict[str, Any],
    *,
    sampler: fitting.SamplerSettings,
    parameterisation: str,
    seed: int,
    n_proposal_draws: int,
) -> pd.DataFrame:
    """SSI, where there is no exact answer, so the estimators are held against each other."""
    data, delays = _setting(analysis)
    R_pre, R_post = _priors(analysis)
    k = float(analysis["fixed_k"])

    idata = fitting.fit_model(
        "ssi",
        data.onsets,
        delays=delays,
        switch_day=data.ert_arrival_day,
        R_pre=R_pre,
        R_post=R_post,
        k=k,
        latent_parameterisation=parameterisation,
        sampler=sampler,
    )
    rows = []
    for estimator in model_evidence.ESTIMATORS:
        started = time.perf_counter()
        estimate = model_evidence.log_evidence(
            "ssi",
            idata,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=parameterisation,
            estimator=estimator,
            n_proposal_draws=n_proposal_draws,
            rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "estimator": estimator,
                "log_evidence": estimate.log_evidence,
                "standard_error": estimate.standard_error,
                "n_draws": estimate.n_draws,
                "seconds": time.perf_counter() - started,
            }
        )
    frame = pd.DataFrame(rows)
    reference = frame.loc[frame["estimator"] == model_evidence.BRIDGE_SAMPLING].iloc[0]
    scale = np.hypot(frame["standard_error"], reference["standard_error"])
    frame["difference_from_bridge"] = frame["log_evidence"] - reference["log_evidence"]
    frame["difference_in_standard_errors"] = (frame["difference_from_bridge"] / scale).abs()
    return frame


# ---------------------------------------------------------------------------------------
# 3. Exact marginalisation leaves the evidence alone
# ---------------------------------------------------------------------------------------


def check_marginalisation(
    analysis: dict[str, Any], *, sampler: fitting.SamplerSettings, seed: int, n_proposal_draws: int
) -> pd.DataFrame:
    """The same SSI evidence, computed with 30 sampled latents and with 31."""
    data, delays = _setting(analysis)
    R_pre, R_post = _priors(analysis)
    k = float(analysis["fixed_k"])
    rows = []

    for parameterisation in ("marginalised_inverse_cdf", "inverse_cdf"):
        started = time.perf_counter()
        idata = fitting.fit_model(
            "ssi",
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=parameterisation,
            sampler=sampler,
        )
        built = pymc_models.build_model(
            "ssi",
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=parameterisation,
        )
        estimate = model_evidence.log_evidence(
            "ssi",
            idata,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=parameterisation,
            n_proposal_draws=n_proposal_draws,
            rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "latent_parameterisation": parameterisation,
                "free_coordinates": model_evidence.UnconstrainedTarget(built).dimension,
                "log_evidence": estimate.log_evidence,
                "standard_error": estimate.standard_error,
                "seconds": time.perf_counter() - started,
            }
        )
    frame = pd.DataFrame(rows)
    difference = float(frame["log_evidence"].iloc[0] - frame["log_evidence"].iloc[1])
    scale = float(np.hypot(*frame["standard_error"]))
    frame["difference"] = [difference, np.nan]
    frame["difference_in_standard_errors"] = [abs(difference) / scale, np.nan]
    return frame


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["all"],
        choices=["all", "quadrature", "estimators", "marginalisation"],
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--quadrature-points",
        type=int,
        default=401,
        help="grid points per axis; the boundary ratio in the CSV says whether it was enough",
    )
    parser.add_argument(
        "--proposal-draws",
        type=int,
        default=200_000,
        help="proposal/prior draws for the estimator comparison, where prior Monte Carlo needs "
        "far more than the posterior draw count to say anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, FIXED_K_ANALYSIS)
    sampler = fitting.SamplerSettings.from_config(analysis["sampler"])
    parameterisation = str(config["latent_parameterisation"])
    requested = set(args.checks)
    selected = {"quadrature", "estimators", "marginalisation"} if "all" in requested else requested
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "quadrature" in selected:
        frame = check_quadrature(
            analysis, sampler=sampler, seed=args.seed, n_points=args.quadrature_points
        )
        frame.to_csv(args.output_dir / "evidence_quadrature.csv", index=False)
        print(frame.to_string(index=False))

    if "estimators" in selected:
        frame = check_estimators(
            analysis,
            sampler=sampler,
            parameterisation=parameterisation,
            seed=args.seed,
            n_proposal_draws=args.proposal_draws,
        )
        frame.to_csv(args.output_dir / "evidence_estimators.csv", index=False)
        print(frame.to_string(index=False))

    if "marginalisation" in selected:
        frame = check_marginalisation(
            analysis, sampler=sampler, seed=args.seed, n_proposal_draws=args.proposal_draws
        )
        frame.to_csv(args.output_dir / "evidence_marginalisation.csv", index=False)
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
