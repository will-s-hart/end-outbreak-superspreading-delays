"""Stage-4 validation of the RAC calculators, in three parts.

Each answers a different question, and none of them substitutes for another::

    python validation/run_rac_validation.py --checks all

``thompson``
    **External validation.** Re-runs the Équateur analysis under Thompson et al.'s own
    conventions — Poisson limit, ``R`` estimated from the pre-ERT data alone with a flat prior,
    risk of cases *on or after* day ``t`` given data strictly before it — and holds the result
    against their published closed form (eqs. (3)–(5)) and against Figs. S3D/E of
    ``thompson_supp.pdf``. Nothing in this project's machinery is exercised by the closed form
    itself, so the comparison tests the whole pipeline against an outside answer.

``equality``
    **The §6.4 equality check**, at fixed ``(R_pre, R_post, k)`` and with the conditioning
    matched. For the latent-free models the reset state is the observed counts, so the closed
    forms are checked directly against forward simulation. For SSI the retained state is
    latent, and a *smoothed* one, so the comparison is against a particle **smoother** — the
    filter's ancestral paths — never against its filtering output. The filtering curve is
    computed too, but as the §5.6 measurement, not as a pass criterion.

``matched_pair``
    **The latent-reconstruction check.** Fits SSI twice, once under
    ``marginalised_inverse_cdf`` (where the uncoupled latents are integrated out and have to be
    rebuilt from their conditional) and once under plain ``inverse_cdf`` (where nothing is
    removed), and requires the RAC curves to agree. Since the two fits share no latent block,
    agreement is an end-to-end check on the reconstruction rather than on the likelihood. A
    second fit under the *same* parameterisation with a different seed calibrates what "within
    Monte-Carlo error" means here, instead of leaving it to judgement.

Writes one CSV per check to ``validation/results/`` plus a figure for the Thompson replication;
``validation/results/rac_validation.md`` is the written summary that goes with them.
"""

from __future__ import annotations

import argparse
import datetime
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from end_of_outbreak import configuration, fitting, outbreak_data, particle_filter, pymc_models
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import LogNormalPrior, TransmissionParameters

matplotlib.use("Agg")

DEFAULT_OUTPUT_DIR = configuration.REPO_ROOT / "validation" / "results"
FIXED_K_ANALYSIS = "naive_models_fixed_k"
CLOSED_FORM_MODELS = ("dlo", "sse", "cori")


def _setting(analysis: dict[str, Any]) -> tuple[outbreak_data.OutbreakData, OnsetAnchoredDelays]:
    data = outbreak_data.load_onset_data(analysis["shared"]["data_file"])
    return data, configuration.onset_anchored_delays_from_config({"shared": analysis["shared"]})


def _dates(data: outbreak_data.OutbreakData) -> list[datetime.date]:
    """Calendar dates of the analysis window, for the CSV index."""
    return [data.date_of(int(day)) for day in data.day_index]


# ---------------------------------------------------------------------------------------
# 1. External validation against Thompson et al. (2024)
# ---------------------------------------------------------------------------------------


def check_thompson(
    data: outbreak_data.OutbreakData,
    delays: OnsetAnchoredDelays,
    *,
    output_dir: Path,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Reproduce their Équateur risk curve through this project's RAC machinery."""
    counts, w = data.onsets, delays.serial_interval
    shape, rate = rac.thompson_reproduction_number_posterior(
        counts, w, last_day=data.ert_arrival_day - 1
    )
    print(
        f"  their R posterior: Gamma(shape={shape:.1f}, rate={rate:.3f}) — "
        f"mean {shape / rate:.3f}, mode {(shape - 1) / rate:.3f}, "
        f"sd {np.sqrt(shape) / rate:.3f}   [Fig. S3D, dashed]"
    )

    theirs = rac.thompson_withdrawal_risk(counts, w, shape=shape, rate=rate)
    rng = np.random.default_rng(seed)
    R = rng.gamma(shape, 1.0 / rate, size=n_draws)
    ours = rac.risk_curve("cori", counts=counts, serial_interval=w, R_pre=R)
    # Their day t is our day t − 1; line them up on our index.
    aligned = np.concatenate((theirs, [np.nan]))

    frame = pd.DataFrame(
        {
            "day": data.day_index,
            "date": _dates(data),
            "onsets": counts,
            "remaining_weight": rac.pooled_remaining_weight(counts.astype(float), w),
            "rac_ours": ours.risk,
            "risk_thompson_next_day": aligned,
            "difference": ours.risk - aligned,
        }
    )
    difference = np.abs(frame["difference"].to_numpy()[:-1])
    print(
        f"  max |ours(t) − theirs(t+1)| = {difference.max():.2e} over {n_draws} draws "
        f"(pure Monte-Carlo error; the identity is exact)"
    )
    for threshold in (0.05, 0.01):
        day = ours.first_day_below(threshold)
        where = "never within the window" if day is None else f"day {day} ({data.date_of(day)})"
        print(f"  RAC first below {threshold:g}: {where}")
    print(f"  RAC on the withdrawal day (day {data.ert_withdrawal_day}): {ours.risk[-1]:.4f}")

    _plot_thompson(data, ours, output_dir / "rac_thompson_replication.png")
    return frame


def _plot_thompson(
    data: outbreak_data.OutbreakData, curve: rac.RiskCurve, destination: Path
) -> None:
    """The curve to hold against Fig. S3E (dashed line) of ``thompson_supp.pdf``."""
    figure, axes = plt.subplots(figsize=(7.0, 3.6))
    axes.plot(data.dates, curve.risk, color="tab:blue", linestyle="--", label="replicated")
    axes.axvline(data.dates[data.ert_withdrawal_day], color="black", linestyle=":")
    axes.annotate(
        "ERT withdrawal",
        (data.dates[data.ert_withdrawal_day], 0.75),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    for threshold in (0.05, 0.01):
        axes.axhline(threshold, color="grey", linewidth=0.6)
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("Risk of withdrawing ERT")
    axes.set_xlabel("Date")
    axes.set_title("Équateur 2018, generic serial interval — cf. Fig. S3E (dashed)", fontsize=10)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------------------
# 2. The §6.4 equality check at fixed parameters
# ---------------------------------------------------------------------------------------


def check_equality(
    data: outbreak_data.OutbreakData,
    delays: OnsetAnchoredDelays,
    *,
    parameters: TransmissionParameters,
    parameterisation: str,
    sampler: fitting.SamplerSettings,
    n_replicates: int,
    n_particles: int,
    simulation_stride: int,
    seed: int,
) -> pd.DataFrame:
    """Closed forms against simulation, and the SSI curve against a particle smoother."""
    counts, w = data.onsets, delays.serial_interval
    rows: list[dict[str, Any]] = []
    columns: dict[str, NDArray[np.float64]] = {}
    # Simulating the reset future costs a full replicate set per conditioning day, so the
    # simulation arm runs on a grid of days rather than on all 111. The closed forms are
    # evaluated on the same grid, and nothing about the comparison depends on which days.
    simulation_days = np.union1d(
        data.day_index[::simulation_stride], [data.ert_arrival_day, data.ert_withdrawal_day]
    ).astype(np.int64)
    columns["simulated_here"] = np.isin(data.day_index, simulation_days).astype(np.float64)

    for model in CLOSED_FORM_MODELS:
        k = None if model == "cori" else parameters.k
        analytic = rac.risk_curve(
            model, counts=counts, serial_interval=w, R_pre=parameters.R_pre, k=k
        )
        simulated = rac.simulated_risk_curve(
            model,
            counts=counts,
            serial_interval=w,
            parameters=TransmissionParameters(parameters.R_pre, parameters.R_post, k),
            days=simulation_days,
            n_replicates=n_replicates,
            rng=np.random.default_rng(seed),
        )
        columns[f"{model}_analytic"] = analytic.risk
        columns[f"{model}_simulated"] = np.where(
            np.isin(data.day_index, simulation_days),
            np.interp(data.day_index, simulation_days, simulated.risk),
            np.nan,
        )
        difference = np.abs(analytic.risk[simulation_days] - simulated.risk)
        standard_error = np.sqrt(analytic.risk * (1 - analytic.risk) / n_replicates).max()
        rows.append(
            {
                "comparison": f"{model}: closed form vs forward simulation",
                "max_abs_difference": difference.max(),
                "monte_carlo_scale": 3 * standard_error,
            }
        )
        # The filter carries no latent state for these models, so its evidence is exact.
        built = pymc_models.build_naive_model(
            model,
            counts,
            serial_interval=w,
            switch_day=data.ert_arrival_day,
            R_pre=parameters.R_pre,
            R_post=parameters.R_post,
            k=k,
        )
        exact = pymc_models.compile_observation_logp(built)()
        filtered = particle_filter.filter_naive(
            model,
            counts,
            TransmissionParameters(parameters.R_pre, parameters.R_post, k),
            serial_interval=w,
            switch_day=data.ert_arrival_day,
            n_particles=8,
            rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "comparison": f"{model}: SMC log-evidence vs closed-form likelihood",
                "max_abs_difference": abs(filtered.log_evidence - exact),
                "monte_carlo_scale": 0.0,
            }
        )

    # SSI: the retained state is latent, so the comparison is MCMC (smoothed) against the
    # filter's ancestral paths (also smoothed). The filtering curve goes in as a measurement.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fitting.fit_model(
            "ssi",
            counts,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=parameters.R_pre,
            R_post=parameters.R_post,
            k=parameters.k,
            latent_parameterisation=parameterisation,
            sampler=sampler,
        )
    state = rac.posterior_state(
        "ssi",
        idata,
        counts,
        delays=delays,
        switch_day=data.ert_arrival_day,
        latent_parameterisation=parameterisation,
        fixed_R_pre=parameters.R_pre,
        fixed_R_post=parameters.R_post,
        fixed_k=parameters.k,
        rng=np.random.default_rng(seed),
    )
    assert state.infectivity is not None
    mcmc_log_probability = rac.log_probability_of_no_further_cases(
        "ssi",
        counts=counts,
        serial_interval=w,
        R_pre=state.R_pre,
        k=state.k,
        infectivity=state.infectivity,
    )
    mcmc = rac.RiskCurve(
        days=data.day_index,
        risk=1.0 - np.exp(mcmc_log_probability).mean(axis=0),
        n_draws=state.n_draws,
    )
    mcmc_error = rac.monte_carlo_standard_error(mcmc_log_probability, n_chains=state.n_chains)
    filtered = particle_filter.filter_naive(
        "ssi",
        counts,
        parameters,
        serial_interval=w,
        switch_day=data.ert_arrival_day,
        n_particles=n_particles,
        rng=np.random.default_rng(seed),
    )
    assert filtered.latent_paths is not None
    smoothed = rac.risk_curve(
        "ssi",
        counts=counts,
        serial_interval=w,
        R_pre=np.full(n_particles, parameters.R_pre),
        k=np.full(n_particles, parameters.k),
        infectivity=filtered.latent_paths,
    )
    filtering_risk = 1.0 - np.exp(-parameters.R_pre * filtered.filtering_remaining_weight).mean(
        axis=1
    )

    columns["ssi_mcmc_smoothed"] = mcmc.risk
    columns["ssi_particle_smoothed"] = smoothed.risk
    columns["ssi_particle_filtering"] = filtering_risk
    columns["ssi_mcmc_standard_error"] = mcmc_error
    rows.append(
        {
            "comparison": "ssi: MCMC smoothed vs particle smoother (matched conditioning)",
            "max_abs_difference": np.abs(mcmc.risk - smoothed.risk).max(),
            "monte_carlo_scale": 3 * float(np.nanmax(mcmc_error)),
        }
    )
    rows.append(
        {
            "comparison": "ssi: smoothed vs filtering (the §5.6 gap — a measurement, not a test)",
            "max_abs_difference": np.abs(mcmc.risk - filtering_risk).max(),
            "monte_carlo_scale": np.nan,
        }
    )
    signed = mcmc.risk - filtering_risk
    late = data.day_index >= 58
    print(
        f"  §5.6 gap (smoothed − filtering): max |·| {np.abs(signed).max():.4f}; "
        f"after the last case, mean {signed[late].mean():+.4f}, "
        f"range [{signed[late].min():+.4f}, {signed[late].max():+.4f}]"
    )
    print(
        f"  filter degeneracy: {filtered.n_distinct[0]} distinct day-0 ancestors of "
        f"{n_particles}; min ESS {filtered.effective_sample_size.min():.0f}; "
        f"{int(filtered.resampled.sum())} resampling steps"
    )
    for row in rows:
        print(
            f"  {row['comparison']}: max |Δ| = {row['max_abs_difference']:.2e}"
            + (
                ""
                if np.isnan(row["monte_carlo_scale"])
                else f" (3 s.e. ≈ {row['monte_carlo_scale']:.2e})"
            )
        )

    curves = pd.DataFrame({"day": data.day_index, "date": _dates(data), **columns})
    curves.attrs["summary"] = pd.DataFrame(rows)
    return curves


def measure_smc_variance(
    data: outbreak_data.OutbreakData,
    delays: OnsetAnchoredDelays,
    *,
    parameters: TransmissionParameters,
    particle_counts: tuple[int, ...],
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """``Var(log L̂)`` for the SSI filter, at a plausible ``θ`` and several particle counts.

    Not a Stage-4 requirement, but it is the measurement Stage 4b's stopping rule turns on
    (PMMH mixes acceptably only when this is around 1–3 at the posterior mode), and the forward
    pass that produces it is already written. Recording it now costs seconds and settles the
    go/no-go before the sampler is written.
    """
    rows = []
    for n_particles in particle_counts:
        estimates = np.array(
            [
                particle_filter.filter_naive(
                    "ssi",
                    data.onsets,
                    parameters,
                    serial_interval=delays.serial_interval,
                    switch_day=data.ert_arrival_day,
                    n_particles=n_particles,
                    rng=np.random.default_rng(seed + repeat),
                ).log_evidence
                for repeat in range(n_repeats)
            ]
        )
        rows.append(
            {
                "n_particles": n_particles,
                "n_repeats": n_repeats,
                "mean_log_evidence": estimates.mean(),
                "sd_log_evidence": estimates.std(ddof=1),
                "var_log_evidence": estimates.var(ddof=1),
            }
        )
        print(
            f"  N = {n_particles:>6}: log L̂ = {estimates.mean():.3f} "
            f"± {estimates.std(ddof=1):.3f} (Var = {estimates.var(ddof=1):.3f})"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# 3. The matched pair: does the reconstruction give back what was integrated out?
# ---------------------------------------------------------------------------------------


def _ssi_rac_curve(
    data: outbreak_data.OutbreakData,
    delays: OnsetAnchoredDelays,
    *,
    idata: Any,
    fixed_k: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(RAC, Monte-Carlo standard error, mean latent path)`` from one SSI fit."""
    parameterisation = fitting.fitted_parameterisation(idata)
    assert parameterisation is not None
    state = rac.posterior_state(
        "ssi",
        idata,
        data.onsets,
        delays=delays,
        switch_day=data.ert_arrival_day,
        latent_parameterisation=parameterisation,
        fixed_k=fixed_k,
        # A separate stream per curve, so that a repeated fit is a genuinely independent
        # replicate: sharing one would correlate the reconstruction draws and flatter the
        # within-parameterisation baseline the comparison is judged against.
        rng=np.random.default_rng(seed),
    )
    assert state.infectivity is not None
    log_probability = rac.log_probability_of_no_further_cases(
        "ssi",
        counts=data.onsets,
        serial_interval=delays.serial_interval,
        R_pre=state.R_pre,
        k=state.k,
        infectivity=state.infectivity,
    )
    return (
        1.0 - np.exp(log_probability).mean(axis=0),
        rac.monte_carlo_standard_error(log_probability, n_chains=state.n_chains),
        state.infectivity.mean(axis=0),
    )


def check_matched_pair(
    data: outbreak_data.OutbreakData,
    delays: OnsetAnchoredDelays,
    *,
    fixed_k: float,
    R_prior: LogNormalPrior,
    sampler: fitting.SamplerSettings,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Fit SSI with and without marginalisation and compare the RAC curves they produce.

    Repeated over ``n_repeats`` independent triples, because a single pair cannot distinguish
    "the reconstruction is exact" from "the two fits happened to land close". Each triple gives
    one *across* difference (marginalised vs plain) and one *within* difference (marginalised
    vs marginalised, different seed); if the reconstruction is exact the two are draws from the
    same distribution, and that is what the comparison reports.
    """
    counts = data.onsets

    def fit(parameterisation: str, chain_seed: int) -> Any:
        settings = fitting.SamplerSettings(
            draws=sampler.draws,
            tune=sampler.tune,
            chains=sampler.chains,
            target_accept=sampler.target_accept,
            seed=chain_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fitting.fit_model(
                "ssi",
                counts,
                delays=delays,
                switch_day=data.ert_arrival_day,
                R_pre=R_prior,
                R_post=R_prior,
                k=fixed_k,
                latent_parameterisation=parameterisation,
                sampler=settings,
            )

    across_maxima, within_maxima = [], []
    first: dict[str, NDArray[np.float64]] = {}
    for repeat in range(n_repeats):
        base = seed + 1000 * repeat
        curves = {}
        errors = []
        for label, parameterisation, offset in (
            ("marginalised", "marginalised_inverse_cdf", 0),
            ("plain", "inverse_cdf", 1),
            ("marginalised_reseeded", "marginalised_inverse_cdf", 2),
        ):
            curve, error, latent = _ssi_rac_curve(
                data,
                delays,
                idata=fit(parameterisation, base + offset),
                fixed_k=fixed_k,
                seed=base + offset,
            )
            curves[label] = curve
            errors.append(error)
            first.setdefault(f"rac_{label}", curve)
            first.setdefault(f"mean_infectivity_{label}", latent)
        # Two independent estimates, so their difference carries √2 standard errors.
        scale = np.sqrt(2.0) * np.maximum.reduce(errors)
        across = np.abs(curves["marginalised"] - curves["plain"]) / scale
        within = np.abs(curves["marginalised"] - curves["marginalised_reseeded"]) / scale
        across_maxima.append(across.max())
        within_maxima.append(within.max())
        print(
            f"  repeat {repeat + 1}: across {across.max():.1f} s.e., within {within.max():.1f} s.e."
        )

    print(
        f"  worst day, in Monte-Carlo standard errors over {n_repeats} triples — "
        f"across parameterisations: {np.mean(across_maxima):.1f} "
        f"(range {min(across_maxima):.1f}–{max(across_maxima):.1f}); "
        f"within one parameterisation: {np.mean(within_maxima):.1f} "
        f"(range {min(within_maxima):.1f}–{max(within_maxima):.1f})"
    )
    difference = np.abs(first["rac_marginalised"] - first["rac_plain"])
    print(
        f"  first triple, absolute scale: max |ΔRAC| = {difference.max():.2e}, "
        f"rms {np.sqrt((difference**2).mean()):.2e}"
    )

    sampled_days = pymc_models.model_days(
        pymc_models.build_naive_model(
            "ssi",
            counts,
            serial_interval=delays.serial_interval,
            switch_day=data.ert_arrival_day,
            R_pre=1.0,
            R_post=1.0,
            k=fixed_k,
            latent_parameterisation="marginalised_inverse_cdf",
        ),
        pymc_models.COHORT_DAY_DIMENSION,
    )
    rebuilt_days, _, _ = pymc_models.marginalised_latent_conditional(
        "ssi",
        counts,
        delays=delays,
        switch_day=data.ert_arrival_day,
        R_pre=1.0,
        R_post=1.0,
        k=fixed_k,
        latent_parameterisation="marginalised_inverse_cdf",
    )
    print(f"  latents: {sampled_days.size} sampled, {rebuilt_days.size} rebuilt {rebuilt_days}")

    rebuilt_gap = np.abs(
        first["mean_infectivity_marginalised"][rebuilt_days]
        - first["mean_infectivity_plain"][rebuilt_days]
    )
    print(
        f"  rebuilt latents: E[Y] = "
        f"{first['mean_infectivity_marginalised'][rebuilt_days].round(4).tolist()} rebuilt vs "
        f"{first['mean_infectivity_plain'][rebuilt_days].round(4).tolist()} sampled; "
        f"max difference {rebuilt_gap.max():.4f}"
    )

    return pd.DataFrame(
        {
            "day": data.day_index,
            "date": _dates(data),
            **first,
            "latent_was_rebuilt": np.isin(data.day_index, rebuilt_days),
        }
    )


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["all"],
        choices=["all", "thompson", "equality", "matched_pair"],
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--thompson-draws", type=int, default=200_000)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--particles", type=int, default=20_000)
    parser.add_argument("--variance-repeats", type=int, default=12)
    parser.add_argument("--matched-pair-repeats", type=int, default=5)
    parser.add_argument(
        "--simulation-stride",
        type=int,
        default=10,
        help="conditioning days to simulate the reset future on, as a stride over the window",
    )
    parser.add_argument(
        "--R-pre",
        type=float,
        default=1.77,
        help="fixed R_pre for the equality check; the SSI posterior mean on this series",
    )
    parser.add_argument("--R-post", type=float, default=0.77)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    config = configuration.load_config(arguments.config)
    analysis = configuration.analysis_config(config, FIXED_K_ANALYSIS)
    data, delays = _setting(analysis)
    sampler = fitting.SamplerSettings.from_config(analysis["sampler"])
    parameterisation = config["latent_parameterisation"]
    fixed_k = float(analysis["fixed_k"])
    wanted = set(arguments.checks)
    output_dir = configuration.ensure_parent(arguments.output_dir / "placeholder").parent

    if wanted & {"all", "thompson"}:
        print("Thompson et al. (2024) replication, under their conventions:")
        frame = check_thompson(
            data,
            delays,
            output_dir=output_dir,
            n_draws=arguments.thompson_draws,
            seed=arguments.seed,
        )
        frame.to_csv(output_dir / "rac_thompson_replication.csv", index=False)

    if wanted & {"all", "equality"}:
        print(
            f"\nEquality check at fixed (R_pre, R_post, k) = "
            f"({arguments.R_pre}, {arguments.R_post}, {fixed_k}):"
        )
        curves = check_equality(
            data,
            delays,
            parameters=TransmissionParameters(
                R_pre=arguments.R_pre, R_post=arguments.R_post, k=fixed_k
            ),
            parameterisation=parameterisation,
            sampler=sampler,
            n_replicates=arguments.replicates,
            n_particles=arguments.particles,
            simulation_stride=arguments.simulation_stride,
            seed=arguments.seed,
        )
        curves.to_csv(output_dir / "rac_equality_check.csv", index=False)
        curves.attrs["summary"].to_csv(output_dir / "rac_equality_summary.csv", index=False)

        print("\nSMC log-evidence variance for SSI (the Stage-4b stopping rule):")
        variance = measure_smc_variance(
            data,
            delays,
            parameters=TransmissionParameters(
                R_pre=arguments.R_pre, R_post=arguments.R_post, k=fixed_k
            ),
            particle_counts=(500, 2_000, 10_000),
            n_repeats=arguments.variance_repeats,
            seed=arguments.seed,
        )
        variance.to_csv(output_dir / "rac_smc_variance.csv", index=False)

    if wanted & {"all", "matched_pair"}:
        print("\nMatched-pair check of the latent reconstruction:")
        frame = check_matched_pair(
            data,
            delays,
            fixed_k=fixed_k,
            R_prior=LogNormalPrior.from_config(analysis["shared"]["priors"]["R_pre"]),
            sampler=sampler,
            n_repeats=arguments.matched_pair_repeats,
            seed=arguments.seed,
        )
        frame.to_csv(output_dir / "rac_latent_reconstruction.csv", index=False)

    print(f"\nwrote checks to {output_dir}")


if __name__ == "__main__":
    main()
