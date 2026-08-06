"""Stage-3 sampler benchmark: which latent parameterisation should the project use?

Runs every candidate of :mod:`end_of_outbreak.latent_parameterisations` against the two models
that carry a latent block on the real Équateur series — SSI (31 latents) and **the actual
SSE-SO model** (110 latents, shapes down to 2.7 × 10⁻⁶) — plus a synthetic SSI history with a
known truth. Records divergences, ``R̂``, effective sample size per second and the posterior
summaries, so the choice of default is auditable rather than asserted.

Two questions, deliberately kept apart (§6.3):

*Is it a warm-up problem?*
    Every configuration is run at a short and a long tuning length. A strategy whose deficit
    disappears with more tuning was fixing where the chain starts, not what it explores.
*Is it a geometry problem?*
    A strategy that still wins at the long tuning length is doing something to the space.

Effective sample size is always measured on the **recovered latent** (``Y`` /
``lambda_tilde``), never on the reparameterised free variable, so the numbers mean the same
thing in every column. Under ``marginalised`` the block is smaller, which is the point: those
latents are integrated out exactly, so their sampling cost is genuinely zero.

Usage::

    python validation/run_sampler_benchmark.py --output validation/results/sampler_benchmark.csv

Not part of the pipeline: this is a one-off study whose conclusion is recorded in
``config/config.yaml`` and ``AGENTS.md``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from numpy.typing import NDArray

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import forward_simulation, outbreak_data, pymc_models
from end_of_outbreak import latent_parameterisations as lp
from end_of_outbreak.model_specifications import LogNormalPrior, TransmissionParameters

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "validation" / "results" / "sampler_benchmark.csv"

LATENT_MODELS = ("ssi", "sse_so")
FIXED_K = 0.18
R_PRIOR = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)
K_PRIOR = LogNormalPrior.from_median_and_quantile(median=0.18, quantile=0.09, probability=0.025)


@dataclass(frozen=True)
class Dataset:
    """One history to benchmark against, with its truth where there is one."""

    name: str
    counts: NDArray[np.int64]
    switch_day: int
    delays: dd.OnsetAnchoredDelays
    models: tuple[str, ...]
    truth: dict[str, float] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------------------
# The histories
# ---------------------------------------------------------------------------------------

MINIMUM_SYNTHETIC_CASES = 25
MAXIMUM_SYNTHETIC_CASES = 200


def _simulate_comparable_history(
    truth: TransmissionParameters,
    delays: dd.OnsetAnchoredDelays,
    data: outbreak_data.OutbreakData,
    seed: int,
) -> NDArray[np.int64]:
    """A synthetic SSI history of roughly the size of the real one, from a known truth.

    Conditioned on not going extinct immediately: with ``k = 0.18`` most trajectories from a
    single index case die out at once, and a one-case history says nothing about how the
    sampler copes with a latent block. Conditioning on the outbreak size is a property of the
    benchmark's *test case*, not of any model being fitted, so it costs nothing here — but it
    does mean the synthetic arm is not a sample from the prior predictive and cannot double as
    simulation-based calibration.
    """
    replicates = forward_simulation.simulate_naive(
        "ssi",
        truth,
        serial_interval=delays.serial_interval,
        n_days=data.n_days,
        switch_day=data.ert_arrival_day,
        n_replicates=2000,
        rng=np.random.default_rng(seed),
    )
    totals = replicates.total_cases
    usable = np.flatnonzero(
        (totals >= MINIMUM_SYNTHETIC_CASES) & (totals <= MAXIMUM_SYNTHETIC_CASES)
    )
    if usable.size == 0:
        raise RuntimeError(
            "no simulated history of a usable size; widen the bounds or change the truth"
        )
    return replicates.counts[usable[0]]


def build_datasets(seed: int) -> list[Dataset]:
    """The real series, a truncated version of it, and a synthetic SSI history."""
    data = outbreak_data.load_onset_data()
    delays = dd.build_onset_anchored_delays()

    # Truncating just past the last observed onset removes the long case-free tail, and with
    # it most of the exactly-marginalisable block. Contrasting the two isolates what the tail
    # is actually costing the sampler.
    truncated_days = 71

    truth = TransmissionParameters(R_pre=2.0, R_post=0.6, k=FIXED_K)
    simulated = _simulate_comparable_history(truth, delays, data, seed)

    return [
        Dataset(
            name="equateur",
            counts=data.onsets,
            switch_day=data.ert_arrival_day,
            delays=delays,
            models=LATENT_MODELS,
            description="the real 2018 Équateur series, days 0-110",
        ),
        Dataset(
            name="equateur_truncated",
            counts=data.onsets[:truncated_days],
            switch_day=data.ert_arrival_day,
            delays=delays,
            models=("sse_so",),
            description=f"the real series cut at day {truncated_days - 1}, before the long tail",
        ),
        Dataset(
            name="synthetic_ssi",
            counts=simulated,
            switch_day=data.ert_arrival_day,
            delays=delays,
            models=("ssi",),
            truth={"R_pre": truth.R_pre, "R_post": truth.R_post},
            description="an SSI history simulated with a known truth",
        ),
    ]


# ---------------------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------------------


def run_one(
    dataset: Dataset,
    model: str,
    parameterisation: lp.LatentParameterisation,
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    estimate_k: bool,
) -> dict[str, Any]:
    """Fit one (dataset, model, parameterisation) and summarise how the sampler coped."""
    built = pymc_models.build_model(
        model,
        dataset.counts,
        delays=dataset.delays,
        switch_day=dataset.switch_day,
        R_pre=R_PRIOR,
        R_post=R_PRIOR,
        k=K_PRIOR if estimate_k else FIXED_K,
        latent_parameterisation=parameterisation,
    )
    latent_variable = (
        pymc_models.TRANSMISSIBILITY_VARIABLE
        if model == "sse_so"
        else pymc_models.INFECTIVITY_VARIABLE
    )
    latent_days = pymc_models.model_days(built, pymc_models.latent_dimension(model))
    sampled_scale = pymc_models.latent_scale_by_day(model, dataset.counts, delays=dataset.delays)[
        latent_days
    ]
    initial_values: dict[Any, Any] = dict(
        lp.suggested_initial_values(
            latent_variable,
            k=FIXED_K,
            sampled_scale=sampled_scale,
            parameterisation=parameterisation,
        )
    )

    parameters = ["R_pre", "R_post"] + (["k"] if estimate_k else [])
    started = time.perf_counter()
    with built, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            initvals=dict(initial_values) if initial_values else None,
            compute_convergence_checks=False,
        )
    wall_clock = time.perf_counter() - started
    # PyMC reports the sampling phase alone. Graph compilation is a fixed one-off cost that
    # differs a lot between parameterisations (the inverse-CDF graph is much larger), so
    # charging it to the sampler would flatter the simpler ones on ESS per second.
    elapsed = float(idata["sample_stats"].attrs.get("sampling_time", wall_clock))

    parameter_summary = az.summary(idata, var_names=parameters)
    latent_summary = az.summary(idata, var_names=[latent_variable])
    divergences = int(np.asarray(idata["sample_stats"]["diverging"]).sum())

    record: dict[str, Any] = {
        "dataset": dataset.name,
        "model": model,
        "parameterisation": parameterisation.name,
        "estimate_k": estimate_k,
        "tune": tune,
        "draws": draws,
        "chains": chains,
        "status": "ok",
        "n_sampled_latents": int(latent_days.size),
        "sampling_seconds": round(elapsed, 2),
        "wall_clock_seconds": round(wall_clock, 2),
        "divergences": divergences,
        "max_rhat_parameters": float(parameter_summary["r_hat"].max()),
        "max_rhat_latents": float(latent_summary["r_hat"].max()),
        "min_ess_bulk_parameters": float(parameter_summary["ess_bulk"].min()),
        "min_ess_tail_parameters": float(parameter_summary["ess_tail"].min()),
        "min_ess_bulk_latents": float(latent_summary["ess_bulk"].min()),
        "min_ess_tail_latents": float(latent_summary["ess_tail"].min()),
        "ess_bulk_per_second_parameters": round(
            float(parameter_summary["ess_bulk"].min()) / elapsed, 2
        ),
        "ess_bulk_per_second_latents": round(float(latent_summary["ess_bulk"].min()) / elapsed, 2),
        "changes_the_start": parameterisation.changes_the_starting_point,
        "changes_the_space": parameterisation.changes_the_space,
    }
    for parameter in parameters:
        record[f"{parameter}_mean"] = round(float(parameter_summary.loc[parameter, "mean"]), 4)
        record[f"{parameter}_sd"] = round(float(parameter_summary.loc[parameter, "sd"]), 4)
    for parameter, value in dataset.truth.items():
        record[f"{parameter}_truth"] = value
    return record


# ---------------------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------------------


def run_benchmark(
    datasets: list[Dataset],
    parameterisations: list[lp.LatentParameterisation],
    *,
    draws: int,
    tune_lengths: list[int],
    chains: int,
    target_accept: float,
    seed: int,
    estimate_k: list[bool],
) -> pd.DataFrame:
    records = []
    for dataset in datasets:
        for model in dataset.models:
            for parameterisation in parameterisations:
                for tune in tune_lengths:
                    for estimating in estimate_k:
                        label = (
                            f"{dataset.name}/{model}/{parameterisation.name}"
                            f"/tune={tune}/k={'free' if estimating else 'fixed'}"
                        )
                        print(f"  {label} ...", flush=True)
                        try:
                            record = run_one(
                                dataset,
                                model,
                                parameterisation,
                                draws=draws,
                                tune=tune,
                                chains=chains,
                                target_accept=target_accept,
                                seed=seed,
                                estimate_k=estimating,
                            )
                        except Exception as failure:
                            # A configuration that cannot even be started is a finding, not a
                            # crash: record why and carry on with the rest of the grid.
                            print(f"    FAILED: {failure}", flush=True)
                            records.append(
                                {
                                    "dataset": dataset.name,
                                    "model": model,
                                    "parameterisation": parameterisation.name,
                                    "estimate_k": estimating,
                                    "tune": tune,
                                    "draws": draws,
                                    "chains": chains,
                                    "status": "failed",
                                    "failure": str(failure).splitlines()[0][:300],
                                }
                            )
                            continue
                        worst_rhat = max(record["max_rhat_parameters"], record["max_rhat_latents"])
                        print(
                            f"    {record['sampling_seconds']:.1f}s,"
                            f" {record['divergences']} divergences,"
                            f" R-hat <= {worst_rhat:.3f},"
                            f" latent ESS {record['min_ess_bulk_latents']:.0f}"
                            f" ({record['ess_bulk_per_second_latents']:.1f}/s)",
                            flush=True,
                        )
                        records.append(record)
    return pd.DataFrame.from_records(records)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, nargs="+", default=[500, 2000])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--parameterisations",
        nargs="+",
        default=sorted(lp.PARAMETERISATIONS),
        help="candidate names; defaults to every registered strategy",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="dataset names to include; defaults to all of them",
    )
    parser.add_argument(
        "--estimate-k",
        choices=["fixed", "free", "both"],
        default="fixed",
        help="whether k is held at 0.18, given its prior, or both (analyses 1/3 vs 2/4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    datasets = build_datasets(arguments.seed)
    if arguments.datasets is not None:
        wanted = set(arguments.datasets)
        datasets = [dataset for dataset in datasets if dataset.name in wanted]
        if not datasets:
            raise SystemExit(f"no datasets named {sorted(wanted)}")
    parameterisations = [lp.parameterisation_of(name) for name in arguments.parameterisations]
    estimate_k = {"fixed": [False], "free": [True], "both": [False, True]}[arguments.estimate_k]

    print(f"benchmarking {len(parameterisations)} parameterisations", flush=True)
    frame = run_benchmark(
        datasets,
        parameterisations,
        draws=arguments.draws,
        tune_lengths=arguments.tune,
        chains=arguments.chains,
        target_accept=arguments.target_accept,
        seed=arguments.seed,
        estimate_k=estimate_k,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)
    print(f"\nwrote {len(frame)} runs to {arguments.output}")


if __name__ == "__main__":
    main()
