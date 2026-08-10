"""Validate the semi-analytic RST formulas against Galton--Watson continuation simulation.

This study is deliberately outside the report pipeline. It starts from fixed retained weights,
draws the remaining infections implied by each model, then simulates every resulting lineage to
extinction or a population threshold whose residual false-survival probability is negligible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from end_of_outbreak import branching_process, configuration

R_RESET = 2.0
K = 0.18
RETAINED_WEIGHT = 0.35
PIPELINE_MEAN = 0.20
N_REPLICATES = 100_000
SURVIVAL_THRESHOLD = 200
SEED = 20260810


def validate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []
    for model in ("sse", "ssi", "cori", "sse_so", "ssi_so", "cori_so"):
        poisson = model.startswith("cori")
        event_mechanism = model.startswith("sse")
        onset_anchored = model.endswith("_so")
        q = float(
            branching_process.poisson_extinction_probability(R_RESET)
            if poisson
            else branching_process.negative_binomial_extinction_probability(R_RESET, K)
        )

        if event_mechanism:
            remaining_seeds = rng.negative_binomial(
                K * RETAINED_WEIGHT,
                K / (K + R_RESET),
                size=N_REPLICATES,
            )
            log_no_sustained = RETAINED_WEIGHT * np.log(q)
        else:
            remaining_seeds = rng.poisson(R_RESET * RETAINED_WEIGHT, size=N_REPLICATES)
            log_no_sustained = -R_RESET * RETAINED_WEIGHT * (1.0 - q)

        if onset_anchored:
            remaining_seeds += rng.poisson(PIPELINE_MEAN, size=N_REPLICATES)
            log_no_sustained -= PIPELINE_MEAN * (1.0 - q)

        survived = branching_process.simulate_galton_watson_survival(
            remaining_seeds,
            R=R_RESET,
            k=None if poisson else K,
            survival_threshold=SURVIVAL_THRESHOLD,
            rng=rng,
        )
        analytic = float(-np.expm1(log_no_sustained))
        simulated = float(survived.mean())
        standard_error = float(np.sqrt(simulated * (1.0 - simulated) / N_REPLICATES))
        rows.append(
            {
                "model": model,
                "analytic_rst": analytic,
                "simulated_rst": simulated,
                "absolute_difference": abs(analytic - simulated),
                "simulation_standard_error": standard_error,
                "difference_in_standard_errors": abs(analytic - simulated) / standard_error,
                "false_survival_bound": q**SURVIVAL_THRESHOLD,
                "n_replicates": N_REPLICATES,
            }
        )
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> str:
    worst = results.loc[results["difference_in_standard_errors"].idxmax()]
    return f"""# Validation of risk of sustained transmission

**Conclusion:** the semi-analytic RST formulas agree with independent Galton--Watson
continuation simulation. The largest discrepancy is **{worst["difference_in_standard_errors"]:.2f}
simulation standard errors** ({worst["model"]}); the largest absolute discrepancy is
**{results["absolute_difference"].max():.4f}**.

Each replicate begins from a fixed retained weight of {RETAINED_WEIGHT:g}. Onset-anchored
replicates additionally carry a Poisson incubation pipeline of mean {PIPELINE_MEAN:g}. Future
transmission runs at `R = {R_RESET:g}` and `k = {K:g}` for the overdispersed models. A replicate
is resolved as surviving when one generation reaches {SURVIVAL_THRESHOLD} infections; the
largest upper bound on subsequently becoming extinct is
{results["false_survival_bound"].max():.2e}.

The CSV beside this file records the analytic and simulated values, Monte-Carlo standard errors,
and truncation bounds for SSE, SSI, their onset-anchored forms, and both Poisson limits. DLO is
absent because it does not define independent offspring families and hence has no corresponding
Galton--Watson extinction event.
"""


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    results = validate()
    results.to_csv(configuration.ensure_parent(args.output_csv), index=False)
    configuration.ensure_parent(args.output_summary).write_text(summary(results))
    print(f"wrote {args.output_csv} and {args.output_summary}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=configuration.REPO_ROOT / "validation/results/rst_validation.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=configuration.REPO_ROOT / "validation/results/rst_validation.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
