Committed analysis outputs, one subdirectory per analysis. Everything here is produced by a
Snakemake rule from a script in `scripts/`, and everything here feeds a figure or the report.

| Path | What it is |
| --- | --- |
| `<analysis>/<model>_posterior.nc` | Tier-1 MCMC draws, one file per model. |
| `<analysis>/<model>_rac.csv` | Tier-2 RAC curve (plus RAT, for the onset-anchored models). |
| `<analysis>/model_evidence.json` | Tier-2 marginal likelihoods and posterior model probabilities. |
| `<analysis>/dispersion_posteriors.json` | Tier-2 `k` posterior summaries and their pairwise divergences — only for the analyses that estimate `k`. |

The per-analysis subdirectories arrive one per stage: `naive_models_fixed_k/` with Stage 5,
`naive_models_estimated_k/` with Stage 7, and the two onset-anchored ones with Stage 9.

**Validation studies live in `validation/`, not here** — sampler benchmarks, the RAC
cross-checks, and the particle-MCMC check write to `validation/results/` with a written summary
beside the data. Keeping them out of this directory means anything under `results/` can be taken
as a report input without further checking.

Git does not preserve mtimes, so a fresh clone looks stale to Snakemake. The default profile in
`config/snakemake_profile/` drops the `mtime` rerun trigger for that reason; `snakemake --touch`
restores mtime consistency after cloning if you want it.
