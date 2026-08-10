Committed analysis outputs, one subdirectory per analysis. Everything here is produced by a
Snakemake rule from a script in `scripts/`, and everything here feeds a figure or the report.

| Path | What it is |
| --- | --- |
| `<analysis>/<model>_posterior.nc` | Tier-1 MCMC draws for the fit to the complete record, one file per model. It feeds the evidence and dispersion steps, and serves as the last conditioning day's fit. |
| `<analysis>/<model>_rac.csv` | Tier-1 RAC curve (plus RAT, for the onset-anchored models), from one fit per conditioning day. |
| `<analysis>/<model>_rac_diagnostics.csv` | Tier-1 sampler diagnostics for those fits, one row per conditioning day. The per-day posteriors themselves are not kept. |
| `<analysis>/model_evidence.json` | Tier-2 marginal likelihoods and posterior model probabilities. |
| `<analysis>/dispersion_posteriors.json` | Tier-2 `k` posterior summaries and their pairwise divergences — only for the analyses that estimate `k`. |
| `delay_distributions.csv` | Tier-2 discrete incubation, TOST, target serial-interval and implied serial-interval distributions for Supplementary Fig. S1. |
| `report_numbers.tex` | Tier-2 LaTeX macros: every number `report/report.tex` quotes, collected from the files above. Spans the analyses rather than sitting inside one, which is why it is not in a subdirectory. |

The per-analysis subdirectories arrive one per stage: `naive_models_fixed_k/` with Stage 5,
`naive_models_estimated_k/` with Stage 7, and the two onset-anchored ones with Stage 9;
`report_numbers.tex` arrives with Stage 10.

`report_numbers.tex` is a results file that happens to be written in LaTeX syntax, not a
document: it holds only `\defresultnum{key}{value}` lines and a header recording the files each
value was read from. The macro machinery that consumes it lives in `report/report.tex`, which
raises a compile error on a key this file does not define. See `report/README.md`.

**Validation studies live in `validation/`, not here** — sampler benchmarks, the RAC
cross-checks, and the particle-MCMC check write to `validation/results/` with a written summary
beside the data. Keeping them out of this directory means anything under `results/` can be taken
as a report input without further checking.

Git does not preserve mtimes, so a fresh clone looks stale to Snakemake. The default profile in
`config/snakemake_profile/` drops the `mtime` rerun trigger for that reason; `snakemake --touch`
restores mtime consistency after cloning if you want it.
