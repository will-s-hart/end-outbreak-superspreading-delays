Committed analysis outputs, one subdirectory per analysis. Everything here is produced by a
Snakemake rule from a script in `scripts/`, and everything here feeds a figure — all but the
three directories noted below as outside the report also feed it.

| Path | What it is |
| --- | --- |
| `<analysis>/<model>_posterior.nc` | Tier-1 MCMC draws for the fit to the complete record, one file per model. It feeds the evidence and dispersion steps, and serves as the last conditioning day's fit. Written gzipped (`fitting.NETCDF_COMPRESSION`), which is lossless and roughly a third of the size — a reporting-sweep fit is 113 MB raw, past GitHub's 100 MB ceiling. The two sweeps are additionally thinned to every 5th draw (`sampler.thin`), which is not lossless and is why it is confined to the analyses computing no model evidence. |
| `<analysis>/<model>_rac.csv` | Tier-1 RAC curve, plus RAT for onset-anchored models and RST for every branching model (never DLO), from one fit per conditioning day. |
| `<analysis>/<model>_rac_diagnostics.csv` | Tier-1 sampler diagnostics for those fits, one row per conditioning day. The per-day posteriors themselves are not kept. |
| `<analysis>/model_evidence.json` | Tier-2 marginal likelihoods and posterior model probabilities. |
| `<analysis>/dispersion_posteriors.json` | Tier-2 `k` posterior summaries and their pairwise divergences — only for the analyses that estimate `k`. |
| `underreporting_60/`, `underreporting_80/` | The reporting sweep, for the naive and the onset-anchored SSE/SSI. Tier-1 files only: comparing a model with itself under a different reporting assumption needs no model evidence and no dispersion summary, so those two are absent by design rather than missing. Their posteriors additionally carry `unreported_incidence` and the `true_incidence` recovered from it, the imputed true onsets — which is what makes these the largest files here. Neither can be dropped to save room: bridge sampling walks the built model's *free* variables and demands each by name, so a thinned fit is unusable for evidence even where evidence is not currently computed. |
| `no_switch_fixed_R/`, `no_switch_single_R/` | The exploratory no-switchpoint variants: tier-1 files only, and no model evidence (`no_switch_fixed_R` fixes every parameter, so there is nothing to integrate over). They feed their own figures and **nothing in the report** — see `AGENTS.md`. |
| `onset_models_uninformative_k/` | Analysis 4 under a `k` prior a decade wider on each side: the full tier-1 and tier-2 set, evidence and `k` summary included. It feeds its own figure and **nothing in the report** yet. |
| `delay_distributions.csv` | Tier-2 discrete incubation, TOST, target serial-interval and implied serial-interval distributions for the supplementary delay figure. |
| `report_numbers.tex` | Tier-2 LaTeX macros: every number `report/report.tex` quotes, collected from the files above. Spans the analyses rather than sitting inside one, which is why it is not in a subdirectory. |

The per-analysis subdirectories arrive one per stage: `naive_models_fixed_k/` with Stage 5,
`naive_models_estimated_k/` with Stage 7, and the two onset-anchored ones with Stage 9;
`report_numbers.tex` arrives with Stage 10.

`report_numbers.tex` is a results file that happens to be written in LaTeX syntax, not a
document: it holds only `\defresultnum{key}{value}` lines and a header recording the files each
value was read from. The macro machinery that consumes it lives in `report/report.tex`, which
raises a compile error on a key this file does not define. See `report/README.md`.

**Validation studies live in `validation/`, not here** — sampler benchmarks, the risk
cross-checks, and the particle-MCMC check write to `validation/results/` with a written summary
beside the data. Keeping them out of this directory means anything under `results/` can be taken
as a report input without further checking.

Git does not preserve mtimes, so a fresh clone looks stale to Snakemake. The default profile in
`config/snakemake_profile/` drops the `mtime` rerun trigger for that reason; `snakemake --touch`
restores mtime consistency after cloning if you want it.
