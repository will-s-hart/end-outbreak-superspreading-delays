# End-of-outbreak decision-making under superspreading and onset-date delays

Research code studying how two modelling choices affect the estimated risk of relaxing
outbreak-control interventions:

1. **The mechanism generating heterogeneity in transmission.** Three models can be tuned to
   the same population-level mean–variance relationship yet imply different end-of-outbreak
   risks, because the *timing* and *grouping* of the excess variance differ.
2. **Treating symptom-onset dates as infection dates.** Renewal models are routinely fitted
   with onset time series substituted directly for infection time series. We quantify the
   consequences.

The case study is the 2018 Équateur Province (DRC) Ebola outbreak — 54 cases with onsets
between 5 April and 2 June 2018, an Ebola Response Team (ERT) present from 8 May, withdrawn
on 24 July.

## Models

All five are driven by the same onset-to-onset serial interval, so the comparison isolates
the mechanism rather than confounding it with a different assumed interval.

| Model | Anchoring | Excess variance acts at the level of |
| --- | --- | --- |
| **DLO** | infections (naive) | the **day** — negative binomial on aggregate incidence |
| **SSE** | infections (naive) | the **transmission event** |
| **SSI** | infections (naive) | the **individual** — latent Gamma infectivity |
| **SSE-SO** | symptom onsets | the transmission event |
| **SSI-SO** | symptom onsets | the individual |

The "naive" models treat onset dates as infection dates; the onset-anchored (`-SO`) models
reference transmission to the infector's own onset via a time-from-onset-to-transmission
(TOST) distribution and model the incubation period explicitly.

A headline result concerns the dispersion parameter `k`. A literature estimate of the
*individual-level* offspring dispersion (`k = 0.18`) is meaningful for SSE/SSI, where `k`
indexes an individual-level offspring distribution — but **not** for DLO, where it controls
day-to-day variation in aggregate incidence. Transplanting the value between them is an error
made in earlier studies.

## Output quantity

The **risk of additional cases (RAC)**: the probability that at least one further case occurs
after a given day, assuming the reproduction number reverts to its pre-intervention value once
interventions are relaxed. It is a real-time quantity — parameters and latent states are fitted
to the record through the day in question and nothing after it — so each curve is one MCMC fit
per day rather than a summary of a single fit. That is why `pixi run pipeline` takes hours.

A companion quantity, the **risk of additional transmission (RAT)**, asks instead for at least
one further *transmission event*. The naive models identify the two by assumption; the
onset-anchored models separate them, and the gap is the contribution of individuals already
infected but not yet symptomatic. RAC is reported throughout; RAT is reported beside it and
RST, both to check that the naive/onset difference is not a property of RAC alone and because the
RAC-RAT gap quantifies the onset-vs-infection conflation directly.

The **risk of sustained transmission (RST)** asks whether the reset future ever becomes extinct.
It is computed from the same per-day posterior fits for SSE, SSI and their onset-anchored forms;
the DLO model is omitted because its day-level negative binomial does not define individual
offspring families. The risks obey RST ≤ RAT ≤ RAC.

## Getting started

The environment is managed by [pixi](https://pixi.sh):

```sh
pixi install
pixi run check           # format, lint, typecheck, test
pixi run pipeline-dry    # what the pipeline would do, and why
pixi run pipeline        # reproduce every result, figure and the report
```

Compiling `report/report.pdf` needs `latexmk` and `pdflatex` from a TeX distribution (MacTeX,
TeX Live). That is the pipeline's only external toolchain dependency, and the `report` rule says
so plainly if it is missing; every other rule runs without it.

`results/` and `figures/` are committed, but git does not preserve modification times, so a
fresh clone can look stale to Snakemake. The default profile
(`config/snakemake_profile/`) drops the `mtime` rerun trigger for this reason; run
`pixi run snakemake --touch` if you want mtime consistency restored anyway.

## Layout

```
config/            pipeline configuration, keyed per analysis, and the Snakemake profile
data/              the padded daily onset series, with provenance in data/README.md
docs/              the detail behind AGENTS.md: the estimand, the models, the findings
end_of_outbreak/   the package: models, delays, inference, RAC/RAT/RST, evidence
scripts/           one compute-and-save script and one load-and-plot script per analysis
results/           committed analysis outputs, including the report's numbers as LaTeX macros
figures/           committed figures (PDF + PNG)
report/            LaTeX methods and results, and the compiled PDF
tests/             pytest suite
validation/        benchmarks and cross-checks: subject is the implementation, not the outbreak
Snakefile          pipeline entry point (three tiers: fit and risk -> derive -> plot)
```

See `AGENTS.md` for conventions and the development workflow, and `docs/` for the detail it
points to: [the risk estimand](docs/risk.md), [the models and fits](docs/models.md), and
[what the study has measured](docs/findings.md).

The report quotes **no number of its own**: `report/report.tex` writes `\resultnum{<key>}` and
`results/report_numbers.tex` says what each key expands to, so a refit that moves a number moves
it in the prose too. See `report/README.md`.

## Status

All four analyses, the RAC/RAT/RST comparisons and the methods-and-results report run end to end
from the raw CSV: `pixi run pipeline` reproduces everything in `results/`, `figures/` and
`report/`. The sensitivity analyses listed at the end of the report have not been run.

## References

- Thompson RN, Hart WS, Keita M, Fall IS, Gueye AS, Chamla D, Mossoko M, Ahuka-Mundeke S,
  Nsio-Mbeta J, Jombart T, Polonsky J (2024). Using real-time modelling to inform the 2017
  Ebola outbreak response in DR Congo. *Nature Communications* **15**:5667.
- Cori A, Ferguson NM, Fraser C, Cauchemez S (2013). A new framework and software to estimate
  time-varying reproduction numbers during epidemics. *Am J Epidemiol* **178**:1505–1512.
- Van Kerkhove MD, Bento AI, Mills HL, Ferguson NM, Donnelly CA (2015). A review of
  epidemiological parameters from Ebola outbreaks to inform early public health
  decision-making. *Scientific Data* **2**:150019.
- WHO Ebola Response Team (2014). Ebola virus disease in West Africa — the first 9 months of
  the epidemic and forward projections. *N Engl J Med* **371**:1481–1495.
- Althaus CL (2015). Ebola superspreading. *Lancet Infect Dis* **15**:507–508.
