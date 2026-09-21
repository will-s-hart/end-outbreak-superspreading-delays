# AGENTS.md

Instructions for coding agents working in this repository. Read all of it; it is short by design.
Detail lives in the files under *Where to read more* — go there when you are about to touch the
thing they describe, not before.

## The study

How do (i) the **mechanism** generating heterogeneity in transmission and (ii) the common
practice of treating **symptom-onset dates as infection dates** affect the estimated risk of
relaxing outbreak-control interventions? The case study is the 2018 Équateur Province (DRC)
Ebola outbreak.

Five models are compared, all driven by the same onset-to-onset serial interval:

| Model | Anchoring | Overdispersion acts at the level of | Latents |
| --- | --- | --- | --- |
| `dlo` | infections (naive) | the day (`NB` on aggregate incidence) | none |
| `sse` | infections (naive) | the transmission event (`NB(RΛ, kΛ)`) | none |
| `ssi` | infections (naive) | the individual (latent Gamma infectivity) | `Y_t` |
| `sse_so` | symptom onsets | the transmission event | `λ̃_t` |
| `ssi_so` | symptom onsets | the individual | `Y_t` |

`cori` and `cori_so` are also built, as the `k → ∞` Poisson limits — **validation targets, not
compared models**, and they stay out of the four analyses. Four analyses, configured in
`config/config.yaml`: the naive models at a transplanted `k` and at an estimated `k`, then the
naive and onset-anchored models likewise.

`config/config.yaml` also carries two **exploratory** analyses, `no_switch_fixed_R` and
`no_switch_single_R`, which remove the `R` switchpoint to ask how much of the naive/onset
difference it accounts for. They are deliberately in neither `IMPLEMENTED_ANALYSES` nor
`SAMPLED_ANALYSES`: no report figure and no report number depends on them, and nothing builds
them unless you name the target. `no_switch_results` is the expensive half (8 fits and 8 risk
curves, so a cluster job: `hpc run no_switch_results`);
`pixi run pipeline-no-switch` builds the lot locally, and `pipeline-no-switch-present` draws just
the figures from pulled curves.

`onset_models_uninformative_k` is kept out of the report the same way. It repeats the onset
analysis with estimated `k` under a prior a decade wider on each side of the same median, to ask
whether that analysis's onset-anchored `k` posteriors — which reproduce their prior — are the data
or the prior. `uninformative_k_results` is its cluster half (4 fits, 4 curves, evidence and the
`k` summary); `pipeline-uninformative-k` and `pipeline-uninformative-k-present` mirror the
no-switch tasks. The report itself carries that analysis's estimated-`k` figure in the supplement
(`SUPPLEMENTARY_ANALYSES` in the `Snakefile`), not the main text.

The output quantity is the **risk of additional cases (RAC)**, a real-time reset posterior
predictive: fit parameters *and* latents to the record through day `t` alone, reset `R` to
`R_pre`, and ask for the posterior probability of at least one further case. **Every conditioning
day gets its own fit** — about 110 per model, which is why the pipeline takes hours. The
companion **risk of additional transmission (RAT)** separates from RAC only under the
onset-anchored models. The **risk of sustained transmission (RST)** is the probability that the
reset future never becomes extinct; it is computed for SSE/SSI and their onset-anchored forms,
but not DLO because DLO has no individual branching-process interpretation. See
[docs/risk.md](docs/risk.md).

## Tooling

The environment is managed by **pixi**. All commands run through `pixi run`:

| Command | What it does |
| --- | --- |
| `pixi run fmt` | `ruff format end_of_outbreak scripts validation tests` |
| `pixi run lint` | `ruff check end_of_outbreak scripts validation tests` |
| `pixi run typecheck` | `ty check end_of_outbreak scripts validation tests` |
| `pixi run test` | `pytest` |
| `pixi run check` | all four of the above, in order |
| `pixi run pipeline` | `snakemake --profile config/snakemake_profile -j4` |
| `pixi run pipeline-dry` | dry run: what would re-run, and why |
| `pixi run pipeline-present` | after pulling cluster results, render figures/report only; compute rules are excluded |
| `pixi run pipeline-present-dry` | dry-run the presentation-only allowlist |
| `pixi run pipeline-no-switch` | the exploratory no-switchpoint variants; in no other target |
| `pixi run pipeline-no-switch-present` | their figures only, after pulling a cluster run of `no_switch_results` |
| `pixi run pipeline-uninformative-k` | the vague-`k` repeat of the onset analysis; in no other target |
| `pixi run pipeline-uninformative-k-present` | its figure only, after pulling a cluster run of `uninformative_k_results` |

**Run `pixi run check` and fix every issue before committing.** No exceptions — a failing lint,
type or test check is not "pre-existing", it is the current state of the tree.

Snakemake and its Slurm executor come from bioconda; the scientific stack is conda-forge, with
`ruff` and `ty` from PyPI. The lock covers macOS ARM, Linux x86-64 and Windows x86-64.

**Python has no upper bound, and the lock is on 3.13 anyway.** The bound existed because
chain-level multiprocessing forks an already-multi-threaded interpreter; nothing in the pipeline
does that now (`fitting.fit_model` samples with `cores=1`). The suite passes on 3.14.6 and an
unbounded solve keeps every numerical package at the version already locked, so the interpreter
can move whenever someone runs `pixi update python` — just not in the middle of a cluster run.

## Running the analyses — read this before starting anything long

A full `refit_daily` pipeline is **hours** of MCMC, and it monopolises the machine while it runs.
So the division of labour is:

> **Agents run the quick route locally to see that a new or edited analysis works and what it
> roughly says. All cluster work is a manual job for the user: agents must not push inputs, start,
> monitor, cancel or pull a cluster run unless the user explicitly requests that exact action in
> the current conversation.**

The quick route is `rac.method: single_fit_filtered` — one fit per model, latents filtered per
conditioning day — which turns hours into minutes:

```sh
pixi run python scripts/run_risk_curves.py --analysis <analysis> --model <model> \
  --posterior results/<analysis>/<model>_posterior.nc \
  --method single_fit_filtered --diagnostics /tmp/d.csv --output /tmp/rac.csv
```

It is a genuine approximation and **never a committed result** ([docs/risk.md](docs/risk.md)); it
is for checking that a change runs, produces sane numbers, and moves things in the direction you
expected. If the edited analysis feeds a figure, agents must also render the applicable quick-route
figure(s) to a clearly named temporary directory, present the PNGs or direct file links to the user
for visual checking, and keep those previews in place through the handoff. Saying only that the
agent inspected a temporary figure is not sufficient. When the answer needs to be the estimand,
give the user the manual commands from `.cluster/CHEATSHEET.md` (local-only like `starter_docs/`) and
stop; do not execute any cluster command on the user's behalf without their explicit request.

**The quick route cannot exercise the convergence gate.** `single_fit_filtered` writes filter
diagnostics with no `R̂` column at all, so every acceptance-criteria bug reaches the cluster
untested. Two have: an SSE sweep fit at `R̂` 1.047, and a nan `R̂` on `no_switch_fixed_R/sse`
that stopped the whole analysis. So **run `refit_daily` locally wherever it is affordable** —
it is not always hours. `no_switch_fixed_R` fixes every parameter, so its SSE curve is 110
fits that sample nothing and takes 22 seconds on a laptop. Check the per-fit `seconds` column
of an existing `<analysis>/<model>_rac_diagnostics.csv` before assuming an analysis needs the
cluster.

Three habits that follow from this, all learned the hard way:

- **Do not edit a rule's inputs while a long run is in flight.** `config/config.yaml` and
  `scripts/analysis_driver.py` are inputs to every `rac` rule, so touching either marks all
  fourteen curves stale even when the change provably cannot alter a number. Land pipeline
  changes *before* starting the run, not during it.
- **Do not delete a committed result to force a re-run.** If an output is stale, let the rerun
  triggers say so; if they do not, the fix is a missing entry in a rule's `params:`.

## Git workflow

- Commit regularly with descriptive messages. Run `pixi run check` first.
- The default branch is `main`. Branch for feature work rather than committing to `main` directly.
- `starter_docs/` is untracked by design and must stay that way.

## Conventions that are easy to get wrong

### Mathematical notation

Ruff ignores `N802`/`N803`/`N806`/`N999`, so identifiers keep the notation of the implementation
plan: `R_pre`, `R_post`, `F_r`, `Y`, `Lambda`, `C_t`. Keep that capitalisation rather than
renaming to snake case. `RUF001`–`RUF003` are ignored too, so prose may use en dashes, accents
(Équateur) and mathematical symbols. Prefer descriptive names over brevity everywhere else, even
at the cost of verbosity.

### Day indexing and the `R` switch

Day 0 is the first observed onset (5 April 2018). The ERT arrived on day 33 and withdrew on day
110; the analysis window is days 0–110 inclusive (111 rows). Day 0 is an initial condition in
every model, so likelihoods run over days 1–110 and **risk curves start at day 1** — a "fit to the
record through day 0" is the prior.

`R` switches from `R_pre` to `R_post` on day 33 **in each model's own time index**;
`renewal.switch_index` is the single home of that convention, so don't re-derive it inline. The
resulting timing asymmetry between the naive and onset-anchored models is a deliberate,
reportable result, not a bug — and it is **~11 days**, the mean **incubation period**, because the
two conventions differ by the gap between a transmission and the resulting onset. It is *not* the
15.3 d serial interval, which is the infector-onset to infectee-onset gap and does not separate
the conventions.

### Delay-weight indexing

Weight arrays are stored densely **from their first supported lag**, which differs between
distributions. Never hard-code the offset; use the constants in `delay_distributions`.

| Array | First lag | Constant |
| --- | --- | --- |
| `w` (serial interval) | 1 | `SERIAL_INTERVAL_FIRST_LAG` |
| `f_inc` (incubation) | 1 | `INCUBATION_FIRST_LAG` |
| `f_tost` (onset → transmission) | 0 | `TOST_FIRST_LAG` |

`f_inc` must have **no mass at lag 0** — that is what makes the onset-anchored recursion well
ordered. `f_tost` may. At most one of the two may be supported at lag 0. `survival_weights` and
`tost_survival_weights` live here too, for the same reason: they are survival functions of these
distributions, and putting them anywhere else drags an unrelated module into a dependency list.

## Where a new script or output file goes

Two trees, and the split is by **purpose, not by cost**:

| | Report analyses | Validation studies |
| --- | --- | --- |
| Scripts | `scripts/run_*.py`, `scripts/plot_*.py` | `validation/run_*.py` |
| Outputs | `results/`, `figures/`, `report/` | `validation/results/` |
| Driven by | a Snakemake rule | run on demand |
| In `rule all` / a tier's dependency list | yes | **never** |

Benchmarks, cross-checks, replications of other people's results and anything else whose subject
is *the implementation rather than the outbreak* go in `validation/`, with a written `.md` summary
beside the data. Anything that feeds a figure or a number in the report goes in `scripts/` and
`results/`. **Do not put a check's output under `results/`** — the point of the split is that
everything there can be taken as a report input without further checking. Validation *method*
still lives in the package (`particle_mcmc.py` is the model to follow); it just stays out of every
rule's dependency list. `validation/` is covered by `pixi run check`, and its outputs are
committed.

## Module layout

Flat package `end_of_outbreak/` (no `src/`), with `scripts/` for analysis and plotting.

- Keep modules **genuinely separable**. The Snakemake rules list the exact package modules each
  rule depends on, so that editing a plotting-facing definition re-runs no MCMC. A kitchen-sink
  `utils.py` inside the package, or a fat `__init__.py` that re-exports everything, would defeat
  that scheme — don't add either.
- **Split by what a thing is, and the tiers follow.** The risk lives in three modules for that
  reason: `risk_of_additional_cases.py` is the estimand (tier 1, since `refit_daily` drives fits
  from it), `refit_risk.py` and `filtered_risk.py` are the two ways of *acquiring the state* it
  evaluates, and `risk_curves.py` is the curve container the figure and report tiers read.
  Keeping `first_day_below` beside the closed forms would put a presentation-facing definition on
  the dependency list of ~1500 fits. **If a rule's dependency list looks wrong, the fix is usually
  a module boundary, not a workaround in the `Snakefile`.**
- Reusable *method* goes in the package, and so does **config parsing** (`configuration.py`):
  both script trees need it, and the Snakemake rules must be able to name it in their `input:`
  lists, which they cannot do for a file under `scripts/`. Presentation-only helpers go in
  `scripts/utils.py`, which computes nothing.
- Strict split between **compute-and-save** and **load-and-plot** scripts.
- Analysis scripts are named for what they do, never for figure numbers.

`particle_filter.py` **is** in `RAC_CORE`, because `single_fit_filtered` is a selectable results
path. `particle_mcmc.py` stays out of every rule's list. `parallel.py` also stays out, for the
opposite reason to `risk_curves.py`: it decides which process runs a fit, never what the fit
returns, so listing it would make a scheduling tweak a reason to re-run hours of MCMC.

## Pipeline

`Snakefile` drives everything in three tiers — `fit`/`rac` (MCMC, minutes–hours),
`evidence`/`dispersion`/`report_numbers` (seconds–minutes), `figure`/`report` (seconds) — so a
change at one tier never re-runs the tiers above it. When adding a module, **add it to the right
dependency list at the top of the `Snakefile`** (`FIT_CORE`, `RAC_CORE`, `EVIDENCE_CORE`,
`DISPERSION_CORE`, `REPORT_NUMBERS_CORE`, `PLOT_CORE`); Snakemake's `code` trigger hashes only a
rule's own body and does not follow Python imports.

**`rac` sits in tier 1, not tier 2, and that is not an oversight.** The estimand is a fit per
conditioning day, so "recomputing a risk curve never re-runs a fit" cannot hold for it. It holds
everywhere else: `fit`, `evidence` and `dispersion` describe the model given the whole record, and
restyling a figure re-runs no MCMC. **`rule fit` still runs** — the evidence and dispersion steps
need the full-record posterior, and `rac` reuses it as the last conditioning day's fit rather than
repeating it, which also keeps the end of the curve and those summaries on one posterior. The
per-day posteriors are not persisted; `<model>_rac_diagnostics.csv` is what survives them.

Three more things the pipeline rests on:

- **`IMPLEMENTED_ANALYSES` at the top of the `Snakefile`, not `config["analyses"]`, is what the
  targets are built from.** Keep future config-only analyses out of it until their `run_`/`plot_`
  scripts exist.
- **Any config value that changes a rule's output must appear in that rule's `params:` — and
  nothing else should.** A key placed too high re-runs work it does not affect: `analysis_params()`
  is shared with the `fit` rule, so a `rac`-only setting belongs in `RAC_PARAMS`, or changing a
  particle count would re-fit every model. A setting that changes no output at all — the
  convergence criteria below — belongs in neither.
- **But note that `config/config.yaml` is itself an `input:` of every rule, and the `input`
  trigger is content-based.** So *any* edit to it — including one to a value in no `params:` at
  all — currently marks every fit stale. Verified with `pipeline-dry`: changing only the
  convergence thresholds reports `reason: Updated input files: config/config.yaml` for all
  fourteen fits. That makes the `params:` lists a belt-and-braces second line of defence rather
  than the mechanism, and it means **a one-line config edit costs a full pipeline run** unless
  you can show the outputs are already current and record that with `snakemake --touch`.
  Dropping the file from the `input:` lists would make `params:` load-bearing and edits cheap, at
  the cost of the safety net; it has not been done, because it would mean betting hours of
  compute on those lists being complete. Weigh that before editing the config casually.
- Scripts take `argparse` arguments and are invoked from `shell:`, never through Snakemake's
  `script:` directive, so every script stays runnable and debuggable on its own.
- **`rule rac` is the only wide rule.** It declares `threads: 8` and passes `--jobs {threads}`,
  which spreads the conditioning days over worker processes; every other rule takes Snakemake's
  default of one thread, because a fit's chains now run in one process. Change the width in the
  *profile* (`set-threads`), not in the `Snakefile`: `threads` is part of a rule's code, so
  editing it there would make a change of core count a reason to re-run every curve. The cluster
  profile deliberately sets no `cpus_per_task`, so a Slurm request follows `threads` and the two
  cannot drift apart.

### The convergence gate is a regression test, not a discovery tool

Each risk curve rests on ~110 fits, so every one is checked and
`<model>_rac_diagnostics.csv` records the result per day. **The curve and the table are always
written**, and are identical whatever the criteria say: `rac.convergence` in the config decides
only whether a shaky-looking set of fits *stops the build*.

```yaml
rac:
  convergence: {max_r_hat: 1.02, divergence_fraction: 0.01, on_failure: error}
```

- **`on_failure: warn` while exploring**, which prints the offending days to stderr and carries
  on. Refusing to write a curve that has already been computed, because one day out of a hundred
  and ten was marginal, costs an hour of sampling and tells you nothing the diagnostics table does
  not — and Snakemake deletes a failed job's outputs, so the evidence goes with it too.
- **`error` once a configuration is known to pass**, which is what it is set to: the gate is a
  regression test on the sampling, not a way of discovering that a model is hard.

**Read the thresholds as a maximum over ~1540 fits × ~60 variables**, not as the familiar
single-fit `R̂ < 1.01`. That multiplicity is why `max_r_hat` is 1.02 rather than 1.01: on a
complete run the observed maximum is **1.0100**, on one day of SSE-SO with **no divergences** —
noise on the hardest model, not a fit that failed. A stuck chain lands far above the threshold,
not just over it. The full calibration is recorded in `config/config.yaml`; re-derive it from
`<model>_rac_diagnostics.csv` before moving the number, and treat a *pattern* of days creeping up
as the sampler saying something rather than as a threshold to raise.

`results/` and `figures/` are committed. Git does not preserve mtimes, so the profile drops the
`mtime` rerun trigger; after a fresh clone, `snakemake --touch` restores consistency if you want
it.

After `hpc pull`, use `pixi run pipeline-present`, not `pipeline`. It
targets only `figures` and `report/report.pdf` and uses `--allowed-rules` to exclude every fit,
RAC and tier-2 compute rule. Pulled results are therefore immutable inputs even though the
cluster's Snakemake provenance database was not pulled.

**Re-run the cheap rules first, though: `evidence`, `dispersion` and `report_numbers`.** None of
the three is in `pipeline-present`'s allowlist, so a pull leaves whatever the cluster produced,
and on 2026-09-18 that was a `dispersion_posteriors.json` inconsistent with the
`ssi_posterior.nc` beside it — `dispersion` is a deterministic read of the stored `k` draws, so
it could not have come from that file. All three take seconds to minutes and recomputing them is
the only way to know tier 2 matches tier 1. Do this **before** any `snakemake --touch`, which
stamps an inconsistent pair as current and hides the problem for good.

## Testing

`pytest`, tests under `tests/`. Validation is by property rather than by golden file wherever
possible: normalisation, moment additivity, limiting cases (`k → ∞` collapsing SSE/SSI onto Cori),
likelihood-vs-simulation agreement, and analytic-vs-particle-filter agreement.

Two rules carry most of the weight, and both are explained where they apply
([docs/models.md](docs/models.md), [docs/risk.md](docs/risk.md)):

- **Never check a reimplementation against itself.** Where a test needs a fast reimplementation of
  something the package computes, pin it against the package's own version at random points
  first.
- **Match the conditioning before calling agreement a pass.** There are two particle-filter
  checks and they answer different questions: the fixed-`θ` arithmetic check compares against the
  particle *smoother* — both sides conditioned on the whole record, which is what makes it a test
  of the closed forms and not of the estimand — while PMMH is a *filtering* route whose target is
  `filtered_risk`. Agreement is required in both, of different things.

Checks that need MCMC on the real series live in `validation/`, not `tests/`; the tests carry the
same comparisons on short histories, which is what keeps `pixi run test` quick.

## Settled decisions — do not silently revisit

Each is argued out in `starter_docs/implementation_plan.md` and, where noted, backed by a
committed measurement.

| Decision | Where it is argued |
| --- | --- |
| **Serial interval:** the generic EVD estimate (mean 15.3 d, SD 9.3 d) for all five models. Its variance budget of 86.49 d² admits published EVD incubation estimates; the outbreak-specific estimate (19.46, 6.08) does not, and is kept only for a sensitivity analysis. | plan §2 |
| **RAC is the headline; RAT and RST separate its components.** RAT is distinct only for onset-anchored models. RST is computed for every branching model, never DLO, and means eventual non-extinction. | [docs/risk.md](docs/risk.md) |
| **Real-time conditioning:** parameters *and* latents fitted to the record through day `t`, one fit per conditioning day. `single_fit_filtered` is a comparison, never a results path. The old smoothed route is gone. | [docs/risk.md](docs/risk.md), plan §5.6 |
| **`R` switches on day 33 in each model's own time index.** | *Day indexing*, above |
| **Naming:** the day-level negative-binomial model is `DLO`, not the older "CIO"; the package is `end_of_outbreak`. | — |
| **SSI-SO infectivity prior** is `Y_t \| D_t ~ Gamma(k D_t, k)`. The `Gamma(k I_t, k)` in `starter_docs/models.jpeg` is a transcription slip. | plan §4 |
| **Particle MCMC is a check, never a results path.** | [docs/risk.md](docs/risk.md) |
| **Latent parameterisation:** `marginalised_inverse_cdf`, with `negligible_latent_threshold = 0.0`. Gamma `icdf` is differentiable in its probability coordinate but not its shape: fixed-`k` fits use NUTS throughout, while estimated-`k` fits deliberately use `Slice: [k]` plus NUTS for the `R` parameters and latent uniforms. That compound sampler is what the benchmark validated. | [docs/models.md](docs/models.md), `validation/results/sampler_benchmark.md` |
| **Model evidence by bridge sampling**, with the other estimators as agreement checks. | [docs/models.md](docs/models.md), `validation/results/evidence_validation.md` |
| **Fits are stored as NETCDF4 via xarray**, engine `h5netcdf`, read back with `xr.open_datatree`. `arviz.InferenceData` is a deprecated alias for `xr.DataTree` in arviz 1.x, so the I/O is xarray's; keep arviz for `az.summary` and the diagnostics. | `fitting.NETCDF_ENGINE` |
| **Under-reporting carries the unreported counts as a scalar-support, vector-batched latent.** Exact Poisson/NB thinning supplies its own `logp`; `true_incidence = reported + unreported` remains a deterministic and no `pm.Potential` is used. RAC still means at least one further **true** case. | [docs/models.md](docs/models.md) |
| **A reporting delay's as-of day is explicit.** A real-time delayed-reporting curve requires historical count snapshots; each runs from day 0 through its as-of day, so the day is inferred as `len(snapshot) - 1`. Our snapshots include their conditioning day; `end-of-outbreak-vbd`'s windows stop strictly before it. | [docs/risk.md](docs/risk.md) |
| **The unreported-count `CustomDist` declares scalar support**, so PyMC's ordinary Metropolis uses coordinate-wise updates. No custom step assignment remains; tests inspect automatic assignment and movement on a long block. | [docs/models.md](docs/models.md) |
| **The filter is adapted in the counts too**, drawing unreported cases from the exact Poisson or negative-binomial conditional and weighting by the corresponding reported marginal. Proposing totals and reweighting by the binomial kills the whole cloud on any busy day. SSE incomplete-reporting filtering is supported; DLO filtered RAC is refused because its retained profile is not stored. | [docs/risk.md](docs/risk.md) |
| **A fit's chains run in one process (`cores=1`); the parallel axis is the conditioning days.** `refit_risk` spreads them over a worker pool, so throughput is capped by the machine rather than by `chains`. Bit-identical draws either way — PyMC seeds chain `c` before it decides how many processes to run. Never move parallelism back inside `pm.sample`. | [docs/models.md](docs/models.md), `tests/test_parallel.py` |
| **The report quotes no literal number.** | `report/README.md` |

## Open items

- **Incubation period.** Currently WHO Ebola Response Team (2014), NEJM 371:1481–1495, gamma with
  mean 11.4 d and SD 8.1 d, leaving a residual TOST of mean 3.9 d and SD 4.57 d. Configurable in
  `config/config.yaml`; `check_delay_budget` rejects any estimate whose variance exceeds the
  serial interval's.
- **Under-reporting is implemented for every model but run only for SSE, SSI, SSE-SO and SSI-SO at 60% and 80%.** The naive models were added after the onset-anchored ones and sit after them in `models:`, because each model's seed stream is its index there. Reporting *delays* are implemented and tested but used by no analysis: there are no reporting dates for Équateur. The reporting probability is a fixed input, not a parameter — it is not identified from these data without an external prior.
- **Six sensitivity analyses, listed in the report and none of them run.** In order of value: the
  outbreak-specific serial interval (naive models only — it is structurally inadmissible for the
  onset-anchored ones, and saying so is itself a result); the incubation/TOST decomposition;
  **retrospective rather than real-time conditioning**, retaining the day-`t` state from a fit to
  the complete record, which asks a different question rather than approximating this one; the
  shifted `R`-switch that separates structure from indexing; a non-empty initial incubation
  pipeline; and prior sensitivity. The first, second, fourth and fifth are one config value apiece
  and no new code — the fourth only since a per-analysis `switch_day` became a setting anything
  reads. The `shared.ert_arrival_day` key that used to look like the switch knob was dead config,
  and is gone; the ERT's dates live in `outbreak_data.py`, where the report reads them from.
- **The two no-switchpoint variants are implemented but not interpreted.** `no_switch_fixed_R`
  holds `R` at 0.95 with `k` at 0.18 and estimates nothing but the latents, so any surviving
  naive/onset gap is retained-state arithmetic alone; `no_switch_single_R` estimates one `R` per
  model. Both move the switch past the end of the window, which also makes RAC's reset to
  `R_pre` a no-op — so they report a forward predictive, not the reset predictive the report's
  analyses do. Nothing is written up until a `refit_daily` run exists; the quick route is never
  a finding.
- **The vague-`k` repeat is implemented but not interpreted.** Under the report's `k` prior the
  onset-anchored posteriors land on its median with almost no evidence gain, which the report now
  says is ambiguous. `onset_models_uninformative_k` settles it: posteriors that widen roughly in
  proportion to the prior mean the data say little about `k` under onset anchoring; posteriors
  that stay near 0.18 and narrow mean they do. Its model probabilities are not comparable with the
  report analysis's, since a wider prior is charged for in the evidence. It is also the first
  piece of the prior-sensitivity analysis listed above.
- **The reset-convention follow-up remains declined** unless asked for. RST is now a reported
  estimand rather than an optional follow-up.
- **Per-fit process spawn dominates the cheap models' cost.** Each conditioning-day fit starts
  four chain processes, which is a large fraction of a two-second fit. If the pipeline's runtime
  ever matters, the lever is reusing a sampler across days in `fitting.fit_model`, not the
  Snakemake scheduling.

## Where to read more

Go to these when you are about to change the thing they describe.

| File | What it covers |
| --- | --- |
| [docs/risk.md](docs/risk.md) | The estimand, the two estimators, the closed forms, the affine representation and the exact latent marginalisation, the onset reset state, matching conditioning against the particle routes |
| [docs/models.md](docs/models.md) | The renewal core, the builders and simulators, the latent block and what the sampler benchmark settled, model evidence, dispersion comparison |
| [docs/findings.md](docs/findings.md) | What the study has measured, and which file holds each number. **Read it before "fixing" a surprising result** |
| `starter_docs/implementation_plan.md` | Legacy rationale only; its warning names the authoritative sources. Untracked, so local only |
| `.cluster/CHEATSHEET.md` | Running the full pipeline on ARC via the `hpc` CLI. Untracked, so local only |
| `scripts/README.md` | The script table, the tier conventions, figure layout rules, the no-fallback rule |
| `validation/README.md` | What each validation study answers and where it writes |
| `report/README.md` | The report's register, and the generated-numbers scheme |
| `results/README.md`, `figures/README.md`, `data/README.md` | What each committed output is |
