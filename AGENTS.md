# AGENTS.md

Instructions for coding agents working in this repository.

## Project

Research code for a study of how (i) the **mechanism** generating heterogeneity in
transmission and (ii) the common practice of treating **symptom-onset dates as infection
dates** affect the estimated risk of relaxing outbreak-control interventions. The case study
is the 2018 Équateur Province (DRC) Ebola outbreak.

Five models are compared, all driven by the same onset-to-onset serial interval:

| Model | Anchoring | Overdispersion acts at the level of | Latents |
| --- | --- | --- | --- |
| `dlo` | infections (naive) | the day (`NB` on aggregate incidence, fixed `k`) | none |
| `sse` | infections (naive) | the transmission event (`NB(RΛ, kΛ)`) | none |
| `ssi` | infections (naive) | the individual (latent Gamma infectivity) | `Y_t`, 31 |
| `sse_so` | symptom onsets | the transmission event | `λ̃_t`, 110 |
| `ssi_so` | symptom onsets | the individual | `Y_t`, 31 |

The output quantity throughout is the **risk of additional cases (RAC)**. It is a
**retrospective reset posterior predictive**, not a filtering probability — the shorthand
`P(· | data up to day t)` is misleading and the estimand is a procedure:

> Fit parameters *and latents* to the complete record (days 0–110). For each day `t`, retain
> the inferred state attached to the history through `t`, discard the realised trajectory
> after `t`, reset `R` to `R_pre`, and simulate a counterfactual future. RAC(t) is the
> posterior probability that this future contains at least one further case.

The retained state is therefore **smoothed** — informed by data after day `t` — for the latent
models. That is a deliberate decision, not an oversight. For the onset-anchored models the
reset state must also reconstruct the **incubation pipeline** (infected before `t`, not yet
symptomatic); restarting the recursion from observed onsets alone silently drops it and
understates RAC.

A companion quantity, the **risk of additional transmission (RAT)**, is the probability of at
least one further *transmission event* after that day. Under the three naive models the two
coincide by assumption — that identification is exactly the conflation this project is about
— and they separate only under `sse_so`/`ssi_so`, where the gap is the contribution of the
latent pipeline of already-infected but not-yet-symptomatic individuals. **Use RAC everywhere
except the supplementary analysis that treats RAT explicitly, and the methodology supporting
it.** `RAC(t) ≥ RAT(t)` always.

`starter_docs/implementation_plan.md` is the **source of truth** for model definitions,
conventions and the staged plan — read the relevant section before changing any analytic
formula or convention. It is untracked (it lives under a directory with its own `.gitignore`),
so it is available locally but not in the published repo.

## Tooling

The environment is managed by **pixi**. All commands run through `pixi run`:

| Command | What it does |
| --- | --- |
| `pixi run fmt` | `ruff format end_of_outbreak scripts tests` |
| `pixi run lint` | `ruff check end_of_outbreak scripts tests` |
| `pixi run typecheck` | `ty check end_of_outbreak scripts tests` (clears `VIRTUAL_ENV` first) |
| `pixi run test` | `pytest` |
| `pixi run check` | all four of the above, in order |
| `pixi run pipeline` | `snakemake --profile config/snakemake_profile -j4` |
| `pixi run pipeline-dry` | dry run: what would re-run, and why |

**Run `pixi run check` and fix every issue before committing.** No exceptions — a failing
lint, type or test check is not "pre-existing", it is the current state of the tree.

`snakemake-minimal` comes from bioconda (conda-forge does not carry it); everything else is
conda-forge, with `ruff` and `ty` from PyPI.

## Conventions

### Mathematical notation

Ruff ignores `N802`/`N803`/`N806`/`N999`, so identifiers keep the notation of the
implementation plan: `R_pre`, `R_post`, `F_r`, `Y`, `Lambda`, `C_t`. Keep that capitalisation
rather than renaming to snake case. `RUF001`–`RUF003` are ignored too, so prose may use en
dashes, accents (Équateur) and mathematical symbols.

Prefer descriptive names over brevity everywhere else, even at the cost of verbosity.

### Delay-weight indexing

Weight arrays are stored densely **from their first supported lag**, which differs between
distributions. Never hard-code the offset; use the constants in `delay_distributions`.

| Array | First lag | Constant |
| --- | --- | --- |
| `w` (serial interval) | 1 | `SERIAL_INTERVAL_FIRST_LAG` |
| `f_inc` (incubation) | 1 | `INCUBATION_FIRST_LAG` |
| `f_tost` (onset → transmission) | 0 | `TOST_FIRST_LAG` |

`f_inc` must have **no mass at lag 0** — that is what makes the onset-anchored recursion well
ordered. `f_tost` may. At most one of the two may be supported at lag 0.

### The naive-model core

`renewal.py` carries the renewal operator in two interchangeable forms. Use
`delay_weighted_sum` when the driving series is observed data (DLO, SSE) and
`delay_design_matrix` when it is latent and the sum must stay symbolic (SSI, and the Stage-8
onset-anchored models); they agree exactly. `switch_index` is the single home of the
`R`-switch convention — don't re-derive it inline.

`pymc_models.build_naive_model` and `forward_simulation.simulate_naive` cover `dlo`, `sse`,
`ssi` and `cori`. Before extending either:

- **`cori` is not one of the five compared models.** It is the Poisson `k → ∞` limit, present
  as the target of the collapse checks. Keep it out of the four analyses.
- **`R_pre`, `R_post` and `k` each take either a fixed `float` or a `LogNormalPrior`.** A fixed
  value becomes a constant in the graph rather than a random variable, so the fixed-`k`
  analyses and the fixed-`θ` particle-filter checks share one builder with the estimated-`k`
  analyses.
- **Days with zero force of infection are dropped from the likelihood**
  (`renewal.likelihood_days`): the observation there is a point mass at 0. A *positive* count
  on such a day raises rather than being dropped silently. Nothing is dropped on the real
  series — the index case drives every day at `max_lag = 110`.
- **The SSI latent block is centred**, inline in `pymc_models._infectivity_latents`. That is
  the Stage-3 baseline and the single call site Stage 3 redirects. A 500-draw smoke fit on the
  real series already produces a few divergences, so the §6.3 risk is real and on schedule.
- `pymc_models.compile_joint_logp` evaluates a built model's joint density at named values on
  their **natural** scale — no log transforms, no Jacobian. It is how the tests compare a
  likelihood with a hand-written density, and what the Stage-6 evidence estimators consume.

### The SSE-SO latent block — read before touching Stage 3

Measured on the committed data with `max_lag = 110`, `k = 0.18`:

- SSE-SO has **110** latents, not the ~85 an early draft assumed. `λ_t > 0` on every inference
  day because the index case contributes `f_tost,t · D_0` at every lag.
- Shapes `k λ_t` run from 0.383 down to **2.2 × 10⁻⁶**; 44 are below 10⁻², 20 below 10⁻⁴.
- A `Gamma(2.2e-6)` sits at `E[log Y] ≈ −4.5 × 10⁵` with `SD ≈ 4.5 × 10⁵` on PyMC's internal
  log scale. **The shapes, not the dimensionality, are the problem.**
- Truncating `f_tost` does not fix it — at `max_lag = 40` there are still 98 latents with a
  smallest shape of 8.9 × 10⁻⁶.

The fix is `negligible_latent_threshold` in `config/config.yaml` (null until Stage 3 sets it):
drop days whose `λ_t` is below it and set `E_t = 0`, with error bounded by
`R_pre · Σ_dropped λ_t`. Benchmark on the **real** SSE-SO model, never a well-conditioned stub.

Also note: mean-1 rescaling is an **initialisation** fix, not a reparameterisation — on the log
scale it is a pure translation, leaving the Gamma shape and hence the geometry untouched. Do
not implement "log-scale latents with a Jacobian"; it duplicates PyMC's default transform.

### Day indexing

Day 0 is the first observed onset (5 April 2018). The ERT arrived on day 33 and withdrew on
day 110; the analysis window is days 0–110 inclusive (111 rows). Day 0 is an initial condition
in every model, so likelihoods run over days 1–110. `R` switches from `R_pre` to `R_post` on
day 33 **in each model's own time index**. The resulting timing asymmetry between the naive and
onset-anchored models is a deliberate, reportable result, not a bug — and it is **~11 days**,
the mean **incubation period**, because the two conventions differ by the gap between a
transmission and the resulting onset. It is *not* the 15.3 d serial interval, which is the
infector-onset to infectee-onset gap and does not separate the conventions.

### Module layout

Flat package `end_of_outbreak/` (no `src/`), with `scripts/` for analysis and plotting.

- Keep modules **genuinely separable**. The Snakemake rules list the exact package modules
  each rule depends on, so that editing `particle_filter.py` re-runs no MCMC fits. A
  kitchen-sink `utils.py` inside the package, or a fat `__init__.py` that re-exports
  everything, would defeat that scheme — don't add either.
- Reusable *method* goes in the package; presentation-only helpers go in `scripts/utils.py`.
- Strict split between **compute-and-save** scripts and **load-and-plot** scripts, so
  restyling a figure never re-runs MCMC.
- Analysis scripts are named for what they do, never for figure numbers.

### Pipeline

`Snakefile` drives everything in three tiers — `fit` (MCMC, minutes–hours), `rac`/`evidence`
(seconds–minutes), `figure` (seconds) — so a change at one tier never re-runs the tiers above
it. When adding a module, **add it to the right dependency list at the top of the `Snakefile`**
(`FIT_CORE`, `RAC_CORE`, `EVIDENCE_CORE`, `PLOT_CORE`); Snakemake's `code` trigger hashes only
a rule's own body and does not follow Python imports.

The `rac` rule writes `results/<analysis>/<model>_rac.csv`. That file carries the
supplementary **RAT** column too, for the onset-anchored models — one derived-results file per
model, named for the headline quantity.

Per-analysis parameters live in `config/config.yaml`, keyed per analysis so that tweaking the
`k` prior for the estimated-`k` analyses does not invalidate the fixed-`k` fits. Seeds live
there too.

**Any config value that changes a rule's output must appear in that rule's `params:`.** The
default profile drops the `mtime` trigger, and the `input` trigger tracks the *set* of input
files rather than their contents — so listing `config.yaml` under `input:` does **not** make a
rule re-run when a value inside it changes. `analysis_params()` at the top of the `Snakefile`
collects these; extend it when you add a config key that affects results.

`results/` and `figures/` are committed. Git does not preserve mtimes, so the default profile
(`config/snakemake_profile/`) drops the `mtime` rerun trigger; after a fresh clone,
`snakemake --touch` restores mtime consistency if you want it.

Scripts take `argparse` arguments and are invoked from `shell:` directives, never via
Snakemake's `script:` directive, so that every script stays runnable and debuggable on its own.

### Testing

`pytest`, tests under `tests/`. Validation is by property rather than by golden file wherever
possible: normalisation, moment additivity, limiting cases (`k → ∞` collapsing SSE/SSI onto
Cori), likelihood-vs-simulation agreement, and analytic-vs-particle-filter agreement.

**Likelihood-vs-simulation, for models with latents.** For the closed-form models the check is
direct: enumerate short histories, evaluate the built model's likelihood for each, compare with
the simulator's frequencies. For SSI (and later the SO models) the simulator has to match the
*marginal* of the counts while the builder supplies the *joint* with the latents. The bridge is
that the latent priors are conditionally independent given the counts —
`p(I) = E_{Y ~ Π Gamma(k I_u, k)}[Π_t Poisson(...)]` — so a naive Monte-Carlo marginalisation is
unbiased and cheap. The fast vectorised integrand used for it is pinned against the built
model's joint density at random points first; don't skip that step, or the test degenerates
into checking a reimplementation against itself.

There are **two** particle-filter checks, and they answer different questions — don't collapse
them into one:

- **Fixed `(R_pre, R_post, k)`, conditioning matched to the pipeline.** The equality check:
  a regression test that the RAC arithmetic is right. Agreement is required.
- **Particle MCMC (PMMH) over the parameters.** The pipeline check: an independent route to
  the posterior sharing no machinery with the PyMC fits, which is the main guard against the
  latent-parameterisation risk. SBC cannot substitute — it validates an implementation
  against itself, so a misconception shared by the model and the simulator survives it.

  For the latent models a PMMH RAC curve **should not** match the MCMC one: the filter
  conditions on *filtering* latents, the pipeline on *smoothed* ones. The gap is the §5.6
  approximation, and measuring it is the point. Don't "fix" it.

PMMH mixes only if the variance of the estimated log-likelihood is roughly 1–3 at the mode.
Measure it before writing the sampler; if it can't be reached at a tractable particle count,
record that and keep the synthetic-data tiers. `particle_mcmc.py` is validation, not a results
path — it stays out of `rule all` and out of every tier's dependency list.

## Decisions already taken

Do not silently revisit these; they are argued out in the implementation plan.

1. **Serial interval.** The generic EVD estimate (mean 15.3 d, SD 9.3 d) is used for all five
   models. Its variance budget of 86.49 d² admits published EVD incubation estimates; the
   outbreak-specific estimate (mean 19.46, SD 6.08) does not, and is retained only for a
   sensitivity analysis.
2. **Naming and reporting of the risk.** **RAC** (risk of additional cases) is the headline
   quantity, used throughout; **RAT** (risk of additional transmission) appears only in the
   supplementary analysis and its methodology. For the onset-anchored models compute **both**
   and report the gap between them.
3. **Latents at the conditioning day.** Use the full-data (smoothed) posterior for every day —
   one fit per model. The methods section must state the approximation and its downward bias.
4. **`R` switch.** Day 33 in each model's own time index (see *Day indexing* above).
5. **Naming.** The day-level negative-binomial model is `DLO`, not the older "CIO"; the package
   is `end_of_outbreak`.
6. **SSI-SO infectivity prior.** `Y_t | D_t ~ Gamma(k D_t, k)`. The `Gamma(k I_t, k)` written
   in `starter_docs/models.jpeg` is a transcription slip.
7. **Particle MCMC is a check, never a results path.** Main analyses stay in PyMC. See
   *Testing* above and §6.6 of the implementation plan.

## Open items

- **Latent-variable parameterisation** (`latent_parameterisation` in `config/config.yaml`) is
  `null` until the Stage-3 sampler benchmark has run. Scripts must fail loudly rather than
  pick a default silently. Record the chosen default here once benchmarked.
- **Incubation period.** Currently WHO Ebola Response Team (2014), NEJM 371:1481–1495, gamma
  with mean 11.4 d and SD 8.1 d, which leaves a residual TOST of mean 3.9 d and SD 4.57 d.
  Configurable in `config/config.yaml`; `check_delay_budget` rejects any estimate whose
  variance exceeds the serial interval's.

## Git workflow

- Commit regularly with descriptive messages. Run `pixi run check` first.
- The default branch is `main`. Branch for feature work rather than committing to `main`
  directly.
- `starter_docs/` is untracked by design and must stay that way.
