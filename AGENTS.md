# AGENTS.md

Instructions for coding agents working in this repository.

## Project

Research code for a study of how (i) the **mechanism** generating heterogeneity in
transmission and (ii) the common practice of treating **symptom-onset dates as infection
dates** affect the estimated risk of relaxing outbreak-control interventions. The case study
is the 2018 Équateur Province (DRC) Ebola outbreak.

Five models are compared, all driven by the same onset-to-onset serial interval:

| Model | Anchoring | Overdispersion acts at the level of | Latents (all / sampled) |
| --- | --- | --- | --- |
| `dlo` | infections (naive) | the day (`NB` on aggregate incidence, fixed `k`) | none |
| `sse` | infections (naive) | the transmission event (`NB(RΛ, kΛ)`) | none |
| `ssi` | infections (naive) | the individual (latent Gamma infectivity) | `Y_t`, 31 / **30** |
| `sse_so` | symptom onsets | the transmission event | `λ̃_t`, 110 / **58** |
| `ssi_so` | symptom onsets | the individual | `Y_t`, 31 / **30** |

The second latent count is what the sampler actually sees: the rest are integrated out in
closed form, exactly (see *The latent block* below). `cori` and `cori_so` are also built, as the
`k → ∞` Poisson limits — they are validation targets, not compared models.

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

### The model core

`renewal.py` carries the renewal operator in two interchangeable forms. Use
`delay_weighted_sum` when the driving series is observed data (DLO, SSE, Cori-SO) and
`delay_design_matrix` when it is latent and the sum must stay symbolic (SSI, SSE-SO, SSI-SO);
they agree exactly. `switch_index` is the single home of the `R`-switch convention — don't
re-derive it inline.

`pymc_models` builds every model: `build_naive_model` for the infection-anchored ones,
`build_onset_anchored_model` for the onset-anchored ones, `build_model` to dispatch on
anchoring from the shared delay triple. `forward_simulation.simulate_naive` covers the naive
models; the onset-anchored simulators arrive with Stage 8. Before extending any of them:

- **`cori` and `cori_so` are not compared models.** They are the Poisson `k → ∞` limits,
  present as the targets of the collapse checks. Keep them out of the four analyses.
- **`R_pre`, `R_post` and `k` each take either a fixed `float` or a `LogNormalPrior`.** A fixed
  value becomes a constant in the graph rather than a random variable, so the fixed-`k`
  analyses and the fixed-`θ` particle-filter checks share one builder with the estimated-`k`
  analyses.
- **`latent_parameterisation` is a required argument for any model with latents.** There is no
  silent default; `config/config.yaml` is the source of truth.
- **Days with zero force of infection are dropped from the likelihood**
  (`renewal.likelihood_days`): the observation there is a point mass at 0. A *positive* count
  on such a day raises rather than being dropped silently. Nothing is dropped on the real
  series — the index case drives every day at `max_lag = 110`.
- `pymc_models.compile_joint_logp` evaluates a built model's joint density at named values on
  their **natural** scale — no log transforms, no Jacobian. `compile_observation_logp` gives the
  likelihood term alone, which is the only way to compare two models whose latent blocks differ
  in size.

### The latent block — settled in Stage 3

**Default: `marginalised_inverse_cdf`.** Recorded in `config/config.yaml`; the comparison it
rests on is `results/checks/sampler_benchmark.md` (74 MCMC runs). Two independent mechanisms,
and they compose:

- **Exact marginalisation.** A latent that reaches no observation day carrying a case enters the
  likelihood only through the `exp(−Σ_j μ_j)` factor, which is linear in the latents and so
  factorises. Its Gamma prior is conjugate to that, giving
  `∫ Gamma(y; k·scale, k) e^{−c y} dy = (1 + c/k)^{−k·scale}` and a conditional posterior
  `Gamma(k·scale, k + c)`. On the real series this removes **52 of SSE-SO's 110** latents —
  exactly the pathological tail, since the last case is on day 58 — and one of SSI's/SSI-SO's
  31. **No approximation whatsoever**, and nothing is lost: recover a removed latent after the
  fit from `latent_parameterisations.conditional_posterior`, which is what the RAC reset state
  needs.
- **Inverse-CDF reparameterisation.** Sample `Uniform(0, 1)` and push through the Gamma quantile
  function. `pm.icdf` *does* have a gradient in this PyTensor, so this runs under NUTS — the
  trade-off §6.3 anticipated (better geometry, gradient-free samplers only) does not arise.

What the benchmark actually established, and should not be re-litigated:

- **It is a geometry problem, not a warm-up problem.** Mean-1 rescaling changes nothing (3999 of
  4000 draws divergent on SSE-SO, same as the untouched model), and *more* tuning makes `R̂`
  worse, not better. Rescaling is a pure translation on the log scale, so the shape and hence
  the curvature are untouched — exactly as §6.3 argued.
- **The geometric-mean start is not representable.** `E[log Y] ≈ −1/α`, and SSE-SO's untouched
  shapes reach `α = 2.7 × 10⁻⁶`, so the typical value is `exp(−3.7 × 10⁵)` — zero in double
  precision. `LOG_UNDERFLOW` marks the wall at `α ≈ 1.4 × 10⁻³`. A coordinate whose typical set
  is outside the floating-point range cannot be tuned into behaving.
- **Divergent runs are not merely noisy.** The untouched SSE-SO fits report `R_post` between
  0.24 and 0.30 where every converged run gives 0.32–0.33.
- **`negligible_latent_threshold` is `0.0` and is not needed.** No threshold both clears the
  underflow wall and leaves the likelihood alone: ~10⁻² is needed for the shapes and costs ~0.1
  nats, which is a 10% shift in every Bayes factor. Marginalisation removes the same coordinates
  for free, and every surviving latent has `λ_t ≥ 0.136`. The knob stays only as a fallback for
  a pathological latent that is genuinely *coupled* to the data, which marginalisation cannot
  touch.
- **Inverse-CDF and mean-1 rescaling do not compose** — the Gamma quantile function is
  scale-equivariant, so stacking them is provably a no-op. `LatentParameterisation` refuses the
  combination rather than pretending to benchmark it.
- Do not implement "log-scale latents with a Jacobian"; it duplicates PyMC's default transform.

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
8. **Latent parameterisation.** `marginalised_inverse_cdf`, chosen by the Stage-3 benchmark
   (`results/checks/sampler_benchmark.md`). `negligible_latent_threshold` is `0.0` — the
   approximation it offered is unnecessary once the uncoupled latents are integrated out
   exactly. See *The latent block* above.

## Open items

- **Incubation period.** Currently WHO Ebola Response Team (2014), NEJM 371:1481–1495, gamma
  with mean 11.4 d and SD 8.1 d, which leaves a residual TOST of mean 3.9 d and SD 4.57 d.
  Configurable in `config/config.yaml`; `check_delay_budget` rejects any estimate whose
  variance exceeds the serial interval's.
- **Onset-anchored forward simulators** (`forward_simulation`) are still to come in Stage 8,
  along with the RAC/RAT calculators, the onset-anchored particle filter and the remaining §4.4
  equivalence tests. The Stage-3 benchmark therefore has no synthetic SSE-SO arm with a known
  truth; the truncated real series stands in as its second case.

## Git workflow

- Commit regularly with descriptive messages. Run `pixi run check` first.
- The default branch is `main`. Branch for feature work rather than committing to `main`
  directly.
- `starter_docs/` is untracked by design and must stay that way.
