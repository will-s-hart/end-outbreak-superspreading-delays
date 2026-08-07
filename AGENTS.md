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
| `pixi run fmt` | `ruff format end_of_outbreak scripts validation tests` |
| `pixi run lint` | `ruff check end_of_outbreak scripts validation tests` |
| `pixi run typecheck` | `ty check end_of_outbreak scripts validation tests` (clears `VIRTUAL_ENV` first) |
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
anchoring from the shared delay triple. `fitting.fit_model` is the one place that turns a built
model into an `InferenceData`, and it records on the result the things the draws do not carry —
the latent parameterisation, and the value of any parameter that was *fixed* rather than
estimated (a fixed parameter is a constant in the graph, so it never appears in the posterior).
`forward_simulation.simulate_naive` covers the naive models; the onset-anchored simulators
arrive with Stage 8. Before extending any of them:

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
rests on is `validation/results/sampler_benchmark.md` (74 MCMC runs). Two independent mechanisms,
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

**Rebuild the marginalised latents before computing RAC or RAT — this is the one way to get the
default silently wrong.** The removed latents are exactly the days from the last observed case
(day 58) onwards, which is exactly the range a late conditioning day `t` needs for its
incubation pipeline. They are *not* in `idata.posterior`; `model_days` returns only the sampled
days, so an array indexed by position will look plausible and be truncated at day 57. Call
`pymc_models.marginalised_latent_conditional` once per posterior draw of `(R_pre, R_post, k)`,
draw from the `Gamma(shape, rate)` it returns, and splice the result onto the sampled block to
get a complete latent path by day. That is exact — the removed latents are conditionally
independent of the sampled ones as well as of each other — and it is done once per draw, not
once per `t`, so the one-fit-serves-every-day economy of §5.6 is untouched.

Validate that reconstruction with a **matched pair of fits**: `marginalised_inverse_cdf` (needs
reconstruction) against plain `inverse_cdf` (nothing removed) must give the same RAC curve
within Monte-Carlo error. That is why both stay in the registry. Stage 4 ran it and it passes;
see *The RAC calculators* below.

`pymc_models.latent_block_structure` is the single source for a block's layout — the days, the
scales, and the two coupling weight vectors with `c_u = R_pre·a_u + R_post·b_u`. The builders
use it too, so it cannot drift from what they build.

### The RAC calculators — Stage 4

`risk_of_additional_cases.py` holds the estimand and nothing else: `pooled_remaining_weight`
(Λ(t)), the closed forms of §5.3, the posterior averaging, the reconstruction the default
parameterisation makes necessary, and `simulated_risk_curve`, which is the cross-check today
and — for the onset-anchored models, which have no closed form — the actual estimator from
Stage 8. Four things to know before touching it:

- **RAC(t) is one number per day, not a distribution.** It is a posterior *probability*, so the
  average over draws happens inside it: `risk_curve` returns `1 − mean_draws exp(log P)`. Use
  `monte_carlo_standard_error` when comparing two curves — the between-chain spread is the
  scale any such comparison has to be judged against.
- **Λ(t) is the whole retained state for SSE and SSI — and is not enough for DLO.** DLO applies
  a fresh, force-of-infection-independent `k` on every future day, so it needs the whole
  profile `(μ_j)_{j>t}` from `future_force_of_infection`. Collapsing it to Λ(t) would silently
  turn DLO into a different model, and the §5.4 headline would vanish with it.
- **A parameter that was *fixed* in the fit is not in the posterior.** Pass `fixed_k` for the
  fixed-`k` analyses, and `fixed_R_pre`/`fixed_R_post` too for the fixed-`θ` cross-checks.
- **`cori`/`cori_so` remain validation targets**, not compared models, here as everywhere.

**Never compare a filtering RAC with a smoothed one and call agreement a pass.** The estimand
resets from a smoothed state (§5.1, §5.6); a particle filter's forward pass gives filtering
states. `particle_filter.ParticleFilterResult` names both — `latent_paths` are the ancestral
(smoothing) draws, `filtering_remaining_weight` is Λ(t) under the filter — so the equality check
uses the former and the §5.6 measurement uses the latter.

What Stage 4 established (`validation/results/rac_validation.md`, and the checks it summarises):

- **The external validation passes exactly.** Thompson et al.'s convention differs from this
  project's by exactly one day, `γ(t) = Λ(t − 1)`, so their eqs. (3)–(5) are this pipeline's
  arithmetic at `k → ∞` with their Gamma posterior for `R`. Their published Équateur `R`
  estimate is recovered (`Gamma(28, 10.65)`, mode 2.53 — Fig. S3D dashed) and so is their risk
  curve (Fig. S3E dashed).
- **§5.4, measured.** At `k = 0.18` and `R = 2.6` on this series, DLO's RAC sits within 10% of
  the Poisson limit while SSE's is three to four times smaller (day 90: 0.41 vs 0.12, Poisson
  0.44). The same `k` does far less work in DLO. The inequality behind it —
  `−RΛ ≤ Σ_j φ(μ_j) ≤ φ(Λ)` for `φ(μ) = −k log(1 + Rμ/k)` — is universal; the *direction* of
  the comparison with SSE is a property of a thin, spread-out profile, not a theorem, so do not
  restate it as one.
- **SSI's RAC is *above* SSE's after the final case, and that is not a bug** (day 90: 0.123 vs
  0.093). It reads like one — inferring that the late cases were not very infectious ought to
  push SSI down — and it does, hard: `E[Λ_Y(t)]/Λ(t) ≈ 0.35` late in the series, worth −0.17 of
  RAC at day 90. It is outweighed by §5.4's convexity, here with **SSE** in the pooled role
  (+0.21). SSE's dispersion parameter is `kΛ(t)`, which shrinks with the transmission that
  *remains* (0.040 by day 90); SSI's total shape is `k · Σ I_u = 9.7`, fixed by the individuals
  who ever existed, so once each residual slice is small SSI collapses to the Poisson limit.
  `R` works slightly against the gap (SSI's mean `R_pre` is 1.78 against SSE's 2.10). **Do not
  present §5.4 as "DLO is the odd one out"** — DLO and SSI are on the same side of SSE, for the
  same reason.
- **The §5.6 gap has both signs, as §5.6 predicted.** Smoothed minus filtering is strongly
  positive early (up to +0.64 around day 2, where the filter has not yet seen the cases that
  reveal a high infectivity) and negative after the last case (mean −0.03, worst −0.15 around
  day 70). Report it as measured, not as a claimed direction.

### The analysis and plotting scripts — Stage 5

`scripts/run_<analysis>.py` carries **one subcommand per pipeline tier** — `fit`, `rac`,
`evidence`, and `dispersion` where `k` is estimated — and each Snakemake rule invokes exactly one
of them. `scripts/plot_<analysis>.py` reads what those wrote and draws it. Four things worth
knowing before writing the next pair:

- **Everything that varies between the four analyses comes from `config/config.yaml`,** keyed by
  a module-level `ANALYSIS` constant. Stage 7 factored the rest into `scripts/analysis_driver.py`,
  so a run script is now a docstring, that constant, and a one-line `main`. The driver could not
  go in `scripts/utils.py`, which is in `PLOT_CORE` — fit-driving logic there would make every
  restyle a reason to re-run MCMC — so it is named in `FIT_CORE`, `RAC_CORE`, `EVIDENCE_CORE` and
  `DISPERSION_CORE` instead. The **plot** scripts stay one per analysis, because what differs
  between figures is the panel *arrangement*; the panels themselves are in
  `scripts/figure_panels.py` (in `PLOT_CORE`), which is also the one place under `scripts/` that
  imports `risk_of_additional_cases`.
- **The RAC step writes the Monte-Carlo standard error beside the curve.** RAC(t) is a posterior
  *average*, so it carries Monte-Carlo error, and a curve published without it cannot be
  compared with another one. The curve and the error come from a single evaluation of the
  per-draw log-probabilities — call `posterior_state` then
  `log_probability_of_no_further_cases`, rather than `risk_curve_from_posterior`, which discards
  them.
- **The latent reconstruction needs a seed, and a different stream per model.** It is a Monte
  Carlo draw like any other; `reconstruction_rng` derives it from the analysis's sampler seed
  and the model's index, so curves that are meant to be independent are not silently correlated
  through a shared stream. (Stage 4 hit exactly that bug in the matched-pair check.)
- **A fit is an `xarray.DataTree`, and its I/O is xarray's.** `arviz.InferenceData` is a
  deprecated alias for `xr.DataTree` in arviz 1.x, so `fitting.save_fit`/`load_fit` use
  `DataTree.to_netcdf` and `xr.open_datatree`, both pinned to `fitting.NETCDF_ENGINE`
  (`h5netcdf` — a fit has groups, so it is NETCDF4 and needs an HDF5 backend). `load_fit` loads
  eagerly rather than leaving a lazy handle on a results file. Keep using arviz for what it is
  still for — `az.summary` and the rest of the diagnostics.
- **A plotting script computes nothing and fabricates nothing.** `utils.read_model_evidence` and
  `utils.read_dispersion_summary` raise when their file is absent instead of falling back: a pie
  chart of placeholder numbers is indistinguishable from a real one on the page, and so is a
  legend quoting a median the figure worked out for itself. The single piece of method the
  figure tier borrows is `RiskCurve.first_day_below`, because "the day a curve settles below a
  threshold" is a definition the report quotes, and it must not drift between the marker on the
  panel and the number in the text. That is why `risk_of_additional_cases` is the one tier-2
  module in `PLOT_CORE`.

### Model evidence — Stage 6

`model_evidence.py` computes `p(D_{1:110} | model)` with the parameters *and* the latents
integrated out, and turns a set of them into posterior model probabilities under §6.2's uniform
prior over models. Four things to know:

- **The density being normalised is the built model's own joint log-density.** The module calls
  `pymc_models.build_model` with the arguments the fit was run with rather than writing the
  likelihood out again, so the integral estimated is the one that was sampled. That is why
  `R_pre`, `R_post` and `k` must be passed **exactly as they were passed to `fit_model`** — a
  `float` where the parameter was fixed (no prior factor) and a `LogNormalPrior` where it was
  estimated. The prior is part of the question and is not recoverable from the draws.
- **The integral is done in PyMC's unconstrained coordinates, Jacobian included.**
  `UnconstrainedTarget` is the one place that knows this. Getting the transform's *direction*
  wrong shifts every evidence by a constant — a perfectly plausible number and a wrong Bayes
  factor — so it is pinned against `compile_joint_logp` in the tests rather than trusted.
- **Bridge sampling is the default**; prior Monte Carlo and importance sampling are the §6.5
  agreement checks, and `log_evidence_by_quadrature` is the deterministic answer for a model
  with at most three free coordinates. Do not add a harmonic-mean estimator.
- **Exact latent marginalisation does not change the evidence** and nothing needs reconstructing
  here — unlike the RAC calculators, which do need the removed latents back.

What Stage 6 established (`validation/results/evidence_validation.md`):

- **Bridge sampling reproduces deterministic quadrature** for DLO and SSE on the real series to
  1.1 × 10⁻³ and 2.9 × 10⁻³ nats (1.4 and 2.6 standard errors), with the quadrature box's
  boundary density 41–44 nats below its peak. This is the only check in the stage that is not
  sampler-against-sampler.
- **The three estimators agree on real-series SSI**, all within their own error bars. Importance
  sampling sits 2.1 combined s.e. below bridge sampling — a pass, but the largest discrepancy
  anywhere in the study; treat a future move past ~3 as a regression, not as noise.
- **The evidence is invariant to the latent parameterisation** (32 vs 33 free coordinates,
  1.3 combined s.e. apart), which is what shows the marginalisation `pm.Potential` carries the
  whole removed factor and not just its shape.

### The dispersion posteriors — Stage 7

`posterior_comparison.py` summarises one positive scalar's posterior and measures how far two of
them differ, and the `dispersion` rule turns the `k` draws of an estimated-`k` analysis into
`results/<analysis>/dispersion_posteriors.json`. Three measures, because none of them says
enough alone: the **median ratio** (location, but not width), the **overlap** `∫ min(p_a, p_b)`
(scale-free, symmetric, and finite where a KL divergence would not be), and
**`P(k_a > k_b)`** for independent draws. All are computed on the log scale, as the density plots
are; the overlap is invariant to that choice, since both densities pick up the same Jacobian.
Every *pair* is compared rather than a designated reference model, so the same file serves
Analysis 4, which has no DLO.

What Stage 7 measured on the real series, under the shared `LogNormal(0.18)` prior:

| | DLO | SSE | SSI |
| --- | --- | --- | --- |
| posterior median `k` (95% CrI) | 0.38 (0.23–0.63) | 0.50 (0.31–0.85) | 0.14 (0.07–0.27) |
| log evidence, `k` fixed → estimated | −97.1 → −93.2 | −106.8 → −99.4 | −84.8 → −84.5 |
| RAC first below 0.05 | day 107 | day 99 | day 96 |
| RAC first below 0.01 | **never** | day 110 | day 107 |

- **The divergence is real but it is not DLO against the rest.** DLO vs SSI: overlap 0.094,
  median ratio 2.7, `P(k_DLO > k_SSI) = 0.99`. SSE vs SSI: overlap 0.032, ratio 3.6, 0.999. But
  **DLO vs SSE: overlap 0.59** and `P(k_DLO > k_SSE) = 0.22` — the two are barely distinguishable.
  Do not write this up as "DLO's `k` is the odd one out"; SSI's is, and DLO and SSE agree.
- **Only SSI's posterior is compatible with the literature `k = 0.18`.** DLO's and SSE's both sit
  above the prior's own 97.5th percentile (0.36) despite it being deliberately informative. The
  evidence gains say the same thing more sharply: letting `k` move is worth **7.4 nats to SSE and
  3.9 to DLO but only 0.26 to SSI**, so Analysis 1 was charging DLO and SSE for a value that was
  never theirs — and `k = 0.18` really is SSI's number on this series.
- **The §5.4 effect survives estimating `k`, which is the point of the Fig. 1 / Fig. 2 pair.**
  Each model at its own best `k` still puts eleven days between DLO's and SSI's 0.05 crossing,
  and DLO still never reaches 0.01 inside the window. The misapplication of a literature `k`
  makes the gap; the day-level *mechanism* keeps it.

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
- Reusable *method* goes in the package, and so does **config parsing**
  (`configuration.py`): both script trees need it, and the Snakemake rules have to be able
  to name it in their `input:` lists, which they cannot do for a file under `scripts/`.
  Presentation-only helpers — figure styling and the like — go in `scripts/utils.py`, which
  computes nothing: every number a figure draws was written to `results/` by a tier-2 rule.
- Strict split between **compute-and-save** scripts and **load-and-plot** scripts, so
  restyling a figure never re-runs MCMC.
- Analysis scripts are named for what they do, never for figure numbers.

### Where a new script or output file goes

Two trees, and the split is by **purpose, not by cost**:

| | Report analyses | Validation studies |
| --- | --- | --- |
| Scripts | `scripts/run_*.py`, `scripts/plot_*.py` | `validation/run_*.py` |
| Outputs | `results/<analysis>/`, `figures/<analysis>/` | `validation/results/` |
| Driven by | a Snakemake rule | run on demand |
| In `rule all` / a tier's dependency list | yes | **never** |

So: benchmarks, cross-checks, replications of other people's results, degeneracy measurements
and anything else whose subject is *the implementation rather than the outbreak* go in
`validation/`, with a written `.md` summary beside the data. Anything that feeds a figure or a
number in the report goes in `scripts/` and `results/`. **Do not put a check's output under
`results/`** — the point of the split is that everything there can be taken as a report input
without further checking. `validation/README.md` states the local conventions; `particle_mcmc.py`
is the model to follow for the package side (validation *method* still lives in the package, it
just stays out of every rule's dependency list).

`validation/` is covered by `pixi run check` like everything else, and its outputs are committed.

### Pipeline

`Snakefile` drives everything in three tiers — `fit` (MCMC, minutes–hours), `rac`/`evidence`
(seconds–minutes), `figure` (seconds) — so a change at one tier never re-runs the tiers above
it. When adding a module, **add it to the right dependency list at the top of the `Snakefile`**
(`FIT_CORE`, `RAC_CORE`, `EVIDENCE_CORE`, `PLOT_CORE`); Snakemake's `code` trigger hashes only
a rule's own body and does not follow Python imports.

The `rac` rule writes `results/<analysis>/<model>_rac.csv`. That file carries the
supplementary **RAT** column too, for the onset-anchored models — one derived-results file per
model, named for the headline quantity.

Two tier-2 rules apply to only some analyses, and both key off the config rather than a hard-coded
list of names: `dispersion` runs where `fixed_k` is null (`estimates_dispersion`), and the
`figure` rule picks its output up through `dispersion_summary_of`, which returns nothing for the
fixed-`k` analyses. `supplementary_figure` invokes the same plot script with `--figure
supplementary`, for the analyses named in `SUPPLEMENTARY_ANALYSES` — one script per analysis, so
a displaced panel cannot drift out of step with the figure it was displaced from.

Per-analysis parameters live in `config/config.yaml`, keyed per analysis so that tweaking the
`k` prior for the estimated-`k` analyses does not invalidate the fixed-`k` fits. Seeds live
there too.

**`IMPLEMENTED_ANALYSES` at the top of the `Snakefile`, not `config["analyses"]`, is what the
targets are built from.** The config describes all four analyses from the start; only the ones
whose `run_`/`plot_` scripts exist can be built. `rule all` and the convenience aggregates
(`fits`, `results`, `figures`) iterate the former. Add an analysis to it as its scripts land
(Stage 9) — otherwise `snakemake fits` dies with a missing-input error naming a script nobody has
written yet.

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
path — it stays out of `rule all` and out of every tier's dependency list. Stage 4 already
measured `Var(log L̂)` for SSI on the real series — see `validation/results/rac_smc_variance.csv`
— so that go/no-go does not need re-deriving.

The equality check itself lives in `validation/run_rac_validation.py` rather than in `tests/`
where it needs MCMC on the real series; the tests carry the same comparisons on short
histories, which is what keeps `pixi run test` quick.

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
   (`validation/results/sampler_benchmark.md`). `negligible_latent_threshold` is `0.0` — the
   approximation it offered is unnecessary once the uncoupled latents are integrated out
   exactly. See *The latent block* above.
9. **Model evidence by bridge sampling**, with prior Monte Carlo and importance sampling kept as
   the §6.5 agreement checks and quadrature as the exact answer where the space is small enough.
   Validated in Stage 6; see *Model evidence* above.
10. **Fits are stored as NETCDF4 via xarray**, engine `h5netcdf`, read back with
    `xr.open_datatree`. See *The analysis and plotting scripts* above.

## Open items

- **Incubation period.** Currently WHO Ebola Response Team (2014), NEJM 371:1481–1495, gamma
  with mean 11.4 d and SD 8.1 d, which leaves a residual TOST of mean 3.9 d and SD 4.57 d.
  Configurable in `config/config.yaml`; `check_delay_budget` rejects any estimate whose
  variance exceeds the serial interval's.
- **The two naive analyses are built; the onset-anchored ones are not.**
  `naive_models_fixed_k` and `naive_models_estimated_k` run end to end — `pixi run pipeline`
  reproduces both figure directories from the raw CSV in about a minute. Stage 9 adds the two
  onset-anchored analyses; each needs a `run_`/`plot_` pair (the run script being a docstring and
  an `ANALYSIS` constant over `scripts/analysis_driver.py`) and an entry in
  `IMPLEMENTED_ANALYSES`.
- **Onset-anchored forward simulators** (`forward_simulation`) are still to come in Stage 8,
  along with the onset-anchored RAC/RAT calculators, the onset-anchored particle filter and the
  remaining §4.4 equivalence tests. The Stage-3 benchmark therefore has no synthetic SSE-SO arm
  with a known truth; the truncated real series stands in as its second case. The naive-model
  RAC calculators, simulators and filter all landed in Stage 4 and raise `NotImplementedError`
  pointing at Stage 8 when handed an onset-anchored model, rather than quietly doing something
  infection-anchored.
## Git workflow

- Commit regularly with descriptive messages. Run `pixi run check` first.
- The default branch is `main`. Branch for feature work rather than committing to `main`
  directly.
- `starter_docs/` is untracked by design and must stay that way.
