# The models, the fits and what is derived from them

Read this before changing `renewal.py`, `pymc_models.py`, `latent_parameterisations.py`,
`fitting.py`, `forward_simulation.py`, `model_evidence.py` or `posterior_comparison.py`.
`AGENTS.md` carries the models table and the conventions; the risk calculators are in
[risk.md](risk.md).

## The renewal core and the builders

`renewal.py` carries the renewal operator in two interchangeable forms. Use `delay_weighted_sum`
when the driving series is observed data (DLO, SSE, Cori-SO) and `delay_design_matrix` when it is
latent and the sum must stay symbolic (SSI, SSE-SO, SSI-SO); they agree exactly, and
`tests/test_renewal.py` checks that they do.

`pymc_models` builds every model: `build_naive_model` for the infection-anchored ones,
`build_onset_anchored_model` for the onset-anchored ones, `build_model` to dispatch on anchoring
from the shared delay triple. `fitting.fit_model` is the one place that turns a built model into
a posterior `DataTree`, and it records the things the draws do not carry — the latent
parameterisation, and the value of any parameter that was *fixed* rather than estimated. It is
called on **windows**, not only the whole record: `refit_risk` hands it `counts[:t + 1]` once per
conditioning day, and nothing in it needs to know which is happening.

`forward_simulation.simulate_naive` covers the naive models.
`simulate_onset_anchored_convenient` draws the direct onset recursion used for inference;
`simulate_onset_anchored_natural` additionally draws explicit infections and their incubation
allocations. They are distributionally equivalent by Poisson splitting, and
`simulate_onset_anchored` dispatches between them.

Before extending any of these:

- **`R_pre`, `R_post` and `k` each take either a fixed `float` or a `LogNormalPrior`.** A fixed
  value becomes a constant in the graph rather than a random variable, so the fixed-`k` analyses
  and the fixed-`θ` particle-filter checks share one builder with the estimated-`k` analyses.
- **`latent_parameterisation` is a required argument for any model with latents.** There is no
  silent default; `config/config.yaml` is the source of truth.
- **Days with zero force of infection are dropped from the likelihood**
  (`renewal.likelihood_days`): the observation there is a point mass at 0. A *positive* count on
  such a day raises rather than being dropped silently. Nothing is dropped on the complete record
  — the index case drives every day at `max_lag = 110`.
- `pymc_models.compile_joint_logp` evaluates a built model's joint density at named values on
  their **natural** scale — no log transforms, no Jacobian. `compile_observation_logp` gives the
  likelihood term alone, which is the only way to compare two models whose latent blocks differ
  in size.

## The latent block

**Default: `marginalised_inverse_cdf`**, recorded in `config/config.yaml`. Two independent
mechanisms, and they compose:

- **Exact marginalisation.** A latent that reaches no observation day carrying a case enters the
  likelihood only through the `exp(−Σ_j μ_j)` factor, which is linear in the latents and so
  factorises. Its Gamma prior is conjugate to that, giving
  `∫ Gamma(y; k·scale, k)·e^{−cy} dy = (1 + c/k)^{−k·scale}` and a conditional posterior
  `Gamma(k·scale, k + c)`. On the complete record this removes **52 of SSE-SO's 110** latents —
  exactly the pathological tail, since the last case is on day 58 — and one of SSI's/SSI-SO's 31.
  **No approximation whatsoever.** Shorter windows remove more.
- **Inverse-CDF reparameterisation.** Sample `Uniform(0, 1)` and push through the Gamma quantile
  function. `pm.icdf` *does* have a gradient in this PyTensor, so this runs under NUTS.

`pymc_models.latent_block_structure` is the single source for a block's layout — the days, the
scales, and the two coupling weight vectors with `c_u = R_pre·a_u + R_post·b_u`. The builders use
it too, so it cannot drift from what they build.

The removed latents still matter to the risk, and are handled there analytically; see
[risk.md](risk.md). Validate that with a **matched pair of fits**: `marginalised_inverse_cdf`
against plain `inverse_cdf` (nothing removed, the sampler carries the lot) must give the same RAC
curve within Monte-Carlo error. That is why both stay in the registry.

### What the sampler benchmark settled, and should not be re-litigated

The comparison is `validation/results/sampler_benchmark.md` (74 MCMC runs).

- **It is a geometry problem, not a warm-up problem.** Mean-1 rescaling changes nothing (3999 of
  4000 draws divergent on SSE-SO, same as the untouched model), and *more* tuning makes `R̂`
  worse, not better. Rescaling is a pure translation on the log scale, so the shape and hence the
  curvature are untouched.
- **The geometric-mean start is not representable.** `E[log Y] ≈ −1/α`, and SSE-SO's untouched
  shapes reach `α = 2.7 × 10⁻⁶`, so the typical value is `exp(−3.7 × 10⁵)` — zero in double
  precision. `LOG_UNDERFLOW` marks the wall at `α ≈ 1.4 × 10⁻³`. A coordinate whose typical set is
  outside the floating-point range cannot be tuned into behaving.
- **Divergent runs are not merely noisy.** The untouched SSE-SO fits report `R_post` between 0.24
  and 0.30 where every converged run gives 0.32–0.33.
- **`negligible_latent_threshold` is `0.0` and is not needed.** No threshold both clears the
  underflow wall and leaves the likelihood alone: ~10⁻² is needed for the shapes and costs ~0.1
  nats, a 10% shift in every Bayes factor. Marginalisation removes the same coordinates for free,
  and every surviving latent has `λ_t ≥ 0.136`. The knob stays only as a fallback for a
  pathological latent genuinely *coupled* to the data, which marginalisation cannot touch.
- **Inverse-CDF and mean-1 rescaling do not compose** — the Gamma quantile function is
  scale-equivariant, so stacking them is provably a no-op. `LatentParameterisation` refuses the
  combination rather than pretending to benchmark it.
- **Do not implement "log-scale latents with a Jacobian"**; it duplicates PyMC's default
  transform.

## Model evidence

`model_evidence.py` computes `p(D_{1:110} | model)` with the parameters *and* the latents
integrated out, and turns a set of them into posterior model probabilities under a uniform prior
over models. It is a property of a model given the **whole record**, so it uses the full-record
fit and is untouched by the conditioning-day machinery.

- **The density being normalised is the built model's own joint log-density.** The module calls
  `pymc_models.build_model` with the arguments the fit was run with rather than writing the
  likelihood out again, so the integral estimated is the one that was sampled. That is why
  `R_pre`, `R_post` and `k` must be passed **exactly as they were passed to `fit_model`** — a
  `float` where fixed, a `LogNormalPrior` where estimated. The prior is part of the question and
  is not recoverable from the draws.
- **The integral is done in PyMC's unconstrained coordinates, Jacobian included.**
  `UnconstrainedTarget` is the one place that knows this. Getting the transform's *direction*
  wrong shifts every evidence by a constant — a perfectly plausible number and a wrong Bayes
  factor — so it is pinned against `compile_joint_logp` in the tests rather than trusted.
- **Bridge sampling is the default**; prior Monte Carlo and importance sampling are the agreement
  checks, and `log_evidence_by_quadrature` is the deterministic answer for a model with at most
  three free coordinates. Do not add a harmonic-mean estimator.
- **Exact latent marginalisation does not change the evidence**, and nothing needs recovering
  here — unlike the risk calculators.

What the model-probability panel reports is a comparison of the models **as specified, priors
included**: identical priors make the Bayes factors well defined, not neutral, because `k` means
an individual-level offspring dispersion in SSE/SSI and a day-level incidence dispersion in DLO.
The caption has to say so.

## Comparing dispersion posteriors

`posterior_comparison.py` summarises one positive scalar's posterior and measures how far two of
them differ; the `dispersion` rule turns the `k` draws of an estimated-`k` analysis into
`results/<analysis>/dispersion_posteriors.json`. Three measures, because none says enough alone:
the **median ratio** (location, not width), the **overlap** `∫ min(p_a, p_b)` (scale-free,
symmetric, and finite where a KL divergence would not be), and **`P(k_a > k_b)`** for independent
draws. All are computed on the log scale, as the density plots are; the overlap is invariant to
that choice, since both densities pick up the same Jacobian. Every *pair* is compared rather than
a designated reference model, so the same file serves the analysis that has no DLO.

## Likelihood-vs-simulation, for models with latents

For the closed-form models the check is direct: enumerate short histories, evaluate the built
model's likelihood for each, compare with the simulator's frequencies. For SSI and the SO models
the simulator has to match the *marginal* of the counts while the builder supplies the *joint*
with the latents. The bridge is that the latent priors are conditionally independent given the
counts — `p(I) = E_{Y ~ Π Gamma(k I_u, k)}[Π_t Poisson(...)]` — so a naive Monte-Carlo
marginalisation is unbiased and cheap. The fast vectorised integrand used for it is pinned
against the built model's joint density at random points first; **don't skip that step**, or the
test degenerates into checking a reimplementation against itself.
