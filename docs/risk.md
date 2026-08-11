# The risk estimand and its calculators

Read this before changing `risk_of_additional_cases.py`, `refit_risk.py`, `filtered_risk.py` or
`risk_curves.py`. `AGENTS.md` carries the one-paragraph version.

## The estimand

The **risk of additional cases (RAC)** is a **real-time reset posterior predictive**:

> Fit parameters *and latents* to the record through day `t` alone. Discard the realised
> trajectory after `t`, reset `R` to `R_pre`, and simulate a counterfactual future. RAC(t) is
> the posterior probability that this future contains at least one further case.

So `P(· | data up to day t)` is the right reading, and **every conditioning day gets its own
fit** — about 110 per model. Day 0 is an initial condition in every model and carries no
likelihood term, so the curve starts at day 1.

A companion quantity, the **risk of additional transmission (RAT)**, is the probability of at
least one further *transmission event* after that day. Under the three naive models the two
coincide by assumption — that identification is exactly the conflation this project is about —
and they separate only under `sse_so`/`ssi_so`, where the gap is the contribution of the latent
pipeline of already-infected but not-yet-symptomatic individuals. Use RAC everywhere except the
supplementary analysis that treats RAT explicitly. `RAC(t) ≥ RAT(t)` always.

The **risk of sustained transmission (RST)** is the late-outbreak analogue: after the same
conditioning and reset, it is the posterior probability that the future transmission process
never becomes extinct. “Sustained” means eventual non-extinction, not reaching a finite case
threshold. RST is available for SSE, SSI and their onset-anchored forms, with Cori/Cori-SO as
Poisson-limit checks. It is not defined for DLO: DLO attaches a fresh negative-binomial draw to
aggregate incidence on each day and does not assign an offspring family to each individual.
Always `RST(t) ≤ RAT(t) ≤ RAC(t)`; reset draws with `R_pre ≤ 1` have RST exactly zero.

## The two estimators

Chosen by `rac.method` in `config/config.yaml`:

| | module | role |
| --- | --- | --- |
| `refit_daily` (default) | `refit_risk.py` | the estimand; every figure and report number |
| `single_fit_filtered` | `filtered_risk.py` | one full-record fit, latents filtered per day |

**Never publish from `single_fit_filtered`.** It approximates on two counts: the parameters see
all 111 days, and conditioning the latents on less data than the parameters ignores their
posterior correlation. `R` and the latents are correlated in the posterior — a history is
explained either by a higher `R` or by more infectious cohorts, and the data through day `t`
constrain that trade-off jointly — so a smoothed-parameter/filtered-latent pair is the day-`t`
posterior of *neither*. It is a different quantity, not a cheaper route to this one.

DLO and SSE have no latent state, so only the first approximation applies to them, which makes
them the **control** when reading the comparison in
`validation/results/rac_method_comparison.csv`.

**Only the conditioning changes.** `Λ(t)`, `W(t)` and `M(t)` sum over days `u ≤ t`, so a day-`t`
estimate is the existing closed form applied to `counts[:t + 1]`. `refit_risk` owns no
mathematics: it loops days, calls `fitting.fit_model` on the truncated series, and hands the
draws to `risk_log_probabilities`.

## The closed forms

`risk_of_additional_cases.py` holds the estimands and nothing else. **`risk_log_probabilities` is
the one entry point** — it applies the closed form to the sampled latents and adds the correction
for the rest, returning a `DailyRiskEstimate` that every route produces.

- **RAC(t) is one number per day, not a distribution.** It is a posterior *probability*, so the
  average over draws happens inside it. Use `monte_carlo_standard_error` when comparing two
  curves — the between-chain spread is the scale any such comparison has to be judged against.
- **Λ(t) is the whole retained state for SSE and SSI — and is not enough for DLO.** DLO applies a
  fresh, force-of-infection-independent `k` on every future day, so it needs the whole profile
  `(μ_j)_{j>t}` from `future_force_of_infection`. Collapsing it to Λ(t) would silently turn DLO
  into a different model, and the mechanism headline would vanish with it.
- **A parameter that was *fixed* in the fit is not in the posterior.** Pass `fixed_k` for the
  fixed-`k` analyses, and `fixed_R_pre`/`fixed_R_post` too for the fixed-`θ` cross-checks.

### Eventual extinction under the reset

Let `q` be the smallest extinction fixed point of the one-case offspring distribution under
`R* = R_pre`. It is solved independently for every posterior draw:

```
q = (1 + R* (1 - q) / k)^(-k)       SSE/SSI
q = exp(-R* (1 - q))                Cori
```

The vectorised bracketed solver in `branching_process.py` returns `q = 1` exactly at and below
criticality. For infection-anchored models the conditional log-probabilities of eventual
extinction are

```
SSE:   Lambda(t) log(q)
SSI:  -R* (1 - q) Lambda_Y(t)
Cori: -R* (1 - q) Lambda(t)
```

For SSE the future seeds are negative binomial and their pgf evaluated at `q` reduces to
`q^Lambda`; for SSI and Cori they are conditionally Poisson. Every seed then starts an
independent family whose extinction probability is `q`.

### The onset reset state

Two pieces: future transmission from retained onset cohorts, and the incubation pipeline of
people infected by `t` but not yet symptomatic. `onset_event_probabilities` gives

```
log P(no further case)        = −H(t) − M(t)
log P(no further transmission) = −H(t) − M(t)·[1 − exp(−c)]
log P(no sustained transmission) = W(t) log(q) − M(t)·(1 − q)       [SSE-SO]
                                  = −R* (1 − q)W(t) − M(t)·(1 − q) [SSI-SO/Cori-SO]
```

Restarting from observed onsets alone drops `M(t)` and understates RAC.

**These forms are exact and are written out in the report**, as Proposition 2 of
`report/report.tex` with a proof: `W(t)` is the TOST-survival-weighted retained driving series,
`M(t)` the incubation-survival-weighted expected infections at the *historical* `R`, and
`c = k log(1 + R*/k)` (or `R*` in the Poisson limit) the per-case zero-offspring exponent, with
`H(t) = c·W(t)` for SSE-SO and `R*·W(t)` for Cori-SO/SSI-SO. **If you change the arithmetic,
change the proposition.** `tests/test_stage8_onset.py` pins them three ways: a transcription of
the proposition as plain double sums (no design matrices, so it checks the formulae rather than a
shared abstraction), a degenerate-delay case where they collapse to one line, and the `R* = 0`
case where RAT is exactly zero.

Generation-time distributions are normalised, so they allocate a case's offspring across time
without changing its total offspring pgf. They therefore change `W(t)` and `M(t)`—and hence when
RST falls—but not `q`, the eventual extinction probability under a constant reset `R*`.

## The latents the fit did not sample

The `marginalised` parameterisations integrate some latents out of the likelihood, so they are
**not** in `idata.posterior`. They are the days from the last observed case in the window
onwards — exactly the range a late conditioning day needs for its incubation pipeline — and
`model_days` returns only the sampled days, so an array indexed by position will look plausible
and be truncated. On an early window the block can be removed **entirely**, and then the variable
is absent from the posterior altogether. `posterior_state` handles that, and refuses any latent
that falls between the two blocks.

**They are not drawn back.** In every model the risk is *affine* in the latent path:

```
log P(no further event after t | θ, Y) = −(α(θ) + Σ_u β_u(θ)·Y_u),    β_u ≥ 0
```

`latent_risk_basis` is that `β`, factored as `R_reset·retained + R_pre·pre + R_post·post` so the
draw dependence is three scalars. RAT scales the pipeline part by `1 − e^{−c}`. For RST the
retained-infectivity coefficient is `R_reset·(1 − q)` and the historical pipeline coefficients
are multiplied by `1 − q`; the existing RAC and RAT coefficients are unchanged.
**Individual latents are therefore never needed** — only linear functionals of them.

The removed latents are conditionally independent `Gamma(A_u, B_u)` given `θ`, so the Gamma
moment generating function integrates them out of the *risk* in closed form:

```
E[exp(−β_u·Y_u) | θ] = (1 + β_u/B_u)^(−A_u)
```

`marginalised_risk_correction` does it. This is a **third exact marginalisation**, composing with
the two the fit already performs: the likelihood integrates these latents out of the observation
density, and this integrates the same latents out of the risk. It is exact where drawing them was
Monte Carlo, and there is no random stream to seed. `reconstruct_latent_paths` survives only for
the matched-conditioning simulation check, which needs an explicit path.

**`latent_risk_basis` is pinned to the closed forms by `β_u = log P(0) − log P(e_u)`**, exact by
linearity. Keep that test if you touch either: an independently derived `β` that drifted from the
closed forms would be invisible in the output.

**SSE-SO carries a boundary latent on the last day of whatever window it was fitted to.** It
cannot reach any fitted onset, because incubation starts at lag 1, but its infections belong to
that day's reset pipeline; its conditional law is therefore its Gamma prior. It moves with the
window under `refit_daily`, and `unsampled_latent_conditional` is the one place that knows it
exists — it returns the marginalised block and this together.

## Comparing against the particle routes

**Match the conditioning before calling agreement a pass.**
`particle_filter.ParticleFilterResult` names its two outputs for what they are — `latent_paths`
are the ancestral (smoothing) draws, `filtering_remaining_weight` and `filtering_pipeline_mean`
are equal-weight filtering snapshots — and which one a check wants depends on what it is
checking:

- the **arithmetic** check fixes `θ` and compares the closed forms against the *smoother*,
  because the MCMC state it is held against is also conditioned on the whole record;
- `filtered_risk` reads the **filtering** snapshots, and so does PMMH, so those two estimate the
  same thing and their agreement *is* a test.

Adaptive resampling may leave the live particle cloud weighted; do not replace the explicitly
resampled filtering snapshots with an unweighted average of that live cloud.

## The bulk failure mode

A hundred and ten windows per model is where a builder edge case hides — most sharply an early
window whose latent block is entirely marginalised. **It is building that breaks, not sampling**,
so `tests/test_refit_risk.py` builds every truncation of the real series for every model without
sampling any of them, which is fast, and separately checks that no latent falls between the
sampled and unsampled blocks.

The tier-1 `rac` step then **fails outright** if any day's fit shows `R̂ > 1.01` or divergences
above 1% of draws — but writes `<model>_rac_diagnostics.csv` first, so a failed run leaves the
evidence of how it failed behind it. A curve built from 110 fits nobody has looked at is not a
result.

## Incomplete reporting

RAC still means **at least one further *true* case**. That is the quantity an end-of-outbreak
declaration is about, and it is what `end-of-outbreak-vbd` computes too — its
`additional_case_prob` takes incidence "with trailing sample dimensions, e.g. posterior draws of
the inferred true incidence".

The consequence is that the driving series of every closed form stops being data and becomes a
posterior quantity, one series per draw. `PosteriorState.true_counts` carries it, and
`risk_log_probabilities` prefers it to `counts` whenever it is there. The arithmetic is
unchanged: `pooled_remaining_weight` and `remaining_tost_weight` already carried a leading draw
axis for the latent paths, and the remaining broadcasts are normalised through `_by_draw`. Only
DLO needed real work — its exponent needs the whole future force-of-infection *profile* rather
than a scalar `Λ(t)`, so the profile is now contracted from
`future_force_of_infection_operator` inside the existing draw-chunk loop, and a per-draw profile
never materialises.

Lower reporting can only raise RAC: the tail of zeros that ends an outbreak may be hiding cases.
A crossing between two reporting levels is a bug, and the quick route checks for one.

### The two estimators under incomplete reporting

`refit_daily` is unchanged in structure. What is worth stating is that **conditioning day `t` is
its window's as-of day**: `counts[:t+1]` includes day `t`, so an onset three days before `t` has
had three days in which to be reported, and a reporting delay's right-truncation therefore
tracks the curve without anything extra. That is *not* the convention of
`end-of-outbreak-vbd`, which conditions on the record strictly before its calculation day — so
the as-of day is an explicit argument rather than `len(counts) - 1`, and a test pins it.

`single_fit_filtered` still works, because the particle filter was extended rather than
refused. Each particle now carries its own history of true counts: the filter *draws* `D_t` from
the model's own count law (`forward_simulation.draw_counts`, shared with the simulators so the
two cannot drift), then weights by `Binomial(c_t; D_t, π_t)`, then draws the day's Gamma latent
from the conditional that `D_t` implies. The ordering still works because `μ_t` depends on the
latents strictly before `t` while the day's latent scale depends on the counts up to and
including it. **DLO is the one model this does not cover** and it raises rather than
approximating: its closed form needs each particle's whole retained profile, not the scalar
remaining weight the filter records.
