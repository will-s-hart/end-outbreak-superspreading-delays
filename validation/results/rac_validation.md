# Validation of the RAC calculators

**Conclusion: the RAC arithmetic, the reset state and the exact latent marginalisation all check
out.** The closed forms agree with forward simulation of the reset future; the marginalised and
un-marginalised parameterisations give the same RAC curve; the analytic marginalisation of the
latents the fit removed agrees with the particle smoother to 2.8 × 10⁻³ against a combined
tolerance of 6.8 × 10⁻³; and the Poisson-limit RAC reproduces Thompson et al.'s published
Équateur curve *exactly* under their conventions, because their convention differs from this
project's by exactly one day.

Three things worth carrying forward beyond the pass/fail:

- **The two estimators of the estimand differ by little where it matters.** `refit_daily` minus
  `single_fit_filtered` is concentrated at the start of the window (max 0.27 for DLO on day 1)
  and negligible after the last onset: ±0.0002 for the latent-free models, −0.006 for SSI. Since
  DLO and SSE have no latent state, their gap *is* the parameter conditioning, so the comparison
  decomposes the approximation — the latent half is about thirty times the parameter half, and
  both are well under a percentage point in the tail.
- **`Var(log L̂)` for the SSI filter on the real series is ~0.02 at 2000 particles**, two orders
  of magnitude below the 1–3 that PMMH needs, so the particle-MCMC route on the real series is
  comfortably feasible and that go/no-go does not have to be re-derived.
- **The equality check's tolerance was wrong until now, and is fixed.** It compared a *single*
  particle-smoother run against the MCMC curve and judged the difference against the MCMC
  standard error alone. One smoother run is about twice as noisy as the MCMC curve — path
  degeneracy leaves the late-window estimate on a fraction of the particles — so the tolerance
  understated the comparison's error and the check would fail on an unlucky seed with nothing
  wrong. It now averages six independent smoother runs and puts their spread into the tolerance.

Produced by

```
python validation/run_rac_validation.py --checks all
```

writing `rac_thompson_replication.csv` (+ `.png`), `rac_equality_check.csv`,
`rac_equality_summary.csv`, `rac_smc_variance.csv`, `rac_method_comparison.csv` and
`rac_latent_reconstruction.csv`.

---

## 1. External validation: Thompson et al. (2024), Supplementary Analysis 2

Their conventions differ from this project's in three ways: the Poisson limit rather than an
overdispersed offspring distribution, `R` estimated from the pre-ERT data alone under a flat
prior, and a risk of cases **on or after** day `t` given data strictly **before** it. The last
of these is a one-day shift and nothing more:

```
γ_theirs(t) = Σ_{u < t} I_u (1 − F_{t−u−1}) = Λ_ours(t − 1),
```

so their eq. (5) is this project's posterior average done in closed form. The pipeline
reproduces it to Monte-Carlo error and no further:

| Quantity | Value |
| --- | --- |
| Their `R` posterior (eq. (3), days 1–32) | `Gamma(shape = 28.0, rate = 10.651)` |
| — mean / mode / sd | 2.629 / 2.535 / 0.497 |
| `max \|RAC_ours(t) − risk_theirs(t+1)\|`, 200 000 draws | 2.17 × 10⁻⁴ |

The `R` estimate matches the dashed line of their Fig. S3D (a peak near 2.5 at a density near
0.8), and `rac_thompson_replication.png` matches the dashed line of Fig. S3E: flat at ~1 until
mid-June, through 0.5 in the first days of July, and still visibly above zero when the ERT
actually left.

Decision-relevant summary of the replicated curve:

| | |
| --- | --- |
| RAC first below 0.05 | day 107 — 21 July 2018 |
| RAC first below 0.01 | never, within the analysis window |
| RAC on the withdrawal day (day 110, 24 July) | 0.0303 |

That is the *generic* serial interval, and it is why Thompson et al. argued for the
outbreak-specific one where it exists: under theirs the risk at withdrawal is far lower. Both
statements are reproduced here.

## 2. The equality check (§6.4), at fixed `(R_pre, R_post, k) = (1.77, 0.77, 0.18)`

**Latent-free models — the closed forms against forward simulation of the reset future.** The
reset state is the observed counts, so nothing has to be reconstructed and the comparison is
direct. 20 000 replicates per conditioning day, on a stride-10 grid of days:

| Model | `max \|closed form − simulated\|` | 3 binomial s.e. |
| --- | --- | --- |
| DLO | 5.75 × 10⁻³ | 1.06 × 10⁻² |
| SSE | 6.99 × 10⁻³ | 1.06 × 10⁻² |
| Cori | 2.55 × 10⁻³ | 1.06 × 10⁻² |

**The filter is exact where the model has no latent state**, which is what licenses trusting it
on SSI. Its SMC log-evidence equals the built model's likelihood to the last bit — a difference
of exactly `0.0` for DLO, SSE and Cori — despite sharing no code with the PyMC graph.

**SSI — matched conditioning.** Both sides condition on the whole record, which is what makes
this a check of the *arithmetic*: the MCMC state and the filter's ancestral paths (a smoother)
are estimates of the same distribution, so they must agree. The filter's *filtering* output
answers a different question and belongs to the estimator comparison in §3.

| Comparison | `max \|Δ\|` | Scale |
| --- | --- | --- |
| MCMC vs particle smoother, 6 runs averaged | 2.79 × 10⁻³ | 3 combined s.e. ≈ 6.81 × 10⁻³ |

The worst day sits at 1.9 combined standard errors. **This is also the check that the exact
latent marginalisation is exact**: the MCMC side no longer draws the latents the fit integrated
out, it integrates them out of the risk in closed form, and the smoother — which shares no
machinery with it — lands in the same place.

**The tolerance is the interesting part.** Until now this compared a *single* smoother run and
judged it against the MCMC standard error alone. Measured over six independent runs, one
smoother run is about twice as noisy as the MCMC curve, and its error is not in that tolerance
at all; the check would fail on an unlucky seed with nothing wrong, and pass on a lucky one with
something wrong. Averaging six runs and combining the two errors is what makes the comparison
mean what it says.

For the record, the earlier committed version of this table (3.23 × 10⁻³ against 5.11 × 10⁻³)
came from a filter that no longer exists: `particle_filter.filter_naive` gained a per-day
resampled snapshot afterwards, which consumes the random stream once per day and shifts the
ancestry. The check had not been re-run since.

**Degeneracy.** 20 000 particles, adaptive resampling at ESS < N/2:

| Diagnostic | Value |
| --- | --- |
| Distinct day-0 ancestors among the final particles | 1085 of 20 000 |
| Resampling steps over 110 days | 9 |
| Minimum ESS after weighting | 1142 |

Mild, because the daily counts are small (maximum 6, 80 of 111 days at zero) and the latent
transition is driven by the *observed* counts rather than by the previous latent state, so the
filter is fully adapted.

## 3. `Var(log L̂)` — the particle-MCMC feasibility rule

PMMH mixes acceptably when the variance of the estimated log-likelihood at the mode is around
1–3 (Doucet et al. 2015). For SSI on the real series, over 12 independent runs per particle
count:

| Particles | `log L̂` | sd | Var |
| --- | --- | --- | --- |
| 500 | −83.008 | 0.321 | 0.103 |
| 2 000 | −82.950 | 0.150 | 0.023 |
| 10 000 | −83.005 | 0.069 | 0.005 |

Even 500 particles overshoots the target by an order of magnitude in the *good* direction. The
worry §6.6 raised — path degeneracy over 111 days with a 40–60 day memory — does not
materialise here. Tier B of the particle-MCMC check should be attempted on the real series, and
a few hundred particles will do.

*Caveat:* this is at one plausible `θ`, not integrated over the posterior, and the onset-anchored
filter of Stage 8 carries an incubation pipeline in its state, so §6.6 is right that the
measurement has to be repeated there rather than assumed to carry over.

## 4. The two estimators of the estimand

`refit_daily` (a fit per conditioning day — the estimand) against `single_fit_filtered` (one
full-record fit, latents filtered per day). Naive models at fixed `k`, every eighth conditioning
day, from `rac_method_comparison.csv`:

| | `max \|gap\|` | on day | after the last onset (days 58–110) |
| --- | ---: | ---: | --- |
| DLO | 0.2652 | 1 | mean **+0.0002**, range [−0.0004, +0.0013] |
| SSE | 0.0640 | 1 | mean **−0.0002**, range [−0.0011, +0.0002] |
| SSI | 0.0513 | 1 | mean **−0.0060**, range [−0.0144, −0.0003] |

Not a pass/fail: the two condition on different things, so the question is how far apart they are
and where. **DLO and SSE are the control** — with no latent state their whole gap is the
parameter conditioning — so the table decomposes the approximation. In the decision-relevant tail
the parameter half is ±0.0002 and the latent half, everything SSI shows on top of it, is about
0.006. Early in the window the gap is large for the reason it should be: `refit_daily` has almost
no data there, while `single_fit_filtered` is using parameters informed by the whole outbreak.

Only SSI's tail gap has a consistent sign (negative — refitting sits below filtering). Report the
comparison as measured rather than as a claimed direction.

## 5. The matched pair: is the latent marginalisation exact in practice?

`marginalised_inverse_cdf` removes the latents the data constrain only through `exp(−Σ_j μ_j)`;
the risk then integrates them back out in closed form, using the same `Gamma(k·scale_u, k + c_u)`
conditional. Plain `inverse_cdf` samples all of them instead. On SSI the removed block is one
latent — the final cohort, day 58 — out of 31. The two must give the same RAC curve.

Judged in Monte-Carlo standard errors (the between-chain spread of the curve, ×√2 for two
independent estimates), over five independent triples of fits:

| Comparison | Worst day, in MCSE | Range |
| --- | --- | --- |
| Across parameterisations (marginalised vs plain) | 2.0 | 1.5 – 2.4 |
| Within one parameterisation (different seed) | 1.9 | 1.2 – 2.5 |

The two distributions are indistinguishable, which is the point: the difference the
reconstruction introduces is the same size as the difference between two runs of the *same*
fit. In absolute terms the curves differ by at most 2.4 × 10⁻³ (rms 8.8 × 10⁻⁴).

The removed latent itself agrees with the sampled one directly: `E[Y_58] = 0.206` from the
closed-form conditional the risk marginalises over, against 0.210 estimated from the draws of the
fit that sampled it. The first carries no Monte-Carlo error at all, which is the point of doing
it analytically.

A single pair of fits would not have supported this claim. The worst-day statistic is a maximum
over 111 correlated days and ranges from 1.5 to 2.4 MCSE across the five triples, so one pair
could easily have read 2.4 and looked like a near-miss — while the *within*-parameterisation
comparison, where nothing is being reconstructed at all, reaches 2.5. Hence the replication:
what is being checked is that the two comparisons have the same spread, not that either is
small.

## What this does *not* cover

- **The onset-anchored models.** Their reset state includes the incubation pipeline, and their
  calculators, simulators and filter are checked separately in
  `onset_particle_mcmc_validation.md`. Everything here refuses `sse_so` and `ssi_so` rather than
  quietly doing something infection-anchored.
- **RAT.** It coincides with RAC under all three naive models by assumption — the conflation the
  project is about — so there is nothing separate to validate here; see the onset check.
- **The estimated-`k` analyses.** Every check here fixes `k`, either at 0.18 or by the fit. The
  RAC machinery takes `k` per draw and the tests cover that path, but the checks on the real
  series do not exercise a `k` posterior.
- **The full conditioning-day grid.** The estimator comparison runs on every eighth day, because
  its refit arm costs one MCMC fit per day kept. The gap is a smooth function of `t`, so the
  stride describes it, but a feature narrower than eight days would be missed.
