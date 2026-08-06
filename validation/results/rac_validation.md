# Stage-4 validation of the RAC calculators

**Conclusion: the RAC arithmetic, the reset state and the latent reconstruction all check out.**
The three "done when" criteria of Stage 4 are met — the closed forms agree with forward
simulation of the reset future, the marginalised and un-marginalised parameterisations give the
same RAC curve, and the Poisson-limit RAC reproduces Thompson et al.'s published Équateur curve
*exactly* under their conventions, because their convention differs from this project's by
exactly one day.

Two things worth carrying forward beyond the pass/fail:

- **The §5.6 smoothing gap has been measured, and it has both signs**, exactly as §5.6 said it
  might. Smoothed-minus-filtering RAC is strongly *positive* early (+0.64 at day 2) and
  *negative* after the last observed case (mean −0.03, worst −0.15). The plan's hypothesis —
  predominantly downward over the decision-relevant late period — survives, and is now a
  measurement rather than an assertion.
- **`Var(log L̂)` for the SSI filter on the real series is ~0.02 at 2000 particles**, two orders
  of magnitude below the 1–3 that PMMH needs. Stage 4b's Tier B (the real series) looks
  comfortably feasible, and that go/no-go no longer has to be re-derived.

Produced by

```
python validation/run_rac_validation.py --checks all
```

writing `rac_thompson_replication.csv` (+ `.png`), `rac_equality_check.csv`,
`rac_equality_summary.csv`, `rac_smc_variance.csv` and `rac_latent_reconstruction.csv`.

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

**SSI — matched conditioning.** RAC resets from a *smoothed* state, so the filter's ancestral
paths (a smoother) are the right comparison and its filtering output is not:

| Comparison | `max \|Δ\|` | Scale |
| --- | --- | --- |
| MCMC smoothed vs particle smoother | 3.23 × 10⁻³ | 3 MCSE ≈ 5.11 × 10⁻³ |
| MCMC smoothed vs particle **filtering** | 6.36 × 10⁻¹ | — (this is the §5.6 gap, not a test) |

**The §5.6 gap, measured.** Smoothed minus filtering, by period:

| Period | Signed gap |
| --- | --- |
| Around day 2 | **+0.64** — the filter has not yet seen the cases that reveal a high early infectivity |
| After the last case (days 58–110) | mean **−0.031**, range [−0.153, −0.0003] |

Both directions appear, and where §5.6 said they would. The decision-relevant late period is
downward: conditioning on the subsequent run of zeros pulls the retained infectivity down and
with it the RAC. Nowhere near enough to move the threshold-crossing dates by more than a day,
but it is the approximation the report has to state, and it now has a number.

**Degeneracy.** 20 000 particles, adaptive resampling at ESS < N/2:

| Diagnostic | Value |
| --- | --- |
| Distinct day-0 ancestors among the final particles | 1085 of 20 000 |
| Resampling steps over 110 days | 9 |
| Minimum ESS after weighting | 1142 |

Mild, because the daily counts are small (maximum 6, 80 of 111 days at zero) and the latent
transition is driven by the *observed* counts rather than by the previous latent state, so the
filter is fully adapted.

## 3. `Var(log L̂)` — the Stage-4b stopping rule, settled early

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

## 4. The matched pair: is the latent reconstruction exact in practice?

`marginalised_inverse_cdf` removes the latents the data constrain only through `exp(−Σ_j μ_j)`
and rebuilds them afterwards from `Gamma(k·scale_u, k + c_u)`; plain `inverse_cdf` samples all
of them. On SSI the removed block is one latent — the final cohort, day 58 — out of 31. The two
must give the same RAC curve.

Judged in Monte-Carlo standard errors (the between-chain spread of the curve, ×√2 for two
independent estimates), over five independent triples of fits:

| Comparison | Worst day, in MCSE | Range |
| --- | --- | --- |
| Across parameterisations (marginalised vs plain) | 2.0 | 1.5 – 2.4 |
| Within one parameterisation (different seed) | 1.9 | 1.2 – 2.5 |

The two distributions are indistinguishable, which is the point: the difference the
reconstruction introduces is the same size as the difference between two runs of the *same*
fit. In absolute terms the curves differ by at most 2.4 × 10⁻³ (rms 8.8 × 10⁻⁴).

The rebuilt latent itself agrees with the sampled one directly: `E[Y_58] = 0.199` rebuilt
against 0.210 sampled, both Monte-Carlo estimates of the same conditional.

A single pair of fits would not have supported this claim. The worst-day statistic is a maximum
over 111 correlated days and ranges from 1.5 to 2.4 MCSE across the five triples, so one pair
could easily have read 2.4 and looked like a near-miss — while the *within*-parameterisation
comparison, where nothing is being reconstructed at all, reaches 2.5. Hence the replication:
what is being checked is that the two comparisons have the same spread, not that either is
small.

## What this does *not* cover

- **The onset-anchored models.** Their RAC has no closed form and their reset state includes the
  incubation pipeline; the calculators, simulators and filter are Stage 8. Everything here
  raises `NotImplementedError` when handed `sse_so` or `ssi_so` rather than quietly doing
  something infection-anchored.
- **RAT.** It coincides with RAC under all three naive models by assumption — the conflation the
  project is about — so there is nothing separate to validate until Stage 8.
- **The estimated-`k` analyses.** Every check here fixes `k`, either at 0.18 or by the fit. The
  RAC machinery takes `k` per draw and the tests cover that path, but the checks on the real
  series do not exercise a `k` posterior.
