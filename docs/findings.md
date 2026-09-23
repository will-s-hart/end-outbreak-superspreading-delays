# What the study has measured

Findings, not conventions. They are here so that a real result is not mistaken for a bug and
"fixed", and each names the file that holds the number, so none of it has to be taken on trust.

## What did real-time conditioning change?

Almost nothing, in the numbers — and that is worth knowing rather than assuming.

Moving from the old retrospective estimand (one full-record fit supplying the state for every
day) to the real-time one (a fit per conditioning day) moved **3 of the 28 threshold crossings,
each by exactly one day**: SSE-SO and SSI-SO at 0.05, and SSE-SO at 0.01, all under estimated
`k`. The other 25 are identical. Every finding below survives to the precision it is quoted at.
So the smoothed approximation was, on this series, very nearly right — which is a fact about this
series and this posterior, not a general licence to use it.

What changed is what the quantity *is*. `P(· | data up to day t)` is now literally true rather
than a shorthand the methods had to disown, and the caveat about a state informed by data after
day `t` is gone.

The visible change is at the **start** of the curve, not the end. RAC used to be pinned at 1 for
every early day by construction: parameters informed by the whole outbreak leave no doubt that
more cases follow. It now climbs from about **0.28 on day 1**, through 0.83 by day 5, to ~0.99 by
day 10. Early on the record is one case and a few days of silence, `R` is prior-dominated, and
the probability of any further case is genuinely well below one —
`−k log(1 + R/k)·Λ(1) ≈ −0.34` at the prior median gives ≈ 0.29. The day-to-day roughness over
those weeks is the estimand too, not the sampler: each day is a separate posterior conditioned on
a separate record.

**Convergence in the four core analyses, over 1820 fits** (14 model/analysis pairs × 130 days,
`<model>_rac_diagnostics.csv`): worst `R̂` over every sampled variable is **1.0124**; 108 days
exceed 1.005; 157 divergences in 14.6M draws, worst single day 38; minimum bulk ESS 356.
**The onset-anchored models under the wide `k` prior are the hard cases**: 51 of SSE-SO's
estimated-`k` days and 13 of SSI-SO's are above 1.005, against 26 for SSE-SO at fixed `k` and
none at all for the infection-anchored models bar 18 days split between the two SSI fits. The
worst single fit anywhere is SSI-SO's day 106 under estimated `k` — `R̂` 1.0123, ESS 356, 38
divergences, which is 0.5% of its draws against a 1% gate. That is the wide prior's sampling
cost, and it is why both estimated-`k` analyses run at `target_accept: 0.99`: at 0.95 the same
fits failed the gate outright (`config/config.yaml` records the measurement). The estimated-`k`
inverse-CDF models use the benchmarked compound sampler (Slice for `k`, NUTS for the `R`
parameters and latent uniforms).

## What did incomplete reporting change?

In the authoritative 60%/80% sweeps, assuming cases went unreported delayed the 0.05 crossing
by nine to thirteen days at 60% reporting, and moved it much further than choosing SSE versus SSI
did. **Every curve now reaches 0.01 as well**, which the 0–110 window could not show: at 60%
reporting the four crossings are days 112 (SSE-SO and SSI-SO), 121 (SSE) and 125 (SSI), all of
them *after* the ERT actually withdrew on day 110 (`results/underreporting_*/<model>_rac.csv`).
A declaration criterion at 0.01 is therefore not robust to a reporting assumption this weak —
the same conclusion as before, now stated as located crossings rather than as four curves left
above the line.

**Onset anchoring's advance is the part the reporting assumption does not touch.** SSE-SO crosses
0.05 seven days before SSE under complete reporting, seven days before it at 80% and eight at
60%; SSI-SO leads SSI by eight, nine and twelve days — the one place the advance *widens* as the
assumption weakens, because SSI loses more to it than SSI-SO does. The RAC-to-RAT gap inside the
onset-anchored models is similarly fixed — six days for SSE-SO at all three levels, six/five/four
for SSI-SO. So the two things the anchoring convention buys survive a reporting assumption strong
enough to move every curve by more than a week.

The fitted curves were empirically ordered `RAC(60%) ≥ RAC(80%) ≥ RAC(100%)` at every day.
Do not present that as a theorem: adding a hidden case conditionally raises risk, but these are
separate fits whose parameter and latent posteriors change with the reporting assumption. A
future crossing would require investigation, not automatic rejection as mathematically
impossible.

The reporting sweeps use latent unreported counts, and now cover the infection-anchored models
as well. All 624 fits pass the configured convergence gate: worst `R̂` 1.009, minimum bulk ESS
567, and 64 divergences in 20.0M draws. That needed 8000 draws after 4000 tuning steps per chain,
against 2000/2000 elsewhere: the discrete unreported-count block mixes far more slowly than the
rest of the model and sets the budget. At 2000/2000 one SSE fit reached `R̂` 1.047 on
`unreported_incidence` and failed the gate outright.

Against the earlier totals-plus-binomial run, no threshold crossing moved by more than one day
and the pointwise RAC differences had no systematic sign; the median absolute difference was 0.77
combined chain-based Monte-Carlo standard errors, with 29 of 312 points beyond two and a maximum
of 5.13. That comparison was made between the two parameterisations at the old sampler budget and
has not been repeated since the budget rose, so the strict pointwise Monte-Carlo check remains
outstanding. The scientific findings above are unchanged across both.

## How far can the model probabilities be trusted?

**Not to the second decimal place, and between the onset-anchored pair not at all.** In
`onset_models_fixed_k` the two are separated by 0.048 nats of log evidence, against
bridge-sampling standard errors of 0.037 (SSE-SO) and 0.019 (SSI-SO) — a combined 0.042, so the
gap is 1.1 standard errors. The probabilities that follow (SSE-SO 0.499, SSI-SO 0.476) are a
tie: **which of the two ranks first is not resolved by this run**, and the earlier statement
that the ranking was real at 2.4 standard errors no longer holds. Under estimated `k` they are
equally close (0.46 and 0.48). What is resolved is the pair against the infection-anchored
models, which are three nats and more away. Raising
`n_proposal_draws` in the evidence step would shrink the error as `1/sqrt(n)`, and the step is
tier-2 — seconds to minutes, not a refit.

The estimator itself is not the problem. It is seeded per model from the analysis seed
(`AnalysisSetting.reconstruction_rng`) and reproduces to 1e-16 across processes and thread counts;
the models with no latent block reproduce to the last bit. It is the *width* of the honest
standard error against the *narrowness* of the gap being measured.

**Re-run `evidence` and `dispersion` after every pull.** On 2026-09-18 the pulled
`dispersion_posteriors.json` reported a `k` median for naive SSI that the `ssi_posterior.nc`
beside it does not produce — and that posterior is bit-identical to the one committed before the
pull. `dispersion` is a pure deterministic read of the stored `k` draws, so this is proof of an
inconsistency rather than an inference about one. Both rules are cheap, so recomputing them is
the only way to know the tier-2 files match the tier-1 files they claim to summarise. Note that
`snakemake --touch` will stamp an inconsistent pair as current and hide exactly this.

## Does the machinery reproduce an outside answer?

**Yes, exactly** (`validation/results/rac_validation.md`). Thompson et al.'s convention differs
from this project's by exactly one day, `γ(t) = Λ(t − 1)`, so their eqs. (3)–(5) are this
pipeline's arithmetic at `k → ∞` with their Gamma posterior for `R`. Their published Équateur `R`
estimate is recovered (`Gamma(28, 10.65)`, mode 2.53 — Fig. S3D dashed) and so is their risk curve
(Fig. S3E dashed).

## What does the mechanism do to the risk?

At `k = 0.18` and `R = 2.6`, **DLO's RAC sits within 10% of the Poisson limit while SSE's is three
to four times smaller** (day 90: 0.41 vs 0.12, Poisson 0.44). The same `k` does far less work in
DLO. The inequality behind it — `−RΛ ≤ Σ_j φ(μ_j) ≤ φ(Λ)` for `φ(μ) = −k log(1 + Rμ/k)` — is
universal; the *direction* of the comparison with SSE is a property of a thin, spread-out profile,
not a theorem, so do not restate it as one.

**SSI's RAC is *above* SSE's after the final case, and that is not a bug** (day 90: 0.123 vs
0.093). Both differences flow from one fact: **SSI's heterogeneity is attached to people you have
been watching; SSE's is attached to events that have not happened yet.** SSI's residual risk is
revised downwards by weeks of silence and SSE's is not (`E[Λ_Y]/Λ ≈ 0.35` late in the series),
while at matched means SSE's remaining risk is a rare burst and SSI's is spread over 54 people who
all still exist. At day 90, matched at 0.40 expected further cases: SSE has a 9% chance of anything
at all but ~4.3 cases if it fires; SSI has a 32% chance of typically 1.2. RAC asks only "any
further case?", so it reads 9% against 32%.

**Do not present this as "DLO is the odd one out"** — DLO and SSI are on the same side of SSE, for
the same reason.

**The ordering is specific to RAC and to this setting** — check both before claiming it anywhere.
Ask "would transmission re-establish?" instead and it reverses on the same data (SSI 0.021 against
SSE 0.036), because a burst of four re-establishes far more readily than a lone case. Neither
difference flips the ordering alone.

## What happens when `k` is estimated rather than transplanted?

On the real series, under the shared `LogNormal` prior with median 0.18 and 95% interval
0.018–1.8 — a decade on each side of the literature value
(`results/naive_models_estimated_k/dispersion_posteriors.json`):

| | DLO | SSE | SSI |
| --- | --- | --- | --- |
| posterior median `k` (95% CrI) | 0.91 (0.40–2.50) | 1.88 (0.78–5.74) | 0.054 (0.013–0.233) |
| log evidence, `k` fixed → estimated | −97.1 → −89.8 | −106.8 → −91.8 | −84.7 → −83.8 |

These are the numbers under the **wide** prior, which is now the only one. Under an earlier prior
only a factor of two wide on each side, the same three medians were 0.38, 0.50 and 0.14 and the
gains 3.9, 7.4 and 0.26 nats: the ordering and every qualitative statement below held, but the
posteriors were compressed towards the prior and the gains understated, which is why the narrow
prior was dropped rather than kept as a sensitivity.

- **The divergence is real but it is not DLO against the rest.** DLO vs SSI: overlap 0.016, median
  ratio 16.8, `P(k_DLO > k_SSI) = 0.999`. SSE vs SSI: overlap 0.004, ratio 35.0, 0.99994. But
  **DLO vs SSE: overlap 0.45** and `P(k_DLO > k_SSE) = 0.14` — barely distinguishable. Do not
  write this up as "DLO's `k` is the odd one out"; SSI's is, and DLO and SSE agree.
- **The split is day-level against individual-level, which is where onset-as-infection bites
  hardest.** DLO and SSE both attach their excess variance to a **day** — DLO to aggregate
  incidence, SSE to the day's pooled transmission through a freshly drawn `λ_t` — while SSI
  attaches it to the **individual**. The series being fitted is *onsets*, and the incubation period
  (mean 11.4 d, SD 8.1 d) scatters each day's infections forward over a wide kernel, so a naive
  model reading onsets as infections sees day-to-day variation the convolution has already largely
  averaged out, and answers with a larger `k`. **Day-level dispersion is attenuated by the
  conflation itself.** SSI is close to invariant to it, because the convolution *regroups*
  individuals across days and the aggregate infectivity of `n` i.i.d. individuals is `Gamma(kn, k)`
  whichever `n` they are: regrouping moves the cohorts, not the variance.
- **Only SSI's posterior is compatible with the literature `k = 0.18`.** DLO's and SSE's sit an
  order of magnitude above it, and SSE's median is above even the 97.5th percentile of its own
  prior (1.8) — under a prior this wide, that is the data's answer and not the prior's. The
  evidence gains say it more sharply: letting `k` move is worth **15.0 nats to SSE and 7.3 to DLO
  but only 0.9 to SSI**, each net of the marginal-likelihood cost of the prior's width. Analysis 1
  was charging DLO and SSE for a value that was never theirs.
- **Estimating `k` redraws the risk gap rather than closing it.** The 0.05 crossings move from
  107/95/97 (DLO/SSE/SSI) at fixed `k` to 107/103/92 estimated: SSI's `k` falls, so it settles
  five days earlier, and SSE's rises, so it moves back eight days towards DLO and the Poisson
  limit.

## What does onset anchoring do?

Analyses 3 and 4 compare `sse`, `ssi`, `sse_so` and `ssi_so`. Under estimated `k`:

| | SSE | SSI | SSE-SO | SSI-SO |
| --- | ---: | ---: | ---: | ---: |
| posterior median `k` | 1.856 | 0.052 | 0.169 | 0.160 |
| posterior median `R_post` | 0.612 | 0.803 | 0.283 | 0.267 |

- **Why onset anchoring makes the risk settle earlier.** The total onset-to-onset serial interval
  is shared, so this is not a shorter generation interval. The decomposition changes which side of
  conditioning day `t` a transmission occupies. The SO models place infection at transmission
  time: infections already generated under historical `R_post` stay in the incubation pipeline when
  controls are reset, and the remaining TOST opportunity of older cohorts is largely exhausted. The
  naive models move those infection decisions to the later onset date and apply the reset `R_pre`
  there, making residual transmission look younger and persist longer. The same indexing error
  applies the ERT effect to onsets rather than transmissions, explaining why naive `R_post` is much
  higher: post-arrival onsets that arose from pre-arrival infections are otherwise charged to the
  controlled period.
- **The onset-anchored `k` posteriors are their prior, and that is now a finding rather than an
  ambiguity.** SSE-SO's is 0.169 (0.025–1.488) and SSI-SO's 0.160 (0.020–1.615) against a prior
  of 0.18 (0.018–1.8), and releasing `k` is worth −0.11 and −0.02 nats. A posterior that
  reproduces an *informative* prior is ambiguous — it looks the same whether the data agree or say
  nothing — and that ambiguity is what the decade-wide prior was introduced to settle. It settles
  it the second way: **under onset anchoring these data do not locate `k` at all.**
- **So the attenuation prediction cannot be tested on the onset-anchored side of this series.**
  SSE's `k` does sit an order of magnitude above SSE-SO's (median ratio 11.0, overlap 0.126,
  `P(k_SSE > k_SSE-SO) = 0.976`), but by the bullet above that comparison holds an identified
  posterior against a prior, so **do not quote it as attenuation measured** — the earlier write-up
  did, on narrow-prior posteriors that were themselves prior-dominated. SSI and SSI-SO overlap
  0.52 and are not separated at all. What survives as evidence for attenuation is Analysis 2's
  day-level/individual-level split, where every posterior compared is identified.
- **The anchoring advance becomes mechanism-dependent when `k` is estimated.** SSE-SO leads SSE by
  15 days at both thresholds, SSI-SO leads SSI by 4, against 7 and 8 at fixed `k`. The asymmetry
  is on the infection-anchored side: releasing `k` moves SSE and SSI in opposite directions, so
  the gap each has to close differs while the onset-anchored curves barely move.
- **Do not universalise the direction.** It is measured under this reset convention, switch
  convention, outbreak history and posterior. The incubation-scale explanation is the mechanism for
  this result, not a theorem for arbitrary time series or interventions.

## What did extending the window change?

The window now runs to day 130 (13 August 2018), twenty case-free days past the ERT's withdrawal,
because ending it at the withdrawal left curves above 0.01 on the last day with no crossing to
report. Three things are worth recording.

- **Every curve in the study now settles below both thresholds inside the window.** The last to
  do so are 60% under-reporting SSI (0.01 on day 125), DLO (118), the Poisson limit Cori (117) and
  wide-prior SSE (114). No macro in `results/report_numbers.tex` is a bound, so the "not before
  ⟨date⟩" wording the generator can emit is currently unused — and that is the check that the
  twenty days were enough.
- **The first 110 days did not move.** At fixed `k`, DLO's and SSI's curves reproduce the
  pre-extension run to four decimal places on every shared day; SSE-SO's differ by at most 0.0041
  overall and 0.0019 after the last onset, because `max_lag` follows the window and the delay
  weights were rediscretised with it. **No threshold crossing moved.** The per-day fits cannot
  change: each conditions on `counts[:t+1]`, which the extension does not touch.
- **The parameter panels, evidences and `k` summaries are conditioned twenty days past the
  withdrawal, and those days are counterfactual in one respect** — `R_post` applies to them,
  though the response had ended. The report says so in Section 2.1. Refitting the fixed-`k`
  analysis to day 110 instead moves every `R_post` median by under 0.005 and every log evidence
  by less than its bridge-sampling standard error, so nothing quoted depends on the choice; it is
  flagged as a limitation rather than corrected, because the alternative would break the
  guarantee that the curve's last conditioning day and the tier-2 summaries rest on one posterior.

## What do the switchpoint and superspreading contribute?

The three variants of Analysis 3 (report Figs. S4–S6) each remove one ingredient. All are at
fixed `k = 0.18`.

- **Most of the anchoring advance is the switchpoint, not the retained state.** Remove the switch
  and SSE-SO leads SSE by 3 days at both thresholds while SSI-SO leads SSI by 0, against 7 and 8
  days with the switch in place. Estimating one `R` per model instead of fixing it at 0.95 changes
  nothing: all four posteriors are 1.00 (0.54–2.03) and every crossing falls on the same day as
  under the fixed `R`. So what the advance mostly measures is the ~11-day displacement of the
  intervention between the two time indices, which is the mechanism the report already named.
- **What the retained state contributes on its own is the RAC/RAT gap**, and that survives the
  switch's removal intact: 8 days for both onset-anchored models, against 6–7 with the switch.
  The incubation pipeline is a property of the model, not of the intervention.
- **Superspreading is decisively better supported than Poisson transmission under infection
  anchoring, and only weakly so under onset anchoring.** SSI beats Cori by 7.0 nats (−84.8 against
  −91.8); SSI-SO beats Cori-SO by 0.8 (−81.8 against −82.6). Normalised over the four together:
  SSI-SO 0.67, Cori-SO 0.30, SSI 0.03, Cori 3 × 10⁻⁵. Once incubation is represented explicitly,
  much of what individual-level dispersion was accounting for is already explained — the same
  result the unidentified `k` posteriors report from the other direction.
- **Anchoring moves the declaration further than superspreading does, and the two interact.**
  With superspreading removed, anchoring alone is worth 16 days at 0.05 (Cori against Cori-SO)
  against 8 days in the models that carry it (SSI against SSI-SO). Superspreading is worth 10 days
  under infection anchoring (Cori against SSI) but 2 under onset anchoring (Cori-SO against
  SSI-SO). Each mechanism brings the declaration forward, and each does so by less in the presence
  of the other: they act on the same residual risk.
- **The no-switch curves are forward predictives, not reset predictives.** With `R` constant the
  reset to `R_pre` is a no-op, so compare those two figures with each other and not with
  Fig. 4.

## Do the numerical routes agree?

**Model evidence** (`validation/results/evidence_validation.md`). Bridge sampling reproduces
deterministic quadrature for DLO and SSE to 1.1 × 10⁻³ and 2.9 × 10⁻³ nats (1.4 and 2.6 standard
errors), with the quadrature box's boundary density 41–44 nats below its peak — the only check
that is not sampler-against-sampler. The three estimators agree on real-series SSI within their
own error bars; importance sampling sits 2.1 combined s.e. below bridge sampling, the largest
discrepancy anywhere in the study, so **treat a future move past ~3 as a regression, not noise**.
The evidence is invariant to the latent parameterisation (32 vs 33 free coordinates, 1.3 combined
s.e. apart), which is what shows the marginalisation `pm.Potential` carries the whole removed
factor and not just its shape.

**Onset models and the particle route**
(`validation/results/onset_particle_mcmc_validation.md`). Natural and convenient onset simulation
agree, and the deterministic one-day-incubation limit recovers Cori/SSE/SSI after shifting the
naive switch by one day. With future `R = 0`, RAT is exactly zero and RAC is exactly the
probability that the retained Poisson incubation pipeline is non-empty; explicit reset simulation
agrees with both. At 250 particles the largest measured `Var(log L̂)` is 0.081; the particle
smoother and MCMC agree within 0.0214 at matched conditioning; PMMH/PyMC 95% intervals overlap for
all six synthetic parameter comparisons. The filtering curve in that table now comes from
`filtered_risk` itself, so the check exercises a results path rather than a reimplementation of
one; it sits up to 0.65 from the matched smoothing curve, which is a difference of estimand and
not of implementation.

**The RAC equality check's tolerance was mis-specified until it was re-run**, and the fix is
worth remembering. It compared a *single* particle-smoother run against the MCMC curve and judged
the difference against the MCMC standard error alone — but one smoother run is about twice as
noisy, because path degeneracy leaves the late-window estimate resting on a fraction of the
particles. It now averages six independent runs and combines both errors (2.79e-03 against
6.81e-03). The same averaging independently confirms the exact latent marginalisation: over six
runs the smoother agrees with it to ±3e-4 with no systematic sign. **A tolerance that omits one
side's error is not a tolerance**, and this one passed for a long time by luck.

**The two estimators of the risk** (`validation/results/rac_method_comparison.csv`, naive models
at fixed `k`, every eighth conditioning day). The gap between `refit_daily` and
`single_fit_filtered` is **concentrated at the start of the window and negligible in the tail**:

| | max \|gap\| | on day | after the last onset (day ≥ 58) |
| --- | ---: | ---: | --- |
| DLO | 0.2652 | 1 | mean **+0.0002**, range [−0.0004, +0.0013] |
| SSE | 0.0640 | 1 | mean **−0.0002**, range [−0.0011, +0.0002] |
| SSI | 0.0513 | 1 | mean **−0.0060**, range [−0.0144, −0.0003] |

**DLO and SSE are the control**: with no latent state their whole gap is the parameter
conditioning, so the comparison decomposes the approximation. In the decision-relevant tail the
parameter half is ±0.0002 and the latent half — everything SSI shows on top of that — is about
0.006, roughly thirty times larger but still well under a percentage point. Early on, where
`refit_daily` has almost no data and `single_fit_filtered` is using a posterior informed by the
whole outbreak, the gap is large for exactly the reason it should be.

Report it as measured. The signs differ by model and by period, and only SSI's tail gap has a
consistent one (negative: refitting sits below filtering).
