# What the study has measured

Findings, not conventions. They are here so that a real result is not mistaken for a bug and
"fixed", and each names the file that holds the number, so none of it has to be taken on trust.

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
SSE 0.036), because a burst of four re-establishes far more readily than a lone case. Run this RAC
code on the companion project's regime — one index case, `R = 2` throughout, no intervention — and
SSI comes out *below* SSE at 0.37 of it, matching `sse-ssi-pmo` Fig. 1. Neither difference flips
the ordering alone.

## What happens when `k` is estimated rather than transplanted?

On the real series, under the shared `LogNormal(0.18)` prior
(`results/naive_models_estimated_k/dispersion_posteriors.json`):

| | DLO | SSE | SSI |
| --- | --- | --- | --- |
| posterior median `k` (95% CrI) | 0.38 (0.23–0.63) | 0.50 (0.31–0.85) | 0.14 (0.07–0.27) |
| log evidence, `k` fixed → estimated | −97.1 → −93.2 | −106.8 → −99.4 | −84.8 → −84.5 |

- **The divergence is real but it is not DLO against the rest.** DLO vs SSI: overlap 0.094, median
  ratio 2.7, `P(k_DLO > k_SSI) = 0.99`. SSE vs SSI: overlap 0.032, ratio 3.6, 0.999. But **DLO vs
  SSE: overlap 0.59** and `P(k_DLO > k_SSE) = 0.22` — barely distinguishable. Do not write this up
  as "DLO's `k` is the odd one out"; SSI's is, and DLO and SSE agree.
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
- **Only SSI's posterior is compatible with the literature `k = 0.18`.** DLO's and SSE's both sit
  above the prior's own 97.5th percentile (0.36) despite it being deliberately informative. The
  evidence gains say it more sharply: letting `k` move is worth **7.4 nats to SSE and 3.9 to DLO
  but only 0.26 to SSI**. Analysis 1 was charging DLO and SSE for a value that was never theirs.

## What does onset anchoring do?

Analyses 3 and 4 compare `sse`, `ssi`, `sse_so` and `ssi_so`. Under estimated `k`:

| | SSE | SSI | SSE-SO | SSI-SO |
| --- | ---: | ---: | ---: | ---: |
| posterior median `k` | 0.501 | 0.140 | 0.178 | 0.175 |
| posterior median `R_post` | 0.634 | 0.731 | 0.279 | 0.255 |

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
- **The attenuation prediction passes.** SSE-SO's `k` moves below SSE's towards 0.18 (median ratio
  2.81, overlap 0.089, `P(k_SSE > k_SSE-SO) = 0.992`), whereas SSI changes much less (overlap
  0.750). Incubation convolution attenuates day-level dispersion when onsets are treated as
  infections; individual-level dispersion is comparatively stable to regrouping.
- **Do not universalise the direction.** It is measured under this reset convention, switch
  convention, outbreak history and posterior. The incubation-scale explanation is the mechanism for
  this result, not a theorem for arbitrary time series or interventions.

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
all six synthetic parameter comparisons.

**The two estimators of the risk** (`validation/results/rac_method_comparison.csv`). The gap
between `refit_daily` and `single_fit_filtered` **has both signs**; report it as measured, not as a
claimed direction. **DLO and SSE are the control** — with no latent state their whole gap is the
parameter conditioning — so anything the latent models show on top of that is the cost of
conditioning latents and parameters differently.
