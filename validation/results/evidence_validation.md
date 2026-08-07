# Stage-6 validation of the marginal likelihoods

**Conclusion: the evidence machinery checks out, on the real series and at the numbers Fig. 1C
is drawn from.** Bridge sampling reproduces deterministic quadrature for DLO and SSE to within
its own standard error; the three estimators agree on SSI, where no exact answer exists; and
integrating a latent out in closed form leaves the evidence where it was.

Run with `python validation/run_evidence_validation.py --checks all` (≈ 50 s in total on an
M-series laptop, including six MCMC fits of the real series). Data in the three CSVs beside this
file. The same three comparisons run on eight-day synthetic histories in
`tests/test_model_evidence.py`, which is what keeps `pixi run test` quick.

Everything here is at `k = 0.18` fixed, with the `naive_models_fixed_k` priors and sampler
settings from `config/config.yaml`, on the padded 111-day Équateur series.

**Why this stage needs its own validation at all.** The evidence *is* a normalising constant, so
the error that matters is not noise but a silent constant offset — a Jacobian dropped, a
transform applied backwards, a prior term counted twice. Each of those produces a perfectly
plausible number and a wrong Bayes factor, and no amount of internal consistency between
estimators would reveal it. That is why check 1 exists and why it comes first.

## 1. Bridge sampling against deterministic quadrature — the §6.5 acceptance criterion

At fixed `k` neither DLO nor SSE has latents, so `log p(D | model)` is an integral over
`(log R_pre, log R_post)` alone and can be done on a tensor-product Simpson grid. That grid
shares only the *integrand* with bridge sampling — none of the proposal fitting, the weighting or
the recursion — so agreement is a check of the estimator rather than of a shared assumption.

| Model | Quadrature | Bridge sampling | Difference | In bridge s.e. |
| --- | --- | --- | --- | --- |
| DLO | −97.103831 | −97.102723 ± 0.000798 | +1.11 × 10⁻³ | 1.39 |
| SSE | −106.784299 | −106.781433 ± 0.001089 | +2.87 × 10⁻³ | 2.63 |

401 grid points per axis, box `mean ± 12 sd` in each unconstrained coordinate. **The box has to
contain the mass before "exact" means anything**, so the CSV also carries the ratio of the
largest boundary density to the peak: −44.1 nats (DLO) and −41.1 nats (SSE). The grid is
therefore not truncating anything that matters, and the comparison is real.

**Would have failed if** the difference exceeded a few times the reported standard error, or the
boundary ratio had come out near zero. Both differences are inside 3 s.e. at the third decimal
place of a number near −100, which is the resolution the Bayes factors in Fig. 1C need.

## 2. Three estimators on SSI, where there is no exact answer

SSI carries 31 latents, so the parameter space is 33-dimensional in the sampled coordinates and
quadrature is out. The three estimators share no proposal and no weighting scheme, so their
agreement is evidence about the answer rather than about a common ingredient.

| Estimator | `log p(D | ssi)` | s.e. | Draws | From bridge | In combined s.e. |
| --- | --- | --- | --- | --- | --- | --- |
| Bridge sampling | −84.771912 | 0.0076 | 204 000 | — | — |
| Importance sampling | −84.805265 | 0.0142 | 200 000 | −0.0334 | 2.07 |
| Prior Monte Carlo | −84.834491 | 0.2400 | 200 000 | −0.0626 | 0.26 |

The ordering of the error bars is the expected one and is itself a check: prior Monte Carlo is
unbiased but its variance is set by how far the posterior has moved from the prior, and 0.24 nats
is what that costs on 111 days of real data. Importance sampling, from a moment-matched
*diagonal* normal, sits between the two — its proposal ignores the posterior correlations that
bridge sampling's full covariance captures.

**The one number worth watching**: importance sampling is 2.07 combined standard errors below
bridge sampling. That is a pass, but it is the largest discrepancy anywhere in this study, and it
is in the direction a slightly too-light proposal tail would produce. It is recorded here so
that a future change which pushes it past ~3 is recognised as a regression rather than as noise.

**Would have failed if** any pair disagreed by more than about four times their combined standard
error. Judging against the estimators' own error bars, rather than a fixed tolerance, is
deliberate: a fixed tolerance would be a test of the tolerance, since prior Monte Carlo is
legitimately wide here.

## 3. Exact marginalisation does not move the evidence

`marginalised_inverse_cdf` — the project default — integrates the uncoupled latents out in closed
form and carries the result as a `pm.Potential`. On the real series that removes one of SSI's 31
latents. Plain `inverse_cdf` samples all 31. The normalising constant is the same integral either
way, so this is a direct test that the potential carries the *whole* of the removed factor
`(1 + c_u/k)^{−k·scale_u}`, normalising constant included.

| Parameterisation | Free coordinates | `log p(D | ssi)` | s.e. |
| --- | --- | --- | --- |
| `marginalised_inverse_cdf` | 32 | −84.771912 | 0.0076 |
| `inverse_cdf` | 33 | −84.757255 | 0.0084 |

Difference −0.0147, i.e. **1.30 combined standard errors**. (Coordinate counts include `R_pre`
and `R_post`; the latent blocks are 30 and 31.)

This is the evidence-side counterpart of the matched-pair check Stage 4 ran on the RAC curves,
and it matters more here than there: a missing constant in the potential would shift *every*
latent model's evidence and none of the closed-form ones, which is exactly the pattern that
would make SSI look artificially good or bad in panel 1C.

**Would have failed if** the two differed by more than their combined Monte-Carlo error. Note
what this check cannot see: an error in the potential that happens to be *shared* by the
likelihood the sampler used, since both fits go through the same builder. Check 1 is what covers
that class of error, for the models where it can.

## The Fig. 1 numbers these produce

For the record, since they are what panel 1C reports:

| Model | `log p(D | model)` | Posterior model probability |
| --- | --- | --- |
| SSI | −84.78 | 0.999996 |
| DLO | −97.10 | 4.4 × 10⁻⁶ |
| SSE | −106.78 | 2.8 × 10⁻¹⁰ |

**Read these as §6.5 insists they be read.** They compare the models *as specified, priors
included*. `k = 0.18` is applied to DLO deliberately, as a demonstration of a mistake made in the
literature, so this is a statement about a set of models one of which is knowingly miscalibrated
— not clean evidence about the mechanism of transmission heterogeneity. Fig. 2, where `k` is
estimated under a common prior, is the closer thing to that comparison.

## What this does *not* cover

- **The estimated-`k` analyses.** Every check here fixes `k` at 0.18, so the integral is over two
  parameters plus latents rather than three. The `k` axis is exercised only by the synthetic
  histories in `tests/`. Re-run this study at Stage 7 rather than assuming it carries over: a
  third dimension changes what a moment-matched Gaussian proposal is worth, and it takes DLO and
  SSE out of quadrature's reach at the default grid size.
- **The onset-anchored models.** Stage 8. SSE-SO has 110 latents on the real series, of which 52
  are marginalised; nothing here says how bridge sampling behaves in that geometry, and the
  quadrature check has no analogue there.
- **Any error shared between the builder and the estimator.** Both go through
  `pymc_models.build_model`, which is the point — the integral estimated is the one that was
  sampled — but it means a misconception in the model itself survives every check in this file.
  The particle filter's SMC log-evidence (§6.6, Tier C) is the structurally independent route,
  and it is not yet wired up to these numbers.
- **The posterior itself.** Every standard error here is the Monte-Carlo error of an evidence
  *given these posterior draws*. None of them charges for an error in the fit.
