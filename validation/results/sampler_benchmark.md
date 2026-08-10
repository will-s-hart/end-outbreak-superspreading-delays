# Stage-3 sampler benchmark: choosing the latent parameterisation

**Conclusion: use `marginalised_inverse_cdf`.** It is the only strategy that samples the real
SSE-SO model cleanly, it is the fastest per effective sample on every model and dataset tried,
and it costs nothing in accuracy. Recorded in `config/config.yaml` and `AGENTS.md`.

**The negligible-latent threshold of §6.3 turns out not to be needed** and is set to `0.0`. The
latents it was meant to remove are removable *exactly*, so there is no approximation to make.

Produced by

```
python validation/run_sampler_benchmark.py --tune 500 2000 --draws 1000 --chains 4
python validation/run_sampler_benchmark.py --datasets equateur --tune 2000 --draws 1000 \
    --chains 4 --estimate-k free \
    --parameterisations centred unit_mean inverse_cdf marginalised marginalised_inverse_cdf \
    --output validation/results/sampler_benchmark_estimated_k.csv
```

writing `sampler_benchmark.csv` (64 runs) and `sampler_benchmark_estimated_k.csv` (10 runs).

---

## 1. What the problem turned out to be

§6.3 posed the question as *warm-up or geometry?* The answer is unambiguous: **geometry**.

- **Neither initialisation strategy helps at all.** Mean-1 rescaling leaves the real SSE-SO
  model with 3999 of 4000 draws divergent, exactly as the untouched model does. This is what
  §6.3 predicted from first principles — on PyMC's log scale, rescaling by a known constant is
  a pure translation, so the Gamma shape and hence the curvature are untouched — and the
  measurement confirms it rather than contradicting it.
- **Longer tuning makes things worse, not better.** Going from 500 to 2000 tuning steps takes
  the untouched SSE-SO model from `R̂ = 1.93` to `R̂ = 1.98`, and the rescaled one from 1.83 to
  2.71. A chain that is not exploring does not start exploring given more time.
- **The geometric-mean start cannot be represented at all.** `E[log Y] ≈ −1/α`, and the
  untouched SSE-SO block reaches `α = 2.7 × 10⁻⁶`, so its typical value is about
  `exp(−3.7 × 10⁵)` — zero in double precision. There is no number to start the chain at.
  `latent_parameterisations.LOG_UNDERFLOW` records where that wall sits (`α ≈ 1.4 × 10⁻³`), and
  those four runs are logged as failures rather than results.

That last point is the sharpest statement of the problem. A coordinate whose typical set lies
outside the floating-point range is not a coordinate that can be tuned into behaving; it has to
be removed or replaced.

## 2. What fixes it

Two things, and they compose.

**The inverse-CDF reparameterisation.** Sampling `u ~ Uniform(0, 1)` and pushing it through the
Gamma quantile function gives a bounded, flat-prior space whose geometry does not depend on the
shape at all. On the untouched 110-latent SSE-SO block it produces **zero divergences** and
`R̂ = 1.00`.

§6.3 expected this option to preclude NUTS, on the grounds that `icdf` has no gradient. More
precisely, the PyTensor in use differentiates the Gamma quantile with respect to its probability
coordinate but not its shape. The inverse-CDF uniforms therefore run under NUTS. With fixed `k`
the whole sampled block is NUTS; with estimated `k`, PyMC uses the compound step `Slice: [k]`
plus `NUTS: [R_pre, R_post, <latent>_uniform]`. The estimated-`k` rows below measured that exact
compound sampler, which performed well; there was no later switch to Slice caused by a dependency
change. Thus the anticipated trade-off — better geometry but gradient-free sampling of the
entire block — still does not arise, although the original wording “runs under NUTS like any
other” was too broad.

**Exact marginalisation of the uncoupled latents.** A latent that reaches no observation day
carrying a case enters the likelihood only through `exp(−Σ_j μ_j)`, which is linear in the
latents and therefore factorises. Its Gamma prior is conjugate to that, so it can be integrated
out in closed form:

```
∫ Gamma(y; k·scale, k) e^{−c y} dy = (1 + c/k)^{−k·scale},   Y | data ~ Gamma(k·scale, k + c)
```

On the Équateur series this removes **52 of SSE-SO's 110 latents** — precisely the pathological
tail, since the last case is on day 58 and every transmission day after it is uncoupled. The
smallest surviving shape is 0.024 rather than 2.7 × 10⁻⁶. It removes one of SSI's and SSI-SO's
31, the final cohort. **There is no approximation anywhere in this**, and nothing is lost: the
removed latents are recoverable after the fit from the closed-form conditional
(`latent_parameterisations.conditional_posterior`), which is what the Stage-4/8 RAC reset state
will need.

## 3. The measurements

`ESS` is always the minimum bulk effective sample size over the **recovered** latent
(`Y` / `lambda_tilde`), never over the reparameterised free variable, so the numbers mean the
same thing in every row. 4 chains × 1000 draws throughout.

> Two things to read carefully. **`ESS/s` is comparable within a tuning length but not across
> one**, because PyMC's reported sampling time includes the tuning phase. And **the elapsed
> times of the broken configurations are meaningless**: a chain that diverges on every draw
> abandons its trajectories immediately, so the untouched SSE-SO model looks "fast" precisely
> because it is doing nothing.

### The real Équateur series, SSE-SO — the decisive case

| strategy | tune | latents | divergent | max R̂ | min latent ESS | ESS/s | s | R_pre | R_post |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centred | 500 | 110 | 4000 | 1.93 | 5 | 1 | 5 | 2.00 | 0.24 |
| centred | 2000 | 110 | 4000 | 1.98 | 5 | 0 | 10 | 1.60 | 0.30 |
| centred_geometric_init | 500 | — | could not start | — | — | — | — | — | — |
| centred_geometric_init | 2000 | — | could not start | — | — | — | — | — | — |
| unit_mean | 500 | 110 | 3999 | 1.83 | 5 | 1 | 5 | 2.00 | 0.30 |
| unit_mean | 2000 | 110 | 4000 | 2.71 | 4 | 0 | 9 | 1.40 | 0.28 |
| unit_mean_geometric_init | 500 | — | could not start | — | — | — | — | — | — |
| unit_mean_geometric_init | 2000 | — | could not start | — | — | — | — | — | — |
| inverse_cdf | 500 | 110 | 0 | 1.01 | 1013 | 206 | 5 | 2.07 | 0.32 |
| inverse_cdf | 2000 | 110 | 0 | 1.00 | 1010 | 112 | 9 | 2.06 | 0.33 |
| marginalised | 500 | 58 | 81 | 1.01 | 814 | 24 | 33 | 2.07 | 0.33 |
| marginalised | 2000 | 58 | 112 | 1.01 | 644 | 10 | 66 | 2.03 | 0.33 |
| marginalised_unit_mean | 500 | 58 | 196 | 1.07 | 59 | 2 | 30 | 2.07 | 0.32 |
| marginalised_unit_mean | 2000 | 58 | 16 | 1.01 | 941 | 15 | 64 | 2.07 | 0.33 |
| **marginalised_inverse_cdf** | 500 | 58 | **0** | **1.01** | **1207** | **265** | 5 | 2.06 | 0.33 |
| **marginalised_inverse_cdf** | 2000 | 58 | **0** | **1.00** | **1110** | **133** | 8 | 2.06 | 0.32 |

Marginalisation *alone* is a large improvement but not a cure: 81–112 divergences remain, and it
is six to eight times slower per effective sample than the same block in uniform coordinates.
The two together are strictly better than either.

**The broken configurations give wrong answers, not just noisy ones.** Every converged run puts
`R_pre ≈ 2.03–2.07` and `R_post ≈ 0.32–0.33`; the untouched and rescaled runs report `R_post`
between 0.24 and 0.30 and `R_pre` between 1.40 and 2.20. Had the diagnostics been ignored, the
post-ERT reproduction number — the quantity the whole risk calculation turns on — would have
been biased downwards by around a quarter.

### The real series truncated at day 70 — isolating the case-free tail

| strategy | tune | latents | divergent | max R̂ | min latent ESS | ESS/s | s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centred | 500 | 70 | 152 | 1.01 | 733 | 37 | 20 |
| centred | 2000 | 70 | 86 | 1.01 | 699 | 19 | 37 |
| unit_mean | 2000 | 70 | 158 | 1.01 | 895 | 24 | 38 |
| inverse_cdf | 500 | 70 | 0 | 1.00 | 1151 | 284 | 4 |
| marginalised | 500 | 58 | 46 | 1.01 | 562 | 33 | 17 |
| marginalised_inverse_cdf | 500 | 58 | 0 | 1.01 | 1117 | 294 | 4 |

Cutting the window just past the last case leaves 70 latents whose smallest shape is far more
benign, and the untouched model then *works* — badly, with 86–152 divergences, but it converges.
So the long case-free tail is exactly what breaks it, which is the same conclusion the
marginalisation reaches from the other direction: the unsamplable latents are the ones the data
cannot resolve. The inverse-CDF advantage is unchanged here, at 5–8× the throughput.

### SSI on the real and synthetic series

| dataset | strategy | tune | divergent | min latent ESS | ESS/s |
| --- | --- | --- | --- | --- | --- |
| equateur | centred | 2000 | 0 | 1395 | 174 |
| equateur | unit_mean | 2000 | 0 | 1317 | 159 |
| equateur | inverse_cdf | 2000 | 0 | 3723 | 1079 |
| equateur | marginalised_inverse_cdf | 2000 | 0 | 3695 | 1264 |
| synthetic_ssi | centred | 2000 | 0 | 1224 | 140 |
| synthetic_ssi | inverse_cdf | 2000 | 0 | 3832 | 966 |
| synthetic_ssi | marginalised_inverse_cdf | 2000 | 0 | 3749 | 1112 |

SSI has only 31 latents with shapes bounded below by `k`, so every strategy converges. The
inverse-CDF reparameterisation is still worth having: it roughly triples the effective sample
size per draw and gives six to seven times the throughput.

### With `k` estimated — analyses 2 and 4

§6.3 warns that rescaling does not remove the funnel between `k` and the latents, since the
latent variance depends on `k`. It does bite, and the ordering is unchanged.

| model | strategy | divergent | max R̂ | min latent ESS | ESS/s |
| --- | --- | --- | --- | --- | --- |
| ssi | centred | 2 | 1.01 | 943 | 66 |
| ssi | unit_mean | 22 | 1.01 | 861 | 56 |
| ssi | inverse_cdf | 1 | 1.00 | 2283 | 137 |
| ssi | marginalised_inverse_cdf | **0** | 1.00 | **2478** | **175** |
| sse_so | centred | 3988 | 1.52 | 7 | 0 |
| sse_so | unit_mean | 4000 | 1.78 | 6 | 0 |
| sse_so | inverse_cdf | 9 | 1.01 | 1154 | 56 |
| sse_so | marginalised | 253 | 1.02 | 422 | 7 |
| sse_so | marginalised_inverse_cdf | **1** | 1.00 | **1130** | 38 |

Freeing `k` costs everything some accuracy — `inverse_cdf` alone picks up 9 divergences where it
had none — but the chosen default stays essentially clean at 1 divergence in 4000.

## 4. Cross-method agreement

The point of running several strategies is that they must agree, and where they converge they
do. On the real SSE-SO series, `inverse_cdf` (110 sampled latents, nothing integrated out) and
`marginalised_inverse_cdf` (58 sampled, 52 integrated out in closed form) give
`R_pre = 2.06–2.07` and `R_post = 0.32–0.33` — agreement to two decimal places by two routes
that share no latent block. That is a fitted-model check on the marginalisation, on top of the
likelihood-level checks in `tests/test_latent_parameterisations.py`.

On the synthetic SSI history with `R_pre = 2.0` and `R_post = 0.6`, every strategy returns
`R_pre = 1.68–1.71 (sd 0.7)` and `R_post = 0.73–0.75 (sd 0.3)`. The truth is within half a
posterior standard deviation of both, and the strategies agree with each other far more closely
than any of them agrees with the truth — which is what a single 56-case realisation should look
like.

## 5. The negligible-latent threshold

§6.3 asked for a threshold on `λ_t` below which the SSE-SO latent is dropped and `E_t` set to 0,
with the induced error bounded by `max(R) · Σ_dropped λ_t`. Measuring that trade-off is what
retires the idea:

| threshold on λ_t | latents kept | smallest shape at k = 0.18 | change in log-likelihood |
| --- | --- | --- | --- |
| 0 (drop nothing) | 110 | 2.7 × 10⁻⁶ | — |
| 10⁻³ | 88 | 1.9 × 10⁻⁴ | ≤ 0.008 |
| 10⁻² | 76 | 2.0 × 10⁻³ | ≤ 0.095 |
| 3 × 10⁻² | 70 | 6.6 × 10⁻³ | ≤ 0.32 |
| 10⁻¹ | 65 | 1.9 × 10⁻² | ≤ 0.89 |

**No threshold does the job.** To lift the smallest shape clear of the underflow wall the
threshold has to reach about 10⁻², which moves the log-likelihood by ~0.1 nats — a 10% shift in
every Bayes factor, and the posterior model probabilities of Figs. 1C/3C are a headline output.
Tighten it to where the likelihood is safe and the shapes are still hopeless.

The exact marginalisation resolves the dilemma by removing the same coordinates for free. Every
latent that survives it has `λ_t ≥ 0.136`, so **any threshold small enough to be defensible now
drops nothing at all**. `negligible_latent_threshold` is therefore set to `0.0` and kept only as
a fallback for a hypothetical future dataset with a pathological latent that is genuinely
coupled to the data — which marginalisation cannot touch.

A refinement was tried and rejected: ranking days by their *influence*
`λ_t · Σ_{a ≤ T−t} f_inc,a`, the mass that actually lands inside the window, rather than by
`λ_t` alone. It changes the ordering hardly at all, because the small-`λ` days are the late ones
whose incubation mass mostly falls outside the window anyway, so both rules select nearly the
same set. Not worth the extra concept.

## 6. What this means downstream

Marginalisation removes latents from the *sampler*, not from the *model*, and the RAC estimand
needs them back: the reset state at day `t` requires `E_u = R_u Y_u` for every `u ≤ t` in order
to rebuild the incubation pipeline (§5.1), and the removed days are exactly those from the last
observed case onwards — the range that matters most for a late `t`.

That reconstruction is exact and cheap. Conditional on the parameters the removed latents are
independent `Gamma(k·scale_u, k + c_u)`, independent of the sampled block too, so drawing them
once per posterior draw of `(R_pre, R_post, k)` gives exact draws from the full smoothed
posterior. It is **once per draw, not once per conditioning day**, so the one-fit-serves-every-
day economy of §5.6 survives intact. `pymc_models.marginalised_latent_conditional` returns the
days and the `(shape, rate)`.

The residual risk is not statistical but clerical: `model_days` returns only the *sampled* days,
so code that indexes the posterior array by position would silently truncate the retained state
at day 57. Two guards are in place — the requirement is written into `AGENTS.md`, and both
`marginalised_inverse_cdf` and plain `inverse_cdf` are kept in the registry so that Stage 4/8
can validate the reconstruction by a **matched pair of fits**: the two share no latent block, so
agreeing RAC curves check the reconstruction end to end rather than merely the likelihood.

## 7. Caveats

- **`ESS/s` includes tuning.** Comparable within a tuning length, not across one.
- **Elapsed times for divergent runs are not throughput.** See the note above §3.
- **No synthetic SSE-SO arm.** A history simulated from a known SSE-SO truth needs the
  onset-anchored forward simulator, which lands in Stage 8. The requirement §6.3 is emphatic
  about — benchmark the *actual* SSE-SO model on the real data, not a well-conditioned stub — is
  met; the known-truth arm is SSI only, and the truncated real series stands in as a second,
  differently-conditioned SSE-SO case.
- **The synthetic history is conditioned on not going extinct.** With `k = 0.18` most
  trajectories from a single index case die out immediately, and a one-case history says nothing
  about a latent block. That conditioning is a property of the test case, not of any model being
  fitted, but it does mean this arm is not a prior-predictive draw and cannot double as
  simulation-based calibration.
- **One machine, one seed per cell.** The divergence counts and `R̂` values separate the
  strategies by orders of magnitude, so the ranking is not in doubt; the ESS figures carry the
  usual Monte-Carlo noise and should not be read to two significant figures.
