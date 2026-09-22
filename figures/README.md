Committed figures (PDF + PNG), one subdirectory per analysis, drawn by `scripts/plot_*.py` from
what the earlier tiers wrote to `results/`.

| Path | What it is |
| --- | --- |
| `model_schematic/model_schematic.pdf`, `.png` | Main text: the methods schematic — how each anchoring convention dates a transmission pair, what state each carries past the last day of the record, and when `R` switches. The only figure here that draws no data, and so the only one whose rule has no `results/` input; it also carries no number, because the incubation period it depends on is quoted in the caption through `\resultnum`. |
| `<analysis>/<analysis>.pdf`, `.png` | The analysis's comparison figure. In the main text for the core analyses except `onset_models_estimated_k`, which the report carries in its supplement with the three variants below (`SUPPLEMENTARY_ANALYSES` in the `Snakefile`). The estimated-`k` figures draw the `k` panel on a log axis, since the shared prior is a decade wide on each side of its median. |
| `sustained_transmission_fixed_k/sustained_transmission_fixed_k.pdf`, `.png` | Main text: RAC/RAT, then RST, for all four branching models at fixed `k` — split by risk rather than by anchoring, so that each panel carries the naive/onset comparison. Where RAC and RAT share a panel RAC is solid and RAT dashed, as in the under-reporting figure's bottom row. |
| `underreporting/underreporting.pdf`, `.png` | Main text, four panels: RAC for the naive and then the onset-anchored models at 100%, 80% and 60% assumed reporting, and RAC/RAT for all four at 80% and at 60%. The 100% curves are `onset_models_fixed_k`'s — at a reporting probability of one the model *is* that analysis's — so this figure's inputs span three results directories. |
| `onset_models_no_superspreading/…` | Supplement: superspreading and onset anchoring, each with the other held fixed — `cori` and `cori_so` beside Analysis 3's `ssi` and `ssi_so`, drawn by `scripts/plot_no_superspreading.py` from both analyses' results. The fixed-`k` four-panel layout: the two `R` posteriors, the pie normalised over all four (`combined_model_evidence.json`), and the RAC curves across the bottom. The Poisson limits are the palette's two greys, so grey reads as "no superspreading". |
| `no_switch_fixed_R/…`, `no_switch_single_R/…` | Supplement: the no-switchpoint variants, drawn by `scripts/plot_no_switch.py`. The fixed-`R` one is the pie beside the RAC panel, since nothing but the latents is estimated; the single-`R` one adds a panel for the one reproduction number it fits. The onset-anchored curves are dash-dotted there, because without the switch they lie on the naive ones; dashes are RAT's everywhere else. |
| `delay_distributions/delay_distributions.pdf`, `.png` | Supplement: incubation, TOST and target/implied serial intervals. |
| `sustained_transmission_estimated_k/sustained_transmission_estimated_k.pdf`, `.png` | Supplement: the fixed-`k` sustained-transmission layout with `k` estimated. |

Every figure here is in the report, and `tests/test_report_numbers.py` holds it to that. Both
formats are written from a single render, so they cannot disagree. Restyling is tier 3 and
costs seconds: nothing here re-runs a fit.
