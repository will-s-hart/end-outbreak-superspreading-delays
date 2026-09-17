Committed figures (PDF + PNG), one subdirectory per analysis, drawn by `scripts/plot_*.py` from
what the earlier tiers wrote to `results/`.

| Path | What it is |
| --- | --- |
| `model_schematic/model_schematic.pdf`, `.png` | Main Fig. 1: the methods schematic — how each anchoring convention dates a transmission pair, what state each carries past the last day of the record, and when `R` switches. The only figure here that draws no data, and so the only one whose rule has no `results/` input; it also carries no number, because the incubation period it depends on is quoted in the caption through `\resultnum`. |
| `<analysis>/<analysis>.pdf`, `.png` | The analysis's main figure. |
| `sustained_transmission_fixed_k/sustained_transmission_fixed_k.pdf`, `.png` | Main Fig. 6: RAC/RAT/RST for infection- and onset-anchored branching models at fixed `k`. |
| `underreporting/underreporting.pdf`, `.png` | Main Fig. 7: RAC for the onset-anchored models at 100%, 80% and 60% assumed reporting. The 100% curves are `onset_models_fixed_k`'s — at a reporting probability of one the model *is* that analysis's — so this figure's inputs span three results directories. |
| `no_switch_fixed_R/…`, `no_switch_single_R/…` | The exploratory no-switchpoint variants, drawn by `scripts/plot_no_switch.py`. In no report figure and in no `rule all` target; built by `pixi run pipeline-no-switch`. The fixed-`R` one is the RAC panel alone, since nothing is estimated; the single-`R` one adds a panel for the one reproduction number it fits. |
| `delay_distributions/delay_distributions.pdf`, `.png` | Supplementary Fig. S1: incubation, TOST and target/implied serial intervals. |
| `sustained_transmission_estimated_k/sustained_transmission_estimated_k.pdf`, `.png` | Supplementary Fig. S2: the Main Fig. 6 layout with `k` estimated. |

Both formats are written from a single render, so they cannot disagree. Restyling is tier 3 and
costs seconds: nothing here re-runs a fit.
