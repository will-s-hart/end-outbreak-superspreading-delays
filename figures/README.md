Committed figures (PDF + PNG), one subdirectory per analysis, drawn by `scripts/plot_*.py` from
what the earlier tiers wrote to `results/`.

| Path | What it is |
| --- | --- |
| `<analysis>/<analysis>.pdf`, `.png` | The analysis's main figure. |
| `sustained_transmission_fixed_k/sustained_transmission_fixed_k.pdf`, `.png` | Main Fig. 5: RAC/RAT/RST for infection- and onset-anchored branching models at fixed `k`. |
| `delay_distributions/delay_distributions.pdf`, `.png` | Supplementary Fig. S1: incubation, TOST and target/implied serial intervals. |
| `sustained_transmission_estimated_k/sustained_transmission_estimated_k.pdf`, `.png` | Supplementary Fig. S2: the Main Fig. 5 layout with `k` estimated. |

Both formats are written from a single render, so they cannot disagree. Restyling is tier 3 and
costs seconds: nothing here re-runs a fit.
