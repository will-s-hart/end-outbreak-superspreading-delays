The methods-and-results write-up, compiled by the `report` rule.

| Path | What it is |
| --- | --- |
| `report.tex` | The document: abstract, short introduction, methods, one results section per analysis with bullet-point findings, and two appendices. |
| `report.pdf` | The compiled output, committed like `results/` and `figures/`. |

**It is written as a paper, not as a description of this repository.** Implementation detail —
pixi, Snakemake, tiers, the generated-numbers scheme, `validation/` — belongs in Appendix B,
"Code and reproducibility". Derivations go in `proposition`/`proof`/`remark` environments, with
the equivalence of the two onset-anchored forms in Appendix A. If a methods paragraph explains
how the code is arranged, it is in the wrong file.

**No number is typed into the prose.** `report.tex` writes `\resultnum{<key>}`, and
`results/report_numbers.tex` — written by the tier-2 `report_numbers` rule from
`scripts/run_report_numbers.py` — supplies what each key expands to, from what the earlier tiers
wrote to `results/`. An unknown key is a *compile error*, not a blank, so a renamed model or a
deleted analysis stops the build instead of leaving a stale figure on the page. Anything that
looks like an input rather than a result — the serial interval, the priors, the sampler settings —
is a macro too, so editing `config/config.yaml` edits the methods section.

Four rules go with that scheme:

- **Keys carry no underscores**, because they are expanded through `\csname`: an analysis or
  model name is slugged to letters and digits only (`onset_models_estimated_k` →
  `onsetmodelsestimatedk`, `sse_so` → `sseso`). The mapping is mechanical, not a table.
- **Rounding lives in the script, not in the prose**, so the same quantity cannot appear to two
  decimal places in one section and three in another. Anything that can be negative goes through
  `math_mode`, so TeX sets a minus rather than a hyphen.
- **`run_report_numbers.py` computes nothing new.** Posterior summaries come from
  `posterior_comparison.summarise_posterior` (the same call `dispersion_posteriors.json` makes);
  crossings from `RiskCurve.first_day_below` (the same call the figure markers make); convergence
  from the per-day diagnostics tables the `rac` step wrote, so the numbers quoted are the ones the
  pipeline's own acceptance gate applied. The cross-analysis quantities it forms are subtractions
  of numbers already in `results/`, done there so no sentence has to do arithmetic.
- **It takes `IMPLEMENTED_ANALYSES` on the command line.** The `Snakefile` owns that list; the
  script must not grow a second copy.

**`latexmk`/`pdflatex` are the pipeline's only external toolchain dependency** — pixi does not
provide TeX, and the `report` rule says so rather than failing with "command not found".

`tests/test_report_numbers.py` checks the two committed files against each other: every key the
report uses is defined, every figure it includes exists, and every file the macros were read from
is still there.

The exception is the handful of measurements quoted from `validation/results/`. Validation
outputs deliberately never feed `rule all` (see `AGENTS.md`), so the report cannot depend on them
the way it depends on `results/`; each is named in the text with the file it came from.

## Building

```sh
pixi run pipeline           # builds report/report.pdf along with everything else
latexmk -pdf -cd report/report.tex   # just the document, from an existing report_numbers.tex
```

The `report` rule is the one place in the pipeline with an **external toolchain dependency**:
`latexmk` and `pdflatex` come from a TeX distribution (MacTeX, TeX Live), which pixi does not
provide. The rule says so rather than failing with "command not found"; everything else in the
pipeline runs without it. Build artefacts (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.out`) land
here and are gitignored.
