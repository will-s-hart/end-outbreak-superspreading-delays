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
`scripts/run_report_numbers.py` — supplies what each key expands to, from the files the other
tier-2 rules produced. An unknown key is a *compile error*, not a blank, so a renamed model or a
deleted analysis stops the build instead of leaving a stale figure on the page. Anything that
looks like an input rather than a result — the serial interval, the priors, the sampler settings —
is a macro too, so editing `config/config.yaml` edits the methods section.

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
