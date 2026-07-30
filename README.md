# paper-saluran-jhr-se16
Code and data underlying estimates of voter turnout and support for the main coalitions (BN, PH, PN), by ethnicity and age, in Johor's 16th state election (2026). Estimated via ecological inference on open polling-stream (saluran) data.

## Layout

| path | what it holds |
|---|---|
| `data/` | saluran-level ballots and stats, one pair per election |
| `estimates/` | the estimation pipeline, and everything it writes to `estimates/output/` — see `estimates/README.md` |
| `dataviz.py` | every figure in the paper, in one script |
| `estimates.py` | orchestrator: one run reproduces the whole pipeline |
| `tex/` | manuscript source, plus `tex/dataviz/` for generated figures |

## Building

```bash
python3 estimates.py --all --figures    # the whole pipeline, then every figure
cd tex && latexmk -pdf -outdir=out manuscript-r0.tex
```

`python3 estimates.py --list` shows the stages; `--all` includes the two slow ones (rebuilding
the panel from the voter roll, and the eiPack benchmark), which are skipped by default. See
`estimates/README.md` for what each stage does.

LaTeX builds go to `tex/out/`, which is gitignored. The tracked PDF at `tex/manuscript-r0.pdf`
is deliberately moved there by hand, whenever a version is worth committing.
