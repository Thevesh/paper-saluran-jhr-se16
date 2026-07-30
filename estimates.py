"""Run the whole estimation pipeline, start to finish.
Usage:
    python3 estimates.py                 everything except the two slow stages
    python3 estimates.py --all           everything, including them
    python3 estimates.py --from joint    resume from a named stage onwards
    python3 estimates.py --list          show the stages and stop

One run reproduces every number, table and figure in the paper from the raw voter roll and
ballot files. Stages run in order; each reads what the ones before it wrote to estimates/output/.

Two stages are skipped by default because they are slow and their outputs are already in the
repo. `data` rebuilds the saluran panel from the external voter roll (~10 min, and needs
ROLL_DIR in estimates/common.py to point at a local copy). `benchmark` fits the eiPack
hierarchical Bayesian model for the comparison in Section 4.4 (~1.5 h, and needs R with eiPack
installed). Everything else runs in a few minutes.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
EST = os.path.join(ROOT, "estimates")

# (stage, what it produces, slow?) in dependency order.
STAGES = [
    ("data",            "saluran panel from the roll and ballots", True),
    ("bounds",          "deterministic Duncan--Davis bounds", False),
    ("models",          "the candidate ecological models", False),
    ("reconcile",       "raking to the exact statewide margins", False),
    ("validate",        "the simulation harness the models are checked against", False),
    ("bivariate",       "the two-group scatter estimate, for comparison", False),
    ("series",          "estimates aligned across the three elections", False),
    ("scaled",          "support as a share of voters rather than registrants", False),
    ("groundtruth",     "the near-pure saluran holdout", False),
    ("joint",           "the joint ethnicity-by-age model", False),
    ("joint_estimates", "the joint model's crosstab, marginals and scaled panel", False),
    ("standdown",       "the PN stand-down checks of Section 5.2", False),
    ("holdout",         "the unrestricted vs locality-fixed holdout designs", False),
    ("kappa",           "the bounded-heterogeneity frontier of Section 4.5", False),
    ("errors",          "the reported error bars, and the purity-threshold check", False),
    ("benchmark",       "the eiPack hierarchical Bayesian comparison", True),
    ("tables",          "tex/tab_results, tab_crosstab, tab_errors", False),
    ("workbook",        "the estimate workbooks in estimates/output/", False),
    ("report",          "the HTML report in estimates/output/", False),
]


def run(stage):
    t = time.time()
    print(f"\n\033[1m>> {stage}\033[0m", flush=True)
    r = subprocess.run([sys.executable, f"{stage}.py"], cwd=EST)
    if r.returncode:
        raise SystemExit(f"\n{stage} failed (exit {r.returncode}); pipeline stopped")
    print(f"   done in {time.time() - t:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="include the two slow stages")
    p.add_argument("--from", dest="start", metavar="STAGE", help="resume from this stage")
    p.add_argument("--list", action="store_true", help="list the stages and exit")
    p.add_argument("--figures", action="store_true", help="also regenerate the figures")
    a = p.parse_args()

    names = [s for s, _, _ in STAGES]
    if a.list:
        for s, what, slow in STAGES:
            print(f"  {s:16} {what}{'   [slow, skipped by default]' if slow else ''}")
        return

    todo = STAGES
    if a.start:
        if a.start not in names:
            raise SystemExit(f"unknown stage {a.start!r}; try --list")
        todo = STAGES[names.index(a.start):]
    if not a.all:
        todo = [s for s in todo if not s[2]]

    t = time.time()
    for stage, _, _ in todo:
        run(stage)
    if a.figures:
        run_figs = subprocess.run([sys.executable, "dataviz.py"], cwd=ROOT)
        if run_figs.returncode:
            raise SystemExit("dataviz failed")
    print(f"\n\033[1mpipeline complete in {time.time() - t:.0f}s\033[0m")
    if not a.all:
        print("(skipped the slow stages: data, benchmark -- rerun with --all to include them)")


if __name__ == "__main__":
    main()
