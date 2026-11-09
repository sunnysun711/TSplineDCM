# paper.py — one command per dataset: tune -> run -> analyze -> extract -> figures -> MANIFEST.md.
"""Runs the paper protocol for one dataset from a manifest (configs/paper/<dataset>.yaml) and writes down where every
number and figure came from. Resumable: tune.py resumes its optuna study, run.py skips finished splits.

    python paper.py configs/paper/SM.yaml                     # everything
    python paper.py configs/paper/SM.yaml --stage tune        # tune | run | analyze | figs | manifest (or several)
    python paper.py configs/paper/SM.yaml --streams 2         # cap on parallel streams (default: as in the manifest)

Manifest:
    dataset: SM                      # results / figures_data / figures name (analyze --dataset, make_all --dataset)
    main: tspline                    # model_name of the paper's TSplineDCM row (analyze --main-name)
    tune: {trials: 30, splits: "0-2"}          # same budget + same tuning splits for every model
    splits: "0-99"                   # for the curve extracts (each run uses its own yaml's splits)
    ind_splits: "0"                  # for the per-individual extracts
    streams:                         # configs/exp stems; a stream runs in order, streams run in parallel
      - [SM_tspline, SM_tspline_level, SM_tspline_homo]
      - [SM_rumboost, SM_rumboost_pcuf, ...]
    no_tune: [SM_mxl_biogeme]        # frozen (MLE, no search space): run <stem>.yaml as-is
    inherit: {SM_rumboost_pcuf: SM_rumboost}   # ablations take another stem's tuned winner (tools/apply_winner.py);
                                               # the source must come earlier in the same stream

Logs: $TSDCM_OUT/logs/paper_<dataset>_<YYMMDD_HHMMSS>.log (+ _stream<i>.log); watch.log at the workspace root gets one
line at start and end when it exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
PY = sys.executable
EXP = ROOT / "configs" / "exp"

sys.path.insert(0, str(ROOT))
from trainer_base import DEFAULT_OUT_ROOT  # noqa: E402

OUT_ROOT = Path(DEFAULT_OUT_ROOT)
LOG_DIR = OUT_ROOT / "logs"
WATCH = ROOT.parent / "watch.log"


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def watch(line: str) -> None:
    """One short line in the workspace watch.log (append-only, <= 200 chars), if that file exists."""
    if WATCH.exists():
        with open(WATCH, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{now()} | {line}"[:200] + "\n")


def _tuned(stem: str) -> str:
    return f"{stem}_tuned"


def _run(cmd: list[str], log, tag: str) -> int:
    log.write(f"=== {tag} start {now()}\n"); log.flush()
    rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"=== {tag} end {now()} (rc {rc})\n"); log.flush()
    return rc


def stream_worker(i: int, stems: list[str], m: dict[str, Any], stages: set[str], stamp: str) -> list[tuple[str, int]]:
    log_path = LOG_DIR / f"paper_{m['dataset']}_{stamp}_stream{i}.log"
    no_tune = set(m.get("no_tune") or [])
    inherit = dict(m.get("inherit") or {})
    trials, tune_splits = int(m["tune"]["trials"]), str(m["tune"]["splits"])
    results = []
    with open(log_path, "a", encoding="utf-8") as log:
        for stem in stems:
            run_stem = stem if stem in no_tune else _tuned(stem)
            if "tune" in stages and stem not in no_tune:
                if stem in inherit:
                    src = _tuned(inherit[stem])
                    if not (EXP / f"{src}.yaml").exists():
                        log.write(f"!!! {stem}: inherit source {src}.yaml missing (must be tuned earlier in this stream)\n")
                        results.append((stem, 99)); continue
                    rc = _run([PY, "tools/apply_winner.py", f"configs/exp/{src}.yaml", f"configs/exp/{stem}.yaml"], log, f"inherit {stem} <- {src}")
                else:
                    rc = _run([PY, "tune.py", f"configs/exp/{stem}.yaml", "--trials", str(trials), "--tune-split", tune_splits], log, f"tune {stem}")
                if rc:
                    results.append((stem, rc)); continue
            if "run" in stages:
                rc = _run([PY, "run.py", f"configs/exp/{run_stem}.yaml"], log, f"run {run_stem}")
                results.append((stem, rc))
    return results


def manifest_md(m: dict[str, Any], stamp: str) -> Path:
    """figures/<dataset>/MANIFEST.md: every model row -> yaml, tuned yaml, HPO log, results dir, splits; figure files -> script."""
    import json

    ds = m["dataset"]
    no_tune = set(m.get("no_tune") or [])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "?"
    rows = []
    for stems in m["streams"]:
        for stem in stems:
            base = yaml.safe_load((EXP / f"{stem}.yaml").read_text(encoding="utf-8"))
            run_stem = stem if stem in no_tune else _tuned(stem)
            model_name = base.get("model_name") or base["model"]
            res = OUT_ROOT / ds / model_name
            n_done = len(list(res.glob("split*/metrics.json"))) if res.exists() else 0
            hpo = OUT_ROOT / "hpo" / base["dataset"] / model_name / "trials.json"
            hpo_note = "-"
            if hpo.exists():
                t = json.loads(hpo.read_text(encoding="utf-8"))
                b = t.get("best") or {}
                hpo_note = f"{t.get('n_trials')} trials, splits {t.get('tune_splits', [t.get('tune_split')])}, best {b.get('val_nll', float('nan')):.4f}"
            splits_file = f"splits/{base.get('splits_name') or base['dataset']}.npz"
            rows.append(f"| {model_name} | {base['model']} | configs/exp/{stem}.yaml | configs/exp/{run_stem}.yaml | {hpo_note} | {splits_file} | {res} | {n_done} |")
    figs_dir = ROOT / "figures" / ds
    kinds = {}
    for p in sorted(figs_dir.glob("*")) if figs_dir.exists() else []:
        if p.name == "MANIFEST.md":
            continue
        kind = p.name[len(ds) + 1:].split("_")[0]
        kinds[kind] = kinds.get(kind, 0) + 1
    script = {"asucf": "plots/fig_asucf.py", "mupanel": "plots/fig_mu_panel.py", "vot": "plots/fig_vot_hist.py / fig_vot_heatmap.py",
              "tau": "plots/fig_tau.py", "z": "plots/fig_z_offset.py", "zsu": "plots/fig_zsu.py", "feature": "plots/fig_feature_table.py",
              "folds": "synthetic/fig_syn_folds.py", "grp": "synthetic/fig_syn_groups.py", "individual": "synthetic/fig_syn_individual.py",
              "asc": "synthetic/asc_table.py"}
    lines = [f"# {ds}: where every number and figure comes from", "",
             f"Written by `paper.py` at {now()}, code commit `{commit}`, manifest `configs/paper/{ds}.yaml`.",
             f"Protocol: {m['tune']['trials']} TPE trials per model, selection = mean validation NLL over splits {m['tune']['splits']}; "
             f"runs over each yaml's own splits; tables in `{OUT_ROOT / ds / '_summary'}`; curves in `figures_data/{ds}/`.", "",
             "| model_name | code | experiment yaml | yaml that ran | HPO (hpo/<dataset>/<model_name>/trials.json) | splits file | results dir | splits done |",
             "|---|---|---|---|---|---|---|---|", *rows, "",
             "| figure file prefix | count | script (reads figures_data csv only) |", "|---|---|---|",
             *[f"| {ds}_{k}_* | {n} | {script.get(k, '?')} |" for k, n in sorted(kinds.items())], "",
             "Every figure script has a parameter block at the top and reads only `figures_data/<dataset>/*.csv`; the csvs were "
             "written by `plots/extract/*.py` (and `synthetic/extract_syn.py`) from the checkpoints in the results dirs above."]
    figs_dir.mkdir(parents=True, exist_ok=True)
    p = figs_dir / "MANIFEST.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="paper protocol for one dataset (tune -> run -> analyze -> figs -> manifest)")
    ap.add_argument("manifest", help="configs/paper/<dataset>.yaml")
    ap.add_argument("--stage", nargs="*", default=None, help="subset of: tune run analyze figs manifest (default all)")
    ap.add_argument("--streams", type=int, default=None, help="max parallel streams (default: number in the manifest)")
    a = ap.parse_args()
    m = yaml.safe_load(Path(a.manifest).read_text(encoding="utf-8"))
    stages = set(a.stage or ["tune", "run", "analyze", "figs", "manifest"])
    ds = m["dataset"]
    stamp = dt.datetime.now().strftime("%y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    main_log = LOG_DIR / f"paper_{ds}_{stamp}.log"
    print(f"[paper] {ds}: stages {sorted(stages)}; log {main_log}")
    with open(main_log, "a", encoding="utf-8") as log:
        log.write(f"=== paper {ds} start {now()} stages={sorted(stages)} manifest={a.manifest}\n"); log.flush()
        if stages & {"tune", "run"}:
            streams = m["streams"]
            watch(f"background job started | paper.py {ds} | {len(streams)} parallel streams: tune + run ({sum(len(s) for s in streams)} experiments) | log logs/{main_log.name}, ends with PAPER_DONE")
            with ThreadPoolExecutor(max_workers=a.streams or len(streams)) as ex:
                futs = [ex.submit(stream_worker, i, s, m, stages, stamp) for i, s in enumerate(streams)]
                for f in futs:
                    for stem, rc in f.result():
                        log.write(f"{'ok ' if rc == 0 else 'FAIL'} {stem} (rc {rc})\n")
            log.flush()
        if "analyze" in stages:
            _run([PY, "analyze.py", "--dataset", ds, "--main-name", m.get("main", "tspline"), "--export"], log, "analyze")
        if "figs" in stages:
            no_tune = set(m.get("no_tune") or [])
            exps = [s if s in no_tune else _tuned(s) for stems in m["streams"] for s in stems]
            _run([PY, "plots/make_all.py", "--dataset", ds, "--exps", *exps, "--splits", str(m.get("splits", "0-99")),
                  "--ind-splits", str(m.get("ind_splits", "0"))], log, "make_all")
        if "manifest" in stages:
            p = manifest_md(m, stamp)
            log.write(f"=== manifest {p}\n")
        log.write(f"=== paper {ds} end {now()}\nPAPER_DONE\n")
    watch(f"background job finished | paper.py {ds} | stages {sorted(stages)} done (PAPER_DONE) | log logs/{main_log.name}; manifest figures/{ds}/MANIFEST.md")
    print(f"[paper] {ds} done; see {main_log}")


if __name__ == "__main__":
    main()
