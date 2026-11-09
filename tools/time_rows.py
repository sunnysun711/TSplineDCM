# -*- coding: utf-8 -*-
"""Re-time model rows on an idle machine, one run at a time. Timing only: results go to a scratch out-root,
nothing is analysed or plotted; the per-split training seconds (and, as a free regression check, the test NLL
against the paper set's metrics.json) are merged into figures_data/timing/timing.csv and written to the log.

    python tools/time_rows.py                                  # the default plan (below)
    python tools/time_rows.py --rows SM:tspline LPMC:rumboost  # a subset
    python tools/time_rows.py --splits 0-2                     # fewer splits

Which yaml: the one the paper set actually ran. configs/paper/<dataset>.yaml lists the models that were not tuned
(`no_tune`); those ran configs/exp/<ds>_<model>.yaml, every other model ran <ds>_<model>_tuned.yaml. Before any
training the chosen yaml's `overrides` are compared, leaf by leaf, with the config stored in the paper set's
split000 metrics.json; a mismatch stops the script.

Regression check: a split counts as reproduced when its test NLL equals the paper set's, or differs by at most
NLL_TOL (prediction-stage rounding). Splits that differ by more are reported and NOT written
to timing.csv. The CSV is merged on (dataset, model, split): new rows replace rows with the same key, every
other row stays.

Why: the paper sets' fit_seconds were measured with 3-6 jobs sharing the GPU/CPU; this gives every model the
same idle-machine condition after the speed-up. Run NOTHING else while it runs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_ROOT = Path(os.environ.get("TSDCM_OUT", "results")) / "tmp" / "timing"
PAPER_ROOT = Path(os.environ.get("TSDCM_OUT", "results"))
LOG_DIR = PAPER_ROOT / "logs"
CSV = ROOT / "figures_data" / "timing" / "timing.csv"
WATCH = ROOT.parent / "watch.log"
NLL_TOL = 1e-8                      # |test NLL - paper test NLL| above this = regression failure, row rejected
                                    # (the 2026-10-07 LPMC tspline_homo / tspline_level reruns differ by 1.1e-9..4.5e-9: prediction-stage rounding)
KEY = ["dataset", "model", "split"]

# dataset tag -> (results dir of the paper set, default splits)
DATASETS = {"SM": ("SM", "0-9"), "LPMC": ("LPMC_officialtest", "0-9"), "SYN": ("SYN", "0-4")}
MODELS = ["tspline", "tspline_level", "tspline_homo", "rumboost", "gbdt", "fdnn"]


def split_ids(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def watch(line: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | {line}"
    assert len(line) <= 200, len(line)
    with open(WATCH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------------------------------------
# which yaml did the paper set run, and does it match what was stored on disk
# ----------------------------------------------------------------------------------------------------------
def select_yaml(ds: str, model: str) -> Path:
    """no_tune models in configs/paper/<ds>.yaml ran the plain yaml, everything else the _tuned one."""
    paper_cfg = _yaml.safe_load(open(ROOT / "configs" / "paper" / f"{ds}.yaml", encoding="utf-8"))
    no_tune = set(paper_cfg.get("no_tune") or [])
    plain, tuned = ROOT / "configs" / "exp" / f"{ds}_{model}.yaml", ROOT / "configs" / "exp" / f"{ds}_{model}_tuned.yaml"
    path = plain if f"{ds}_{model}" in no_tune else tuned
    if not path.exists():
        raise FileNotFoundError(f"{ds}:{model}: expected {path.name} (no_tune={f'{ds}_{model}' in no_tune}) but it does not exist")
    return path


def _leaves(d: dict, prefix: tuple = ()) -> dict[tuple, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_leaves(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def _get(d: Any, path: tuple) -> Any:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return "<missing>"
        d = d[p]
    return d


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


def check_config(ds: str, model: str, yaml_path: Path) -> list[str]:
    """Compare the yaml's overrides (and model / model_name) with the paper set's split000 metrics.json config.
    Returns the list of mismatches (empty = ok)."""
    results_name = DATASETS[ds][0]
    ref = PAPER_ROOT / results_name / model / "split000_seed42" / "metrics.json"
    if not ref.exists():
        return [f"paper-set reference missing: {ref}"]
    stored = json.load(open(ref, encoding="utf-8"))["config"]
    exp = _yaml.safe_load(open(yaml_path, encoding="utf-8"))
    bad = []
    for k in ("model", "model_name"):
        if exp.get(k) != stored.get(k):
            bad.append(f"{k}: yaml {exp.get(k)!r} vs stored {stored.get(k)!r}")
    for path, val in _leaves(exp.get("overrides") or {}).items():
        got = _get(stored, path)
        if got == "<missing>" and path[0] == "biogeme_config":      # the biogeme backend stores its config flat
            got = _get(stored, path[1:])
        if not _same(val, got):
            bad.append(f"{'.'.join(path)}: yaml {val!r} vs stored {got!r}")
    return bad


# ----------------------------------------------------------------------------------------------------------
def collect(results_name: str, model: str, ids: list[int], out_root: Path) -> list[dict]:
    rows = []
    for sid in ids:
        f = out_root / results_name / model / f"split{sid:03d}_seed42" / "metrics.json"
        if not f.exists():
            rows.append({"split": sid, "fit_seconds": np.nan}); continue
        m = json.load(open(f, encoding="utf-8"))
        fi = m.get("fit_info", {}) or {}
        ref = PAPER_ROOT / results_name / model / f"split{sid:03d}_seed42" / "metrics.json"
        ref_nll = json.load(open(ref, encoding="utf-8"))["test"]["nll"] if ref.exists() else np.nan
        rows.append({"split": sid, "fit_seconds": float(m["fit_seconds"]), "test_nll": float(m["test"]["nll"]),
                     "paper_test_nll": ref_nll, "epochs": fi.get("epochs_run", fi.get("best_iteration")),
                     "n_params": fi.get("n_params"), "backend": m.get("config", {}).get("backend")})
    return rows


def nll_status(r: dict) -> str:
    """'identical' | 'close' (within NLL_TOL) | 'FAIL' | 'noref' | 'missing'."""
    if not np.isfinite(r.get("fit_seconds", np.nan)):
        return "missing"
    ref = r.get("paper_test_nll", np.nan)
    if not np.isfinite(ref):
        return "noref"
    if r["test_nll"] == ref:
        return "identical"
    return "close" if abs(r["test_nll"] - ref) <= NLL_TOL else "FAIL"


def merge_csv(new: pd.DataFrame) -> pd.DataFrame:
    """Rows keyed (dataset, model, split): new replaces old, everything else is kept."""
    if CSV.exists():
        old = pd.read_csv(CSV)
        keys = set(map(tuple, new[KEY].itertuples(index=False)))
        old = old[[tuple(k) not in keys for k in old[KEY].itertuples(index=False)]]
        new = pd.concat([old, new], ignore_index=True)
    cols = [c for c in new.columns if c not in KEY]
    return new[KEY + cols].sort_values(KEY).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=None, help="DATASET:model entries; default = every dataset x every model")
    ap.add_argument("--splits", default=None, help="override the per-dataset default splits")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    rows = args.rows or [f"{d}:{m}" for d in DATASETS for m in MODELS]
    out_root = Path(args.out_root)
    stamp = dt.datetime.now().strftime("%y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"timing_{stamp}.log"

    def say(msg: str) -> None:
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    CSV.parent.mkdir(parents=True, exist_ok=True)
    say(f"=== timing {stamp}: {len(rows)} rows, out-root {out_root}, splits {args.splits or 'per-dataset default'}")

    # --- choose the yaml per row and check it against the paper set BEFORE any training
    plan: list[tuple[str, str, Path]] = []
    problems = []
    for entry in rows:
        ds, model = entry.split(":")
        yml = select_yaml(ds, model)
        bad = check_config(ds, model, yml)
        say(f"    {ds}:{model:16s} -> {yml.name:36s} config vs paper split000: {'ok' if not bad else 'MISMATCH'}")
        for b in bad:
            say(f"        {b}")
        problems += [f"{ds}:{model} {b}" for b in bad]
        plan.append((ds, model, yml))
    if problems:
        say(f"=== STOP: {len(problems)} config mismatch(es) between chosen yaml and the paper set; nothing was run")
        sys.exit(2)

    watch(f"background job started | tools/time_rows.py | timing {len(rows)} rows one by one (idle GPU, serial) | log logs/timing_{stamp}.log, ends with TIMING_DONE")
    records: list[dict] = []
    rejected: list[dict] = []
    for ds, model, yml in plan:
        results_name, default_splits = DATASETS[ds]
        ids = split_ids(args.splits or default_splits)
        say(f"--- {ds}:{model} splits {ids[0]}-{ids[-1]} ({yml.name}) start {dt.datetime.now():%H:%M:%S}")
        t0 = time.time()
        cmd = [PY, str(ROOT / "run.py"), str(yml), "--splits", f"{ids[0]}-{ids[-1]}", "--out-root", str(out_root), "--overwrite"]
        with open(log, "a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        got = collect(results_name, model, ids, out_root)
        for r in got:
            r.update({"dataset": ds, "results_name": results_name, "model": model, "wall_seconds_total": wall,
                      "nll_status": nll_status(r), "yaml": yml.name})
        fit = np.array([r["fit_seconds"] for r in got], dtype=float)
        n_id = sum(r["nll_status"] == "identical" for r in got)
        n_close = sum(r["nll_status"] == "close" for r in got)
        say(f"    rc {rc} | wall {wall:.0f} s | fit mean {np.nanmean(fit):.1f} s, median {np.nanmedian(fit):.1f}, "
            f"min {np.nanmin(fit):.1f}, max {np.nanmax(fit):.1f} | test NLL identical to paper set: {n_id}/{len(got)}, "
            f"within {NLL_TOL:g}: {n_id + n_close}/{len(got)}")
        for r in got:
            if r["nll_status"] in ("FAIL", "missing"):
                say(f"    REJECTED split {r['split']}: {r['nll_status']}"
                    + (f" test NLL {r['test_nll']!r} vs paper {r['paper_test_nll']!r} (diff {r['test_nll'] - r['paper_test_nll']:.3e})"
                       if r["nll_status"] == "FAIL" else ""))
                rejected.append(r)
            else:
                records.append(r)
        pd.DataFrame(records + rejected).to_csv(out_root / f"timing_partial_{stamp}.csv", index=False)   # survives a crash

    if not records:
        say("=== no accepted rows; timing.csv untouched")
        say("TIMING_DONE")
        return
    new = pd.DataFrame(records)
    merged = merge_csv(new)
    merged.to_csv(CSV, index=False)
    say("\n=== summary of this run (mean fit seconds per split, idle machine; accepted rows only)")
    piv = new.groupby(["dataset", "model"]).agg(n=("split", "size"), fit_s=("fit_seconds", "mean"),
                                                fit_sem=("fit_seconds", "sem"), epochs=("epochs", "mean"),
                                                identical=("nll_status", lambda s: int((s == "identical").sum())),
                                                within_tol=("nll_status", lambda s: int(s.isin(["identical", "close"]).sum())))
    say(piv.to_string(float_format=lambda x: f"{x:.1f}"))
    if rejected:
        say(f"rejected (not written): " + ", ".join(f"{r['dataset']}:{r['model']}:split{r['split']}" for r in rejected))
    say(f"csv: {CSV} ({len(merged)} rows total, {len(new)} written by this run)")
    say("TIMING_DONE")
    watch(f"background job finished | tools/time_rows.py | timing done (TIMING_DONE), csv figures_data/timing/timing.csv | log logs/timing_{stamp}.log")


if __name__ == "__main__":
    main()
