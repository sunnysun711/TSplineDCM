# plots/extract/timing_paper.py — training seconds per split recorded in the paper sets -> csv (for fig_timing).
"""The paper sets' metrics.json carry fit_seconds measured while 3-6 jobs shared the machine. This copies the
per-model mean / SEM / split count (and n_params, epochs) into one csv so the timing figure can show them
next to the idle-machine re-timing from tools/time_rows.py without reading the results root itself.

    python plots/extract/timing_paper.py            # all three paper sets

Writes figures_data/timing/timing_paper.csv
    columns: dataset, results_name, model, n_splits, fit_s_mean, fit_s_sem, n_params, epochs_mean, test_nll_mean
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path(os.environ.get("TSDCM_OUT", "results"))
SETS = {"SM": "SM", "LPMC": "LPMC_officialtest", "SYN": "SYN"}


def main() -> None:
    rows = []
    for ds, results_name in SETS.items():
        base = OUT_ROOT / results_name
        for mdir in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
            fit, ep, nll, npar = [], [], [], []
            for f in sorted(mdir.glob("split*_seed42/metrics.json")):
                m = json.load(open(f, encoding="utf-8"))
                fi = m.get("fit_info", {}) or {}
                fit.append(float(m["fit_seconds"])); nll.append(float(m["test"]["nll"]))
                ep.append(fi.get("epochs_run", fi.get("best_iteration", np.nan)))
                npar.append(fi.get("n_params", np.nan))
            if not fit:
                continue
            rows.append({"dataset": ds, "results_name": results_name, "model": mdir.name, "n_splits": len(fit),
                         "fit_s_mean": float(np.mean(fit)), "fit_s_sem": float(np.std(fit, ddof=1) / np.sqrt(len(fit))) if len(fit) > 1 else 0.0,
                         "n_params": float(np.nanmean([x for x in npar if x is not None])) if any(x is not None for x in npar) else np.nan,
                         "epochs_mean": float(np.nanmean([x for x in ep if x is not None])) if any(x is not None for x in ep) else np.nan,
                         "test_nll_mean": float(np.mean(nll))})
    out = ROOT / "figures_data" / "timing" / "timing_paper.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
