# plots/extract/learning_curve.py — gather the per-split test metrics of the LPMC learning curve into one csv.
"""Learning curve: the same models on six sizes of the LPMC training pool, official test unchanged.
Reads $TSDCM_OUT/<results name>/_summary/per_split.csv (analyze.py --export) of every size and the split files
(for the actual train+val rows of each split); touches no checkpoint and no model code.

    python plots/extract/learning_curve.py        -> figures_data/LPMC_curve/learning_curve.csv
        columns: results_name, pool_rows_nominal, pool_rows (train+val of that split), model, split_id,
                 test_nll, test_acc, val_nll, fit_seconds, n_params
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== parameters ==================================
SIZES = {                            # results name -> nominal train+val rows (splits.py SUBSAMPLE_POOL_ROWS; full pool last)
    "LPMC_n2000_officialtest": 2000, "LPMC_n4000_officialtest": 4000, "LPMC_sub_officialtest": 7677,
    "LPMC_n15000_officialtest": 15000, "LPMC_n30000_officialtest": 30000, "LPMC_officialtest": 54766,
}
MODELS = ["tspline", "tspline_homo", "rumboost", "gbdt", "fdnn"]
OUT_NAME = "LPMC_curve"
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("TSDCM_OUT", ROOT / "results"))
KEEP = ["model", "split_id", "test_nll", "test_acc", "val_nll", "fit_seconds", "n_params"]


def _pool_rows(results_name: str) -> dict[int, int]:
    """split_id -> train + val rows, from splits/<results name>.npz."""
    p = ROOT / "splits" / f"{results_name}.npz"
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith("train"):
            sid = int("".join(ch for ch in k if ch.isdigit()) or 0)
            out[sid] = int(len(z[k]) + len(z[k.replace("train", "val")]))
    return out


def main() -> Path:
    parts = []
    for name, nominal in SIZES.items():
        p = RESULTS / name / "_summary" / "per_split.csv"
        if not p.exists():
            print(f"[learning_curve] {name}: no _summary/per_split.csv yet (run paper.py --stage analyze); skipped")
            continue
        d = pd.read_csv(p)
        d = d[d.model.isin(MODELS)][KEEP].copy()
        rows = _pool_rows(name)
        d.insert(0, "pool_rows", d.split_id.map(rows).fillna(nominal).astype(int))
        d.insert(0, "pool_rows_nominal", nominal)
        d.insert(0, "results_name", name)
        parts.append(d)
        print(f"[learning_curve] {name}: {d.split_id.nunique()} splits x {d.model.nunique()} models")
    out = ROOT / "figures_data" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(parts, ignore_index=True).sort_values(["pool_rows_nominal", "model", "split_id"])
    df.to_csv(out / "learning_curve.csv", index=False)
    print(f"[learning_curve] -> {out / 'learning_curve.csv'} ({len(df)} rows)")
    return out / "learning_curve.csv"


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
