# synthetic/asc_table.py — ASC recovery table of the synthetic experiment (paper Table 2, plus TasteNet / FDNN rows).
"""Reads figures_data/<DATASET>/z_offset.csv (the ASC rows every model's z_offset extract wrote, one per split) and
syn_truth_asc.csv, and reports every alternative's ASC RELATIVE TO BASE_ALT (the only identified quantity; the DGP
pins ASC_CAR = 0) as mean +/- SEM over splits, with the truth row on top. FDNN has no ASC: its row is the
pre-softmax output at an all-zero input (models/fdnn.py z_offset_params).
Output: figures/<DATASET>/<DATASET>_asc_table.csv and .tex (booktabs). No torch, no model code.

Edit the parameter block, then:   python synthetic/asc_table.py
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SYN"
MODELS = None                       # None = every model in z_offset.csv; or ["tspline", "mnl", "tastenet", "fdnn"]
BASE_ALT = "CAR"
DIGITS = 3
OUT_DIR = None
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "plots"))
from style import display, tex  # noqa: E402

DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def relative_asc(z: pd.DataFrame) -> pd.DataFrame:
    """One row per (model_name, split, alt != BASE_ALT): value minus the base's value (0 when the base is pinned)."""
    rows = []
    for (m, s), g in z[z.param == "ASC"].groupby(["model_name", "split"], sort=False):
        vals = dict(zip(g.alt, g.value))
        masked = dict(zip(g.alt, g.masked))
        base = 0.0 if masked.get(BASE_ALT, False) else float(vals.get(BASE_ALT, 0.0))
        for alt, v in vals.items():
            if alt != BASE_ALT:
                rows.append({"model_name": m, "split": s, "alt": alt, "value": float(v) - base})
    return pd.DataFrame(rows)


def main() -> None:
    z = pd.read_csv(DATA / "z_offset.csv")
    truth = pd.read_csv(DATA / "syn_truth_asc.csv")
    rel = relative_asc(z)
    models = list(MODELS) if MODELS else list(dict.fromkeys(rel.model_name))
    alts = [a for a in truth.alt if a != BASE_ALT]
    t_base = float(truth.loc[truth.alt == BASE_ALT, "asc"].iloc[0]) if (truth.alt == BASE_ALT).any() else 0.0
    out_rows = [{"source": "truth", "alt": a, "mean": float(truth.loc[truth.alt == a, "asc"].iloc[0]) - t_base, "sem": 0.0, "n": 0}
                for a in alts]
    for m in models:
        for a in alts:
            v = rel[(rel.model_name == m) & (rel.alt == a)]["value"].to_numpy()
            if len(v) == 0:
                continue
            sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            out_rows.append({"source": m, "alt": a, "mean": float(v.mean()), "sem": sem, "n": int(len(v))})
    table = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"{DATASET}_asc_table.csv", index=False, lineterminator="\n", float_format="%.6g")

    fmt = f"{{:.{DIGITS}f}}"
    lines = [f"% ASC relative to {BASE_ALT} (pinned to 0 in the DGP and in the models that fix it); mean +/- SEM over splits.",
             "% FDNN: pre-softmax output at an all-zero input (no structural ASC).",
             "\\begin{tabular}{l" + "c" * len(alts) + "}", "\\toprule",
             " & " + " & ".join(alts) + " \\\\", "\\midrule"]
    for src in ["truth"] + models:
        cells = []
        for a in alts:
            r = table[(table.source == src) & (table.alt == a)]
            if r.empty:
                cells.append("--")
            elif src == "truth":
                cells.append(fmt.format(r["mean"].iloc[0]))
            else:
                cells.append(fmt.format(r["mean"].iloc[0]) + " $\\pm$ " + fmt.format(r["sem"].iloc[0]))
        name = "Truth" if src == "truth" else tex(display(src))
        n = int(table[table.source == src]["n"].max()) if src != "truth" else 0
        lines.append(f"{name}" + (f" ({n} splits)" if n else "") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / f"{DATASET}_asc_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[asc_table] {OUT / f'{DATASET}_asc_table.csv'} and .tex")


if __name__ == "__main__":
    main()
