# plots/fig_feature_table.py — LaTeX table: which variables enter which part of each model.
"""Input: figures_data/<DATASET>/feature_table.csv (from plots/extract/feature_table.py, configs only).
Output: figures/<DATASET>/<DATASET>_feature_table.tex, a booktabs tabular to \\input in the paper.

Edit the parameter block, then:   python plots/fig_feature_table.py
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
MODELS = None                       # None = all rows in csv; or ["tspline", "mnl", "fdnn"]
SHOW_MONOTONE = True                # append the monotone directions column (TSplineDCM rows)
OUT_DIR = None
# ============================================================================

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import display  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("->", r"$\rightarrow$")


def main() -> None:
    df = pd.read_csv(DATA / "feature_table.csv").fillna("")
    if MODELS:
        df = df[df.model.isin(MODELS)]
    cols = ["Model", "Alternative", "Attributes $x$ (utility)", "Characteristics $z$: role"]
    if SHOW_MONOTONE:
        cols.append("Monotone dirs.")
    lines = [r"\begin{tabular}{l l l p{6cm}" + (" l" if SHOW_MONOTONE else "") + "}", r"\toprule",
             " & ".join(cols) + r" \\", r"\midrule"]
    for m, g in df.groupby("model", sort=False):
        first = True
        for _, r in g.iterrows():
            row = [_esc(display(m)) if first else "", _esc(r["alt"]), _esc(r["x_attrs"]),
                   _esc(r["z_role"]) if first else r"\textquotedbl"]
            if SHOW_MONOTONE:
                row.append(_esc(r["notes"].replace("monotone: ", "")))
            lines.append(" & ".join(row) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{DATASET}_feature_table.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fig_feature_table] {p}")


if __name__ == "__main__":
    main()
