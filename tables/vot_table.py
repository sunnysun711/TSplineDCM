# tables/vot_table.py — value-of-time summary table: median / mean VoT per alternative and the share of individuals kept.
"""Paper tables SM_vot.tex / LPMC_vot.tex. Same input and the SAME filter as plots/fig_vot_hist.py
(figures_data/<DATASET>/vot_individual.csv.gz: one row per individual x split x alternative, raw model units):
an individual is kept when the cost slope is clearly negative (mu_cost < MU_COST_MAX) and the VoT lies in VOT_RANGE.
Writes figures/<DATASET>/<DATASET>_vot_table.tex (and .csv). No torch, no model code.
Edit the parameter block, then:   python tables/vot_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== parameters ==================================
DATASETS = ["SM", "LPMC_officialtest"]
MODELS = ["tspline", "tspline_homo", "rumboost_pcuf", "fdnn", "mnl"]     # rows, in this order (missing ones skipped)
MU_COST_MAX = -1e-6                 # keep individuals whose cost slope is clearly negative   (= fig_vot_hist.py)
VOT_RANGE = (0.1, 100.0)            # keep VoT within this range (display units)             (= fig_vot_hist.py)
OUT_DIR = None                      # None -> figures/<DATASET>/
# ============================================================================
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "plots"))               # style.py (display names, VoT units) lives with the figure scripts
from style import display, vot_unit  # noqa: E402


def one_dataset(dataset: str) -> None:
    data = ROOT / "figures_data" / dataset
    out = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / dataset
    out.mkdir(parents=True, exist_ok=True)
    factor, unit = vot_unit(dataset)
    d = pd.read_csv(data / "vot_individual.csv.gz")
    alts = list(dict.fromkeys(d["alt"]))
    recs = []
    for m in MODELS:
        dm = d[d.model_name == m]
        if dm.empty:
            continue
        for alt in alts:
            g = dm[dm.alt == alt]
            if g.empty:
                continue
            v = g["vot"] * factor
            keep = (g["mu_cost"] < MU_COST_MAX) & (v >= VOT_RANGE[0]) & (v <= VOT_RANGE[1])
            vk = v[keep]
            recs.append({"model": m, "alt": alt, "n": int(len(g)), "kept": float(keep.mean()),
                         "median": float(vk.median()) if len(vk) else np.nan,
                         "mean": float(vk.mean()) if len(vk) else np.nan})
    df = pd.DataFrame(recs)
    df.to_csv(out / f"{dataset}_vot_table.csv", index=False)
    # LaTeX: one row per model, per alternative "median (mean) [kept %]"
    cols = " & ".join(f"\\multicolumn{{2}}{{c}}{{{a}}}" for a in alts)
    sub = " & ".join(["median (mean)", "kept"] * len(alts))
    lines = [r"\begin{tabular}{l" + "rr" * len(alts) + "}", r"\toprule",
             f"Model & {cols} \\\\", f" & {sub} \\\\", r"\midrule"]
    for m in MODELS:
        r = df[df.model == m]
        if r.empty:
            continue
        cells = []
        for a in alts:
            x = r[r.alt == a]
            if x.empty or not np.isfinite(x["median"].iloc[0]):
                cells += ["--", f"{100 * x['kept'].iloc[0]:.0f}\\%" if not x.empty else "--"]
            else:
                cells += [f"{x['median'].iloc[0]:.1f} ({x['mean'].iloc[0]:.1f})", f"{100 * x['kept'].iloc[0]:.0f}\\%"]
        name = display(m) if m != "mnl" else "MNL (SGD)"
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / f"{dataset}_vot_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[vot_table] {dataset}: {len(df)} rows, unit {unit} -> {out / (dataset + '_vot_table.tex')}")


def main() -> None:
    for ds in DATASETS:
        one_dataset(ds)


if __name__ == "__main__":
    main()
