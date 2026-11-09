# plots/fig_learning_curve.py — test NLL against the size of the LPMC training pool, and what the heterogeneity term is worth.
"""Paper figure LPMC_learning_curve.pdf. Input: figures_data/LPMC_curve/learning_curve.csv from
plots/extract/learning_curve.py. Left: test NLL (mean over resamples, bars = standard error) of each model against
the train+val rows, log axis. Right: the value of the heterogeneity term, NLL(no heter.) - NLL(TSplineDCM) per
resample (paired), mean with standard error, and the share of resamples in which the term helps. No model code.

Edit the parameter block, then:   python plots/fig_learning_curve.py
"""
from __future__ import annotations

# ============================== parameters ==================================
MODELS = ["tspline", "tspline_homo", "rumboost", "gbdt", "fdnn"]     # drawing order of the left panel
MAIN, HOMO = "tspline", "tspline_homo"                                # the pair of the right panel
FIG_SIZE = (6.6, 2.6)               # inches, two panels
Y_LIM_NLL = None                    # e.g. (0.65, 0.78); None = auto
OUT_DIR = None                      # None -> figures/LPMC_curve/
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import display, latexify, line_style, model_color  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / "LPMC_curve" / "learning_curve.csv"
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / "LPMC_curve"


def main() -> None:
    latexify(FONT_SIZE)
    import matplotlib.pyplot as plt

    d = pd.read_csv(DATA)
    sizes = sorted(d.pool_rows_nominal.unique())
    x_of = d.groupby("pool_rows_nominal").pool_rows.mean()              # actual mean train+val rows per size
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE, sharex=True)     # the same x range in both panels

    for m in MODELS:
        g = d[d.model == m].groupby("pool_rows_nominal").test_nll.agg(["mean", "sem", "count"])
        if g.empty:
            continue
        st = line_style(m)
        color = st.pop("color", None) or model_color(m)
        ax.errorbar(x_of.loc[g.index], g["mean"], yerr=g["sem"], color=color, marker="o", markersize=3, linewidth=1.2,
                    capsize=2, label=display(m), **st)
    for a in (ax, ax2):                      # log axis with one tick per size (the default minor labels overlap)
        a.set_xscale("log")
        a.set_xticks(x_of.loc[sizes].to_list())
        a.set_xticklabels([f"{v / 1000:.0f}k" if v >= 10000 else f"{v / 1000:.1f}k".replace(".0k", "k") for v in x_of.loc[sizes]],
                          fontsize=FONT_SIZE - 2, rotation=30, ha="right", rotation_mode="anchor")
        a.minorticks_off()
    ax.set_xlim(x_of.min() / 1.3, x_of.max() * 1.3)   # shared: the same explicit range in both panels
    ax.set_xlabel("Training pool")
    ax.set_ylabel("Test NLL")
    if Y_LIM_NLL:
        ax.set_ylim(*Y_LIM_NLL)
    ax.legend(fontsize=FONT_SIZE - 2, frameon=False)
    ax.set_title("(a) Test NLL against the training size", fontsize=FONT_SIZE)

    # paired gain of the heterogeneity term, per resample
    a = d[d.model == MAIN].set_index(["pool_rows_nominal", "split_id"]).test_nll
    b = d[d.model == HOMO].set_index(["pool_rows_nominal", "split_id"]).test_nll
    gain = (b - a).dropna().rename("gain").reset_index()
    g = gain.groupby("pool_rows_nominal").gain.agg(["mean", "sem", "count", lambda v: float((v > 0).mean())])
    g.columns = ["mean", "sem", "n", "share"]
    ax2.axhline(0, color="0.5", linewidth=0.6, linestyle="--")
    ax2.errorbar(x_of.loc[g.index], g["mean"], yerr=g["sem"], color=model_color(MAIN), marker="o", markersize=3,
                 linewidth=1.2, capsize=2)
    for s, r in g.iterrows():
        ax2.annotate(f"{r['share'] * 100:.0f}\\%" if plt.rcParams.get("text.usetex") else f"{r['share'] * 100:.0f}%",
                     (x_of.loc[s], r["mean"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=FONT_SIZE - 2)
    ax2.set_xlabel("Training pool")
    ax2.set_ylabel("NLL gain")                 # NLL(no heter.) - NLL(TSplineDCM), said in the caption
    ax2.set_title("(b) Value of the heterogeneity term", fontsize=FONT_SIZE)
    fig.tight_layout(w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"LPMC_learning_curve.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_learning_curve] {p}")
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
