# plots/fig_timing.py — training seconds per split: idle-machine re-timing vs the paper sets' recorded times.
"""Input: figures_data/timing/timing.csv (tools/time_rows.py: one row per split, idle machine, speed-up code)
and figures_data/timing/timing_paper.csv (plots/extract/timing_paper.py: the paper sets' recorded mean,
measured while 3-6 jobs shared the machine).

One panel per dataset, horizontal bars on a log axis: filled bar = idle-machine mean (with the min-max range
of the splits), hollow bar behind it = paper-set recorded mean. Models not re-timed show only the hollow bar.
Also writes the LaTeX table figures/timing/timing_table.tex (per model and dataset: idle seconds, recorded
seconds, epochs or boosting rounds, parameters) — the compute-table numbers for the paper.

Edit the parameter block, then:   python plots/fig_timing.py
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASETS = ["SM", "LPMC", "SYN"]
MODELS = ["tspline", "tspline_level", "tspline_homo", "fdnn", "rumboost", "gbdt"]   # top to bottom; None = all re-timed
EXTRA_PAPER_MODELS = ["rumboost_pcuf", "rumboost_nomono", "rumboost_fe", "rumboost_noz", "mxl_biogeme", "mnl_biogeme",
                      "mnl", "reslogit", "l_mnl", "tastenet", "taste_dcm"]   # table only (not in the bar figure)
NLL_TOL = 1e-8                      # same_nll: |test NLL - paper test NLL| <= NLL_TOL counts as reproduced (rounding is ~1e-9)
SHOW_RANGE = True                   # thin line = min..max over the re-timed splits
ANNOTATE_RATIO = True               # "x N" next to TSplineDCM bars = idle seconds / RUMBoost idle seconds
SHOW_PAPER = False                  # hollow dashed bar = paper-set recorded mean (shared machine); off once every model is re-timed
FIG_SIZE_PER_PANEL = (2.9, 2.5)
OUT_DIR = None
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import display, latexify, model_color, tex  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / "timing"
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / "timing"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    idle = pd.read_csv(DATA / "timing.csv")
    paper = pd.read_csv(DATA / "timing_paper.csv")
    agg = idle.groupby(["dataset", "model"]).agg(n=("split", "size"), idle_mean=("fit_seconds", "mean"),
                                                 idle_min=("fit_seconds", "min"), idle_max=("fit_seconds", "max"),
                                                 epochs=("epochs", "mean"),
                                                 same_nll=("paper_test_nll", lambda s: int(((idle.loc[s.index, "test_nll"] - s).abs() <= NLL_TOL).sum()))).reset_index()
    return agg, paper


def figure(agg: pd.DataFrame, paper: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    models = MODELS or sorted(agg.model.unique())
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(FIG_SIZE_PER_PANEL[0] * len(DATASETS), FIG_SIZE_PER_PANEL[1]),
                             sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        a = agg[agg.dataset == ds].set_index("model")
        p = paper[paper.dataset == ds].set_index("model")
        rb = a.idle_mean.get("rumboost", np.nan)
        for yi, m in zip(y, models):
            c = model_color(m)
            if SHOW_PAPER and m in p.index:
                ax.barh(yi, p.loc[m, "fit_s_mean"], height=0.62, fill=False, edgecolor=c, linewidth=0.8, linestyle=(0, (2, 1.5)))
            if m in a.index:
                ax.barh(yi, a.loc[m, "idle_mean"], height=0.62, color=c, alpha=0.9)
                if SHOW_RANGE:
                    ax.plot([a.loc[m, "idle_min"], a.loc[m, "idle_max"]], [yi, yi], color="black", linewidth=0.6)
                if ANNOTATE_RATIO and m.startswith("tspline") and np.isfinite(rb) and rb > 0:
                    ax.text(a.loc[m, "idle_max"] * 1.25, yi, rf"$\times${a.loc[m, 'idle_mean'] / rb:.0f}", va="center",
                            fontsize=FONT_SIZE - 2)
        ax.set_xscale("log")
        ax.set_title(ds)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_yticks(y)
    np.atleast_1d(axes)[0].set_yticklabels([tex(display(m)) for m in models])
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#777777", label="idle machine, speed-up code")] if SHOW_PAPER else []
    if SHOW_PAPER:
        handles += [
               Patch(facecolor="none", edgecolor="#777777", linestyle=(0, (2, 1.5)), label="paper run (3--6 jobs sharing the machine)")]
    fig.supxlabel("training seconds per split", fontsize=FONT_SIZE, y=-0.02)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14), frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"timing_bars.{FMT}"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def table(agg: pd.DataFrame, paper: pd.DataFrame) -> None:
    models = list(MODELS or sorted(agg.model.unique())) + [m for m in EXTRA_PAPER_MODELS if m in set(paper.model)]
    lines = [r"\begin{tabular}{l" + "rrr" * len(DATASETS) + "}", r"\toprule",
             "Model & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{ds}}}" for ds in DATASETS) + r" \\",
             " & " + " & ".join(["s (idle) & s (paper) & params"] * len(DATASETS)) + r" \\", r"\midrule"]
    for m in models:
        cells = []
        for ds in DATASETS:
            a = agg[(agg.dataset == ds) & (agg.model == m)]
            p = paper[(paper.dataset == ds) & (paper.model == m)]
            idle = f"{a.idle_mean.iloc[0]:.0f}" if len(a) else "--"
            rec = f"{p.fit_s_mean.iloc[0]:.0f}" if len(p) else "--"
            npar = f"{p.n_params.iloc[0]:,.0f}" if len(p) and np.isfinite(p.n_params.iloc[0]) else "--"
            cells.append(f"{idle} & {rec} & {npar}")
        lines.append(display(m).replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "timing_table.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)
    # a plain-text copy of the same numbers for the md record
    rows = []
    for m in models:
        for ds in DATASETS:
            a = agg[(agg.dataset == ds) & (agg.model == m)]
            p = paper[(paper.dataset == ds) & (paper.model == m)]
            rows.append({"model": m, "dataset": ds,
                         "idle_s": a.idle_mean.iloc[0] if len(a) else np.nan, "idle_min": a.idle_min.iloc[0] if len(a) else np.nan,
                         "idle_max": a.idle_max.iloc[0] if len(a) else np.nan, "n_idle": int(a.n.iloc[0]) if len(a) else 0,
                         "same_nll": int(a.same_nll.iloc[0]) if len(a) else np.nan,
                         "paper_s": p.fit_s_mean.iloc[0] if len(p) else np.nan, "n_params": p.n_params.iloc[0] if len(p) else np.nan,
                         "epochs_idle": a.epochs.iloc[0] if len(a) else np.nan, "epochs_paper": p.epochs_mean.iloc[0] if len(p) else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "timing_table.csv", index=False)


def main() -> None:
    agg, paper = load()
    figure(agg, paper)
    table(agg, paper)


if __name__ == "__main__":
    main()
