# synthetic/fig_syn_folds.py — truth vs the fitted curve of every split (paper SYN_folds_*: Fig. 4 and Fig. 11).
"""Per model and (alt, attr). Top panel: the DGP's population curve (red) inside the P0-20 / P20-80 / P80-100 bands
of the individual truth curves, one fitted curve per split (blues) and the truth knots (dashed). Bottom panel, only
for models with spline parameters (TSplineDCM): the truth heterogeneity profile k_g at its Greville points (red x)
against each split's learned k_g (blue o).
A histogram of the attribute (the fixed batch) sits on top so the reader sees where the data ends: beyond it the fitted
curves and k_g are extrapolation, not recovery.
Input: figures_data/<DATASET>/{syn_truth_curves, syn_truth_heter, syn_truth_knots}.csv + asu_curves.csv.gz
       (+ heter_params.csv for the bottom panel, attr_samples.csv for the histogram). No torch, no model code.

Edit the parameter block, then:   python synthetic/fig_syn_folds.py
Only some figures (substrings of the output file name):   python synthetic/fig_syn_folds.py --dataset SYN --only tspline_WALK
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SYN"
MODELS = None                       # None = every model_name in asu_curves.csv.gz; or ["tspline", "fdnn"]
PAIRS = None                        # None = every (alt, attr); or [("WALK", "TT-WALK")]
SPLITS = None                       # None = every split in the csv; or [0, 1, 2]
FIT_CURVE = "auto"                  # "general" (tau = 0) | "mean" (mean individual) | "auto" = general when the model has one
HETER_STAT = "k"                    # "k" = k_g (comparable across knot vectors) | "k_abs_a" = k_g * |a_g|
SHOW_HIST = True                    # attribute-distribution panel on top (attr_samples.csv): shows where the data ends
K_MIN_INCR_SHARE = 0.005            # a basis whose increment carries < this share of the curve's total drop multiplies nothing:
                                    # its k_g never gets a gradient and stays at the initial value -> not drawn (0 = draw all)
SHOW_TRUTH_BANDS = True
SHOW_TRUTH_KNOTS = True
TRUTH_COLORS = ("#FFCDD2", "#E57373", "#B71C1C")            # P0-20, P20-80, P80-100
SPLIT_COLORS = ("#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1")
FIG_SIZE = (3.4, 3.2)               # + 35 % height for the k panel, + 20 % for the histogram
K_YLIM = (-0.05, 0.85)              # y range of the k_g panel, the same in every figure (truth k_g <= 0.78); None = auto
OUT_DIR = None                      # None -> figures/<DATASET>/
ONLY = None                         # None = every figure; or parts of the output file name, e.g. ["overlay_TRAIN_TT"]; "=name" = whole name
FMT = "pdf"
FONT_SIZE = 9
CURVE_LEGEND_LOC = "lower left"     # legend of the curve panel; same corner in every figure (curves decrease, so it is free)
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "plots"))
from style import cli, display, latexify, tex, wanted, x_unit, y_lim  # noqa: E402

DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _sel(df, **kv):
    m = np.ones(len(df), dtype=bool)
    for k, v in kv.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def _heter_value(df):
    return df["k"].to_numpy() if HETER_STAT == "k" else df["k"].to_numpy() * np.abs(df["a"].to_numpy())


def _one(m, alt, attr, truth, theter, tknots, raw, hp, hist):
    stem = f"{DATASET}_folds_{m}_{alt}_{attr}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    r = _sel(raw, model_name=m, alt=alt, attr=attr)
    t = _sel(truth, alt=alt, attr=attr).sort_values("x")
    if r.empty or t.empty:
        return
    splits = list(SPLITS) if SPLITS else sorted(r["split"].unique())
    has_heter = hp is not None and not _sel(hp, model_name=m, alt=alt, attr=attr).empty
    w, h = FIG_SIZE
    hv = hist[(hist.alt == alt) & (hist.attr == attr)]["value"].to_numpy() if hist is not None else None
    rows, ratios, scale = [], [], 1.0
    if hv is not None:
        rows.append("hist"); ratios.append(1.0); scale += 0.2
    rows.append("curve"); ratios.append(4.0)
    if has_heter:
        rows.append("heter"); ratios.append(1.4); scale += 0.35
    axes = plt.subplots(len(rows), 1, sharex=True, figsize=(w, h * scale), height_ratios=ratios)[1]
    axes = [axes] if len(rows) == 1 else list(axes)
    fig = axes[0].figure
    ax = axes[rows.index("curve")]
    ax2 = axes[rows.index("heter")] if has_heter else None
    if hv is not None:
        ax_h = axes[rows.index("hist")]
        ax_h.hist(hv, bins=120, color="#3E7CB1")
        ax_h.set_ylabel("Freq.")

    x = t["x"].to_numpy()
    if SHOW_TRUTH_BANDS:
        for (lo, hi, c, lab) in ((t["q00"], t["q20"], TRUTH_COLORS[0], "Truth P0-20"),
                                 (t["q20"], t["q80"], TRUTH_COLORS[1], "Truth P20-80"),
                                 (t["q80"], t["q100"], TRUTH_COLORS[2], "Truth P80-100")):
            ax.fill_between(x, lo, hi, color=c, alpha=0.5, linewidth=0, label=tex(lab), zorder=1)
    ax.plot(x, t["general"], color="#D32F2F", linewidth=1.0, label="Truth (population)", zorder=3)
    if SHOW_TRUTH_KNOTS:
        kn = _sel(tknots, alt=alt, attr=attr)["knot"].to_numpy()
        for i, k in enumerate(kn):
            ax.axvline(k, color="#90CAF9", linewidth=0.5, linestyle="--", alpha=0.8, zorder=0,
                       label="Truth knots" if i == 0 else None)
    for i, s in enumerate(splits):
        g = r[r["split"] == s].sort_values("x")
        if g.empty:
            continue
        col = "general" if FIT_CURVE == "general" or (FIT_CURVE == "auto" and np.isfinite(g["general"]).all()) else "mean"
        ax.plot(g["x"], g[col], color=SPLIT_COLORS[i % len(SPLIT_COLORS)], linewidth=1.0, alpha=0.9, zorder=4,
                label=f"Split {s}" + ("" if col == "general" else " (mean)"))
    ax.set_ylim(*y_lim(DATASET))
    ax.set_ylabel("Utility contribution")
    ax.legend(loc=CURVE_LEGEND_LOC, fontsize=FONT_SIZE - 3, ncols=2)
    xlabel = tex(f"{display(m)}: {alt} - {attr.split('-')[0]}{x_unit(DATASET, attr)}")

    if ax2 is not None:
        th = _sel(theter, alt=alt, attr=attr).sort_values("xi")
        ax2.plot(th["xi"], _heter_value(th), "x-", color="#D32F2F", markersize=3.5, linewidth=0.75, label="Truth")
        for i, s in enumerate(splits):
            hs = _sel(hp, model_name=m, alt=alt, attr=attr, split=s).sort_values("xi")
            if hs.empty:
                continue
            if K_MIN_INCR_SHARE > 0 and "a" in hs:                 # keep only the bases that actually carry a drop
                share = hs["a"].abs() / max(float(hs["a"].abs().sum()), 1e-12)
                hs = hs[share >= K_MIN_INCR_SHARE]
            ax2.plot(hs["xi"], _heter_value(hs), "o-", color=SPLIT_COLORS[i % len(SPLIT_COLORS)], markersize=2.5,
                     linewidth=0.8, alpha=0.7)
        ax2.axhline(0.0, color="0.5", linewidth=0.5, linestyle=":")
        if K_YLIM is not None:
            ax2.set_ylim(*K_YLIM)
        ax2.set_ylabel(r"$k_g$" if HETER_STAT == "k" else r"$k_g\,|a_g|$")
        ax2.set_xlabel(xlabel)
        ax2.legend(loc="best", fontsize=FONT_SIZE - 3)
    else:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_syn_folds] {p}")


def main() -> None:
    latexify(FONT_SIZE)
    truth = pd.read_csv(DATA / "syn_truth_curves.csv")
    theter = pd.read_csv(DATA / "syn_truth_heter.csv")
    tknots = pd.read_csv(DATA / "syn_truth_knots.csv")
    raw = pd.read_csv(DATA / "asu_curves.csv.gz")
    hp = pd.read_csv(DATA / "heter_params.csv") if (DATA / "heter_params.csv").exists() else None
    hist = pd.read_csv(DATA / "attr_samples.csv") if SHOW_HIST and (DATA / "attr_samples.csv").exists() else None
    models = list(MODELS) if MODELS else list(dict.fromkeys(raw.model_name))
    pairs = [tuple(p) for p in PAIRS] if PAIRS else list(dict.fromkeys(zip(truth.alt, truth.attr)))
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        for alt, attr in pairs:
            _one(m, alt, attr, truth, theter, tknots, raw, hp, hist)


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
