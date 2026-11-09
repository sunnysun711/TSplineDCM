# plots/fig_asucf.py — AS-UCF figure: utility contribution vs attribute, mean over splits + band (+ general, + spaghetti).
"""Paper figures SM_100_*, LPMC_100_*, SYN_ASUCF_*, SYN_folds_*, SYN_bench_detail_* all come from this file.
Input: figures_data/<DATASET>/asu_curves_agg.csv (+ asu_curves.csv.gz for per-split lines, attr_samples.csv for the
histogram, asu_individual_<model>_split<k>.npz for the spaghetti). No torch, no model code.

Two layouts, both on by default:
    EACH    one figure per model and (alt, attr): histogram + 16-84 band + mean + general (+ individuals)
    OVERLAY one figure per (alt, attr) with every model's mean + band on the same axes
Post-hoc curves (style.PROBE_MODELS: F-DNN partial dependences, TasteNet mean of individual lines) are drawn dashed.
The overlay y-range is shared per attribute kind (time / cost) where style.Y_LIM_OVERLAY lists the dataset.
With OVERLAY_LEGEND = "separate" the overlays carry no legend; one legend figure per overlay group is written instead.

Edit the parameter block, then:   python plots/fig_asucf.py
Only some figures (substrings of the output file name):   python plots/fig_asucf.py --dataset SM --only overlay_TRAIN_TT
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
MODELS = None                       # None = single figures for every model, overlays of style.OVERLAY_GROUPS; or a list
EACH = True                         # single-model figures
OVERLAY = True                      # all MODELS on one axes per (alt, attr)
PAIRS = None                        # None = every (alt, attr) in the csv; or [("CAR", "CO"), ("TRAIN", "TT")]
SHOW_HIST = True                    # attribute-distribution panel on top (single-model layout only)
SHOW_BAND = True                    # 16-84 percentile band across splits
SHOW_GENERAL = True                 # population curve (tau = 0), red (single-model layout only)
SHOW_SPLITS = False                 # every split's mean curve as a thin line (the SYN_folds_* look)
SPAGHETTI_SPLIT = 0                 # per-individual curves of this split, if the npz exists; None = off
SPAGHETTI_MAX = 2000                # at most this many individual lines
FIG_SIZE = (3.4, 3.2)               # inches; hist panel adds 25 % height (single-model layout)
OVERLAY_FIG_SIZE = (2.3, 2.1)       # inches; the paper prints three overlays per row (.32 textwidth = 2.07 in)
OVERLAY_LEGEND = "separate"         # "panel": legend inside every overlay; "separate": none inside, one legend figure
                                    #   <DATASET>_asucf_overlay_legend.pdf (per model: its line drawn over its band)
OVERLAY_LEGEND_NCOL = {"SM": 1}     # columns of that legend figure by dataset prefix; not listed = one row, all models
OVERLAY_LEGEND_ROW_SIZE = (6.4, 0.5)  # inches of the one-row legend figure (cropped to the legend on saving)
BAND_ALPHA = 0.18                   # opacity of the 16-84 band, in the panels and in the legend
OUT_DIR = None                      # None -> figures/<DATASET>/
ONLY = None                         # None = every figure; or parts of the output file name, e.g. ["overlay_TRAIN_TT"]; "=name" = whole name
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import OVERLAY_GROUPS, PALETTE, cli, display, latexify, line_style, model_color, select_overlay, tex, wanted, x_unit, y_lim, y_lim_overlay  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _load():
    agg = pd.read_csv(DATA / "asu_curves_agg.csv")
    models = list(MODELS) if MODELS else list(dict.fromkeys(agg.model_name))
    missing = [m for m in models if m not in set(agg.model_name)]
    if missing:
        raise SystemExit(f"{missing} not in {DATA / 'asu_curves_agg.csv'}; run plots/extract/asu_curves.py for them")
    raw = pd.read_csv(DATA / "asu_curves.csv.gz") if SHOW_SPLITS and (DATA / "asu_curves.csv.gz").exists() else None
    hist = pd.read_csv(DATA / "attr_samples.csv") if SHOW_HIST and (DATA / "attr_samples.csv").exists() else None
    return agg, raw, hist, models


def _pairs(agg, models):
    if PAIRS:
        return [tuple(p) for p in PAIRS]
    sub = agg[agg.model_name.isin(models)]
    return list(dict.fromkeys(zip(sub.alt, sub.attr)))


def _draw_one(ax, g, color, label, show_general, style=None):
    x = g["x"].to_numpy()
    n = int(g["n_splits"].iloc[0])
    if SHOW_BAND and n > 1:
        ax.fill_between(x, g["lo"], g["hi"], color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, g["mean"], color=color, linewidth=1.6, zorder=4,
            label=label or (f"Mean ({n} splits)" if n > 1 else "Mean"), **(style or {}))
    if show_general and np.isfinite(g["general"]).all():
        ax.plot(x, g["general"], color="red", linewidth=1.0, zorder=3, label="General")


def _single(agg, raw, hist, m, alt, attr):
    stem = f"{DATASET}_asucf_{m}_{alt}_{attr}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    g = agg[(agg.model_name == m) & (agg.alt == alt) & (agg.attr == attr)].sort_values("x")
    if g.empty:
        return
    w, h = FIG_SIZE
    if hist is not None:
        fig, (ax_h, ax) = plt.subplots(2, 1, sharex=True, figsize=(w, h * 1.25), height_ratios=[1, 4])
        ax_h.hist(hist[(hist.alt == alt) & (hist.attr == attr)]["value"], bins=120, color="#3E7CB1")
        ax_h.set_ylabel("Freq.")
    else:
        fig, ax = plt.subplots(figsize=(w, h))
    npz = DATA / f"asu_individual_{m}_split{SPAGHETTI_SPLIT}.npz"
    if SPAGHETTI_SPLIT is not None and npz.exists():
        z = np.load(npz)
        key = f"{alt}__{attr}"
        if key in z:
            Y, xs = z[key], z[f"x__{key}"]
            step = max(1, len(Y) // SPAGHETTI_MAX)
            ax.plot(xs, Y[::step].T, color="0.6", alpha=0.03, linewidth=0.5, zorder=1)
            ax.plot([], [], color="0.6", alpha=0.6, linewidth=0.5, label=f"Individuals (split {SPAGHETTI_SPLIT})")
    if raw is not None:
        r = raw[(raw.model_name == m) & (raw.alt == alt) & (raw.attr == attr)]
        for _, s in r.groupby("split"):
            s = s.sort_values("x")
            ax.plot(s["x"], s["mean"], color=PALETTE[0], alpha=0.15, linewidth=0.5, zorder=2)
    _draw_one(ax, g, PALETTE[0], None, SHOW_GENERAL)
    ax.set_ylim(*y_lim(DATASET))
    ax.set_ylabel("Utility contribution" + (" (post hoc)" if line_style(m).get("linestyle") == "--" else ""))
    ax.set_xlabel(tex(f"{display(m)}: {alt} - {attr.split('-')[0]}{x_unit(DATASET, attr)}"))
    ax.legend(loc="best")
    fig.tight_layout()
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_asucf] {p}")


def _overlay(agg, models, alt, attr, tag="overlay"):
    stem = f"{DATASET}_asucf_{tag}_{alt}_{attr}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=OVERLAY_FIG_SIZE)
    drawn = 0
    for i, m in enumerate(models):
        g = agg[(agg.model_name == m) & (agg.alt == alt) & (agg.attr == attr)].sort_values("x")
        if g.empty:
            continue
        st = line_style(m)
        color = st.pop("color", None) or model_color(m, i)
        _draw_one(ax, g, color, tex(display(m)), show_general=False, style=st)
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    ax.set_ylim(*y_lim_overlay(DATASET, attr))      # shared per attribute kind (style.Y_LIM_OVERLAY); clips TasteNet
    ax.set_ylabel("Utility contribution")
    ax.set_xlabel(tex(f"{alt} - {attr.split('-')[0]}{x_unit(DATASET, attr)}"))
    if OVERLAY_LEGEND == "panel":
        ax.legend(loc="best", fontsize=FONT_SIZE - 2, framealpha=0.85)
    fig.tight_layout()
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_asucf] {p}")


def _legend_figure(models, tag="overlay"):
    """One legend for all overlays of `tag`: each entry is the model's line drawn over its 16-84 band.
    ncol = 1 gives a panel of OVERLAY_FIG_SIZE (drops into a grid of overlays); otherwise a strip cropped to the legend."""
    stem = f"{DATASET}_asucf_{tag}_legend"
    if not wanted(stem, ONLY):
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = [], []
    for i, m in enumerate(models):
        st = line_style(m)
        color = st.pop("color", None) or model_color(m, i)
        band = Patch(facecolor=color, alpha=BAND_ALPHA, linewidth=0)
        line = Line2D([], [], color=color, linewidth=1.6, **st)
        handles.append((band, line))
        labels.append(tex(display(m)))
    ncol = next((n for k, n in OVERLAY_LEGEND_NCOL.items() if DATASET.startswith(k)), len(models))
    as_panel = ncol == 1
    fig = plt.figure(figsize=OVERLAY_FIG_SIZE if as_panel else OVERLAY_LEGEND_ROW_SIZE)
    fig.legend(handles, labels, loc="center", ncol=ncol, fontsize=FONT_SIZE - 1, frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, handlelength=2.6, handleheight=1.3,
               columnspacing=1.2, labelspacing=0.8 if as_panel else 0.5)
    p = OUT / f"{stem}.{FMT}"
    with mpl.rc_context({"savefig.bbox": "standard" if as_panel else "tight"}):   # panel keeps the full panel size
        fig.savefig(p)
    plt.close(fig)
    print(f"[fig_asucf] {p}")


def main() -> None:
    latexify(FONT_SIZE)
    agg, raw, hist, models = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    # MODELS given: one overlay of exactly those; MODELS None: the curated main set + the RUMBoost variants
    groups = ({"overlay": list(models)} if MODELS else
              {("overlay" if g == "main" else f"overlay_{g}"): select_overlay(models, g) for g in OVERLAY_GROUPS})
    for alt, attr in _pairs(agg, models):
        if EACH:
            for m in models:
                _single(agg, raw, hist, m, alt, attr)
        if OVERLAY:
            for tag, ms in groups.items():
                if len(ms) > 1:
                    _overlay(agg, ms, alt, attr, tag)
    if OVERLAY and OVERLAY_LEGEND == "separate":
        for tag, ms in groups.items():
            if len(ms) > 1:
                _legend_figure(ms, tag)


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
