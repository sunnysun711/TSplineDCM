# plots/fig_vot_hist.py — histogram of individual values of time at each traveller's observed attributes.
"""Paper figures SM_VoT_hist.pdf / LPMC_VoT_hist.pdf. One panel per alternative. Input:
figures_data/<DATASET>/vot_individual.csv.gz (one row per individual x split x alt, raw model units).
Filtering happens HERE (parameter block), so the extract never has to be re-run to change the rule.
With several splits in the csv, the legend reports each statistic as median [min, max] across splits and the
histogram pools all splits; with one split it is that split's numbers.

Layouts (both on by default): EACH = one figure per model; OVERLAY = all MODELS as step histograms on one figure.
A model whose VoT is the same for everyone (MNL) is drawn as a single vertical line.

Edit the parameter block, then:   python plots/fig_vot_hist.py
Only some figures (substrings of the output file name):   python plots/fig_vot_hist.py --dataset SM --only overlay
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
MODELS = None                       # None = every model in the csv; or ["TSplineDCM-ispline", "MNL"]
EACH = True
OVERLAY = True
ALTS = None                         # None = all in csv
MU_COST_MAX = -1e-6                 # keep individuals whose cost slope is clearly negative
VOT_RANGE = (0.1, 100.0)            # keep VoT within this range (display units, e.g. CHF/h)
TIME_RANGE = None                   # (lo, hi) on the observed time value (raw units); None = no filter
COST_RANGE = None                   # (lo, hi) on the observed cost value (raw units); None = no filter
BINS = 80                           # bin edges are shared by every model and panel (over VOT_RANGE)
DENSITY = True
ALPHA = 0.7                         # transparency of the overlay outlines / constant-VoT lines
SHOW_STATS = ("mean", "median")     # vertical lines + legend entries
FIG_SIZE_PER_PANEL = (3.0, 2.6)
OUT_DIR = None
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
from style import PALETTE, cli, display, latexify, model_color, select_overlay, tex, vot_unit, wanted  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _filter(d, factor):
    keep = d["mu_cost"] < MU_COST_MAX
    v = d["vot"] * factor
    keep &= (v >= VOT_RANGE[0]) & (v <= VOT_RANGE[1])
    if TIME_RANGE:
        keep &= d["x_time"].between(*TIME_RANGE)
    if COST_RANGE:
        keep &= d["x_cost"].between(*COST_RANGE)
    return d.assign(vot_disp=v)[keep]


def _stat_label(name, per_split):
    vals = np.array(per_split)
    if len(vals) == 1:
        return f"{name.capitalize()}: {vals[0]:.2f}"
    return f"{name.capitalize()}: {np.median(vals):.2f} [{vals.min():.2f}, {vals.max():.2f}]"


def _figure(df, models, alts, factor, unit, tag):
    stem = f"{DATASET}_vot_hist_{tag}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    w, h = FIG_SIZE_PER_PANEL
    fig, axes = plt.subplots(1, len(alts), figsize=(w * len(alts), h), squeeze=False)
    single = len(models) == 1
    edges = np.linspace(VOT_RANGE[0], VOT_RANGE[1], BINS + 1)   # same bins everywhere: densities comparable across models
    heights: list[float] = []                                   # pooled over panels -> one y range for the whole row
    for ax, alt in zip(axes[0], alts):
        for i, m in enumerate(models):
            d = _filter(df[(df.model_name == m) & (df.alt == alt)], factor)
            if d.empty:
                print(f"[fig_vot_hist] {m} {alt}: nothing left after filtering")
                continue
            color = model_color(m, i)
            n_split = d["split"].nunique()
            v = d["vot_disp"].to_numpy()
            degenerate = v.std() < 1e-4 * max(1.0, abs(v.mean()))     # e.g. MNL: one VoT for everyone (up to fp noise)
            if degenerate:
                ax.axvline(v.mean(), color=color, linestyle="-", linewidth=1.5, alpha=ALPHA,
                           label=tex(f"{display(m)}: {v.mean():.1f} (constant)"))
                continue
            per = d.groupby("split")["vot_disp"]
            if single:
                ax.hist(v, bins=edges, density=DENSITY, alpha=0.75, color=color, histtype="stepfilled",
                        edgecolor=color, linewidth=0.5)
                for s_name, ls in zip(SHOW_STATS, ("--", ":")):
                    per_split = per.mean() if s_name == "mean" else per.median()
                    ax.axvline(np.median(per_split), color="red" if s_name == "mean" else "green",
                               linestyle=ls, linewidth=1.2, label=_stat_label(s_name, per_split))
                ax.set_title(tex(f"{alt} (n = {len(d)}, {n_split} split{'s' if n_split > 1 else ''})"), fontsize=FONT_SIZE - 1)
            else:
                # overlay: translucent outlines, one legend entry per model with its mean; y range from the bulk
                # of the bins so a model with a few spiky bins (discrete tastes) does not flatten everyone else
                counts, _, _ = ax.hist(v, bins=edges, density=DENSITY, histtype="step", color=color, linewidth=1.0,
                                       alpha=ALPHA, label=tex(f"{display(m)}: {np.median(per.mean()):.1f} (mean)"))
                heights.append(np.percentile(counts, 95))
                ax.set_title(alt, fontsize=FONT_SIZE - 1)
        ax.set_xlabel(f"VoT ({unit})")
        ax.set_ylabel("Density" if DENSITY else "Count")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=FONT_SIZE - 3)
    if not single and heights:
        for ax in axes[0]:                                      # one y range for all panels
            ax.set_ylim(0, 1.3 * max(heights))
    if single:
        fig.suptitle(tex(display(models[0])), fontsize=FONT_SIZE)
    fig.tight_layout()
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_vot_hist] {p}")


def main() -> None:
    latexify(FONT_SIZE)
    df = pd.read_csv(DATA / "vot_individual.csv.gz")
    models = list(MODELS) if MODELS else list(dict.fromkeys(df.model_name))
    factor, unit = vot_unit(DATASET)
    alts = ALTS or list(dict.fromkeys(df.alt))
    OUT.mkdir(parents=True, exist_ok=True)
    if EACH:
        for m in models:
            _figure(df, [m], alts, factor, unit, m)
    overlay = list(models) if MODELS else select_overlay(models, "main")
    if OVERLAY and len(overlay) > 1:
        _figure(df, overlay, alts, factor, unit, "overlay")


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
