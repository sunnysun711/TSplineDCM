# plots/fig_vot_heatmap.py — value-of-time surface VoT(t, c) = mu_time(t) / mu_cost(c) from the population curves.
"""Paper figures SM_VoT_heatmap.pdf / LPMC_VoT_heatmap.pdf. One figure per model, one panel per alternative
that has both a time and a cost attribute. The marginal utilities come from figures_data/<DATASET>/mu_curves_agg.csv
(column CURVE: "general" = tau-0 population curve, falls back to "mean" for models without one, e.g. FDNN).
Optional moving-average smoothing of each 1-D marginal-utility curve before taking the ratio (the old script
smoothed the utility curves with alpha = 0.1).

Edit the parameter block, then:   python plots/fig_vot_heatmap.py
Only some figures (substrings of the output file name):   python plots/fig_vot_heatmap.py --dataset SM --only tspline
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
MODELS = None                       # None = every model in mu_curves_agg.csv; or ["TSplineDCM-ispline"]
CURVE = "general"                   # "general" | "mean"  (general falls back to mean when NaN)
ALTS = None                         # None = all alts with time+cost; or ["TRAIN", "CAR"]
TIME_SLOTS = ("TT", "RailT", "BusT", "AccessT")   # which attribute counts as "time" (first present wins)
COST_SLOT = "CO"
SMOOTH_WINDOW = 9                   # moving-average window (grid points) on each marginal-utility curve; 1 = off
VOT_CLIP = (0.0, 100.0)             # colour range in display units (CHF/h, GBP/h); None = data range
X_MAX = None                        # cost axis upper limit in raw csv units; None = data max
Y_MAX = None                        # time axis upper limit in raw csv units; None = data max
CMAP = "viridis"
FIG_SIZE_PER_PANEL = (3.0, 2.8)
TITLE = False                       # "<model> (general curve)" over the first panel; the paper caption says it instead
PANEL_GAP = 0.08                    # gap between panels as a fraction of a panel's width (constrained layout w_pad)
OUT_DIR = None
ONLY = None                         # None = every figure; or parts of the output file name, e.g. ["tspline"]; "=name" = whole name
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import cli, display, latexify, tex, vot_unit, wanted, x_unit  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _smooth(y, w):
    if w <= 1:
        return y
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def _time_cost(attrs):
    slots = {a.split("-")[0]: a for a in attrs}
    t = next((slots[s] for s in TIME_SLOTS if s in slots), None)
    return t, slots.get(COST_SLOT)


def _curve(g):
    y = g[CURVE].to_numpy()
    if CURVE == "general" and not np.isfinite(y).all():
        y = g["mean"].to_numpy()
    return y


def _one_model(mu_all, m, factor, unit):
    stem = f"{DATASET}_vot_heatmap_{m}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    mu = mu_all[mu_all.model_name == m]
    panels = []
    for alt in (ALTS or list(dict.fromkeys(mu.alt))):
        t_attr, c_attr = _time_cost(mu[mu.alt == alt].attr.unique())
        if t_attr and c_attr:
            panels.append((alt, t_attr, c_attr))
    if not panels:
        print(f"[fig_vot_heatmap] {m}: no alternative with both a time and a cost attribute")
        return
    w, h = FIG_SIZE_PER_PANEL
    fig, axes = plt.subplots(1, len(panels), figsize=(w * len(panels), h), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=PANEL_GAP * w, wspace=0.0)     # room for each panel's y label
    last = None
    used = None
    for ax, (alt, t_attr, c_attr) in zip(axes[0], panels):
        gt = mu[(mu.alt == alt) & (mu.attr == t_attr)].sort_values("x")
        gc = mu[(mu.alt == alt) & (mu.attr == c_attr)].sort_values("x")
        used = "general" if CURVE == "general" and np.isfinite(gt["general"]).all() else "mean"
        t, mt = gt["x"].to_numpy(), _smooth(_curve(gt), SMOOTH_WINDOW)
        c, mc = gc["x"].to_numpy(), _smooth(_curve(gc), SMOOTH_WINDOW)
        with np.errstate(divide="ignore", invalid="ignore"):
            vot = factor * mt[:, None] / mc[None, :]            # (time, cost)
        vot[:, np.abs(mc) < 1e-10] = np.nan
        if X_MAX is not None:
            keep = c <= X_MAX
            c, vot = c[keep], vot[:, keep]
        if Y_MAX is not None:
            keep = t <= Y_MAX
            t, vot = t[keep], vot[keep, :]
        vmin, vmax = VOT_CLIP if VOT_CLIP else (np.nanmin(vot), np.nanmax(vot))
        last = ax.pcolormesh(c, t, vot, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        ax.set_xlabel(tex(f"{alt} - {c_attr.split('-')[0]}{x_unit(DATASET, c_attr)}"))
        ax.set_ylabel(tex(f"{alt} - {t_attr.split('-')[0]}{x_unit(DATASET, t_attr)}"))
    if TITLE:
        axes[0][0].set_title(tex(f"{display(m)} ({used} curve)"))
    cb = fig.colorbar(last, ax=axes[0].tolist(), shrink=0.9, pad=0.02)
    cb.set_label(f"VoT ({unit})")
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_vot_heatmap] {p}")


def main() -> None:
    latexify(FONT_SIZE)
    mu = pd.read_csv(DATA / "mu_curves_agg.csv")
    models = list(MODELS) if MODELS else list(dict.fromkeys(mu.model_name))
    factor, unit = vot_unit(DATASET)
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        if m not in set(mu.model_name):
            print(f"[fig_vot_heatmap] {m} not in mu_curves_agg.csv; run plots/extract/asu_curves.py")
            continue
        _one_model(mu, m, factor, unit)


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
