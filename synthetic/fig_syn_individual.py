# synthetic/fig_syn_individual.py — individual curves of one split: truth and every model (paper DGP_asucf_* / SYN_ASUCF_* / SYN_bench_detail_*).
"""One figure per (source, alt, attr): the fixed batch's individual curves (thin), their mean (dashed) and the
population curve (solid; tau = 0, when the model has one). Model panels add the truth population curve (red dashed)
and, for TSplineDCM, that split's learned knots; the truth panel shows the truth knots. The truth and every model
use the SAME individuals (plots/extract/_common.fixed_batch), so the spaghetti is comparable panel to panel.
Input: figures_data/<DATASET>/syn_individual_truth.npz, asu_individual_<model>_split<SPLIT>.npz, syn_truth_curves.csv,
       asu_curves.csv.gz (+ syn_truth_knots.csv, heter_knots.csv).

Edit the parameter block, then:   python synthetic/fig_syn_individual.py
Only some figures (substrings of the output file name):   python synthetic/fig_syn_individual.py --dataset SYN --only tspline_WALK
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SYN"
SOURCES = None                      # None = "truth" + every model with an asu_individual_<model>_split<SPLIT>.npz
SPLIT = 0
PAIRS = None                        # None = every (alt, attr) in syn_truth_curves.csv
IND_MAX = 2000
SHOW_TRUTH_REF = True               # model panels: truth population curve, red dashed
SHOW_KNOTS = True                   # truth knots / the split's learned knots (TSplineDCM)
FIG_SIZE = (2.8, 2.8)
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "plots"))
from style import PALETTE, cli, display, latexify, line_style, tex, wanted, x_unit, y_lim  # noqa: E402

DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _npz_path(src):
    return DATA / ("syn_individual_truth.npz" if src == "truth" else f"asu_individual_{src}_split{SPLIT}.npz")


def _one(src, alt, attr, truth, raw, tknots, hknots, z):
    stem = f"{DATASET}_individual_{src}_{alt}_{attr}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    key = f"{alt}__{attr}"
    if key not in z:
        return
    Y, xs = z[key], z[f"x__{key}"]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    step = max(1, len(Y) // IND_MAX)
    ax.plot(xs, Y[::step].T, color="#90CAF9", alpha=0.05, linewidth=0.4, zorder=1)
    ax.plot([], [], color="#90CAF9", alpha=0.8, linewidth=0.6, label=f"Individuals ({min(len(Y), IND_MAX)})")
    ax.plot(xs, Y.mean(0), color=PALETTE[0], linestyle="--", linewidth=1.2, zorder=4, label="Mean")
    t = truth[(truth.alt == alt) & (truth.attr == attr)].sort_values("x")
    if src == "truth":
        ax.plot(t["x"], t["general"], color=PALETTE[0], linewidth=1.4, zorder=3, label="General")
        knots = tknots[(tknots.alt == alt) & (tknots.attr == attr)]["knot"].to_numpy() if tknots is not None else []
    else:
        g = raw[(raw.model_name == src) & (raw.alt == alt) & (raw.attr == attr) & (raw.split == SPLIT)].sort_values("x")
        if not g.empty and np.isfinite(g["general"]).all():
            ax.plot(g["x"], g["general"], color=PALETTE[0], linewidth=1.4, zorder=3, label="General")
        if SHOW_TRUTH_REF and not t.empty:
            ax.plot(t["x"], t["general"], color="#D32F2F", linestyle="--", linewidth=1.0, zorder=5, label="Truth (population)")
        knots = (hknots[(hknots.model_name == src) & (hknots.alt == alt) & (hknots.attr == attr) & (hknots.split == SPLIT)]["knot"].to_numpy()
                 if hknots is not None else [])
    if SHOW_KNOTS:
        for i, k in enumerate(knots):
            ax.axvline(k, color="#90CAF9", linewidth=0.4, linestyle="--", alpha=0.8, zorder=0,
                       label=("Truth knots" if src == "truth" else f"Knots (split {SPLIT})") if i == 0 else None)
    ax.set_ylim(*y_lim(DATASET))
    probe = src != "truth" and line_style(src).get("linestyle") == "--"
    ax.set_ylabel("Utility contribution" + (" (PDP probe)" if probe else ""))
    ax.set_xlabel(tex(f"{display(src)}: {alt} - {attr.split('-')[0]}{x_unit(DATASET, attr)}"))
    ax.legend(loc="best", fontsize=FONT_SIZE - 3)
    fig.tight_layout()
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_syn_individual] {p}")


def main() -> None:
    latexify(FONT_SIZE)
    truth = pd.read_csv(DATA / "syn_truth_curves.csv")
    raw = pd.read_csv(DATA / "asu_curves.csv.gz") if (DATA / "asu_curves.csv.gz").exists() else pd.DataFrame(
        columns=["model_name", "alt", "attr", "split", "x", "mean", "general"])
    tknots = pd.read_csv(DATA / "syn_truth_knots.csv") if (DATA / "syn_truth_knots.csv").exists() else None
    hknots = pd.read_csv(DATA / "heter_knots.csv") if (DATA / "heter_knots.csv").exists() else None
    if SOURCES:
        sources = list(SOURCES)
    else:
        sources = ["truth"] + sorted(p.name[len("asu_individual_"):-len(f"_split{SPLIT}.npz")]
                                     for p in DATA.glob(f"asu_individual_*_split{SPLIT}.npz"))
    pairs = [tuple(p) for p in PAIRS] if PAIRS else list(dict.fromkeys(zip(truth.alt, truth.attr)))
    OUT.mkdir(parents=True, exist_ok=True)
    for src in sources:
        p = _npz_path(src)
        if not p.exists():
            print(f"[fig_syn_individual] {src}: no {p.name}; skip")
            continue
        z = np.load(p)
        for alt, attr in pairs:
            _one(src, alt, attr, truth, raw, tknots, hknots, z)


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
