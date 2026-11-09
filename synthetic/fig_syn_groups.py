# synthetic/fig_syn_groups.py — truth vs learned AS-UCF by population group (paper DGP_grp_* / SYN_grp_*: Fig. 5, Fig. 13).
"""One small figure per (source, alt, attr, group): the group's individual curves (thin), their 16-84 % band, the
group mean (dashed) and the model's population curve (solid; tau = 0) — for the truth ("source" = truth) and for
each model's chosen split. Model panels also show the truth group mean (red dashed) so recovery is read directly.
The groups (start time / age below vs above the median) are defined in synthetic/extract_syn.py GROUPS and recorded
in syn_groups_meta.csv.
Input: figures_data/<DATASET>/syn_groups.csv (+ syn_groups_individual_<source>.npz for the thin lines).

Edit the parameter block, then:   python synthetic/fig_syn_groups.py
Only some figures (substrings of the output file name):   python synthetic/fig_syn_groups.py --dataset SYN --only tspline_WALK
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SYN"
SOURCES = None                      # None = every source in syn_groups.csv ("truth" + models); or ["truth", "tspline"]
PANELS = None                       # None = every (alt, attr, group); or [("WALK", "TT-WALK", "st_below"), ("CAR", "TT-CAR", "age_above")]
SPLIT = 0                           # model curves of this split (truth rows carry split -1)
SHOW_INDIVIDUALS = True             # thin lines from syn_groups_individual_<source>.npz when it exists
IND_MAX = 2000
SHOW_BAND = True                    # 16-84 % of the group's individuals
SHOW_TRUTH_REF = True               # model panels: truth group mean, red dashed
GROUP_LABELS = {"st_below": "ST $<$ median", "st_above": "ST $\\geq$ median",
                "age_below": "age $<$ median", "age_above": "age $\\geq$ median"}
FIG_SIZE = (2.8, 2.8)
LEGEND = dict(loc="lower left", frameon=False)   # legend placement; font size is FONT_SIZE - 2
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
from style import PALETTE, cli, display, latexify, tex, wanted, x_unit, y_lim  # noqa: E402

DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _one(src, alt, attr, group, df, npz_cache):
    stem = f"{DATASET}_grp_{src}_{alt}_{attr}_{group}"
    if not wanted(stem, ONLY):
        return
    import matplotlib.pyplot as plt

    split = -1 if src == "truth" else SPLIT
    g = df[(df.source == src) & (df.alt == alt) & (df.attr == attr) & (df.group == group) & (df.split == split)].sort_values("x")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = g["x"].to_numpy()
    if SHOW_INDIVIDUALS:
        if src not in npz_cache:
            p = DATA / f"syn_groups_individual_{src}.npz"
            npz_cache[src] = np.load(p) if p.exists() else None
        z = npz_cache[src]
        key = f"{group}__{alt}__{attr}"
        if z is not None and key in z:
            Y, xs = z[key], z[f"x__{key}"]
            step = max(1, len(Y) // IND_MAX)
            ax.plot(xs, Y[::step].T, color="#90CAF9", alpha=0.05, linewidth=0.4, zorder=1)
            ax.plot([], [], color="#90CAF9", alpha=0.8, linewidth=0.6, label=f"Individuals ({min(len(Y), IND_MAX)})")
    if SHOW_BAND:
        ax.fill_between(x, g["lo"], g["hi"], color=PALETTE[0], alpha=0.18, linewidth=0, zorder=2, label=tex("16-84% individuals"))
    ax.plot(x, g["mean"], color=PALETTE[0], linestyle="--", linewidth=1.2, zorder=4, label=f"Mean ({int(g['n'].iloc[0])})")
    if np.isfinite(g["general"]).all():
        ax.plot(x, g["general"], color=PALETTE[0], linewidth=1.4, zorder=3, label="General")
    if SHOW_TRUTH_REF and src != "truth":
        t = df[(df.source == "truth") & (df.alt == alt) & (df.attr == attr) & (df.group == group)].sort_values("x")
        if not t.empty:
            ax.plot(t["x"], t["mean"], color="#D32F2F", linestyle="--", linewidth=1.0, zorder=5, label="Truth mean")
    ax.set_ylim(*y_lim(DATASET))
    ax.set_ylabel("Utility contribution")
    # two lines: the group part alone on the second line would otherwise run past the figure edge and be clipped
    ax.set_xlabel(tex(f"{display(src)}: {alt} - {attr.split('-')[0]}{x_unit(DATASET, attr)}") + "\n" + GROUP_LABELS.get(group, group))
    ax.legend(fontsize=FONT_SIZE - 2, **LEGEND)
    fig.tight_layout()
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_syn_groups] {p}")


def main() -> None:
    latexify(FONT_SIZE)
    df = pd.read_csv(DATA / "syn_groups.csv")
    sources = list(SOURCES) if SOURCES else list(dict.fromkeys(df.source))
    panels = [tuple(p) for p in PANELS] if PANELS else list(dict.fromkeys(zip(df.alt, df.attr, df.group)))
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for src in sources:
        for alt, attr, group in panels:
            _one(src, alt, attr, group, df, cache)


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
