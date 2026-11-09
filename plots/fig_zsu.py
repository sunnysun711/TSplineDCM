# plots/fig_zsu.py — z-spline ASC head: utility offset of each alternative as a function of one characteristic.
"""Paper figures LPMC_zsu_*.pdf. Input: figures_data/<DATASET>/zsu_curves_agg.csv (mean + 16-84 band over splits).
One figure per z name, all alternatives on one axes. Z_UNIT rescales the x axis back to natural units
(the model sees age/100, distance/1000, ... as in the old explain.plot_zsu_for_model).

Edit the parameter block, then:   python plots/fig_zsu.py
Only some figures (substrings of the output file name):   python plots/fig_zsu.py --dataset LPMC_officialtest --only _age
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "LPMC"
MODELS = None                       # None = every model in zsu_curves_agg.csv (only zspline-head TSplineDCM runs have one)
Z_NAMES = ["age", "distance", "day_of_week", "start_time_linear"]   # None = all in csv
Z_UNIT = {"age": (100, "years"), "start_time_linear": (100, "h"), "start_time": (100, "h"),
          "distance": (1000, "m"), "day_of_week": (10, "")}          # x scale factor, unit label
SHOW_BAND = True
FIG_SIZE = (3.4, 2.8)
OUT_DIR = None
ONLY = None                         # None = every figure; or parts of the output file name, e.g. ["overlay_TRAIN_TT"]; "=name" = whole name
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import PALETTE, cli, display, latexify, tex, wanted  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    df_all = pd.read_csv(DATA / "zsu_curves_agg.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    for MODEL in (MODELS or list(dict.fromkeys(df_all.model_name))):
        _one_model(df_all[df_all.model_name == MODEL], MODEL)


def _one_model(df, MODEL):
    import matplotlib.pyplot as plt

    if df.empty:
        print(f"[fig_zsu] {MODEL} not in zsu_curves_agg.csv; run plots/extract/zsu_curves.py")
        return
    for z in (Z_NAMES or list(dict.fromkeys(df.z))):
        stem = f"{DATASET}_zsu_{MODEL}_{z}"
        if not wanted(stem, ONLY):
            continue
        d = df[df.z == z]
        if d.empty:
            print(f"[fig_zsu] {z} not in csv")
            continue
        scale, unit = Z_UNIT.get(z, (1, ""))
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for i, (alt, g) in enumerate(d.groupby("alt", sort=False)):
            g = g.sort_values("z_value")
            x = g["z_value"] * scale
            c = PALETTE[i % len(PALETTE)]
            if SHOW_BAND and g["n_splits"].iloc[0] > 1:
                ax.fill_between(x, g["lo"], g["hi"], color=c, alpha=0.15, linewidth=0)
            ax.plot(x, g["mean"], color=c, linewidth=1.3, label=alt)
        ax.set_xlabel(tex(f"{display(MODEL)}: {z}" + (f" ({unit})" if unit else "")))
        ax.set_ylabel("Utility contribution")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        p = OUT / f"{stem}.{FMT}"
        fig.savefig(p)
        plt.close(fig)
        print(f"[fig_zsu] {p}")


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
