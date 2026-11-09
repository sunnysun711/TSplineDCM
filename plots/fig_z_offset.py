# plots/fig_z_offset.py — ASC constants and linear z-head coefficients across splits (dot = mean, bar = 16-84 pct).
"""Input: figures_data/<DATASET>/z_offset.csv (model_name, split, alt, param, value, masked). One panel per
non-base alternative, side by side; parameters on the x axis (labels rotated 90 degrees), utility offset on
the shared y axis; several models side by side at each parameter.

Identification: only DIFFERENCES between alternatives are identified in a logit, so every value is shown
relative to BASE_ALT. A model that pins the base alternative's offset to 0 (TSplineDCM/TasteNet: `masked`
= True, the whole ASC + z head of that alt is multiplied by 0) contributes its raw rows unchanged; a model that
learns all rows (torch MNL) has its base row subtracted split by split. The base panel is therefore all zeros
and is not drawn.

Edit the parameter block, then:   python plots/fig_z_offset.py
Only some figures (substrings of the output file name):   python plots/fig_z_offset.py --dataset SM --only z_offset_all
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
PAPER_MODELS = ["tspline", "rumboost_pcuf", "mnl"]   # the paper's set: F-DNN / TasteNet have only a constant, ResLogit adds noise
MODELS = PAPER_MODELS               # None = the curated "main" overlay group; or any list of csv model names
BASE_ALT = "CAR"                    # everything is "alt minus BASE_ALT"; None = use each model's masked alt, else the last alt
PARAMS = None                       # None = all; or ["ASC", "MALE", "FIRST"]
PARAM_LABELS = {                    # x-axis text per csv param name (Swissmetro, RUMBoost preprocessing)
    "ASC": "ASC", "MALE": "Male", "FIRST": "First class",
    "PURPOSE_1": "Commute", "PURPOSE_2": "Shopping", "PURPOSE_3": "Business", "PURPOSE_4": "Leisure",
    "PURPOSE_5": "Return from work", "PURPOSE_6": "Return from shopping", "PURPOSE_7": "Return from business",
    "PURPOSE_8": "Return from leisure",
    # Swissmetro AGE codes: 1 = up to 24, 2 = 25-39 (base = 40 and older); checked against swissmetro.dat
    "AGE_1": r"Age $\leq$ 24", "AGE_2": "Age 25--39",
    # LPMC (RUMBoost preprocessing)
    "day_of_week": "Day of week", "start_time_linear": "Start time", "age": "Age", "female": "Female",
    "driving_license": "Driving licence", "car_ownership": "Car ownership", "distance": "Distance",
    "fueltype_Average": "Fuel: average", "fueltype_Diesel": "Fuel: diesel", "fueltype_Hybrid": "Fuel: hybrid",
    "fueltype_Petrol": "Fuel: petrol", "purpose_B": "Business", "purpose_HBE": "Home-based education",
    "purpose_HBO": "Home-based other", "purpose_HBW": "Home-based work", "purpose_NHBO": "Non-home-based other",
}
FIG_SIZE_PER_PANEL = (3.2, 3.6)     # (width of ONE panel, figure height); panels side by side
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
from style import PALETTE, cli, display, latexify, model_color, select_overlay, tex, wanted  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def relative_to_base(d: pd.DataFrame, base: str) -> pd.DataFrame:
    """alt-minus-base per (split, param). A masked base row counts as 0 (its raw numbers are dead parameters)."""
    b = d[d.alt == base][["split", "param", "value", "masked"]].copy()
    b["value"] = np.where(b["masked"], 0.0, b["value"])
    b = b.rename(columns={"value": "base_value"}).drop(columns="masked")
    out = d[d.alt != base].merge(b, on=["split", "param"], how="left")
    out["value"] = out["value"] - out["base_value"].fillna(0.0)
    return out


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    df = pd.read_csv(DATA / "z_offset.csv")
    models = [m for m in (MODELS or list(dict.fromkeys(df.model_name))) if m in set(df.model_name)]
    if MODELS is None:
        models = select_overlay(models, "main") or models          # curated set; every model would not fit one axis
    if not models:
        raise SystemExit(f"none of {MODELS} in z_offset.csv; run plots/extract/z_offset.py")
    stem = f"{DATASET}_z_offset_{'all' if MODELS is None or MODELS == PAPER_MODELS else '_'.join(models)}"
    if not wanted(stem, ONLY):
        return
    df = df[df.model_name.isin(models)]
    parts = []
    for m in models:
        d = df[df.model_name == m]
        base = BASE_ALT or (d[d.masked].alt.iloc[0] if d.masked.any() else d.alt.iloc[-1])
        parts.append(relative_to_base(d, base).assign(base=base))
    rel = pd.concat(parts, ignore_index=True)
    alts = list(dict.fromkeys(rel.alt))
    params = PARAMS or list(dict.fromkeys(rel.param))
    w, h = FIG_SIZE_PER_PANEL
    fig, axes = plt.subplots(1, len(alts), figsize=(w * len(alts), h), sharey=True, squeeze=False)
    x = np.arange(len(params))
    off = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else [0.0]
    for ax, alt in zip(axes[0], alts):
        for i, m in enumerate(models):
            d = rel[(rel.model_name == m) & (rel.alt == alt)]
            stats = d.groupby("param")["value"].agg(mean="mean", lo=lambda v: np.percentile(v, 16), hi=lambda v: np.percentile(v, 84))
            stats = stats.reindex(params)
            yerr = [np.clip(stats["mean"] - stats["lo"], 0, None), np.clip(stats["hi"] - stats["mean"], 0, None)]
            ax.errorbar(x + off[i], stats["mean"], yerr=yerr,
                        fmt="o", ms=3, capsize=2, linewidth=0.8, color=model_color(m, i), label=tex(display(m)))
        ax.axhline(0, color="0.5", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([tex(PARAM_LABELS.get(p, p)) for p in params], rotation=90)
        ax.set_xlim(-0.6, len(params) - 0.4)
        ax.set_title(tex(alt))                                  # "minus base" is said in the paper caption
        ax.grid(True, axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Utility offset")
    axes[0, -1].legend(loc="upper right", fontsize=FONT_SIZE - 3, frameon=False, handletextpad=0.3, borderaxespad=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{stem}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_z_offset] {p}")


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
