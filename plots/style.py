# plots/style.py — shared, torch-free styling + label tables for the paper figures.
"""Everything a fig_*.py needs besides its csv: matplotlib rc, palette, axis units, y-limits,
value-of-time unit factors, purpose labels, the "structural vs probe" line-style policy, and the two hooks that let a
fig script draw only some of its figures: `wanted(stem, ONLY)` (name filter, checked before anything is drawn) and
`cli(globals())` (--dataset / --only / --out from the command line, written back into the script's parameter block).
No model code is imported here, so the fig scripts stay runnable without torch.
"""
from __future__ import annotations

import os
from pathlib import Path


def wanted(stem: str, only) -> bool:
    """Draw the figure whose output file is `stem`.pdf? ONLY = None draws everything; otherwise any entry hits:
    a plain entry as a substring ("overlay_TRAIN"), an entry starting with "=" only as the whole name
    ("=SM_vot_heatmap_tspline" does not hit SM_vot_heatmap_tspline_level)."""
    return not only or any(stem == s[1:] if s.startswith("=") else s in stem for s in only)


def cli(g: dict) -> None:
    """Command-line overrides of a fig script's parameter block (call as `cli(globals())` in its __main__).
    --dataset X   = DATASET (DATA / OUT follow);  --only a b   = ONLY (substrings of the output file name);
    --out DIR     = OUT_DIR.  With no arguments nothing changes, so `python plots/fig_x.py` behaves as before."""
    import argparse

    ap = argparse.ArgumentParser(description="draw this figure type; defaults come from the parameter block")
    ap.add_argument("--dataset", default=None, help="figures_data / figures dir name (parameter DATASET)")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of the output file name (parameter ONLY)")
    ap.add_argument("--out", default=None, help="output directory (parameter OUT_DIR)")
    a = ap.parse_args()
    if a.dataset:
        g["DATASET"] = a.dataset
        g["DATA"] = g["ROOT"] / "figures_data" / a.dataset
        g["OUT"] = Path(g["OUT_DIR"]) if g.get("OUT_DIR") else g["ROOT"] / "figures" / a.dataset
    if a.out:
        g["OUT_DIR"] = a.out
        g["OUT"] = Path(a.out)
    if a.only:
        g["ONLY"] = list(a.only)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

# display name per results-directory name (model_name); anything else is shown as-is
DISPLAY = {
    "tspline": "TSplineDCM", "tspline_level": "TSplineDCM (B-spline)", "tspline_homo": "TSplineDCM (no heter.)",
    "TSplineDCM-ispline": "TSplineDCM", "TSplineDCM-level": "TSplineDCM (B-spline)",
    "tspline_n40": "TSplineDCM (40 knots)", "tspline_n40p": "TSplineDCM (40 knots, pen.)",
    "tspline_k1n60": "TSplineDCM (k=1, 60 knots)", "tspline_k1n60p": "TSplineDCM (k=1, 60 knots, pen.)",
    "tspline_k1n100p": "TSplineDCM (k=1, 100 knots, pen.)",
    "mnl": "MNL", "MNL": "MNL", "mnl_biogeme": "MNL (biogeme)", "MNLBiogeme": "MNL (biogeme)",
    "fdnn": "F-DNN", "FDNN": "F-DNN", "tastenet": "TasteNet", "TasteNet": "TasteNet",
    "taste_dcm": "TasteDCM", "TasteDCM": "TasteDCM", "l_mnl": "L-MNL", "reslogit": "ResLogit",
    "gbdt": "LightGBM", "rumboost": "RUMBoost", "rumboost_pcuf": "RUMBoost (PCUF)",
    "rumboost_nomono": "RUMBoost (no mono.)", "rumboost_noz": "RUMBoost (no z)", "rumboost_fe": "RUMBoost (FE)",
    "mxl_biogeme": "Mixed logit",
    "truth": "DGP truth",
}

# models whose "AS-UCF" is a post-hoc construct, not a structural population utility: a partial-dependence PROBE
# (FDNN, ...; DCM2 docs/explainability_plots.md) or the mean of individual curves of a model with no population curve
# (TasteNet: beta(z) * x, asu_has_general_trend() is False). Drawn dashed so they are never mistaken for a
# learned population utility.
PROBE_MODELS = {"fdnn", "FDNN", "reslogit", "gbdt", "tb_resnet", "cdnn", "tastenet", "TasteNet"}

# results-dir names -> one canonical key per model family (the v1 reference dirs use CamelCase names)
CANON = {
    "tspline": "tspline", "TSplineDCM-ispline": "tspline", "TSplineDCM": "tspline",
    "tspline_level": "tspline_level", "TSplineDCM-level": "tspline_level", "tspline_homo": "tspline_homo",
    "tspline_n40": "tspline_n40", "tspline_n40p": "tspline_n40p", "tspline_k1n60": "tspline_k1n60",
    "tspline_k1n60p": "tspline_k1n60p", "tspline_k1n100p": "tspline_k1n100p",
    "mnl": "mnl", "MNL": "mnl", "mnl_biogeme": "mnl_biogeme", "MNLBiogeme": "mnl_biogeme", "mxl_biogeme": "mxl_biogeme",
    "fdnn": "fdnn", "FDNN": "fdnn", "tastenet": "tastenet", "TasteNet": "tastenet", "taste_dcm": "taste_dcm", "TasteDCM": "taste_dcm",
    "l_mnl": "l_mnl", "reslogit": "reslogit", "gbdt": "gbdt", "truth": "truth",
    "rumboost": "rumboost", "rumboost_pcuf": "rumboost_pcuf", "rumboost_nomono": "rumboost_nomono",
    "rumboost_noz": "rumboost_noz", "rumboost_fe": "rumboost_fe",
}

# one colour per model family, the same in every figure (no two models share a colour in one overlay)
MODEL_COLOR = {
    "tspline": "#4C72B0", "tspline_level": "#64B5CD", "tspline_homo": "#2A4D7F",
    "tspline_n40": "#E6A100", "tspline_n40p": "#B8860B", "tspline_k1n60": "#00A6A6", "tspline_k1n60p": "#006D6D",
    "tspline_k1n100p": "#7B1FA2",
    "rumboost": "#C44E52", "rumboost_pcuf": "#DD8452", "rumboost_nomono": "#17BECF", "rumboost_noz": "#BCBD22",
    "rumboost_fe": "#000000",
    "fdnn": "#8C8C8C", "reslogit": "#937860", "gbdt": "#555555",
    "mnl": "#55A868", "mnl_biogeme": "#2E7D4F", "mxl_biogeme": "#98D189",
    "tastenet": "#8172B3", "taste_dcm": "#B39DDB", "l_mnl": "#DA8BC3", "truth": "#D62728",
}

# which models go on the shared axes when a fig script's MODELS is None (single-model figures still cover all)
OVERLAY_GROUPS = {
    "main": ["tspline", "rumboost_pcuf", "fdnn", "tastenet", "mnl", "reslogit"],
    "rumboost": ["rumboost", "rumboost_pcuf", "rumboost_nomono", "rumboost_noz", "rumboost_fe"],
    # knot-selection study (SM): the retuned baseline, its many-knot / degree-1 variants, and the tree reference
    "knots": ["tspline", "tspline_n40", "tspline_n40p", "tspline_k1n60", "tspline_k1n60p", "tspline_k1n100p", "rumboost_pcuf"],
}


def canon(model_name: str) -> str:
    return CANON.get(model_name, model_name)


def display(model_name: str) -> str:
    return DISPLAY.get(model_name, model_name)


def model_color(model_name: str, i: int = 0) -> str:
    """Fixed colour of a model family; unknown names fall back to the palette by position."""
    return MODEL_COLOR.get(canon(model_name), PALETTE[i % len(PALETTE)])


def line_style(model_name: str) -> dict:
    return {"linestyle": "--", "color": model_color(model_name)} if model_name in PROBE_MODELS else {"linestyle": "-"}


def select_overlay(models, group: str = "main") -> list:
    """The models of `group` that are present in `models`, in the group's drawing order."""
    want = OVERLAY_GROUPS[group]
    by_key = {}
    for m in models:
        by_key.setdefault(canon(m), m)
    return [by_key[k] for k in want if k in by_key]


# ---- axis units / limits per dataset (v1 explain._get_unit_x / _get_ylim) ----
def attr_slot(x_name: str) -> str:
    """'TT-TRAIN' -> 'TT' (drop the per-alternative suffix used when tastes are split by alt)."""
    return x_name.split("-")[0]


def x_unit(dataset: str, x_name: str) -> str:
    slot = attr_slot(x_name)
    if dataset.startswith("SM"):
        return " (100 min)" if slot in ("TT", "HE") else (" (100 CHF)" if slot == "CO" else "")
    if dataset.startswith(("LPMC", "SYN")):
        if slot in ("TT", "RailT", "BusT", "AccessT", "TransWaitT", "TransWalkT"):
            return " (h)"
        if slot == "CO":
            return " (GBP)"
    return ""


def y_lim(dataset: str) -> tuple[float | None, float | None]:
    if dataset.startswith("SYN"):
        return (-3.8, 0.1)          # the DGP truth curves reach -3.7 (CAR, tau near 1)
    return (None, 0.1)


# Overlay figures (fig_asucf OVERLAY): one y-range per attribute kind, shared by every panel of that kind, so the
# panels compare directly and a baseline with huge slopes (TasteNet runs to -70 on LPMC walking time) does not
# flatten the useful curves onto the top edge. Curves beyond the range are clipped. Datasets / kinds not listed
# fall back to y_lim(dataset) (auto range per panel). The LPMC time range runs to -10.
Y_LIM_OVERLAY: dict[str, dict[str, tuple[float, float]]] = {
    "LPMC": {"time": (-10.0, 0.5), "cost": (-2.0, 2.0)},
    "SM": {"time": (-10.0, 0.5), "cost": (-6.0, 0.5), "headway": (-1.0, 0.1)},
}
TIME_SLOTS = ("TT", "RailT", "BusT", "AccessT", "TransWaitT", "TransWalkT")      # headway (HE) is its own kind


def y_lim_overlay(dataset: str, x_name: str) -> tuple[float | None, float | None]:
    slot = attr_slot(x_name)
    kind = "time" if slot in TIME_SLOTS else {"CO": "cost", "HE": "headway"}.get(slot, "other")
    for key, lims in Y_LIM_OVERLAY.items():
        if dataset.startswith(key) and kind in lims:
            return lims[kind]
    return y_lim(dataset)


# ---- value of time: (dU/dTT)/(dU/dCO) in the csv's raw units -> money per hour ----
# SM: TT in 100 min, CO in 100 CHF -> ratio is CHF per min -> x60 = CHF/h.  LPMC: TT in h, CO in GBP -> GBP/h.
VOT_UNIT = {"SM": (60.0, "CHF/h"), "LPMC": (1.0, "GBP/h"), "SYN": (1.0, "GBP/h")}


def vot_unit(dataset: str) -> tuple[float, str]:
    for k, v in VOT_UNIT.items():
        if dataset.startswith(k):
            return v
    return (1.0, "money/h")


PURPOSE_LABELS = {1: "Commute", 2: "Shopping", 3: "Business", 4: "Leisure",
                  5: "Return\nwork", 6: "Return\nshopping", 7: "Return\nbusiness", 8: "Return\nleisure"}


def latexify(fontsize: int = 9, usetex: bool | None = None) -> None:
    """Paper-ready rcParams with a LaTeX/serif (Computer Modern) look; falls back to mathtext CM
    when no TeX is installed. PLOT_USETEX=0 forces the fallback (robust to '_' / '%' in labels)."""
    import matplotlib as mpl

    if usetex is None:
        usetex = os.environ.get("PLOT_USETEX", "1") != "0"
    rc = {
        "font.family": "serif", "font.serif": ["Computer Modern Roman"], "font.size": fontsize,
        "axes.titlesize": fontsize, "axes.labelsize": fontsize, "legend.fontsize": fontsize - 2,
        "xtick.labelsize": fontsize - 1, "ytick.labelsize": fontsize - 1, "axes.linewidth": 0.5,
        "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    }
    if usetex:
        try:
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams.update(rc)
            import io
            from matplotlib.figure import Figure  # really render once now, not at savefig time

            probe = Figure()
            probe.text(0.5, 0.5, r"probe $x_1$")
            probe.savefig(io.BytesIO(), format="pdf")       # runs latex; raises when TeX is missing or half-updated
            return
        except Exception as ex:
            mpl.rcParams["text.usetex"] = False
            print(f"[plot] latex unavailable ({type(ex).__name__}); falling back to mathtext CM")
    rc["mathtext.fontset"] = "cm"
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams.update(rc)


def tex(s: str) -> str:
    """Escape '_' and '%' only when usetex is on (z names like PURPOSE_1 would otherwise abort LaTeX)."""
    import matplotlib as mpl

    return s.replace("_", r"\_").replace("%", r"\%") if mpl.rcParams.get("text.usetex") else s
