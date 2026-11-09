# plots/fig_mu_panel.py — the I-spline story in one column: AS-UCF, its slope (AS-MUF), and where heterogeneity sits.
"""Stacked panels per model and (alt, attr):
    (1) utility contribution  U(x): general (tau=0) + mean over individuals + 16-84 band across splits
    (2) marginal utility     dU/dx: same three, from mu_curves_agg.csv (gradient taken per split in the extract)
    (3) heterogeneity per basis at the Greville abscissae (HETER_STAT), one dot per split + the mean, plus the knot
        lines — TSplineDCM models only; every other model gets the two-panel version.
On the synthetic dataset the DGP truth is drawn on top of every panel (red dashed): the population curve, its slope,
and the truth heterogeneity profile on the truth's own bases.
Input: figures_data/<DATASET>/{asu_curves_agg, mu_curves_agg}.csv (+ heter_params, heter_knots, heter_meta for
TSplineDCM; + syn_truth_curves / syn_truth_heter for the truth).

HETER_STAT:
    "k_abs_a_per_x"  k_g |a_g| / (xi_g - xi_{g-1}): heterogeneity of the SLOPE near basis g per unit of x. Comparable
                     across knot vectors (and therefore with the truth, whose knots differ from the model's).
    "k_abs_a"        k_g |a_g| per basis (an increment scales with the knot spacing, so heights depend on the knots)
    "k"              k_g alone (dimensionless; where heterogeneity is allowed, not how much slope it moves)

Edit the parameter block, then:   python plots/fig_mu_panel.py
Only some figures (substrings of the output file name):   python plots/fig_mu_panel.py --dataset SM --only tspline_CAR
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
MODELS = None                       # None = every model in asu_curves_agg.csv; or ["TSplineDCM-ispline"]
PAIRS = None                        # None = all (alt, attr); or [("CAR", "TT")]
HETER_STAT = "k_abs_a_per_x"        # "k_abs_a_per_x" | "k_abs_a" | "k"   (see the docstring)
TRUTH = "auto"                      # "auto" = draw the DGP truth when syn_truth_curves.csv exists; True / False to force
KNOT_SPLIT = 0                      # whose knots to draw as dotted lines; None = none
FIG_SIZE = (3.4, 6.0)               # three-panel size; the two-panel version is 2/3 as tall
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
from style import PALETTE, cli, display, latexify, line_style, tex, wanted, x_unit, y_lim  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET
TRUTH_STYLE = dict(color="#D32F2F", linestyle="--", linewidth=1.0, zorder=5)


def _sel(df, m, alt, attr):
    return df[(df.model_name == m) & (df.alt == alt) & (df.attr == attr)]


def _curve_panel(ax, g, ylabel):
    x = g["x"].to_numpy()
    if g["n_splits"].iloc[0] > 1:
        ax.fill_between(x, g["lo"], g["hi"], color=PALETTE[0], alpha=0.18, linewidth=0, label=tex("16-84% splits"))
    ax.plot(x, g["mean"], color=PALETTE[0], linewidth=1.6, label="Mean individual")
    if np.isfinite(g["general"]).all():
        ax.plot(x, g["general"], color="red", linewidth=1.0, label="General")
    ax.set_ylabel(ylabel)


def _read(name, optional=False):
    p = DATA / name
    if not p.exists():
        if optional:
            return None
        raise SystemExit(f"missing {p}; run the extract first")
    return pd.read_csv(p)


def heter_value(h: pd.DataFrame) -> np.ndarray:
    """HETER_STAT of one spline's bases (rows of one split, or the truth), in basis order."""
    h = h.sort_values("g") if "g" in h else h.sort_values("xi")
    k, a, xi = h["k"].to_numpy(dtype=float), h["a"].to_numpy(dtype=float), h["xi"].to_numpy(dtype=float)
    if HETER_STAT == "k":
        return k
    if HETER_STAT == "k_abs_a":
        return k * np.abs(a)
    if HETER_STAT == "k_abs_a_per_x":
        dxi = np.diff(xi, prepend=xi[0] - (xi[1] - xi[0]) if len(xi) > 1 else 1.0)
        return k * np.abs(a) / np.maximum(dxi, 1e-9)
    raise SystemExit(f"unknown HETER_STAT {HETER_STAT!r}")


def _heter_label() -> str:
    return {"k": r"$k_g$", "k_abs_a": r"$k_g\,|a_g|$", "k_abs_a_per_x": r"$k_g\,|a_g| / \Delta\xi_g$"}[HETER_STAT]


def _truth_tables():
    want = TRUTH if TRUTH != "auto" else (DATA / "syn_truth_curves.csv").exists()
    if not want:
        return None, None
    return _read("syn_truth_curves.csv"), _read("syn_truth_heter.csv", True)


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    asu, mu = _read("asu_curves_agg.csv"), _read("mu_curves_agg.csv")
    hp, knots, meta = _read("heter_params.csv", True), _read("heter_knots.csv", True), _read("heter_meta.csv", True)
    t_curves, t_heter = _truth_tables()
    models = list(MODELS) if MODELS else list(dict.fromkeys(asu.model_name))
    spline_models = set(hp.model_name) if hp is not None else set()
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        sub = asu[asu.model_name == m]
        pairs = [tuple(p) for p in PAIRS] if PAIRS else list(dict.fromkeys(zip(sub.alt, sub.attr)))
        three = m in spline_models
        for alt, attr in pairs:
            stem = f"{DATASET}_mupanel_{m}_{alt}_{attr}"
            if not wanted(stem, ONLY):
                continue
            g_u, g_m = _sel(asu, m, alt, attr).sort_values("x"), _sel(mu, m, alt, attr).sort_values("x")
            if g_u.empty or g_m.empty:
                continue
            tc = t_curves[(t_curves.alt == alt) & (t_curves.attr == attr)].sort_values("x") if t_curves is not None else None
            w, h = FIG_SIZE
            if three:
                fig, (a1, a2, a3) = plt.subplots(3, 1, sharex=True, figsize=(w, h), height_ratios=[3, 3, 2])
            else:
                fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(w, h * 2 / 3), height_ratios=[1, 1])
                a3 = None
            probe = line_style(m).get("linestyle") == "--"
            _curve_panel(a1, g_u, "Utility contribution" + (" (PDP probe)" if probe else ""))
            if tc is not None and not tc.empty:
                a1.plot(tc["x"], tc["general"], label="Truth (population)", **TRUTH_STYLE)
            a1.set_ylim(*y_lim(DATASET))
            a1.legend(loc="best")
            _curve_panel(a2, g_m, "Marginal utility")
            if tc is not None and not tc.empty:
                a2.plot(tc["x"], np.gradient(tc["general"].to_numpy(dtype=float), tc["x"].to_numpy(dtype=float)),
                        label="Truth", **TRUTH_STYLE)
            a2.axhline(0, color="0.5", linewidth=0.5)
            tail = f"{alt} - {attr.split('-')[0]}{x_unit(DATASET, attr)}"
            if three:
                h_ = _sel(hp, m, alt, attr)
                parts = []
                for s, hs in h_.groupby("split"):
                    hs = hs.sort_values("g")
                    parts.append(pd.DataFrame({"g": hs["g"].to_numpy(), "xi": hs["xi"].to_numpy(), "v": heter_value(hs)}))
                hv = pd.concat(parts, ignore_index=True)
                a3.scatter(hv["xi"], hv["v"], s=8, color=PALETTE[1], alpha=0.5, linewidths=0, zorder=3)
                hm = hv.groupby("g").agg(xi=("xi", "mean"), v=("v", "mean"))
                a3.plot(hm["xi"], hm["v"], color=PALETTE[1], linewidth=1.0, zorder=4, label="Mean over splits")
                if t_heter is not None:
                    th = t_heter[(t_heter.alt == alt) & (t_heter.attr == attr)]
                    if not th.empty:
                        th = th.sort_values("g")
                        a3.plot(th["xi"], heter_value(th), marker="x", markersize=3.5, label="Truth", **TRUTH_STYLE)
                if KNOT_SPLIT is not None and knots is not None:
                    kk = knots[(knots.model_name == m) & (knots.alt == alt) & (knots.attr == attr) & (knots.split == KNOT_SPLIT)]
                    for t in kk["knot"]:
                        a3.axvline(t, color="0.7", linestyle=":", linewidth=0.5, zorder=1)
                a3.set_ylabel(_heter_label())
                a3.set_ylim(bottom=0)
                a3.legend(loc="best")
                mrow = meta[(meta.model_name == m) & (meta.alt == alt) & (meta.attr == attr)] if meta is not None else None
                tgt = mrow["heter_target"].iloc[0] if mrow is not None and not mrow.empty else "?"
                a3.set_xlabel(tex(f"{display(m)} ({tgt}): {tail}"))
            else:
                a2.set_xlabel(tex(f"{display(m)}: {tail}"))
            fig.tight_layout()
            p = OUT / f"{stem}.{FMT}"
            fig.savefig(p)
            plt.close(fig)
            print(f"[fig_mu_panel] {p}")


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
