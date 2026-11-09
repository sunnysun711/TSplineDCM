# plots/extract/heter_params.py — TSplineDCM spline parameters per split: k_g (heterogeneity), c_g (levels), a_g (increments).
"""For every (alternative, attribute) of a TSplineDCM run, read three learned profiles off each split's model:

    k_g   heterogeneity weight per basis (k_gpi_cap already applied; k_0 = 0 when the left end is anchored)
    c_g   constrained coefficient LEVELS, anchored like SplineUtility.forward_coef (c_0 = 0), all bases
    a_g   INCREMENTS a_g = c_g - c_{g-1} (a_0 = c_0): what the I-spline actually scales; k_g * |a_g| is the
          "heterogeneity of the marginal utility near basis g" bar of the marginal-utility panel

Knot vectors differ per split (grid update), so profiles live on different abscissae. The raw csv keeps them as-is;
the aggregated csv interpolates every split onto one dense grid (np.interp, edge extension) before mean / 16-84 band.

    python plots/extract/heter_params.py configs/exp/SM_tspline.yaml --splits 0-99

Writes figures_data/<dataset>/
    heter_params.csv      long: model_name, alt, attr, attr_slot, split, g, xi, k, c, a
    heter_knots.csv       long: model_name, alt, attr, split, knot                (interior knots incl. both ends)
    heter_params_agg.csv  per (model_name, alt, attr, x): k_mean, k_lo, k_hi, c_mean, c_lo, c_hi, a_mean, a_lo, a_hi, n_splits
    heter_meta.csv        per (model_name, alt, attr): mono, heter_target, heter_mode, spline_degree, keep_first_c_zero
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plots.extract._common import (  # noqa: E402
    add_common_args, alt_attr_pairs, band, experiment_from, iter_loaded_splits, load_prepared, merge_csv,
    model_name_of, out_dir_for, split_list,
)
from plots.style import attr_slot  # noqa: E402

KEYS = ["model_name", "alt", "attr", "attr_slot"]


def _greville_full(t: np.ndarray, k: int) -> np.ndarray:
    """Greville abscissae of ALL G+k basis functions (xi_i = mean(t[i+1 : i+k+1])), untrimmed."""
    n_coef = t.size - k - 1
    if k > 0:
        return np.array([t[i + 1: i + k + 1].mean() for i in range(n_coef)], dtype=np.float64)
    return 0.5 * (t[:-1] + t[1:])


def _anchored_levels(mc, keep_first_c_zero: bool) -> np.ndarray:
    """Constrained levels exactly as SplineUtility.forward_coef anchors them:
    keep_first_c_zero -> degree-0 shifts all by c0, k>0 pins c[0]=0; otherwise untouched."""
    with torch.no_grad():
        c = mc().detach().cpu().numpy().astype(np.float64)
    if not keep_first_c_zero:
        return c
    if mc.k == 0:
        return c - c[:1]
    c = c.copy()
    c[0] = 0.0
    return c


def extract(exp, splits, seed=42, results_root=None, out_dir=None, n_common=200) -> dict[str, Path]:
    tt, e = load_prepared(exp, results_root, seed)
    dataset, model_name = tt.results_name, model_name_of(e)   # results/figures dir: splits name (+ _tag) when pinned
    model, dcfg = tt.model, tt.dcfg
    if not hasattr(model, "asu_heterogeneity_profile") or not hasattr(model, "sp_utils"):
        raise SystemExit(f"{model_name}: not a TSplineDCM (no spline parameters to extract)")
    pairs = alt_attr_pairs(dcfg)
    out = out_dir_for(dataset, out_dir)

    rows, knots, meta = [], [], {}
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="heter"):
            for alt, x in pairs:
                sp, attr_idx, mc = model._resolve_sp(alt, x)
                xi_trim, k_trim, grid_knots = model.asu_heterogeneity_profile(alt_name=alt, x_name=x)
                n_basis = mc.G + mc.k
                k_full = sp.forward_k_gpi()[attr_idx][:n_basis].detach().cpu().numpy().astype(np.float64)
                t = mc.grid.squeeze(0).detach().cpu().numpy().astype(np.float64)
                xi = _greville_full(t, mc.k)                                   # all G+k bases
                c = _anchored_levels(mc, sp.keep_first_c_zero)
                a = np.diff(c, prepend=0.0)                                    # increments, forward_coef's convention
                assert np.allclose(xi[1:-1], xi_trim.detach().cpu().numpy(), atol=1e-6), f"Greville mismatch {alt}/{x} split {sid}"
                assert np.allclose(k_full[1:-1], k_trim.detach().cpu().numpy(), atol=1e-6), f"k_gpi mismatch {alt}/{x} split {sid}"
                rows.append(pd.DataFrame({"model_name": model_name, "alt": alt, "attr": x, "attr_slot": attr_slot(x),
                                          "split": sid, "g": np.arange(n_basis), "xi": xi, "k": k_full, "c": c, "a": a}))
                knots.append(pd.DataFrame({"model_name": model_name, "alt": alt, "attr": x, "split": sid,
                                           "knot": grid_knots.detach().cpu().numpy().astype(np.float64)}))
                meta[(alt, x)] = {"mono": int(mc.mono), "heter_target": sp.heter_target, "heter_mode": sp.heter_mode,
                                  "spline_degree": int(mc.k), "keep_first_c_zero": bool(sp.keep_first_c_zero)}
    long = pd.concat(rows, ignore_index=True)

    # aggregate on a common grid per (alt, attr)
    agg_rows = []
    for key, g in long.groupby(KEYS, sort=False):
        x_max = float(g["xi"].max())
        xc = np.linspace(0.0, x_max, n_common)
        per = {name: [] for name in ("k", "c", "a")}
        for _, s in g.groupby("split"):
            s = s.sort_values("xi")
            for name in per:
                per[name].append(np.interp(xc, s["xi"].to_numpy(), s[name].to_numpy()))
        rec = {**dict(zip(KEYS, key)), "x": xc, "n_splits": g["split"].nunique()}
        for name, arrs in per.items():
            M = np.vstack(arrs)
            lo, hi = band(M) if len(M) > 1 else (M[0], M[0])
            rec.update({f"{name}_mean": M.mean(0), f"{name}_lo": lo, f"{name}_hi": hi})
        agg_rows.append(pd.DataFrame(rec))
    meta_df = pd.DataFrame([{"model_name": model_name, "alt": a_, "attr": x_, "attr_slot": attr_slot(x_), **m}
                            for (a_, x_), m in meta.items()])
    return {
        "raw": merge_csv(out / "heter_params.csv", long),
        "knots": merge_csv(out / "heter_knots.csv", pd.concat(knots, ignore_index=True)),
        "agg": merge_csv(out / "heter_params_agg.csv", pd.concat(agg_rows, ignore_index=True)),
        "meta": merge_csv(out / "heter_meta.csv", meta_df),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="TSplineDCM k_g / c_g / a_g per split -> figures_data csv")
    add_common_args(ap)
    ap.add_argument("--n-common", type=int, default=200, help="common x grid for the aggregated csv")
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, results_root=a.results_root, out_dir=a.out_dir,
            n_common=a.n_common)


if __name__ == "__main__":
    main()
