# plots/extract/zsu_curves.py — z-spline ASC head curves (utility offset vs one individual characteristic) -> csv.
"""Only for TSplineDCM runs with asc_learner: zspline (LPMC, SYN). For each alternative j and characteristic z_q
the head adds a monotone-or-free spline f_jq(z_q) anchored at 0 on the left. The curve is read off each split's
checkpoint on a dense grid over the head's z range.

    python plots/extract/zsu_curves.py configs/exp/LPMC_tspline.yaml --splits 0-99

Writes figures_data/<dataset>/
    zsu_curves.csv.gz   long: model_name, z, alt, split, z_value, y          (z_value in the model's scaled units)
    zsu_curves_agg.csv  per (model_name, z, alt, z_value): mean, lo, hi, n_splits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.tspline.b_spline import coef2curve  # noqa: E402
from plots.extract._common import (  # noqa: E402
    add_common_args, band, experiment_from, iter_loaded_splits, load_prepared, merge_csv, model_name_of, out_dir_for,
    split_list,
)


def extract(exp, splits, seed=42, z_names=None, n_points=200, results_root=None, out_dir=None) -> dict[str, Path]:
    tt, e = load_prepared(exp, results_root, seed)
    dataset, model_name = tt.results_name, model_name_of(e)   # results/figures dir: splits name (+ _tag) when pinned
    model, dcfg = tt.model, tt.dcfg
    head = getattr(model, "asc_learner", None)
    if head is None or not hasattr(head, "mono_coefs"):
        raise SystemExit(f"{model_name}: asc_learner is {dcfg.asc_learner!r}, not a z-spline head")
    z_names = list(z_names or dcfg.z_names)
    out = out_dir_for(dataset, out_dir)
    # the base alternative (fix_asc_zero) has its whole offset, z-spline head included, multiplied by asc_mask = 0:
    # its head parameters are dead, so they are not written (same rule as z_offset / fig_z_offset)
    mask = model.asc_mask.view(-1).detach().cpu().tolist() if hasattr(model, "asc_mask") else [1.0] * len(dcfg.alt_names)
    live_alts = [(j, a) for j, (a, m) in enumerate(zip(dcfg.alt_names, mask)) if m != 0.0]
    skipped = [a for a, m in zip(dcfg.alt_names, mask) if m == 0.0]
    if skipped:
        print(f"[zsu] {model_name}: {skipped} pinned to 0 (fix_asc_zero); their z-spline heads are not written")

    rows = []
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="zsu"):
            for z_name in z_names:
                q = dcfg.z_names.index(z_name)
                lo, hi = head.z_ranges[q]
                for j, alt in live_alts:
                    mc = head.mono_coefs[j][q]
                    zs = torch.linspace(float(lo), float(hi), n_points, device=mc.grid.device, dtype=mc.grid.dtype).unsqueeze(1)
                    coef = mc().unsqueeze(0).unsqueeze(0)
                    coef = coef - coef[..., :1]
                    coef[..., :1] = 0.0                                   # same left anchor as ZSplineHead.forward
                    y = coef2curve(zs, mc.grid, coef=coef, k=mc.k)[:, 0, 0]
                    rows.append(pd.DataFrame({"model_name": model_name, "z": z_name, "alt": alt, "split": sid,
                                              "z_value": zs[:, 0].detach().cpu().numpy().astype(np.float64),
                                              "y": y.detach().cpu().numpy().astype(np.float64)}))
    long = pd.concat(rows, ignore_index=True)
    agg = []
    for (m, zn, alt), g in long.groupby(["model_name", "z", "alt"], sort=False):
        piv = g.pivot(index="split", columns="z_value", values="y")
        M = piv.to_numpy(dtype=float)
        lo_, hi_ = band(M) if len(M) > 1 else (M[0], M[0])
        agg.append(pd.DataFrame({"model_name": m, "z": zn, "alt": alt, "z_value": piv.columns.to_numpy(dtype=float),
                                 "mean": M.mean(0), "lo": lo_, "hi": hi_, "n_splits": len(M)}))
    return {"raw": merge_csv(out / "zsu_curves.csv.gz", long),
            "agg": merge_csv(out / "zsu_curves_agg.csv", pd.concat(agg, ignore_index=True))}


def main() -> None:
    ap = argparse.ArgumentParser(description="z-spline ASC head curves -> figures_data/zsu_curves.csv")
    add_common_args(ap)
    ap.add_argument("--z-names", nargs="*", default=None, help="subset of dcfg.z_names (default: all)")
    ap.add_argument("--n-points", type=int, default=200)
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, z_names=a.z_names, n_points=a.n_points,
            results_root=a.results_root, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
