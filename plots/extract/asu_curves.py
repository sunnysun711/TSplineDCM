# plots/extract/asu_curves.py — AS-UCF curves (utility contribution vs attribute) and their slopes, per split -> csv.
"""Reload every split's checkpoint, evaluate each model's asu_* interface on a common x grid,
and write the curves. Also writes their derivative (the AS-MUF, dU/dx) — np.gradient is taken
PER SPLIT before aggregating (percentiles do not commute with the gradient).

    python plots/extract/asu_curves.py configs/exp/SM_tspline.yaml --splits 0-99
    python plots/extract/asu_curves.py --dataset SM --model fdnn --splits 0-99
    python plots/extract/asu_curves.py ... --individual-split 0      # also dump the spaghetti (N x P) of one split

Writes figures_data/<dataset>/
    asu_curves.csv.gz    long: model_name, alt, attr, attr_slot, split, x, mean, general   (gzip: ~1M rows per dataset)
                         mean    = average of the sampled individuals' curves (tau at its posterior mean)
                         general = the population curve (tau = 0); NaN for probe models (FDNN, ...)
    asu_curves_agg.csv   per (model_name, alt, attr, x): mean, lo, hi (16/84 pct over splits), general, n_splits
    mu_curves.csv.gz / mu_curves_agg.csv     same layout for dU/dx
    attr_samples.csv     the sampled batch's attribute values (for the histogram panel), once per dataset
    asu_individual_<model_name>_split<k>.npz   optional: per-individual curves of one split
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
    add_common_args, alt_attr_pairs, band, experiment_from, fixed_batch, iter_loaded_splits,
    load_prepared, merge_csv, model_name_of, out_dir_for, split_list,
)
from plots.style import attr_slot  # noqa: E402

KEYS = ["model_name", "alt", "attr", "attr_slot"]


def _aggregate(long: pd.DataFrame) -> pd.DataFrame:
    """per (model, alt, attr, x): mean of per-split means, 16/84 band, mean general, n_splits."""
    rows = []
    for key, g in long.groupby(KEYS, sort=False):
        piv = g.pivot(index="split", columns="x", values="mean")          # (S, P)
        xs = piv.columns.to_numpy(dtype=float)
        M = piv.to_numpy(dtype=float)
        lo, hi = band(M) if len(M) > 1 else (M[0], M[0])
        gen = g.pivot(index="split", columns="x", values="general").to_numpy(dtype=float)
        gen_mean = np.nanmean(gen, axis=0) if np.isfinite(gen).any() else np.full(len(xs), np.nan)
        rows.append(pd.DataFrame({**dict(zip(KEYS, key)), "x": xs, "mean": M.mean(0), "lo": lo, "hi": hi,
                                  "general": gen_mean, "n_splits": len(M)}))
    return pd.concat(rows, ignore_index=True)


def extract(exp, splits, seed=42, n_samples=5000, n_points=200, extend_left_to_zero=True,
            results_root=None, out_dir=None, individual_split=None) -> dict[str, Path]:
    tt, e = load_prepared(exp, results_root, seed)
    dataset, model_name = tt.results_name, model_name_of(e)   # results/figures dir: splits name (+ _tag) when pinned
    ds, model, dcfg, dev = tt.ds, tt.model, tt.dcfg, tt.device
    pairs = alt_attr_pairs(dcfg)
    out = out_dir_for(dataset, out_dir)

    batch = fixed_batch(ds, n_samples, seed)
    x_evals: dict[tuple[str, str], torch.Tensor] = {}
    for alt, x in pairs:
        x_min, x_max = model.x_bounds_from_dataset(ds, alt, x)
        x_evals[(alt, x)] = torch.linspace(0.0 if extend_left_to_zero else float(x_min), float(x_max), n_points, device=dev)
    has_general = hasattr(model, "asu_has_general_trend") and model.asu_has_general_trend()

    parts: list[pd.DataFrame] = []
    spaghetti: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="asu"):
            for alt, x in pairs:
                xe = x_evals[(alt, x)]
                y = model.asu_individual_curves(batch=batch, alt_name=alt, x_name=x, x_eval=xe, sample_on=False)
                y = y.detach().cpu().numpy().astype(np.float64)                     # (N, P)
                gen = (model.asu_general_trend(alt_name=alt, x_name=x, x_eval=xe).detach().cpu().numpy().astype(np.float64)
                       if has_general else np.full(len(xe), np.nan))
                parts.append(pd.DataFrame({"model_name": model_name, "alt": alt, "attr": x, "attr_slot": attr_slot(x),
                                           "split": sid, "x": xe.detach().cpu().numpy().astype(np.float64),
                                           "mean": y.mean(0), "general": gen}))
                if individual_split is not None and sid == individual_split:
                    spaghetti[f"{alt}__{x}"] = y.astype(np.float32)
                    spaghetti[f"x__{alt}__{x}"] = xe.detach().cpu().numpy().astype(np.float32)
    long = pd.concat(parts, ignore_index=True)

    # derivatives per split, then aggregate
    mu_parts = []
    for key, g in long.groupby(KEYS + ["split"], sort=False):
        g = g.sort_values("x")
        xs = g["x"].to_numpy()
        d = g.copy()
        d["mean"] = np.gradient(g["mean"].to_numpy(), xs)
        d["general"] = np.gradient(g["general"].to_numpy(), xs) if np.isfinite(g["general"]).all() else np.nan
        mu_parts.append(d)
    mu_long = pd.concat(mu_parts, ignore_index=True)

    paths = {
        "asu": merge_csv(out / "asu_curves.csv.gz", long),          # per split: big -> gzip
        "asu_agg": merge_csv(out / "asu_curves_agg.csv", _aggregate(long)),
        "mu": merge_csv(out / "mu_curves.csv.gz", mu_long),
        "mu_agg": merge_csv(out / "mu_curves_agg.csv", _aggregate(mu_long)),
    }
    # attribute values of the shared batch (histogram panel) — dataset-level, model-independent
    samples = pd.concat([
        pd.DataFrame({"alt": alt, "attr": x, "attr_slot": attr_slot(x),
                      "value": batch["x"][dcfg.alt_names.index(alt)][:, dcfg.alt_xs[alt].index(x)].detach().cpu().numpy()})
        for alt, x in pairs], ignore_index=True)
    samples.to_csv(out / "attr_samples.csv", index=False, lineterminator="\n")
    paths["samples"] = out / "attr_samples.csv"
    if spaghetti:
        p = out / f"asu_individual_{model_name}_split{individual_split}.npz"
        np.savez_compressed(p, **spaghetti)
        print(f"[extract] wrote {p} ({len(spaghetti) // 2} pairs, N={n_samples})")
        paths["individual"] = p
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="AS-UCF / AS-MUF curves per split -> figures_data csv")
    add_common_args(ap)
    ap.add_argument("--n-samples", type=int, default=5000, help="individuals in the fixed batch")
    ap.add_argument("--n-points", type=int, default=200, help="x grid points")
    ap.add_argument("--no-extend-left", action="store_true", help="start the x grid at the data min instead of 0")
    ap.add_argument("--individual-split", type=int, default=None, help="dump per-individual curves of this split (npz)")
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, n_samples=a.n_samples, n_points=a.n_points,
            extend_left_to_zero=not a.no_extend_left, results_root=a.results_root, out_dir=a.out_dir,
            individual_split=a.individual_split)


if __name__ == "__main__":
    main()
