# synthetic/extract_syn.py — truth curves of the synthetic DGP, and truth-vs-model curves by population group -> csv.
"""Extract layer of the synthetic experiment. Two halves, both write figures_data/SYN/:

TRUTH (numpy only; needs synthetic/syn_truth.json + data/syn_data.csv)
    syn_truth_curves.csv        alt, attr, attr_slot, x, general, mean, q00, q20, q80, q100
                                general = population curve (tau = 0); mean / quantiles over the individuals of the
                                fixed batch (same rows as plots/extract/asu_curves.py: torch.manual_seed(seed); randperm)
    syn_truth_heter.csv         alt, attr, attr_slot, g, xi, k, c, a       (Greville point, k profile, levels, increments)
    syn_truth_knots.csv         alt, attr, knot                            (interior knots incl. both ends)
    syn_truth_asc.csv           alt, asc
    syn_individual_truth.npz    <alt>__<attr>: (N, P) truth curves of the fixed batch; x__<alt>__<attr>; idx

GROUPS (truth, plus ONE model per call when an experiment yaml is given)
    syn_groups.csv              source, group, alt, attr, attr_slot, split, x, mean, general, lo, hi, n
                                source = "truth" (split -1) or the model_name; lo/hi = 16/84 pct over the group's
                                individuals; n = individuals in the group sample
    syn_groups_meta.csv         group, column, op, threshold, n_rows
    syn_groups_individual_<source>.npz   <group>__<alt>__<attr>: (min(n, N_IND), P) individual curves (spaghetti)

    python synthetic/extract_syn.py                                              # truth + truth groups
    python synthetic/extract_syn.py configs/exp/SYN_tspline.yaml --splits 0-4    # + that model's group curves
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plots.extract._common import (  # noqa: E402
    add_common_args, band, experiment_from, iter_loaded_splits, load_prepared, merge_csv, model_name_of, out_dir_for,
    split_list,
)
from plots.style import attr_slot  # noqa: E402
from run import load_dataset_config  # noqa: E402
from synthetic.generate import OUT_CSV, TRUTH_JSON, SynDGP  # noqa: E402

DATASET = "SYN"
# population groups: name -> (csv column, op, "median" | number)
GROUPS: dict[str, tuple[str, str, str | float]] = {
    "st_below": ("start_time_linear", "<", "median"),
    "st_above": ("start_time_linear", ">=", "median"),
    "age_below": ("age", "<", "median"),
    "age_above": ("age", ">=", "median"),
}


def _fixed_idx(n_rows: int, n_samples: int | None, seed: int) -> np.ndarray:
    """Same draw as plots/extract/_common.fixed_batch (so truth and model spaghetti share individuals)."""
    torch.manual_seed(seed)
    if n_samples is None or n_samples >= n_rows:
        return np.arange(n_rows)
    return torch.randperm(n_rows)[:n_samples].numpy()


def _group_masks(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    masks, meta = {}, []
    for g, (col, op, thr) in GROUPS.items():
        t = float(df[col].median()) if thr == "median" else float(thr)
        v = df[col].to_numpy(dtype=float)
        m = v < t if op == "<" else (v >= t if op == ">=" else (v <= t if op == "<=" else v > t))
        masks[g] = m
        meta.append({"group": g, "column": col, "op": op, "threshold": t, "n_rows": int(m.sum())})
    return masks, pd.DataFrame(meta)


def _group_idx(mask: np.ndarray, n_group: int | None, seed: int) -> np.ndarray:
    idx = np.where(mask)[0]
    if n_group is not None and n_group < len(idx):
        idx = np.random.default_rng(seed).permutation(idx)[:n_group]
    return np.sort(idx)


def _pairs(dcfg) -> list[tuple[str, str, str]]:
    """(alt, attr as the model names it e.g. TT-WALK, truth key e.g. WALK_TT)."""
    return [(alt, x, f"{alt}_{attr_slot(x)}") for alt in dcfg.alt_names for x in dcfg.alt_xs[alt]]


def _stats(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = band(Y)
    return Y.mean(0), lo, hi


def extract_truth(df: pd.DataFrame, dgp: SynDGP, truth: dict, dcfg, out: Path, n_samples: int, n_points: int,
                  seed: int) -> dict[str, np.ndarray]:
    """Writes the truth csvs + the fixed-batch npz. Returns the x grids per (alt, attr)."""
    idx = _fixed_idx(len(df), n_samples, seed)
    curves, heter, knots, npz, grids = [], [], [], {"idx": idx}, {}
    for alt, x, key in _pairs(dcfg):
        xs = np.linspace(0.0, dgp.ranges[key][1], n_points)
        grids[(alt, x)] = xs
        tau = df[f"tau_{key}"].to_numpy(dtype=float)[idx]
        Y = dgp.individual_curves(key, xs, tau)                                   # (N, P)
        q = np.percentile(Y, [0, 20, 80, 100], axis=0)
        curves.append(pd.DataFrame({"alt": alt, "attr": x, "attr_slot": attr_slot(x), "x": xs,
                                    "general": dgp.general_curve(key, xs), "mean": Y.mean(0),
                                    "q00": q[0], "q20": q[1], "q80": q[2], "q100": q[3]}))
        k = dgp.degree[key]
        T = dgp.knots[key]
        heter.append(pd.DataFrame({"alt": alt, "attr": x, "attr_slot": attr_slot(x), "g": np.arange(len(dgp.incr[key])),
                                   "xi": dgp.greville(key), "k": dgp.kcoef[key], "c": dgp.coef[key], "a": dgp.incr[key]}))
        knots.append(pd.DataFrame({"alt": alt, "attr": x, "knot": T[k:-k] if k > 0 else T}))
        npz[f"{alt}__{x}"] = Y.astype(np.float32)
        npz[f"x__{alt}__{x}"] = xs.astype(np.float32)
    pd.concat(curves, ignore_index=True).to_csv(out / "syn_truth_curves.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.concat(heter, ignore_index=True).to_csv(out / "syn_truth_heter.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.concat(knots, ignore_index=True).to_csv(out / "syn_truth_knots.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.DataFrame({"alt": list(truth["asc"]), "asc": list(truth["asc"].values())}).to_csv(out / "syn_truth_asc.csv", index=False, lineterminator="\n")
    np.savez_compressed(out / "syn_individual_truth.npz", **npz)
    print(f"[extract_syn] wrote syn_truth_{{curves,heter,knots,asc}}.csv + syn_individual_truth.npz (N={len(idx)}) in {out}")
    return grids


def extract_truth_groups(df: pd.DataFrame, dgp: SynDGP, dcfg, out: Path, grids, n_group: int | None, n_ind: int,
                         seed: int) -> None:
    masks, meta = _group_masks(df)
    meta.to_csv(out / "syn_groups_meta.csv", index=False, lineterminator="\n")
    rows, npz = [], {}
    for g, mask in masks.items():
        gidx = _group_idx(mask, n_group, seed)
        npz[f"idx__{g}"] = gidx
        for alt, x, key in _pairs(dcfg):
            xs = grids[(alt, x)]
            tau = df[f"tau_{key}"].to_numpy(dtype=float)[gidx]
            Y = dgp.individual_curves(key, xs, tau)
            mean, lo, hi = _stats(Y)
            rows.append(pd.DataFrame({"source": "truth", "group": g, "alt": alt, "attr": x, "attr_slot": attr_slot(x),
                                      "split": -1, "x": xs, "mean": mean, "general": dgp.general_curve(key, xs),
                                      "lo": lo, "hi": hi, "n": len(gidx)}))
            npz[f"{g}__{alt}__{x}"] = Y[:n_ind].astype(np.float32)
            npz[f"x__{g}__{alt}__{x}"] = xs.astype(np.float32)
    merge_csv(out / "syn_groups.csv", pd.concat(rows, ignore_index=True), key="source")
    np.savez_compressed(out / "syn_groups_individual_truth.npz", **npz)
    print(f"[extract_syn] truth group curves: {meta.to_dict('records')}")


def extract_model_groups(exp, splits, out: Path, dgp: SynDGP, grids, n_group: int | None, n_ind: int, seed: int,
                         group_split: int | None, results_root=None) -> None:
    tt, e = load_prepared(exp, results_root, seed)
    model_name = model_name_of(e)
    model, dcfg, dev = tt.model, tt.dcfg, tt.device
    df = tt.df
    masks, _ = _group_masks(df)
    gidx = {g: _group_idx(m, n_group, seed) for g, m in masks.items()}
    has_general = hasattr(model, "asu_has_general_trend") and model.asu_has_general_trend()
    rows, npz = [], {}
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="extract_syn"):
            for g, idx in gidx.items():
                batch = tt.ds.make_batch_from_indices(idx)
                for alt, x, key in _pairs(dcfg):
                    xs = grids[(alt, x)]
                    xe = torch.as_tensor(xs, dtype=torch.float32, device=dev)
                    Y = model.asu_individual_curves(batch=batch, alt_name=alt, x_name=x, x_eval=xe, sample_on=False)
                    Y = Y.detach().cpu().numpy().astype(np.float64)
                    gen = (model.asu_general_trend(alt_name=alt, x_name=x, x_eval=xe).detach().cpu().numpy().astype(np.float64)
                           if has_general else np.full(len(xs), np.nan))
                    mean, lo, hi = _stats(Y)
                    rows.append(pd.DataFrame({"source": model_name, "group": g, "alt": alt, "attr": x, "attr_slot": attr_slot(x),
                                              "split": sid, "x": xs, "mean": mean, "general": gen, "lo": lo, "hi": hi, "n": len(idx)}))
                    if group_split is not None and sid == group_split:
                        npz[f"{g}__{alt}__{x}"] = Y[:n_ind].astype(np.float32)
                        npz[f"x__{g}__{alt}__{x}"] = xs.astype(np.float32)
                        npz[f"idx__{g}"] = idx
    merge_csv(out / "syn_groups.csv", pd.concat(rows, ignore_index=True), key="source")
    if npz:
        p = out / f"syn_groups_individual_{model_name}.npz"
        np.savez_compressed(p, **npz)
        print(f"[extract_syn] wrote {p} (split {group_split}, <= {n_ind} individuals per group)")


def main() -> None:
    ap = argparse.ArgumentParser(description="synthetic truth curves + truth/model curves by group -> figures_data/SYN")
    add_common_args(ap, default_splits="0-4")
    ap.add_argument("--truth", default=str(TRUTH_JSON))
    ap.add_argument("--csv", default=None, help="synthetic csv (default: the one named in the truth json)")
    ap.add_argument("--n-samples", type=int, default=5000, help="fixed batch size (same as asu_curves)")
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--n-group", type=int, default=10000, help="individuals sampled per group (0 = all)")
    ap.add_argument("--n-ind", type=int, default=2000, help="individual curves kept in the group npz")
    ap.add_argument("--group-split", type=int, default=0, help="split whose model group curves go to the npz; -1 = none")
    ap.add_argument("--skip-truth", action="store_true", help="only the model's group curves (truth csvs already there)")
    a = ap.parse_args()

    truth = json.loads(Path(a.truth).read_text(encoding="utf-8"))
    dgp = SynDGP.from_truth(truth)
    dcfg = load_dataset_config(a.dataset or DATASET)
    out = out_dir_for(a.dataset or DATASET, a.out_dir)
    csv = Path(a.csv) if a.csv else (ROOT / truth.get("csv", str(OUT_CSV.relative_to(ROOT))))
    df = pd.read_csv(csv)
    n_group = a.n_group if a.n_group > 0 else None
    grids = {(alt, x): np.linspace(0.0, dgp.ranges[key][1], a.n_points) for alt, x, key in _pairs(dcfg)}
    if not a.skip_truth:
        grids = extract_truth(df, dgp, truth, dcfg, out, a.n_samples, a.n_points, a.seed)
        extract_truth_groups(df, dgp, dcfg, out, grids, n_group, a.n_ind, a.seed)
    if a.experiment or (a.dataset and a.model):
        extract_model_groups(experiment_from(a), split_list(a.splits), out, dgp, grids, n_group, a.n_ind, a.seed,
                             None if a.group_split < 0 else a.group_split, results_root=a.results_root)


if __name__ == "__main__":
    main()
