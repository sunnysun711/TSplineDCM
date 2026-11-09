# plots/extract/vot_individual.py — individual value of time at each person's OWN observed (time, cost) -> csv.
"""For every alternative with a time and a cost attribute: evaluate each sampled individual's AS-UCF on a dense
grid (the model's asu_individual_curves, tau at its posterior mean), differentiate per individual, read the slope
at that individual's observed x, and take VoT = mu_time / mu_cost (both negative -> positive VoT).

Works for every model with the asu_* interface (TSplineDCM, TasteNet, MNL, TasteDCM, FDNN-probe), so the
same figure can compare models. No filtering here: the fig script decides the valid range (parameter block).

    python plots/extract/vot_individual.py configs/exp/SM_tspline.yaml --splits 0        # one split, all individuals
    python plots/extract/vot_individual.py configs/exp/SM_tspline.yaml --splits 0-99     # all splits (big csv)

Writes figures_data/<dataset>/
    vot_individual.csv.gz  long: model_name, alt, time_attr, cost_attr, split, idx, x_time, x_cost, mu_time, mu_cost, vot
                         (raw model units; convert with plots.style.vot_unit in the fig script)
    individuals.csv      the fixed batch (idx, ID, z columns, observed attributes) for joins/grouping
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
    add_common_args, experiment_from, fixed_batch, iter_loaded_splits, load_prepared, merge_csv, model_name_of,
    out_dir_for, slope_at_points, split_list, time_cost_attrs, write_individuals,
)


def extract(exp, splits, seed=42, n_samples=None, n_points=1000, results_root=None, out_dir=None) -> Path:
    tt, e = load_prepared(exp, results_root, seed)
    dataset, model_name = tt.results_name, model_name_of(e)   # results/figures dir: splits name (+ _tag) when pinned
    ds, model, dcfg, dev = tt.ds, tt.model, tt.dcfg, tt.device
    out = out_dir_for(dataset, out_dir)
    batch = fixed_batch(ds, n_samples, seed)
    write_individuals(batch, dcfg, out)
    idx = batch["idx"].detach().cpu().numpy()

    pairs = []
    for alt in dcfg.alt_names:
        t_attr, c_attr = time_cost_attrs(dcfg, alt)
        if t_attr and c_attr:
            pairs.append((alt, t_attr, c_attr))
    if not pairs:
        raise SystemExit(f"{dataset}: no alternative has both a time and a cost attribute")
    grids = {}
    for alt, t_attr, c_attr in pairs:
        for x in (t_attr, c_attr):
            lo, hi = model.x_bounds_from_dataset(ds, alt, x)
            grids[(alt, x)] = torch.linspace(float(lo), float(hi), n_points, device=dev)

    rows = []
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="vot"):
            for alt, t_attr, c_attr in pairs:
                i_alt = dcfg.alt_names.index(alt)
                mu = {}
                for x in (t_attr, c_attr):
                    xs = grids[(alt, x)]
                    y = model.asu_individual_curves(batch=batch, alt_name=alt, x_name=x, x_eval=xs, sample_on=False)
                    x_obs = batch["x"][i_alt][:, dcfg.alt_xs[alt].index(x)].detach().cpu().numpy().astype(np.float64)
                    mu[x] = (x_obs, slope_at_points(y.detach().cpu().numpy().astype(np.float64),
                                                    xs.detach().cpu().numpy().astype(np.float64), x_obs))
                (x_t, mu_t), (x_c, mu_c) = mu[t_attr], mu[c_attr]
                with np.errstate(divide="ignore", invalid="ignore"):
                    vot = mu_t / mu_c
                rows.append(pd.DataFrame({"model_name": model_name, "alt": alt, "time_attr": t_attr, "cost_attr": c_attr,
                                          "split": sid, "idx": idx, "x_time": x_t, "x_cost": x_c,
                                          "mu_time": mu_t, "mu_cost": mu_c, "vot": vot}))
    df = pd.concat(rows, ignore_index=True)
    return merge_csv(out / "vot_individual.csv.gz", df)


def main() -> None:
    ap = argparse.ArgumentParser(description="individual VoT at observed points -> figures_data/vot_individual.csv")
    add_common_args(ap, default_splits="0")
    ap.add_argument("--n-samples", type=int, default=None, help="individuals in the fixed batch (default: all rows)")
    ap.add_argument("--n-points", type=int, default=1000, help="dense grid for the per-individual derivative")
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, n_samples=a.n_samples, n_points=a.n_points,
            results_root=a.results_root, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
