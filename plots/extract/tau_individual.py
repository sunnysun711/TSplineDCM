# plots/extract/tau_individual.py — TSplineDCM taste index per individual and attribute (posterior mean + sigma) -> csv.
"""tau_p = tanh(mu_p(z) - ema + sigma_p(z) * eps) is the per-individual scale of the increments of attribute p.
This writes the deterministic part per individual, so a fig script can (a) plot tanh(mu) by group, or
(b) re-sample eps in numpy to show the posterior spread (as the old SwissmetroTauAnalysis notebook did).

    python plots/extract/tau_individual.py configs/exp/SM_tspline.yaml --splits 0

Writes figures_data/<dataset>/
    tau_individual.csv.gz  long: model_name, split, idx, attr, mu, sigma, tau   (attr = dcfg.x_names entry, e.g. TT or TT-CAR)
    individuals.csv      the fixed batch (idx, z columns, observed attributes) for grouping
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
    out_dir_for, split_list, write_individuals,
)


def extract(exp, splits, seed=42, n_samples=None, results_root=None, out_dir=None) -> Path:
    tt, e = load_prepared(exp, results_root, seed)
    dataset, model_name = tt.results_name, model_name_of(e)   # results/figures dir: splits name (+ _tag) when pinned
    ds, model, dcfg = tt.ds, tt.model, tt.dcfg
    if not hasattr(model, "taste_encoder"):
        raise SystemExit(f"{model_name} has no taste_encoder (tau is a TSplineDCM quantity)")
    out = out_dir_for(dataset, out_dir)
    batch = fixed_batch(ds, n_samples, seed)
    write_individuals(batch, dcfg, out)
    idx = batch["idx"].detach().cpu().numpy()
    p0 = next(model.parameters())
    z = batch["z"].to(device=p0.device, dtype=p0.dtype)

    rows = []
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="tau"):
            mu, logvar = model.taste_encoder(z)
            if hasattr(model, "tau_mean_ema"):
                mu = mu - model.tau_mean_ema
            mu = mu.detach().cpu().numpy().astype(np.float64)
            sigma = (0.5 * logvar).exp().detach().cpu().numpy().astype(np.float64)
            for j, attr in enumerate(dcfg.x_names):
                rows.append(pd.DataFrame({"model_name": model_name, "split": sid, "idx": idx, "attr": attr,
                                          "mu": mu[:, j], "sigma": sigma[:, j], "tau": np.tanh(mu[:, j])}))
    return merge_csv(out / "tau_individual.csv.gz", pd.concat(rows, ignore_index=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="per-individual taste index -> figures_data/tau_individual.csv")
    add_common_args(ap, default_splits="0")
    ap.add_argument("--n-samples", type=int, default=None, help="individuals in the fixed batch (default: all rows)")
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, n_samples=a.n_samples,
            results_root=a.results_root, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
