# plots/extract/z_offset.py — the individual-characteristics offset (ASC + linear z head) per split -> csv.
"""V_offset[:, i] = const[i] + z @ weight[i] for models with a linear z head (MNL, TSplineDCM with
asc_learner=linear); TasteNet reports only the ASC consts (z modulates tastes there).

    python plots/extract/z_offset.py configs/exp/SM_tspline.yaml --splits 0-99

Writes figures_data/<dataset>/z_offset.csv
    long: model_name, split, alt, param ("ASC" or a z name), value, masked (ASC pinned to 0), note
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
    add_common_args, experiment_from, iter_loaded_splits, load_prepared, merge_csv, model_name_of, out_dir_for,
    split_list,
)


def extract(exp, splits, seed=42, results_root=None, out_dir=None) -> Path:
    tt, e = load_prepared(exp, results_root, seed)
    dataset, model_name = tt.results_name, model_name_of(e)   # results/figures dir: splits name (+ _tag) when pinned
    model = tt.model
    if not hasattr(model, "z_offset_params"):
        raise SystemExit(f"{model_name} has no z_offset_params interface")
    out = out_dir_for(dataset, out_dir)
    rows = []
    with torch.no_grad():
        for sid in iter_loaded_splits(tt, splits, seed, tag="z_offset"):
            p = model.z_offset_params()
            if p is None:
                raise SystemExit(f"{model_name}.z_offset_params() returned None - no individual-characteristics offset")
            alts, masked, note = list(p["alt_names"]), set(p.get("masked_alts", [])), str(p.get("note", ""))
            const = np.asarray(p["const"], dtype=np.float64)
            for i, alt in enumerate(alts):
                rows.append({"model_name": model_name, "split": sid, "alt": alt, "param": "ASC", "value": const[i],
                             "masked": alt in masked, "note": note})
            if p["weight"] is not None:
                W = np.asarray(p["weight"], dtype=np.float64)
                for i, alt in enumerate(alts):
                    for j, z in enumerate(p["z_names"]):
                        rows.append({"model_name": model_name, "split": sid, "alt": alt, "param": z, "value": W[i, j],
                                     "masked": alt in masked, "note": note})
    return merge_csv(out / "z_offset.csv", pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="z-head offsets per split -> figures_data/z_offset.csv")
    add_common_args(ap)
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, results_root=a.results_root, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
