# plots/extract/feature_table.py — which variables enter which part of each model -> csv.
"""Reads only the configs (no checkpoints). For a dataset, lists per alternative the x attributes that
enter the utility, the z characteristics, and for each model which sub-network / term consumes them.

    python plots/extract/feature_table.py --dataset SM --models tspline fdnn tastenet mnl mnl_biogeme

Writes figures_data/<dataset>/feature_table.csv
    columns: dataset, model, alt, x_attrs, z_role, z_names, notes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plots.extract._common import out_dir_for  # noqa: E402
from run import load_dataset_config, load_model_spec  # noqa: E402

# how each model family uses the individual characteristics z (documented, not inferred from code)
Z_ROLE = {
    "tspline": "beta-VAE encoder -> taste index tau (scales I-spline increments); linear/zspline ASC head",
    "tastenet": "taste heads beta_i(z) per alternative (multiplies x); ASC",
    "taste_dcm": "encoder -> taste factor (scales linear utility); ASC",
    "fdnn": "concatenated with all x as DNN input (no utility structure)",
    "mnl": "linear ASC head: ASC_i + z @ gamma_i",
    "mnl_biogeme": "socio-demographic dummies in the utility spec (see train_biogeme.py builder)",
    "l_mnl": "DNN on z adds a per-alternative term to the linear utility",
    "reslogit": "residual layers on top of the linear utility",
    "gbdt": "flat features together with x (multiclass LightGBM, no utility structure)",
    "rumboost": "one unconstrained tree ensemble per z inside each utility (a non-linear ASC); attributes: one monotone ensemble each",
    "mxl_biogeme": "socio-demographic dummies (identified spec); normal random coefficients on time / cost, panel by ID",
}


def extract(dataset: str, models: list[str], out_dir=None) -> Path:
    dcfg = load_dataset_config(dataset)
    out = out_dir_for(dataset, out_dir)
    rows = []
    for m in models:
        spec = load_model_spec(m)
        role = Z_ROLE.get(m, "?")
        for alt in dcfg.alt_names:
            rows.append({"dataset": dataset, "model": m, "backend": spec.get("backend", "torch"), "alt": alt,
                         "x_attrs": " + ".join(dcfg.alt_xs[alt]), "z_role": role, "z_names": " ".join(dcfg.z_names),
                         "notes": ("monotone: " + ", ".join(f"{x}:{dcfg.mono_dirs.get(x.split('-')[0], dcfg.mono_dirs.get(x, '?'))}"
                                                          for x in dcfg.alt_xs[alt])) if m.startswith("tspline") else ""})
    df = pd.DataFrame(rows)
    p = out / "feature_table.csv"
    df.to_csv(p, index=False, lineterminator="\n")
    print(f"[extract] wrote {p} ({len(df)} rows)")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="feature allocation table -> figures_data/feature_table.csv")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", default=["tspline", "mnl", "mnl_biogeme", "mxl_biogeme", "tastenet", "fdnn",
                                                    "taste_dcm", "l_mnl", "reslogit", "gbdt", "rumboost"])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    extract(a.dataset, a.models, a.out_dir)


if __name__ == "__main__":
    main()
