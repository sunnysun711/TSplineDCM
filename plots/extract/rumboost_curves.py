# plots/extract/rumboost_curves.py — RUMBoost utilities (stairs or PCUF splines) per split -> the same csvs the
# torch extracts write, so fig_asucf / fig_mu_panel / fig_vot_* / fig_z_offset / fig_zsu draw RUMBoost next to the others.
"""Reads each run's rumboost_utility.json (written by train_boost.RUMBoostTrainer: explicit one-dimensional stairs
anchored at x = 0, or PCHIP knots for the PCUF variant) — no lightgbm needed except for the functional-effect
variant, whose z ensemble has interactions and is evaluated through its saved boosters (as a slice: one z varies,
the others at 0).

    python plots/extract/rumboost_curves.py configs/exp/SM_rumboost.yaml --splits 0-99

Writes figures_data/<results_name>/
    asu_curves.csv.gz / asu_curves_agg.csv     mean == general (no taste heterogeneity), curve(0) = 0
    mu_curves.csv.gz  / mu_curves_agg.csv      PCUF variant only (the stairs have no derivative)
    vot_individual.csv.gz                       PCUF variant only, at each sampled individual's own x (--vot-splits, default 0)
    z_offset.csv                                ASC (utility at an all-zero input) + one row per BINARY z: f(1) - f(0)
                                                (not for rumboost_fe: its z ensemble has interactions)
    zsu_curves.csv.gz / zsu_curves_agg.csv      continuous z: offset curve over the z range, anchored at its left end
    rumboost_splits.csv                         split thresholds per (alt, attr, split): the "knots" of the stairs
    attr_samples.csv, individuals.csv.gz        written if missing (same fixed batch as the torch extracts)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plots.extract._common import (  # noqa: E402
    add_common_args, band, experiment_from, merge_csv, model_name_of, out_dir_for, split_list, time_cost_attrs,
)
from plots.extract.asu_curves import _aggregate  # noqa: E402
from plots.style import attr_slot  # noqa: E402
from run import load_experiment  # noqa: E402
from train_boost import RUMBoostTrainer, eval_piece  # noqa: E402


def _fixed_idx(n_rows: int, n_samples: int | None, seed: int) -> np.ndarray:
    import torch  # same draw as plots/extract/_common.fixed_batch

    torch.manual_seed(seed)
    if n_samples is None or n_samples >= n_rows:
        return np.arange(n_rows)
    return torch.randperm(n_rows)[:n_samples].numpy()


def _boosters(run_dir: Path):
    import lightgbm as lgb

    d = json.loads((run_dir / "rumboost_model.json").read_text(encoding="utf-8"))
    return [lgb.Booster(model_str=s) for s in d["boosters"]]


def _slice(ens: dict, boosters, col: str, xs: np.ndarray, best_it: int | None) -> np.ndarray:
    """Utility of a 'trees' ensemble along one column, every other column at 0 (a conditional slice)."""
    X = pd.DataFrame(np.zeros((len(xs), len(ens["columns"]))), columns=ens["columns"])
    X[col] = xs
    b = boosters[int(ens["booster"])]
    return b.predict(X, raw_score=True, num_iteration=best_it)


def _curve(ens: dict, boosters, col: str, xs: np.ndarray, best_it) -> np.ndarray:
    if ens["kind"] == "pieces":
        return eval_piece(ens["pieces"][col], xs)
    return _slice(ens, boosters, col, xs, best_it)


def _zero_utility(ens: dict, boosters, best_it) -> float:
    if ens["kind"] == "pieces":
        return float(ens["const"]) + float(sum(eval_piece(pc, np.array([0.0]))[0] for pc in ens["pieces"].values()))
    X = pd.DataFrame(np.zeros((1, len(ens["columns"]))), columns=ens["columns"])
    return float(boosters[int(ens["booster"])].predict(X, raw_score=True, num_iteration=best_it)[0])


def extract(exp, splits, seed=42, n_samples=5000, n_points=200, results_root=None, out_dir=None,
            vot_splits=None) -> dict[str, Path]:
    vot_splits = set(splits if vot_splits is None else vot_splits)      # per-individual VoT only for these (default: split 0)
    plan = load_experiment(exp, out_root=results_root)
    tt = plan["trainer"]
    if not isinstance(tt, RUMBoostTrainer):
        raise SystemExit(f"{plan['experiment']['model']} is not a rumboost experiment")
    e = plan["experiment"]
    dataset, model_name = tt.results_name, model_name_of(e)
    dcfg, X = tt.dcfg, tt.X
    out = out_dir_for(dataset, out_dir)
    idx = _fixed_idx(len(X), n_samples, seed)

    # x grids: [0, max] like asu_curves (extend_left_to_zero)
    grids = {(alt, x): np.linspace(0.0, float(X[c].max()), n_points) for alt in dcfg.alt_names
             for c, x in zip(tt.x_cols[alt], tt.x_names[alt])}
    binary_z = {c for c in tt.z_cols if X[c].nunique() <= 2}
    z_name_of = dict(zip(tt.z_cols, dcfg.z_names))

    asu, mu, zoff, zsu, vot, thr_rows = [], [], [], [], [], []
    n_done = 0
    for sid in splits:
        rd = tt.run_dir(sid, seed)
        uj = rd / "rumboost_utility.json"
        if not uj.exists():
            print(f"[rumboost] skip split {sid}: no {uj}")
            continue
        n_done += 1
        util = json.loads(uj.read_text(encoding="utf-8"))
        best_it = util.get("best_iteration")
        smooth = bool(util.get("smooth"))
        boosters = _boosters(rd) if any(en["kind"] == "trees" for en in util["ensembles"]) else None
        per_alt = {}
        for en in util["ensembles"]:
            per_alt.setdefault(en["alt"], []).append(en)

        for alt in dcfg.alt_names:
            ens_list = per_alt[alt]
            # ---- attributes ----
            for c, x in zip(tt.x_cols[alt], tt.x_names[alt]):
                xs = grids[(alt, x)]
                en = next(en for en in ens_list if c in en["columns"])
                y = _curve(en, boosters, c, xs, best_it)
                y = y - y[0]                                             # anchored at x = 0
                asu.append(pd.DataFrame({"model_name": model_name, "alt": alt, "attr": x, "attr_slot": attr_slot(x),
                                         "split": sid, "x": xs, "mean": y, "general": y}))
                if smooth and en["kind"] == "pieces" and "x_knots" in en["pieces"][c]:
                    mu.append(pd.DataFrame({"model_name": model_name, "alt": alt, "attr": x, "attr_slot": attr_slot(x),
                                            "split": sid, "x": xs, "mean": np.gradient(y, xs), "general": np.gradient(y, xs)}))
                piece = en["pieces"][c] if en["kind"] == "pieces" else None
                if piece is not None:
                    th = piece.get("stairs", piece).get("thresholds", [])
                    thr_rows += [{"model_name": model_name, "alt": alt, "attr": x, "split": sid, "threshold": float(t)} for t in th]
            # ---- individual characteristics (skipped for the functional-effect variant: its z ensemble lets z
            #      interact, so "one z with the others at 0" is not an additive offset and would mislead the figure) ----
            if any(en["kind"] == "trees" for en in ens_list):
                continue
            asc = sum(_zero_utility(en, boosters, best_it) for en in ens_list)
            zoff.append({"model_name": model_name, "split": sid, "alt": alt, "param": "ASC", "value": asc, "masked": False,
                         "note": "utility at an all-zero input"})
            for c in tt.z_cols:
                en = next((en for en in ens_list if c in en["columns"]), None)
                if en is None:
                    continue
                if c in binary_z:
                    v = _curve(en, boosters, c, np.array([0.0, 1.0]), best_it)
                    zoff.append({"model_name": model_name, "split": sid, "alt": alt, "param": z_name_of[c],
                                 "value": float(v[1] - v[0]), "masked": False, "note": "step f(1) - f(0)"})
                else:
                    lo, hi = float(X[c].min()), float(X[c].max())
                    zs = np.linspace(lo, hi, n_points)
                    v = _curve(en, boosters, c, zs, best_it)
                    zsu.append(pd.DataFrame({"model_name": model_name, "z": z_name_of[c], "alt": alt, "split": sid,
                                             "z_value": zs, "y": v - v[0]}))
            # ---- VoT at each sampled individual's own x (PCUF only) ----
            t_attr, c_attr = time_cost_attrs(dcfg, alt)
            if smooth and t_attr and c_attr and sid in vot_splits:
                i_alt = dcfg.alt_names.index(alt)
                cols = dict(zip(tt.x_names[alt], tt.x_cols[alt]))
                obs, slope = {}, {}
                for x in (t_attr, c_attr):
                    xs = np.linspace(0.0, float(X[cols[x]].max()), 1000)
                    en = next(en for en in ens_list if cols[x] in en["columns"])
                    dy = np.gradient(_curve(en, boosters, cols[x], xs, best_it), xs)
                    obs[x] = X[cols[x]].to_numpy(dtype=np.float64)[idx]
                    slope[x] = np.interp(obs[x], xs, dy)
                with np.errstate(divide="ignore", invalid="ignore"):
                    vot.append(pd.DataFrame({"model_name": model_name, "alt": alt, "time_attr": t_attr, "cost_attr": c_attr,
                                             "split": sid, "idx": idx, "x_time": obs[t_attr], "x_cost": obs[c_attr],
                                             "mu_time": slope[t_attr], "mu_cost": slope[c_attr],
                                             "vot": slope[t_attr] / slope[c_attr]}))
                del i_alt
    if n_done == 0:
        raise FileNotFoundError(f"no rumboost_utility.json under {tt.out_root / tt.results_name / tt.model_name}")

    paths: dict[str, Path] = {}
    long = pd.concat(asu, ignore_index=True)
    paths["asu"] = merge_csv(out / "asu_curves.csv.gz", long)
    paths["asu_agg"] = merge_csv(out / "asu_curves_agg.csv", _aggregate(long))
    if mu:
        mu_long = pd.concat(mu, ignore_index=True)
        paths["mu"] = merge_csv(out / "mu_curves.csv.gz", mu_long)
        paths["mu_agg"] = merge_csv(out / "mu_curves_agg.csv", _aggregate(mu_long))
    if zoff:
        paths["z_offset"] = merge_csv(out / "z_offset.csv", pd.DataFrame(zoff))
    if zsu:
        zl = pd.concat(zsu, ignore_index=True)
        agg = []
        for (m, zn, alt), g in zl.groupby(["model_name", "z", "alt"], sort=False):
            piv = g.pivot(index="split", columns="z_value", values="y")
            M = piv.to_numpy(dtype=float)
            lo_, hi_ = band(M) if len(M) > 1 else (M[0], M[0])
            agg.append(pd.DataFrame({"model_name": m, "z": zn, "alt": alt, "z_value": piv.columns.to_numpy(dtype=float),
                                     "mean": M.mean(0), "lo": lo_, "hi": hi_, "n_splits": len(M)}))
        paths["zsu"] = merge_csv(out / "zsu_curves.csv.gz", zl)
        paths["zsu_agg"] = merge_csv(out / "zsu_curves_agg.csv", pd.concat(agg, ignore_index=True))
    if vot:
        paths["vot"] = merge_csv(out / "vot_individual.csv.gz", pd.concat(vot, ignore_index=True))
    if thr_rows:
        paths["splits"] = merge_csv(out / "rumboost_splits.csv", pd.DataFrame(thr_rows))

    # dataset-level files the fig scripts expect (same rows as the torch extracts)
    if not (out / "attr_samples.csv").exists():
        samples = pd.concat([pd.DataFrame({"alt": alt, "attr": x, "attr_slot": attr_slot(x), "value": X[c].to_numpy()[idx]})
                             for alt in dcfg.alt_names for c, x in zip(tt.x_cols[alt], tt.x_names[alt])], ignore_index=True)
        samples.to_csv(out / "attr_samples.csv", index=False, lineterminator="\n")
        paths["samples"] = out / "attr_samples.csv"
    if not (out / "individuals.csv.gz").exists():
        ind = pd.DataFrame({"idx": idx, "ID": tt.df.iloc[idx, dcfg.ID_col].to_numpy()})
        for c, zn in zip(tt.z_cols, dcfg.z_names):
            ind[zn] = X[c].to_numpy()[idx]
        for alt in dcfg.alt_names:
            for c, x in zip(tt.x_cols[alt], tt.x_names[alt]):
                ind[f"{alt}__{x}"] = X[c].to_numpy()[idx]
        av = tt.av[idx]
        ind["av"] = [" ".join(a for a, m in zip(dcfg.alt_names, row) if m) for row in av]
        ind.to_csv(out / "individuals.csv.gz", index=False, lineterminator="\n", float_format="%.6g")
        paths["individuals"] = out / "individuals.csv.gz"
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="RUMBoost utilities per split -> figures_data csv")
    add_common_args(ap)
    ap.add_argument("--n-samples", type=int, default=5000, help="individuals in the fixed batch (VoT / samples)")
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--vot-splits", default="0", help="splits whose per-individual VoT rows are written (big table)")
    a = ap.parse_args()
    extract(experiment_from(a), split_list(a.splits), seed=a.seed, n_samples=a.n_samples, n_points=a.n_points,
            results_root=a.results_root, out_dir=a.out_dir, vot_splits=split_list(a.vot_splits))


if __name__ == "__main__":
    main()
