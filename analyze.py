# analyze.py — summarize a benchmark (mean +/- SEM, paired tests, params, time) and export paper tables.
"""Summarize a benchmark comparison and run paired significance tests.

Every run writes metrics.json under $TSDCM_OUT/{dataset}/{model_name}/split{nnn}_seed{s}/.
Pairing is by split_id (same fixed data split -> same draw), so a paired t-test / Wilcoxon
is valid and far more powerful than an unpaired comparison.

    python analyze.py --dataset SM                       # print table + paired tests vs tspline
    python analyze.py --dataset SM --ref tspline --metric test_nll
    python analyze.py --dataset SM --export              # write _summary/ (csv + tex)
    python analyze.py --dataset SM --results-dir /other/root     # e.g. another results root, for cross-checking

Export (in {results-dir}/{dataset}/_summary/):
    per_split.csv      one row per (model, split, seed): all test_/val_ metrics + n_params
    summary.csv        per model x metric: count/mean/sem/std/median/min/max (+ n_params, hpo seconds/trials)
    paired_tests.csv   paired-by-split tests of every model vs --ref, one row per (metric, model)
    main_table.tex     paper main table: mean +/- SEM of test metrics, n_params, fit seconds, HPO seconds
    ablation_table.tex the tspline variants (tspline_level / tspline_homo / ...) — never in the main table
    hpo_table.tex      each model's search space + winner (from hpo/*/trials.json)
    compute_table.tex  per-model fit seconds and HPO seconds
    summary.json       everything above, machine-readable

The main table takes the tspline row ONLY from model_name == "tspline"; any other run of the
tspline code (level / homo / ...) goes to the ablation table; and the main tspline row must have
SP_heter_target == increment with a non-zero k cap, otherwise this script refuses to export.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trainer_base import DEFAULT_OUT_ROOT, collect_results

try:
    from scipy import stats
except ImportError:  # scipy optional; table still prints, tests are skipped
    stats = None

LOWER_IS_BETTER = ("test_nll", "test_brier", "val_nll", "val_brier")
TABLE_METRICS = ("test_nll", "test_acc", "test_f1", "test_brier", "test_gmpca", "test_rho2")
TSPLINE_KEY = "tspline"           # registry name of the proposed model
MAIN_TSPLINE_NAME = "tspline"     # the ONE model_name allowed to represent it in the main table


def _sem(a: np.ndarray) -> float:
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


# --------------------------------------------------------------------------- #
# classification of runs
# --------------------------------------------------------------------------- #
def _is_ablation(row: pd.Series) -> bool:
    """A run of the tspline code under any model_name other than the canonical one."""
    return str(row.get("model_key", "")) == TSPLINE_KEY and str(row["model"]) != MAIN_TSPLINE_NAME


def _check_main_tspline(df: pd.DataFrame) -> None:
    main = df[(df["model"] == MAIN_TSPLINE_NAME)]
    if main.empty:
        return
    if "heter_target" in main and not (main["heter_target"].fillna("increment") == "increment").all():
        bad = main[main["heter_target"] != "increment"]["split_id"].tolist()
        raise RuntimeError(f"main-table tspline rows with SP_heter_target != increment (splits {bad[:5]}...) - "
                           "the paper model is the I-spline variant; run the level variant under model_name tspline_level")
    if "k_all_zero" in main and main["k_all_zero"].fillna(False).astype(bool).any():
        raise RuntimeError("main-table tspline rows have k_gpi_cap all zero (no heterogeneity) - that is the "
                           "tspline_homo ablation, not the paper model")


# --------------------------------------------------------------------------- #
# paired tests
# --------------------------------------------------------------------------- #
def _paired_rows(df: pd.DataFrame, ref: str, metric: str) -> list[dict]:
    lower_is_better = metric in LOWER_IS_BETTER
    base = df[df.model == ref].set_index("split_id")[metric]
    rows: list[dict] = []
    for m in sorted(df["model"].unique()):
        if m == ref:
            continue
        other = df[df.model == m].set_index("split_id")[metric]
        common = base.index.intersection(other.index)
        if len(common) < 2:
            rows.append({"model": m, "n_pairs": int(len(common))})
            continue
        b, o = base.loc[common].to_numpy(), other.loc[common].to_numpy()
        wins = int((o < b).sum()) if lower_is_better else int((o > b).sum())
        row = {"model": m, "n_pairs": int(len(common)), "ref_mean": float(b.mean()), "model_mean": float(o.mean()),
               "delta": float(o.mean() - b.mean()), "ref_wins": f"{len(common) - wins}/{len(common)}"}
        if stats is not None:
            row["t_pvalue"] = float(stats.ttest_rel(o, b).pvalue)
            try:
                row["wilcoxon_p"] = float(stats.wilcoxon(o, b).pvalue)
            except ValueError:  # all-zero differences
                row["wilcoxon_p"] = float("nan")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# HPO logs
# --------------------------------------------------------------------------- #
def _hpo_info(results_root: Path, dataset: str, model_name: str) -> dict:
    p = results_root / "hpo" / dataset / model_name / "trials.json"
    if not p.exists():
        # results name = splits name (+ _tag), HPO dir = dataset name: LPMC_officialtest -> hpo/LPMC (longest prefix)
        hpo = results_root / "hpo"
        cands = sorted((d.name for d in hpo.iterdir() if d.is_dir() and dataset.startswith(d.name)), key=len) if hpo.exists() else []
        p = hpo / cands[-1] / model_name / "trials.json" if cands else p
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "hpo_trials": int(d.get("n_trials") or 0),
        "hpo_seconds": float(d.get("total_fit_seconds") or 0.0),
        "hpo_sampler": d.get("sampler"),
        "hpo_best": (d.get("best") or {}).get("params"),
        "hpo_space": d.get("search_space") or [],
    }


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def summarize(dataset: str, ref: str = MAIN_TSPLINE_NAME, metric: str = "test_nll",
              results_root: Path = Path(DEFAULT_OUT_ROOT)) -> pd.DataFrame:
    df = collect_results(out_root=results_root, dataset=dataset)
    if df.empty:
        print(f"[analyze] no results under {results_root}/{dataset}/ yet")
        return df
    metric_cols = [c for c in TABLE_METRICS if c in df]
    agg = df.groupby("model")[metric_cols].agg(["mean", "sem"])
    agg.insert(0, ("n", ""), df.groupby("model").size())
    if "n_params" in df and df["n_params"].notna().any():
        agg.insert(1, ("n_params", ""), df.groupby("model")["n_params"].median())
    print(f"\n=== {dataset}: per-model summary ({len(df)} runs) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}", "display.width", 220):
        print(agg)

    lower = metric in LOWER_IS_BETTER
    print(f"\n=== paired tests vs ref={ref!r} on {metric} ({'lower' if lower else 'higher'} is better) ===")
    if ref not in df["model"].unique():
        print(f"[analyze] ref model {ref!r} not found; available: {sorted(df['model'].unique())}")
        return df
    rows = []
    for r in _paired_rows(df, ref, metric):
        if r["n_pairs"] < 2:
            print(f"  {r['model']:20s} only {r['n_pairs']} paired split(s) - need >=2, skipped")
            continue
        row = {"model": r["model"], "n_pairs": r["n_pairs"], f"mean_{metric}": r["model_mean"],
               f"delta_vs_{ref}": r["delta"], "ref_wins": r["ref_wins"]}
        row.update({k: r[k] for k in ("t_pvalue", "wilcoxon_p") if k in r})
        rows.append(row)
    if rows:
        with pd.option_context("display.float_format", lambda x: f"{x:.4g}", "display.width", 220):
            print(pd.DataFrame(rows).to_string(index=False))
        print(f"\n  delta_vs_{ref} > 0 means the baseline is worse than {ref} on {metric} "
              f"({'lower' if lower else 'higher'} better); ref_wins = splits where {ref} beats the baseline.")
    return df


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _fmt(m: float, s: float, nd: int = 4) -> str:
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def _tex_table(rows: list[dict], cols: list[tuple[str, str]], caption: str, label: str) -> str:
    """rows: list of {col_key: str}; cols: [(key, header)]."""
    head = " & ".join(h for _, h in cols) + r" \\"
    body = "\n".join(" & ".join(str(r.get(k, "")) for k, _ in cols) + r" \\" for r in rows)
    return ("\\begin{table}[htbp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n\\toprule\n" + head + "\n\\midrule\n" + body
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def export_summary(dataset: str, ref: str = MAIN_TSPLINE_NAME, out_dir: str | Path | None = None,
                   results_root: Path = Path(DEFAULT_OUT_ROOT)) -> Path | None:
    df = collect_results(out_root=results_root, dataset=dataset)
    if df.empty:
        print(f"[analyze] no results under {results_root}/{dataset}/ yet")
        return None
    _check_main_tspline(df)
    out = Path(out_dir) if out_dir is not None else results_root / dataset / "_summary"
    out.mkdir(parents=True, exist_ok=True)

    metric_cols = [c for c in df.columns
                   if (c.startswith(("test_", "val_")) and not c.endswith("_n")) or c in ("fit_seconds", "n_params")]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]

    # 1. tidy per-run table
    per_split = df.drop(columns=["path"]).sort_values(["model", "split_id", "init_seed"])
    per_split.to_csv(out / "per_split.csv", index=False)

    # 2. per-model aggregation (+ HPO cost)
    agg = df.groupby("model")[metric_cols].agg(["count", "mean", "sem", "std", "median", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    models = list(agg["model"])
    hpo = {m: _hpo_info(results_root, dataset, m) for m in models}
    agg["hpo_trials"] = [hpo[m].get("hpo_trials", 0) for m in models]
    agg["hpo_seconds"] = [hpo[m].get("hpo_seconds", 0.0) for m in models]
    abl = {m for m in models if _is_ablation(df[df.model == m].iloc[0])}
    agg["model_key"] = [df.loc[df.model == m, "model_key"].iloc[0] for m in models]
    agg["ablation"] = [m in abl for m in models]
    agg.to_csv(out / "summary.csv", index=False)

    # 3. paired tests vs ref, every test_ metric
    paired_rows: list[dict] = []
    test_metrics = [c for c in metric_cols if c.startswith("test_")]
    if ref in df["model"].unique():
        for metric in test_metrics:
            for r in _paired_rows(df, ref, metric):
                paired_rows.append({"metric": metric, "better": "lower" if metric in LOWER_IS_BETTER else "higher",
                                    "ref": ref, **r})
    else:
        print(f"[analyze] ref model {ref!r} not found; paired_tests.csv will be empty")
    pd.DataFrame(paired_rows).to_csv(out / "paired_tests.csv", index=False)

    # 4. LaTeX: main / ablation / hpo / compute
    def _row(m: str) -> dict:
        g = df[df.model == m]
        r = {"model": m.replace("_", r"\_"), "n": str(len(g))}
        for c in TABLE_METRICS:
            if c in g and g[c].notna().any():
                r[c] = _fmt(g[c].mean(), _sem(g[c].to_numpy()))
        r["n_params"] = f"{int(g['n_params'].median()):,}" if "n_params" in g and g["n_params"].notna().any() else ""
        r["fit_seconds"] = f"{g['fit_seconds'].mean():.1f}"
        r["hpo_seconds"] = f"{hpo[m].get('hpo_seconds', 0.0):.0f}" if hpo[m] else "--"
        r["hpo_trials"] = str(hpo[m].get("hpo_trials", 0)) if hpo[m] else "--"
        return r

    main_models = [m for m in models if m not in abl]
    cols = [("model", "Model"), ("test_nll", "NLL"), ("test_acc", "Acc."), ("test_f1", "F1"), ("test_brier", "Brier"),
            ("test_gmpca", "GMPCA"), ("test_rho2", r"$\rho^2$"), ("n_params", "\\#params"), ("fit_seconds", "Fit (s)"),
            ("hpo_seconds", "HPO (s)")]
    cols = [(k, h) for k, h in cols if k in ("model", "n_params", "fit_seconds", "hpo_seconds") or k in df]
    n_splits = df["split_id"].nunique()
    (out / "main_table.tex").write_text(
        _tex_table([_row(m) for m in main_models], cols,
                   f"{dataset}: test-set performance, mean $\\pm$ SEM over {n_splits} splits.", f"tab:{dataset}_main"),
        encoding="utf-8")
    abl_models = sorted((abl | {MAIN_TSPLINE_NAME}) & set(models))
    (out / "ablation_table.tex").write_text(
        _tex_table([_row(m) for m in abl_models], cols, f"{dataset}: TSplineDCM variants (ablation).",
                   f"tab:{dataset}_ablation"), encoding="utf-8")
    hpo_rows = []
    for m in models:
        for e in hpo[m].get("hpo_space", []):
            rng = (f"[{e.get('lo')}, {e.get('hi')}]" if "lo" in e else str(e.get("choices"))).replace("_", r"\_")
            best = (hpo[m].get("hpo_best") or {}).get(e["param"], "")
            hpo_rows.append({"model": m.replace("_", r"\_"), "param": e["param"].replace("_", r"\_"),
                             "type": e["type"], "range": rng, "best": str(best).replace("_", r"\_")})
    (out / "hpo_table.tex").write_text(
        _tex_table(hpo_rows, [("model", "Model"), ("param", "Hyperparameter"), ("type", "Sampling"),
                              ("range", "Range / choices"), ("best", "Selected")],
                   f"{dataset}: hyperparameter search spaces and selected values (TPE, split 0 val NLL).",
                   f"tab:{dataset}_hpo"), encoding="utf-8")
    (out / "compute_table.tex").write_text(
        _tex_table([_row(m) for m in models], [("model", "Model"), ("n_params", "\\#params"),
                                               ("fit_seconds", "Fit (s / split)"), ("hpo_trials", "HPO trials"),
                                               ("hpo_seconds", "HPO (s)")],
                   f"{dataset}: computational cost.", f"tab:{dataset}_compute"), encoding="utf-8")

    # 5. machine-readable bundle
    bundle = {
        "dataset": dataset, "ref": ref, "results_root": str(results_root),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "n_runs": int(len(df)),
        "models": {m: {"n_runs": int((df.model == m).sum()), "ablation": m in abl,
                       "splits": sorted(df.loc[df.model == m, "split_id"].astype(int).tolist()),
                       "hpo": {k: v for k, v in hpo[m].items() if k != "hpo_space"},
                       "metrics": {c: {"mean": float(g[c].mean()), "sem": _sem(g[c].dropna().to_numpy()),
                                       "std": float(g[c].std(ddof=1)) if len(g) > 1 else 0.0,
                                       "median": float(g[c].median()), "min": float(g[c].min()), "max": float(g[c].max())}
                                   for c in metric_cols if g[c].notna().any()}}
                   for m, g in df.groupby("model")},
        "paired_tests": paired_rows,
    }
    (out / "summary.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"[analyze] wrote {out}/: per_split.csv, summary.csv, paired_tests.csv, main_table.tex, "
          f"ablation_table.tex, hpo_table.tex, compute_table.tex, summary.json ({len(df)} runs, {df['model'].nunique()} models"
          f"{', ablation: ' + ', '.join(sorted(abl)) if abl else ''})")
    return out


def main() -> None:
    global MAIN_TSPLINE_NAME
    ap = argparse.ArgumentParser(description="Summarize + paired-test a benchmark comparison")
    ap.add_argument("--dataset", required=True, help="e.g. SM")
    ap.add_argument("--ref", default=None, help="reference model_name for paired tests (default: --main-name)")
    ap.add_argument("--main-name", default=MAIN_TSPLINE_NAME,
                    help="model_name of the paper TSplineDCM run in this results dir (default tspline)")
    ap.add_argument("--metric", default="test_nll", help="test_nll | test_acc | test_gmpca | test_rho2 | test_brier")
    ap.add_argument("--results-dir", default=str(DEFAULT_OUT_ROOT), help="results root (default $TSDCM_OUT)")
    ap.add_argument("--export", action="store_true", help="write _summary/ (csv + tex)")
    ap.add_argument("--export-dir", default=None, help="override export directory")
    args = ap.parse_args()
    root = Path(args.results_dir)
    MAIN_TSPLINE_NAME = args.main_name           # every tspline run under another name is an ablation row
    if args.ref is None:
        args.ref = args.main_name
    summarize(args.dataset, ref=args.ref, metric=args.metric, results_root=root)
    if args.export or args.export_dir:
        export_summary(args.dataset, ref=args.ref, out_dir=args.export_dir, results_root=root)


if __name__ == "__main__":
    main()
