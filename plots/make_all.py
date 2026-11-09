# plots/make_all.py — run every extract for a dataset's experiments, then every fig script.
"""Two stages, each skippable:

    python plots/make_all.py --dataset SM --exps SM_tspline_tuned SM_mnl_tuned SM_fdnn_tuned --splits 0-99
    python plots/make_all.py --dataset SM --figs-only
    python plots/make_all.py --dataset SM --extract-only --exps ...

Stage 1 (extract, needs torch + checkpoints): asu_curves for every experiment; heter_params / z_offset / tau /
vot / zsu for the ones whose model supports them (others are skipped with a note). Stage 2 (figs, csv only):
runs each plots/fig_*.py with its parameter block overridden to DATASET, so the per-figure defaults still
come from the files themselves. Edit those files for anything beyond the dataset.
    --figs fig_asucf fig_tau          only these fig modules
    --only overlay_TRAIN_TT legend    only the figures whose output file name contains one of these (sets ONLY)
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

EXTRACTS_ALL = ["asu_curves", "vot_individual", "z_offset"]              # any torch model with the asu_* interface
EXTRACTS_TSPLINE = ["heter_params", "tau_individual"]                     # TSplineDCM only
EXTRACTS_ZSPLINE = ["zsu_curves"]                                         # TSplineDCM with asc_learner: zspline
FIGS = ["fig_asucf", "fig_mu_panel", "fig_vot_heatmap", "fig_vot_hist", "fig_tau", "fig_z_offset", "fig_zsu",
        "fig_feature_table"]
FIGS_SYN = ["fig_syn_folds", "fig_syn_groups", "fig_syn_individual", "asc_table"]   # synthetic/, SYN datasets only


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def _backend(yaml: Path) -> str:
    import yaml as _yaml
    sys.path.insert(0, str(ROOT))
    from run import load_model_spec

    exp = _yaml.safe_load(yaml.read_text(encoding="utf-8")) or {}
    return str(load_model_spec(exp["model"]).get("backend", "torch"))


def extract(dataset: str, exps: list[str], splits: str, ind_splits: str, individual_split: int | None,
            dataset_config: str | None = None, n_individuals: int | None = 20000) -> None:
    for e in exps:
        yaml = ROOT / "configs" / "exp" / f"{e}.yaml"
        if not yaml.exists():
            print(f"[make_all] no {yaml}; skip")
            continue
        backend = _backend(yaml)
        if backend == "rumboost":                                        # explicit utilities, no torch
            _run([PY, str(HERE / "extract" / "rumboost_curves.py"), str(yaml), "--splits", splits,
                  "--vot-splits", ind_splits, "--n-samples", str(n_individuals or 5000)])
            continue
        if backend != "torch":                                           # biogeme / lightgbm: tables only
            print(f"[make_all] {e}: backend {backend} has no curve extract; tables only")
            continue
        for name in EXTRACTS_ALL:
            s = ind_splits if name == "vot_individual" else splits
            extra = ["--individual-split", str(individual_split)] if name == "asu_curves" and individual_split is not None else []
            if name == "vot_individual" and n_individuals:
                extra += ["--n-samples", str(n_individuals)]
            _run([PY, str(HERE / "extract" / f"{name}.py"), str(yaml), "--splits", s, *extra])
        for name in EXTRACTS_TSPLINE + EXTRACTS_ZSPLINE:
            s = ind_splits if name == "tau_individual" else splits
            extra = ["--n-samples", str(n_individuals)] if name == "tau_individual" and n_individuals else []
            rc = _run([PY, str(HERE / "extract" / f"{name}.py"), str(yaml), "--splits", s, *extra])
            if rc:
                print(f"[make_all] {name} not applicable to {e} (exit {rc}); continuing")
    _run([PY, str(HERE / "extract" / "feature_table.py"), "--dataset", dataset_config or dataset,
          "--out-dir", str(ROOT / "figures_data" / dataset)])
    if dataset.startswith("SYN"):                                        # truth curves + per-group curves (synthetic/extract_syn.py)
        syn = str(ROOT / "synthetic" / "extract_syn.py")
        gs = str(individual_split if individual_split is not None else -1)
        yamls = [ROOT / "configs" / "exp" / f"{e}.yaml" for e in exps]
        yamls = [y for y in yamls if y.exists()]
        if not yamls:
            _run([PY, syn, "--dataset", dataset])
        for i, y in enumerate(yamls):
            _run([PY, syn, str(y), "--splits", splits, "--group-split", gs, *(["--skip-truth"] if i else [])])


def figs(dataset: str, only: list[str] | None, only_names: list[str] | None = None) -> None:
    """Draw the fig modules `only` (None = all for the dataset); `only_names` = substrings of the output file name
    (each module's ONLY), None = every figure of each module."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "synthetic"))
    for name in (only or (FIGS + FIGS_SYN if dataset.startswith("SYN") else FIGS)):
        mod = importlib.import_module(name)
        mod.DATASET = dataset
        mod.DATA = ROOT / "figures_data" / dataset
        mod.OUT = ROOT / "figures" / dataset
        mod.ONLY = only_names                    # always set: the module may be reused for another call
        try:
            mod.main()
        except SystemExit as ex:
            print(f"[make_all] {name}: {ex}")
        except FileNotFoundError as ex:
            print(f"[make_all] {name}: missing input ({ex.filename}); run the extract first")
        except Exception as ex:  # one broken figure must not stop the others
            print(f"[make_all] {name} FAILED: {type(ex).__name__}: {str(ex)[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="extract + draw everything for one dataset")
    ap.add_argument("--dataset", required=True,
                    help="results / figures_data / figures dir name = the splits name when the runs pinned one "
                         "(e.g. LPMC_officialtest), else the dataset config name")
    ap.add_argument("--dataset-config", default=None,
                    help="configs/datasets/<name>.yaml for the feature table when it differs from --dataset")
    ap.add_argument("--exps", nargs="*", default=[], help="configs/exp/<name>.yaml stems to extract from")
    ap.add_argument("--splits", default="0-99", help="splits for the curve-type extracts")
    ap.add_argument("--ind-splits", default="0", help="splits for the individual-level extracts (vot, tau)")
    ap.add_argument("--individual-split", type=int, default=0,
                    help="also dump each model's per-individual AS-UCF curves of this split (npz for the spaghetti); -1 = off")
    ap.add_argument("--n-individuals", type=int, default=20000,
                    help="cap on sampled individuals for vot/tau (keeps figures_data small); 0 = every row")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--figs-only", action="store_true")
    ap.add_argument("--figs", nargs="*", default=None, help="subset of fig_* modules")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substrings of the output file name: draw only the matching figures of each fig module")
    a = ap.parse_args()
    if not a.figs_only:
        extract(a.dataset, a.exps, a.splits, a.ind_splits, None if a.individual_split < 0 else a.individual_split,
                a.dataset_config, a.n_individuals or None)
    if not a.extract_only:
        figs(a.dataset, a.figs, a.only)


if __name__ == "__main__":
    main()
