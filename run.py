# run.py — the one entry point for running experiments over the fixed splits.
"""Run one experiment (dataset x model x splits x seeds), resumable.

    # from an experiment yaml (paper configs live in configs/exp/)
    python run.py configs/exp/SM_tspline_tuned.yaml
    python run.py configs/exp/SM_fdnn_tuned.yaml --splits 0-4 --seeds 42 123 --overwrite

    # without a yaml: dataset yaml + model yaml defaults, splits 0-99, seed 42
    python run.py --dataset SM --model fdnn --splits 0-4
    python run.py --dataset SM --model tspline --model-name tspline_try1

    # change any config item from the command line (dotted path into the experiment overrides;
    # values are parsed as yaml, so numbers / lists / booleans work)
    python run.py --dataset SM --model tspline --splits 0 --set model_config.T_beta=1.0
    python run.py configs/exp/SM_tspline_tuned.yaml --set "dcm_config.k_gpi_cap.TRAIN=[0.5, 0, 0.3]"

Three config layers, composed by an experiment (yaml file or --dataset/--model):

    configs/datasets/<dataset>.yaml   -> DcmConfig fields (which csv column is what, monotone directions, knots)
    configs/models/<model>.yaml       -> backend + model_class + model_config / train_config / benchmark_config
                                         (+ search_space for tune.py)
    experiment (yaml or CLI)          -> dataset + model + model_name + splits + init_seeds + overrides

Experiment yaml schema::

    dataset: SM              # configs/datasets/<dataset>.yaml
    splits_name: SM          # optional, splits/<splits_name>.npz; default = dataset
    model: tspline                    # configs/models/<model>.yaml (= registry name for torch models)
    model_name: tspline               # optional, results dir name; default = model. Variants (tspline_level,
                                      #   tspline_homo) differ ONLY here + in overrides, never in `model`.
    splits: 0-99                      # "a-b" | [0, 1, 2] | int n (= range(n))
    init_seeds: [42]
    out_root: /path/to/results        # optional; default $TSDCM_OUT (or ./results)
    tag: smoke                        # optional; temporary experiment -> results/figures dirs get a _smoke suffix
    overrides:                        # deep-merged onto the model yaml; dcm_config onto the dataset yaml
      dcm_config: {...}
      model_config: {...}
      train_config: {...}
      benchmark_config: {...}
      biogeme_config: {...}

Rule: every config item has exactly ONE override path — the one above. Completed runs
(existing metrics.json) are skipped, so the same command is safely resumable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dcm_data import DcmConfig
from models import resolve_model_class
from models.config import ModelConfig, TrainConfig
from trainer_base import BaseTrainer

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"


# --------------------------------------------------------------------------- #
# yaml / dict helpers
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        out = yaml.safe_load(f)
    if not isinstance(out, dict):
        raise ValueError(f"{path} must contain a yaml mapping")
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins, nested dicts merge key-wise (a scalar replaces a dict)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_splits(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return list(range(spec))
    if isinstance(spec, str):
        if "-" in spec:
            lo, hi = spec.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(spec)]
    if isinstance(spec, list):
        return [int(s) for s in spec]
    raise ValueError(f"Cannot parse splits spec: {spec!r}")


def set_path(d: dict, dotted: str, value: Any) -> None:
    """Insert value at a dotted path ('a.b.c'), creating nested dicts as needed."""
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_set_args(items: list[str] | None) -> dict[str, Any]:
    """['model_config.T_beta=1.0', 'dcm_config.k_gpi_cap.TRAIN=[0.5,0,0.3]'] -> nested overrides dict.
    Values are parsed with yaml, so 1.0 / true / [1, 2] / null all become the right python type."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects path=value, got {item!r}")
        path, raw = item.split("=", 1)
        set_path(out, path.strip(), yaml.safe_load(raw))
    return out


# --------------------------------------------------------------------------- #
# the three layers
# --------------------------------------------------------------------------- #
def load_dataset_config(name: str, overrides: dict | None = None) -> DcmConfig:
    fields = _load_yaml(CONFIGS_DIR / "datasets" / f"{name}.yaml")
    if overrides:
        fields = _deep_merge(fields, overrides)
    return DcmConfig(**fields)


def load_model_spec(name: str, overrides: dict | None = None) -> dict[str, Any]:
    spec = _load_yaml(CONFIGS_DIR / "models" / f"{name}.yaml")
    if overrides:
        # dcm_config overrides target the dataset yaml, not the model spec
        spec = _deep_merge(spec, {k: v for k, v in overrides.items() if k != "dcm_config"})
    return spec


def build_trainer(
    dataset: str,
    model_spec: dict[str, Any],
    dcm_overrides: dict | None = None,
    splits_name: str | None = None,
    model_name: str | None = None,
    out_root: str | Path | None = None,
    model: str | None = None,
    tag: str | None = None,
) -> BaseTrainer:
    """Resolve the three layers into a ready-to-run trainer (no training happens here)."""
    backend = model_spec.get("backend", "torch")
    dcfg = load_dataset_config(dataset, dcm_overrides)
    model_key = model or model_spec.get("model_class") or "unknown"

    base_kwargs: dict[str, Any] = {"model_key": model_key}
    if out_root is not None:
        base_kwargs["out_root"] = out_root
    if tag:
        base_kwargs["tag"] = str(tag)          # results dir becomes <dataset or splits_name>_<tag>

    if backend == "torch":
        from train_torch import TorchDCMTrainer

        class_name = model_spec["model_class"]
        # baseline hyperparameters travel with the instance (BaseDCM.cfg), never via module globals
        bench_cfg = model_spec.get("benchmark_config") or {}
        return TorchDCMTrainer(
            dataset_name=splits_name or dataset,
            model_class=resolve_model_class(class_name),
            model_config=ModelConfig(**(model_spec.get("model_config") or {})),
            dcm_config=dcfg,
            train_config=TrainConfig(**(model_spec.get("train_config") or {})),
            model_name=model_name or model_key,
            model_kwargs={"cfg": bench_cfg} if bench_cfg else None,
            **base_kwargs,
        )

    if backend == "biogeme":
        from train_biogeme import BiogemeMNLTrainer

        return BiogemeMNLTrainer(
            dataset_name=splits_name or dataset,
            dcm_config=dcfg,
            model_name=model_name or model_key,
            spec=model_spec.get("biogeme_config") or {},
            **base_kwargs,
        )

    if backend in ("rumboost", "lightgbm"):
        from train_boost import LightGBMTrainer, RUMBoostTrainer

        cls = RUMBoostTrainer if backend == "rumboost" else LightGBMTrainer
        return cls(
            dataset_name=splits_name or dataset,
            dcm_config=dcfg,
            model_name=model_name or model_key,
            spec=model_spec,
            **base_kwargs,
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def experiment_from_args(dataset: str, model: str, model_name: str | None = None,
                         splits: str = "0-99", seeds: list[int] | None = None,
                         splits_name: str | None = None, tag: str | None = None) -> dict[str, Any]:
    """An experiment dict equivalent to a minimal yaml: dataset + model defaults, no overrides."""
    exp: dict[str, Any] = {
        "dataset": dataset, "model": model, "model_name": model_name or model,
        "splits": splits, "init_seeds": list(seeds or [42]), "overrides": {},
    }
    if splits_name:
        exp["splits_name"] = splits_name
    if tag:
        exp["tag"] = tag
    return exp


def load_experiment(
    exp: str | Path | dict[str, Any],
    sets: list[str] | dict[str, Any] | None = None,
    out_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve an experiment (yaml path or dict) + optional --set overrides into a trainer + run plan."""
    exp = dict(exp) if isinstance(exp, dict) else _load_yaml(Path(exp))
    overrides = dict(exp.get("overrides") or {})
    if sets:
        overrides = _deep_merge(overrides, sets if isinstance(sets, dict) else parse_set_args(sets))
    exp["overrides"] = overrides

    trainer = build_trainer(
        dataset=exp["dataset"],
        model=exp["model"],
        model_spec=load_model_spec(exp["model"], overrides),
        dcm_overrides=overrides.get("dcm_config"),
        splits_name=exp.get("splits_name"),
        model_name=exp.get("model_name"),
        out_root=out_root if out_root is not None else exp.get("out_root"),
        tag=exp.get("tag"),
    )
    return {
        "trainer": trainer,
        "split_ids": parse_splits(exp.get("splits", 100)),
        "init_seeds": [int(s) for s in exp.get("init_seeds", [42])],
        "experiment": exp,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run one experiment yaml (or --dataset/--model) over fixed splits")
    ap.add_argument("experiment", nargs="?", default=None, help="configs/exp/*.yaml (omit when using --dataset/--model)")
    ap.add_argument("--dataset", default=None, help="configs/datasets/<name>.yaml")
    ap.add_argument("--model", default=None, help="configs/models/<name>.yaml (registry name)")
    ap.add_argument("--model-name", default=None, help="results directory name (default: --model)")
    ap.add_argument("--splits", default=None, help='override split spec, e.g. "0-4" or "7"')
    ap.add_argument("--splits-name", default=None,
                    help="splits/<name>.npz to use with --dataset/--model (e.g. LPMC_officialtest)")
    ap.add_argument("--tag", default=None,
                    help="temporary-experiment tag: results go to <dataset or splits name>_<tag>/ (e.g. smoke)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override init seeds")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="PATH=VALUE",
                    help="override any config item, e.g. --set model_config.T_beta=1.0 (repeatable)")
    ap.add_argument("--out-root", default=None, help="results root (default $TSDCM_OUT or ./results)")
    ap.add_argument("--overwrite", action="store_true", help="re-run completed runs")
    args = ap.parse_args()

    if args.experiment is None:
        if not (args.dataset and args.model):
            ap.error("give an experiment yaml, or both --dataset and --model")
        exp: str | dict = experiment_from_args(args.dataset, args.model, args.model_name,
                                               splits=args.splits or "0-99", seeds=args.seeds,
                                               splits_name=args.splits_name, tag=args.tag)
    else:
        exp = args.experiment
        if args.dataset or args.model or args.splits_name or args.tag:
            ap.error("--dataset/--model/--splits-name/--tag are for yaml-less runs; edit the yaml or use --set instead")

    plan = load_experiment(exp, sets=args.sets, out_root=args.out_root)
    trainer = plan["trainer"]
    if args.model_name and args.experiment is not None:
        trainer.model_name = args.model_name
    split_ids = parse_splits(args.splits) if args.splits is not None else plan["split_ids"]
    init_seeds = args.seeds if args.seeds is not None else plan["init_seeds"]

    n_total = len(split_ids) * len(init_seeds)
    mcfg = getattr(trainer, "mcfg", None)
    heter = getattr(mcfg, "SP_heter_target", None)  # say out loud which variant is running
    print(
        f"[plan] {trainer.results_name} / {trainer.model_name} ({trainer.model_key}): "
        f"{len(split_ids)} splits x {len(init_seeds)} seeds = {n_total} runs -> {trainer.out_root}"
        + (f" | SP_heter_target={heter}" if heter is not None else "")
        + (f" | --set {args.sets}" if args.sets else "")
    )

    records = []
    for split_id in split_ids:
        for init_seed in init_seeds:
            print(f"\n[{len(records) + 1}/{n_total}] split {split_id}, seed {init_seed}")
            rec = trainer.run(split_id=split_id, init_seed=init_seed, overwrite=args.overwrite)
            records.append(rec)

    test_nll = np.array([r["test"]["nll"] for r in records], dtype=float)
    test_acc = np.array([r["test"]["acc"] for r in records], dtype=float)
    sem = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0  # noqa: E731
    print(
        f"\n[summary] {trainer.results_name}/{trainer.model_name} over {len(records)} runs: "
        f"test nll {test_nll.mean():.4f} +/- {sem(test_nll):.4f} | "
        f"test acc {test_acc.mean():.4f} +/- {sem(test_acc):.4f}"
    )


if __name__ == "__main__":
    main()
