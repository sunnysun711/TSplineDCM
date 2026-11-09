# tune.py — one hyperparameter-search protocol for every model (optuna TPE, resumable).
"""Unified HPO: optuna TPE on ONE tuning split's validation NLL.

Fairness is the whole point of a benchmark table: every model (baselines and the
proposed one) gets the SAME sampler (TPE, fixed seed), the SAME trial budget, and is
selected on the SAME held-out signal — one tuning split's val NLL. The test set is
never touched here. The winner is frozen to an experiment yaml ready to run over all
splits, plus a full trial log (with per-trial fit seconds) for the paper appendix.

    python tune.py --dataset SM --model fdnn --trials 100
    python tune.py configs/exp/SM_tspline.yaml --trials 100
    python tune.py ... --max-epochs 40          # speed cap during search (biases selection; re-train winner uncapped)

Search space: the `search_space:` block of configs/models/<model>.yaml. Keys are dotted
override paths (same paths as run.py --set), values are sampling specs:

    train_config.lr:                  {type: loguniform, lo: 1.0e-3, hi: 1.0e-1}
    model_config.SP_k_gpi_start:      {type: uniform,    lo: 0.1,    hi: 0.8}
    benchmark_config.hidden_layers:   {type: cat_list,   choices: [[100, 100], [250, 250]]}
    benchmark_config.activation:      {type: cat,        choices: [relu, tanh]}
    model_config.MC_k:                {type: int,        lo: 2,      hi: 4}
    dcm_config.k_gpi_cap:             {type: uniform,    lo: 0.0,    hi: 1.0}   # tspline: shared k cap

A model with no search_space (biogeme MLE) has nothing to tune: its base yaml is frozen
verbatim so the protocol still covers it uniformly.

Writes:
    $TSDCM_OUT/hpo/<dataset>/<model_name>/study.db     optuna study (kill it, re-run the same command, it resumes)
    $TSDCM_OUT/hpo/<dataset>/<model_name>/trials.json  space + budget + every trial (val_nll, fit_seconds)
    configs/exp/<stem>_tuned.yaml                      frozen winner (review it, then rename to the paper yaml)
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import yaml

from run import (
    parse_splits,
    CONFIGS_DIR,
    ROOT,
    _deep_merge,
    _load_yaml,
    experiment_from_args,
    load_experiment,
    set_path,
)
from trainer_base import DEFAULT_OUT_ROOT

HPO_DIR = Path(DEFAULT_OUT_ROOT) / "hpo"


# --------------------------------------------------------------------------- #
# search space
# --------------------------------------------------------------------------- #
def load_search_space(model: str) -> list[dict[str, Any]]:
    """configs/models/<model>.yaml `search_space:` -> list of {param, type, ...}; [] if none."""
    raw = _load_yaml(CONFIGS_DIR / "models" / f"{model}.yaml").get("search_space") or {}
    return [{"param": path, **dict(spec)} for path, spec in raw.items()]


def _suggest(trial, entry: dict[str, Any]) -> Any:
    path, kind = entry["param"], entry["type"]
    if kind in ("loguniform", "uniform"):
        return trial.suggest_float(path, float(entry["lo"]), float(entry["hi"]), log=(kind == "loguniform"))
    if kind == "int":
        return trial.suggest_int(path, int(entry["lo"]), int(entry["hi"]))
    if kind == "cat":
        return trial.suggest_categorical(path, list(entry["choices"]))
    if kind == "cat_list":  # optuna can't take lists as categorical choices: json round-trip
        return json.loads(trial.suggest_categorical(path, [json.dumps(c) for c in entry["choices"]]))
    raise ValueError(f"unknown search type {kind!r} for {path!r}")


def _decode(flat: dict[str, Any], space: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = {e["param"]: e["type"] for e in space}
    return {k: (json.loads(v) if kinds.get(k) == "cat_list" and isinstance(v, str) else v) for k, v in flat.items()}


def _nest(flat: dict[str, Any], space: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in _decode(flat, space).items():
        set_path(out, k, v)
    return out


def _round_sig(x: Any, sig: int = 4) -> Any:
    if isinstance(x, float) and x != 0 and math.isfinite(x):
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))
    return x


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def tune_experiment(
    exp: str | Path | dict[str, Any],
    n_trials: int = 30,
    tune_split: int | list[int] = 0,
    seed: int = 42,
    max_epochs: int | None = None,
    splits: str = "0-99",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    import optuna

    exp_path = None if isinstance(exp, dict) else Path(exp)
    base = dict(exp) if isinstance(exp, dict) else _load_yaml(exp_path)
    dataset, model = base["dataset"], base["model"]
    model_name = base.get("model_name") or model
    base_overrides = dict(base.get("overrides") or {})
    tune_splits = [int(s) for s in (tune_split if isinstance(tune_split, (list, tuple)) else [tune_split])]
    tune_split = tune_splits[0]                      # kept in the log for older readers
    space = load_search_space(model)
    extra = base.get("search_space") or {}          # experiment-level additions / overrides (null = drop a key)
    if extra:
        by_path = {e["param"]: e for e in space}
        for path, spec in extra.items():
            if spec is None:
                by_path.pop(path, None)
            else:
                by_path[path] = {"param": path, **dict(spec)}
        space = list(by_path.values())

    if out_path is None:
        out_path = (exp_path.with_name(f"{exp_path.stem}_tuned.yaml") if exp_path is not None
                    else CONFIGS_DIR / "exp" / f"{dataset}_{model_name}_tuned.yaml")
    out_path = Path(out_path)
    hpo_dir = HPO_DIR / dataset / model_name
    hpo_dir.mkdir(parents=True, exist_ok=True)
    log_path = hpo_dir / "trials.json"

    if not space:
        print(f"[tune] {dataset}/{model_name}: no search_space in configs/models/{model}.yaml - freezing base yaml as-is")
        _write_tuned_yaml(out_path, base, base_overrides, splits, note=f"no hyperparameters to tune for {model}; base frozen verbatim")
        result = {"dataset": dataset, "model": model, "model_name": model_name, "sampler": "none", "n_trials": 0,
                  "search_space": [], "trials": [], "best": None, "total_fit_seconds": 0.0}
        log_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    study_name = f"{dataset}__{model_name}"
    storage = f"sqlite:///{(hpo_dir / 'study.db').as_posix()}"
    n_done = 0
    if (hpo_dir / "study.db").exists():
        try:
            n_done = len(optuna.load_study(study_name=study_name, storage=storage).trials)
        except KeyError:          # a study.db without the study record yet (another tuner just created the file)
            n_done = 0
    # seed offset by n_done: a re-seeded TPE would otherwise re-propose the same start-up points on resume
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed + n_done),
        storage=storage, study_name=study_name, load_if_exists=True,
    )
    n_done = sum(1 for t in study.trials if t.state.is_finished())
    remaining = max(0, n_trials - n_done)
    print(f"[tune] {study_name} ({model}): TPE seed {seed}, target {n_trials} trials ({n_done} done, {remaining} to run); "
          f"select on split(s) {tune_splits} mean val NLL, init seed {seed}"
          + (f", epochs capped at {max_epochs}" if max_epochs else ""))

    if remaining:
        with tempfile.TemporaryDirectory(prefix="hpo_") as tmp:
            def objective(trial) -> float:
                sampled: dict[str, Any] = {}
                for e in space:
                    set_path(sampled, e["param"], _suggest(trial, e))
                over = _deep_merge(base_overrides, sampled)
                if max_epochs is not None:
                    over = _deep_merge(over, {"train_config": {"epochs": int(max_epochs)}})
                trial_exp = {**base, "overrides": over, "model_name": model_name}
                plan = load_experiment(trial_exp, out_root=Path(tmp) / f"trial{trial.number}")
                vals, accs, secs = [], [], 0.0
                for sid in tune_splits:              # mean over the tuning splits (one split alone is +-0.02 noisy)
                    rec = plan["trainer"].run(split_id=sid, init_seed=seed, overwrite=True)
                    vals.append(float(rec["val"]["nll"]))
                    accs.append(float(rec["val"]["acc"]))
                    secs += float(rec.get("fit_seconds", 0.0) or 0.0)
                    if not math.isfinite(vals[-1]):  # diverged -> drop this trial, don't poison best
                        raise optuna.TrialPruned()
                trial.set_user_attr("fit_seconds", secs)
                trial.set_user_attr("val_acc", float(sum(accs) / len(accs)))
                trial.set_user_attr("val_nll_splits", vals)
                return float(sum(vals) / len(vals))

            # one erroring trial is logged FAILED and the search goes on
            study.optimize(objective, n_trials=remaining, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"all {len(study.trials)} trials failed - check the search space / model; study: {hpo_dir / 'study.db'}")
    best = study.best_trial
    best_params = {k: _round_sig(v) for k, v in _decode(best.params, space).items()}
    winner = _deep_merge(base_overrides, _nest(best_params, space))
    total_fit = sum(float(t.user_attrs.get("fit_seconds", 0.0) or 0.0) for t in study.trials)
    note = (f"optuna TPE, {n_trials} trials, mean val NLL over split(s) {tune_splits}, seed {seed}; "
            f"winner val_nll={best.value:.4f} (trial {best.number}); HPO fit time {total_fit:.0f}s; log: {log_path}")
    _write_tuned_yaml(out_path, base, winner, splits, note=note)

    result = {
        "dataset": dataset, "model": model, "model_name": model_name, "sampler": "TPE", "seed": seed,
        "n_trials": n_trials, "tune_split": tune_split, "tune_splits": tune_splits,
        "selection_metric": "mean val_nll over tune_splits", "max_epochs": max_epochs,
        "search_space": space, "total_fit_seconds": total_fit,
        "best": {"number": best.number, "val_nll": best.value, "params": best_params},
        "trials": [{"number": t.number, "state": t.state.name, "val_nll": t.value,
                    "params": _decode(t.params, space), **t.user_attrs} for t in study.trials],
    }
    log_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n[tune] best val_nll={best.value:.4f}  params={best_params}")
    print(f"[tune] wrote {out_path}  (log: {log_path})")
    return result


def _write_tuned_yaml(out_path: Path, base: dict, overrides: dict, splits: str, note: str) -> None:
    doc: dict[str, Any] = {"dataset": base["dataset"], "model": base["model"]}
    for k in ("splits_name", "tag", "model_name", "out_root"):     # tag: a smoke experiment stays a smoke experiment
        if k in base:
            doc[k] = base[k]
    doc["splits"] = splits
    doc["init_seeds"] = base.get("init_seeds", [42])
    if overrides:
        doc["overrides"] = overrides
    try:
        run_target = out_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        run_target = str(out_path)
    header = (f"# AUTO-GENERATED by tune.py — review, then rename to the paper yaml (drop _tuned).\n"
              f"# HPO: {note}\n# Run: python run.py {run_target}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified HPO: optuna TPE on one tuning split's val NLL")
    ap.add_argument("experiment", nargs="?", default=None, help="base configs/exp/*.yaml (or use --dataset/--model)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--trials", type=int, default=30, help="search budget (same for every model)")
    ap.add_argument("--tune-split", default="0",
                    help='split(s) whose MEAN val NLL is optimised: "0" | "0,1,2" | "0-2"')
    ap.add_argument("--seed", type=int, default=42, help="sampler + init seed (reproducible)")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap epochs during search (biases selection)")
    ap.add_argument("--splits", default=None,
                    help="splits written into the frozen yaml (default: the base yaml's splits, else 0-99)")
    ap.add_argument("--out", default=None, help="output yaml path (default: <stem>_tuned.yaml)")
    a = ap.parse_args()
    if a.experiment is None:
        if not (a.dataset and a.model):
            ap.error("give a base experiment yaml, or both --dataset and --model")
        exp: str | dict = experiment_from_args(a.dataset, a.model, a.model_name, splits=a.splits or "0-99")
    else:
        exp = a.experiment
    splits = a.splits or (str((_load_yaml(Path(a.experiment)) or {}).get("splits") or "0-99") if a.experiment else "0-99")
    tune_experiment(exp, n_trials=a.trials, tune_split=parse_splits(str(a.tune_split)) if any(c in str(a.tune_split) for c in ',-') else int(a.tune_split), seed=a.seed,
                    max_epochs=a.max_epochs, splits=splits, out_path=a.out)


if __name__ == "__main__":
    main()
