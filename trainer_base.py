# experiments/base_trainer.py
"""Minimal model-agnostic trainer protocol.

The comparison currency between models is the test-set probability matrix.
``BaseTrainer.run(split_id, init_seed)`` does the invariant orchestration:

    load fixed split -> _fit -> _predict_proba(val/test) -> dcm_metrics
    -> persist predictions.npz + metrics.json under one run directory

Subclasses implement only ``_fit`` and ``_predict_proba`` (plus optional
``_config_snapshot``). Nothing here assumes torch — biogeme / boosting /
closed-form models plug in the same way.

Run directory layout::

    {out_root}/{dataset}/{model_name}/split{split_id:03d}_seed{init_seed}/
        predictions.npz   P_test, P_val, test_idx, val_idx, y_test, av_test, ...
        metrics.json      test/val metrics + timing + fit_info + config snapshot
        (subclass artifacts: checkpoints, converge curves, biogeme html, ...)
"""
from __future__ import annotations

import dataclasses
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd

from dcm_data import DcmConfig

from metrics import dcm_metrics
from splits import DATA_DIR, SPLITS_DIR, load_split

DEFAULT_OUT_ROOT = Path(os.environ.get("TSDCM_OUT", Path(__file__).resolve().parents[0] / "results"))


class BaseTrainer(ABC):
    """One instance = one (dataset, model spec); ``run`` once per (split, seed)."""

    def __init__(
        self,
        dataset_name: str,
        dcm_config: DcmConfig,
        model_name: str,
        out_root: str | Path = DEFAULT_OUT_ROOT,
        splits_dir: str | Path = SPLITS_DIR,
        model_key: str | None = None,
        tag: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name        # also the splits/<name>.npz to use
        self.tag = tag or None                  # e.g. "smoke": temporary experiment, kept apart from the paper runs
        self.results_name = f"{dataset_name}_{tag}" if tag else dataset_name   # results / figures_data / figures dir
        self.dcfg = dcm_config
        self.model_name = model_name            # results directory name (variant), e.g. tspline_level
        self.model_key = model_key or model_name  # registry / model-yaml name (the code), e.g. tspline
        self.out_root = Path(out_root)
        self.splits_dir = Path(splits_dir)

        # labels / availability straight from the csv — shared ground truth
        # for scoring, independent of any model's data pipeline
        self.df: pd.DataFrame = pd.read_csv(DATA_DIR / self.dcfg.file_name)
        self.y: np.ndarray = self.df.iloc[:, self.dcfg.alt_label_cols].to_numpy(dtype=np.float64)
        self.av: np.ndarray = self.df.iloc[:, self.dcfg.alt_av_cols].to_numpy(dtype=np.float64)
        assert set(np.unique(self.y)) <= {0.0, 1.0}, "labels must be one-hot"

    # ---------- protocol to implement ----------

    @abstractmethod
    def _fit(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        init_seed: int,
        run_dir: Path,
    ) -> dict[str, Any]:
        """Train on train_idx (val_idx for model selection). Return fit_info
        (best_epoch, n_params, loglike, ... — anything json-serializable).
        ``test_idx`` is provided for trainers that log test curves; it must
        NOT influence fitting."""

    @abstractmethod
    def _predict_proba(self, idx: np.ndarray) -> np.ndarray:
        """Choice probabilities [len(idx), I] for csv rows ``idx``, aligned
        row-by-row with ``idx``. Unavailable alternatives must carry ~0 mass."""

    def _config_snapshot(self) -> dict[str, Any]:
        """Json-serializable hyperparameter snapshot stored in metrics.json."""
        return {}

    # ---------- invariant orchestration ----------

    def run_dir(self, split_id: int, init_seed: int) -> Path:
        return (
            self.out_root / self.results_name / self.model_name
            / f"split{split_id:03d}_seed{init_seed}"
        )

    def run(self, split_id: int, init_seed: int = 42, overwrite: bool = False) -> dict[str, Any]:
        train_idx, val_idx, test_idx = load_split(
            self.dataset_name, split_id, splits_dir=self.splits_dir
        )
        rd = self.run_dir(split_id, init_seed)
        metrics_path = rd / "metrics.json"
        if metrics_path.exists() and not overwrite:
            with open(metrics_path, encoding="utf-8") as f:
                record = json.load(f)
            print(f"[skip] {rd} already complete (overwrite=False)")
            return record
        rd.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        fit_info = self._fit(train_idx, val_idx, test_idx, init_seed, rd) or {}
        fit_seconds = time.time() - t0

        P_test = np.asarray(self._predict_proba(test_idx), dtype=np.float64)
        P_val = np.asarray(self._predict_proba(val_idx), dtype=np.float64)

        m_test = dcm_metrics(P_test, self.y[test_idx], self.av[test_idx])
        m_val = dcm_metrics(P_val, self.y[val_idx], self.av[val_idx])

        np.savez_compressed(
            rd / "predictions.npz",
            P_test=P_test, test_idx=test_idx, y_test=self.y[test_idx], av_test=self.av[test_idx],
            P_val=P_val, val_idx=val_idx, y_val=self.y[val_idx], av_val=self.av[val_idx],
        )

        # the COMPLETE resolved configuration goes into every metrics.json ("what exactly ran" = this file)
        config: dict[str, Any] = {"model": self.model_key, "model_name": self.model_name,
                                  "dcm_config": dataclasses.asdict(self.dcfg), **self._config_snapshot()}
        heter = (config.get("model_config") or {}).get("SP_heter_target")
        record: dict[str, Any] = {
            "dataset": self.dataset_name,
            "results_name": self.results_name,
            "tag": self.tag,
            "model": self.model_name,
            "model_key": self.model_key,
            "split_id": int(split_id),
            "init_seed": int(init_seed),
            **({"heter_target": heter} if heter is not None else {}),  # visible at the top of the file
            "test": m_test,
            "val": m_val,
            "fit_seconds": round(fit_seconds, 2),
            "fit_info": fit_info,
            "config": config,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

        print(
            f"[done] {self.dataset_name}/{self.model_name} split{split_id:03d} "
            f"seed{init_seed}: test nll={m_test['nll']:.4f} acc={m_test['acc']:.4f} "
            f"({fit_seconds:.0f}s)"
        )
        return record


def collect_results(
    out_root: str | Path = DEFAULT_OUT_ROOT,
    dataset: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Flatten all metrics.json under out_root into one tidy DataFrame
    (one row per run; columns test_nll, test_acc, val_nll, ...). The paired
    comparison key across models is (dataset, split_id)."""
    rows = []
    pattern = f"{dataset or '*'}/{model or '*'}/*/metrics.json"
    for p in sorted(Path(out_root).glob(pattern)):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        row = {k: r[k] for k in ["dataset", "model", "split_id", "init_seed", "fit_seconds"]}
        # the results DIRECTORY is the model_name (a dir renamed after the run, e.g. tspline -> tspline_stage4, must
        # not merge with the current tspline rows); the name recorded at run time is kept for reference
        row["model_recorded"] = row["model"]
        row["model"] = p.parent.parent.name
        row.update({f"test_{k}": v for k, v in r["test"].items()})
        row.update({f"val_{k}": v for k, v in r["val"].items()})
        cfg = r.get("config") or {}
        row["model_key"] = r.get("model_key") or cfg.get("model")
        row["n_params"] = (r.get("fit_info") or {}).get("n_params")
        row["heter_target"] = r.get("heter_target")
        caps = (cfg.get("dcm_config") or {}).get("k_gpi_cap")
        row["k_all_zero"] = (all(float(c) == 0.0 for v in caps.values() for c in v)
                             if isinstance(caps, dict) else None)
        row["path"] = str(p.parent)
        rows.append(row)
    return pd.DataFrame(rows)
