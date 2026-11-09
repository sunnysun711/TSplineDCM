# plots/extract/_common.py — shared helpers for the extract layer.
"""The extract layer is the ONLY place that touches checkpoints and model classes.
Each extract/<kind>.py rebuilds a model from its experiment yaml, reloads every split's best.pt,
computes numbers, and writes csv/npz into figures_data/<dataset>/. The fig_*.py scripts then
read those files and nothing else.

csv convention: long format, one row per value, columns
    model_name, alt, attr, split, <x columns>, <value columns>
Re-running an extract for one model_name replaces only that model's rows (merge_csv), so
make_all.py can run models one at a time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import experiment_from_args, load_experiment, parse_splits  # noqa: E402
from trainer_base import DEFAULT_OUT_ROOT  # noqa: E402

FIG_DATA = ROOT / "figures_data"


# --------------------------------------------------------------------------- #
# CLI shared by every extract script
# --------------------------------------------------------------------------- #
def add_common_args(ap: argparse.ArgumentParser, default_splits: str = "0-99") -> None:
    ap.add_argument("experiment", nargs="?", default=None, help="configs/exp/*.yaml (or --dataset/--model)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-name", default=None, help="results dir name (default: --model)")
    ap.add_argument("--splits-name", default=None,
                    help="splits/<name>.npz the runs used (results live under <name>/), e.g. LPMC_officialtest")
    ap.add_argument("--splits", default=default_splits, help='"0-99" | "0" | "0,5,9"')
    ap.add_argument("--seed", type=int, default=42, help="init seed of the runs to load (results dir suffix)")
    ap.add_argument("--results-root", default=None,
                    help=f"where run.py wrote results (default: the experiment's out_root, else $TSDCM_OUT = {DEFAULT_OUT_ROOT})")
    ap.add_argument("--out-dir", default=None, help="default figures_data/<dataset>/")


def experiment_from(args: argparse.Namespace) -> str | dict[str, Any]:
    if args.experiment is None:
        if not (args.dataset and args.model):
            raise SystemExit("give an experiment yaml, or both --dataset and --model")
        return experiment_from_args(args.dataset, args.model, args.model_name, splits_name=args.splits_name)
    return args.experiment


def split_list(spec: str) -> list[int]:
    if "," in spec:
        return [int(s) for s in spec.split(",")]
    return parse_splits(spec)


# --------------------------------------------------------------------------- #
# model / checkpoints / batch
# --------------------------------------------------------------------------- #
def load_prepared(exp: str | dict[str, Any], results_root: str | Path | None = None, seed: int = 42):
    """Trainer with dataset + untrained model on the device (no training). Returns (trainer, experiment dict)."""
    plan = load_experiment(exp, out_root=results_root)
    tt = plan["trainer"]
    if not hasattr(tt, "prepare"):
        raise SystemExit(f"{plan['experiment']['model']} is not a torch model; the asu_* interface needs one")
    tt.prepare(seed)
    return tt, plan["experiment"]


def checkpoint_path(tt, split_id: int, seed: int) -> Path:
    return tt.run_dir(split_id, seed) / "checkpoints" / "best.pt"


def iter_loaded_splits(tt, splits: Iterable[int], seed: int, tag: str = "extract") -> Iterator[int]:
    """Load each split's best.pt into tt.model (eval mode) and yield the split id; missing ones are skipped."""
    n = 0
    for sid in splits:
        p = checkpoint_path(tt, sid, seed)
        if not p.exists():
            print(f"[{tag}] skip split {sid}: no checkpoint at {p}")
            continue
        tt.load_checkpoint(p)
        tt.model.eval()
        n += 1
        yield sid
    if n == 0:
        raise FileNotFoundError(f"no checkpoints under {tt.out_root / tt.results_name / tt.model_name} (run run.py first)")


def fixed_batch(ds, n_samples: int | None, seed: int) -> dict:
    """One fixed random sample of rows, shared across splits/models (same draw as v1 asu.py:
    torch.manual_seed(seed); torch.randperm(N)[:n])."""
    torch.manual_seed(seed)
    n_rows = len(ds)
    if n_samples is None or n_samples >= n_rows:
        idx = torch.arange(n_rows)
    else:
        idx = torch.randperm(n_rows)[:n_samples]
    return ds.make_batch_from_indices(idx.tolist())


def alt_attr_pairs(dcfg) -> list[tuple[str, str]]:
    return [(a, x) for a in dcfg.alt_names for x in dcfg.alt_xs[a]]


def model_name_of(exp: dict[str, Any]) -> str:
    return exp.get("model_name") or exp["model"]


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def out_dir_for(dataset: str, out_dir: str | Path | None) -> Path:
    d = Path(out_dir) if out_dir else FIG_DATA / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def merge_csv(path: Path, df: pd.DataFrame, key: str = "model_name") -> Path:
    """Write df to path, replacing existing rows that share df's `key` values (other models' rows survive).
    A `.csv.gz` path is written/read gzip-compressed (pandas infers it) — used for the big per-split /
    per-individual tables so figures_data stays committable; 6 significant digits everywhere."""
    if path.exists():
        old = pd.read_csv(path)
        keep = old[~old[key].isin(df[key].unique())] if key in old else old.iloc[0:0]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6g")
    print(f"[extract] wrote {path} ({len(df)} rows)")
    return path


def band(a: np.ndarray, pcts: tuple[float, float] = (16.0, 84.0)) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(a, pcts[0], axis=0), np.percentile(a, pcts[1], axis=0)


# --------------------------------------------------------------------------- #
# the fixed batch as a table (joins for the individual-level csvs)
# --------------------------------------------------------------------------- #
TIME_SLOTS = ("TT", "RailT", "BusT", "AccessT")   # priority order when picking "the" time attribute of an alternative
COST_SLOT = "CO"


def time_cost_attrs(dcfg, alt: str) -> tuple[str | None, str | None]:
    """(time attr, cost attr) of an alternative by slot name; None when the alt has no such attribute."""
    slots = {x.split("-")[0]: x for x in dcfg.alt_xs[alt]}
    tt = next((slots[s] for s in TIME_SLOTS if s in slots), None)
    co = slots.get(COST_SLOT)
    return tt, co


def write_individuals(batch: dict, dcfg, out: Path) -> Path:
    """individuals.csv.gz: one row per sampled individual (row index `idx` in the dataset csv, ID, z columns, and
    every alternative's observed attributes as <alt>__<attr>). Model-independent, so written once per dataset."""
    df = pd.DataFrame({"idx": batch["idx"].detach().cpu().numpy(), "ID": batch["ID"].detach().cpu().numpy()})
    z = batch["z"].detach().cpu().numpy()
    for j, name in enumerate(dcfg.z_names):
        df[name] = z[:, j]
    for i, alt in enumerate(dcfg.alt_names):
        xa = batch["x"][i].detach().cpu().numpy()
        for j, x in enumerate(dcfg.alt_xs[alt]):
            df[f"{alt}__{x}"] = xa[:, j]
    av = batch["av"].detach().cpu().numpy()
    df["av"] = [" ".join(a for a, m in zip(dcfg.alt_names, row) if m) for row in av]
    p = out / "individuals.csv.gz"
    df.to_csv(p, index=False, lineterminator="\n", float_format="%.6g")
    print(f"[extract] wrote {p} ({len(df)} rows)")
    return p


def slope_at_points(y: np.ndarray, xs: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
    """Per-row derivative of y (N, P) on grid xs (P,), linearly interpolated at each row's own x_obs (N,)."""
    dy = np.gradient(y, xs, axis=1)
    j = np.clip(np.searchsorted(xs, x_obs) - 1, 0, len(xs) - 2)
    w = (x_obs - xs[j]) / (xs[j + 1] - xs[j])
    rows = np.arange(len(x_obs))
    return dy[rows, j] * (1.0 - w) + dy[rows, j + 1] * w
