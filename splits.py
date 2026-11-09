# experiments/splits.py
"""Fixed GROUP-WISE train/val/test splits shared by ALL models.

Every model (torch / biogeme / boosting / anything) must consume these
pre-generated splits so that comparisons are paired by ``split_id``.

Real datasets are split GROUP-WISE by individual/household: all rows of one
group land in exactly one of train/val/test, preventing person-level leakage
(Swissmetro has 9 panel responses per individual, LPMC ~4.6 trips per
household). The synthetic dataset is deliberately ROW-LEVEL — see
DATASET_SPEC notes.

Generate once::

    python splits.py                  # all default datasets
    python splits.py --dataset SM --n-splits 100

Load anywhere::

    from splits import load_split
    train_idx, val_idx, test_idx = load_split("SM", split_id=0)

File format ``splits/{name}.npz`` (compressed):
    - ``meta``                          json string, see ``_build_meta``
    - ``split_{i:03d}_train/_val/_test`` sorted int32 row indices into the csv

Protocol:
    - group-wise two-stage GroupShuffleSplit, ratios (0.70, 0.15, 0.15)
      interpreted as fractions of GROUPS, via
      ``tspline.config.data._two_stage_group_shuffle_split``
      (row-level ``_two_stage_shuffle_split`` where group column is None)
    - ``split_seed == split_id`` (0..n_splits-1)
    - ``*_officialtest`` variants pin the test set to rows where the csv's
      ``dataset`` column equals "test" (LPMC official benchmark split, already
      household-disjoint from train) and re-randomize train/val group-wise
      inside the official train pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from dcm_data import _two_stage_group_shuffle_split, _two_stage_shuffle_split

ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"

DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_N_SPLITS = 100

# dataset name -> (csv path relative to data/, group column name or None)
# group column None => row-level split. SYN is deliberately row-level: labels
# are independent draws from the known DGP per repeated covariate row (same
# protocol as the public RUMBoost code, which repeats rows 5x), so the
# estimand is fit-to-DGP on the covariate support, not new-individual
# generalization — repeated x in train and test is not label leakage there.
# csv paths match configs/datasets/<name>.yaml file_name
DATASET_SPEC: dict[str, tuple[str, str | None]] = {
    "SM": ("Swissmetro/swissmetro_processed_rumboost.csv", "ID"),
    "LPMC": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    "LPMC_officialtest": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    # LPMC shrunk to Swissmetro's training size (identifiability experiment): the official test set
    # stays, but train+val are drawn from a random subset of households whose rows add up to SM's train+val (7,677)
    "LPMC_sub_officialtest": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    # learning curve: the same shrinking to 2,000 / 4,000 / 15,000 / 30,000 pool rows; together with
    # LPMC_sub_officialtest (7,677) and LPMC_officialtest (the full pool, 54,766) they give six sizes
    "LPMC_n2000_officialtest": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    "LPMC_n4000_officialtest": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    "LPMC_n15000_officialtest": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    "LPMC_n30000_officialtest": ("LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv", "household_id"),
    "SYN": ("syn_data.csv", None),          # synthetic/generate.py output; row-level, 10 splits are enough (paper uses 0-4)
}

# datasets whose test set is pinned by a csv column instead of randomized
OFFICIAL_TEST_CRITERION: dict[str, tuple[str, str]] = {
    "LPMC_officialtest": ("dataset", "test"),
    "LPMC_sub_officialtest": ("dataset", "test"),
    "LPMC_n2000_officialtest": ("dataset", "test"),
    "LPMC_n4000_officialtest": ("dataset", "test"),
    "LPMC_n15000_officialtest": ("dataset", "test"),
    "LPMC_n30000_officialtest": ("dataset", "test"),
}

# datasets whose train+val pool is first shrunk to about this many rows (whole groups, seed = split_id)
SUBSAMPLE_POOL_ROWS: dict[str, int] = {
    "LPMC_sub_officialtest": 6318 + 1359,   # Swissmetro split 0: 6318 train + 1359 val rows
    "LPMC_n2000_officialtest": 2000,
    "LPMC_n4000_officialtest": 4000,
    "LPMC_n15000_officialtest": 15000,
    "LPMC_n30000_officialtest": 30000,
}


def _csv_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_meta(
    name: str,
    csv_rel: str,
    csv_md5: str,
    n_rows: int,
    n_splits: int,
    ratios: tuple[float, float, float],
    method: str,
    group_col: str,
) -> dict:
    return {
        "dataset": name,
        "csv": csv_rel,
        "csv_md5": csv_md5,
        "n_rows": n_rows,
        "n_splits": n_splits,
        "ratios": list(ratios),
        "method": method,
        "group_col": group_col,
        "seed_scheme": "split_seed == split_id",
    }


def _group_splits(
    ids: np.ndarray, n_splits: int, ratios: tuple[float, float, float]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group-wise two-stage splits, seed = split_id."""
    out = []
    for sid in range(n_splits):
        tr, va, te = _two_stage_group_shuffle_split(
            ids=ids, seed=sid, val_ratio=ratios[1], test_ratio=ratios[2]
        )
        out.append((tr, va, te))
    return out


def _official_test_group_splits(
    df: pd.DataFrame,
    criterion: tuple[str, str],
    group_ids: np.ndarray,
    n_splits: int,
    ratios: tuple[float, float, float],
    pool_rows: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pin test by csv column; group-wise re-randomize train/val inside the pool. With pool_rows, the pool is
    first shrunk to a random subset of whole groups adding up to about that many rows (new subset per split)."""
    col, val = criterion
    test_mask = (df[col] == val).to_numpy()
    test_idx = np.sort(np.where(test_mask)[0])
    pool_idx_all = np.where(~test_mask)[0]
    if test_idx.size == 0:
        raise ValueError(f"No rows match official-test criterion {col} == {val!r}")

    # val share of pool GROUPS so that val keeps roughly ratios[1] of total
    val_rel = ratios[1] / (ratios[0] + ratios[1])

    out = []
    for sid in range(n_splits):
        pool_idx = pool_idx_all
        if pool_rows is not None:
            rng = np.random.default_rng(sid)
            groups_all = np.unique(group_ids[pool_idx_all])
            order = rng.permutation(groups_all)
            sizes = pd.Series(group_ids[pool_idx_all]).value_counts()
            n_keep = int(np.searchsorted(np.cumsum(sizes.loc[order].to_numpy()), pool_rows)) + 1
            keep = set(order[:n_keep].tolist())
            pool_idx = pool_idx_all[np.fromiter((g in keep for g in group_ids[pool_idx_all]), dtype=bool, count=pool_idx_all.size)]
        pool_groups = group_ids[pool_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=sid)
        (tr_rel, va_rel) = next(gss.split(pool_idx, groups=pool_groups))
        out.append((np.sort(pool_idx[tr_rel]), np.sort(pool_idx[va_rel]), test_idx))
    return out


def generate(
    name: str,
    n_splits: int = DEFAULT_N_SPLITS,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    out_dir: Path = SPLITS_DIR,
) -> Path:
    """Generate and save the fixed splits for one dataset. Returns the npz path."""
    csv_rel, group_col = DATASET_SPEC[name]
    csv_path = DATA_DIR / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    group_ids = df[group_col].to_numpy() if group_col is not None else None

    if name in OFFICIAL_TEST_CRITERION:
        assert group_ids is not None
        pool_rows = SUBSAMPLE_POOL_ROWS.get(name)
        splits = _official_test_group_splits(
            df, OFFICIAL_TEST_CRITERION[name], group_ids, n_splits, ratios, pool_rows=pool_rows
        )
        method = (
            "official-test pinned by csv column "
            f"{OFFICIAL_TEST_CRITERION[name][0]} == {OFFICIAL_TEST_CRITERION[name][1]!r}; "
            + (f"pool first shrunk to whole {group_col} groups adding up to ~{pool_rows} rows (rng seed = split_id); " if pool_rows else "")
            + f"train/val GroupShuffleSplit by {group_col} inside pool, seed = split_id"
        )
    elif group_ids is not None:
        splits = _group_splits(group_ids, n_splits, ratios)
        method = (
            f"group-wise two-stage GroupShuffleSplit by {group_col} "
            "(_two_stage_group_shuffle_split), seed = split_id"
        )
    else:
        splits = [
            _two_stage_shuffle_split(n_rows, seed=sid, val_ratio=ratios[1], test_ratio=ratios[2])
            for sid in range(n_splits)
        ]
        method = (
            "row-level two-stage ShuffleSplit (_two_stage_shuffle_split), seed = split_id; "
            "deliberate for synthetic data with independently re-drawn labels per repeated row "
            "(matches public RUMBoost protocol)"
        )

    # sanity: disjoint indices; group-disjointness where a group column applies
    for sid, (tr, va, te) in enumerate(splits):
        union = np.concatenate([tr, va, te])
        assert len(np.unique(union)) == len(union), f"split {sid}: overlapping indices"
        if group_ids is not None:
            g_tr, g_va, g_te = map(lambda a: set(group_ids[a]), (tr, va, te))
            assert not (g_tr & g_te), f"split {sid}: groups straddle train/test"
            assert not (g_tr & g_va), f"split {sid}: groups straddle train/val"
            assert not (g_va & g_te), f"split {sid}: groups straddle val/test"

    arrays: dict[str, np.ndarray] = {}
    for sid, (tr, va, te) in enumerate(splits):
        arrays[f"split_{sid:03d}_train"] = tr.astype(np.int32)
        arrays[f"split_{sid:03d}_val"] = va.astype(np.int32)
        arrays[f"split_{sid:03d}_test"] = te.astype(np.int32)

    meta = _build_meta(
        name, csv_rel, _csv_md5(csv_path), n_rows, n_splits, ratios, method, group_col
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    kind = "group-wise" if group_ids is not None else "row-level"
    print(f"[splits] {name}: {n_splits} {kind} splits x {n_rows} rows -> {out_path}")
    return out_path


# ---------- loading ----------

def load_split_meta(name: str, splits_dir: Path | str = SPLITS_DIR) -> dict:
    with np.load(Path(splits_dir) / f"{name}.npz") as z:
        return json.loads(str(z["meta"]))


def load_split(
    name: str,
    split_id: int,
    splits_dir: Path | str = SPLITS_DIR,
    verify_csv: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (train_idx, val_idx, test_idx) for one split_id.

    ``verify_csv=True`` recomputes the csv md5 and raises if the data file has
    changed since the splits were generated (indices would silently misalign).
    """
    path = Path(splits_dir) / f"{name}.npz"
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        if not (0 <= split_id < meta["n_splits"]):
            raise IndexError(f"split_id {split_id} out of range [0, {meta['n_splits']})")
        if verify_csv:
            csv_path = DATA_DIR / meta["csv"]
            current = _csv_md5(csv_path)
            if current != meta["csv_md5"]:
                raise RuntimeError(
                    f"{meta['csv']} changed since splits were generated "
                    f"(md5 {current} != {meta['csv_md5']}). Regenerate {path.name} "
                    "or restore the csv; indices would misalign otherwise."
                )
        key = f"split_{split_id:03d}"
        return z[f"{key}_train"], z[f"{key}_val"], z[f"{key}_test"]


def list_available_splits(splits_dir: Path | str = SPLITS_DIR) -> list[str]:
    d = Path(splits_dir)
    return sorted(p.stem for p in d.glob("*.npz")) if d.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate fixed group-wise splits")
    ap.add_argument("--dataset", choices=sorted(DATASET_SPEC), default=None,
                    help="single dataset; default: all")
    ap.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    args = ap.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASET_SPEC)
    for t in targets:
        generate(t, n_splits=args.n_splits)
