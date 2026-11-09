# dcm_data.py — DcmConfig (which csv column plays which role) + DcmDataset (tensors) + split helpers.
# Merged from v1 tspline/config/{dcm,data}.py; the 7 hard-coded get_*_dcfg() are gone
# (configs/datasets/*.yaml replaced them in 2026-06).


from dataclasses import dataclass, field
from typing import Optional, Any

import pandas as pd

from models.config import AscLearnerType, MonotoneFlag, MonoType, get_device

@dataclass
class DcmConfig:
    name: str
    file_name: str
    alt_names: list[str]
    alt_av_cols: list[int]
    alt_label_cols: list[int]
    x_names: list[str]
    alt_xs: dict[str, list[str]]
    alt_x_cols: dict[str, list[int]]
    z_names: list[str]
    z_cols: list[int] | None
    ID_col: int  # indicator for individuals, used for controlling randomness
    share_mc: bool  # whether to share the same MonoCoef for same attributes (across alternatives)
    mono_dirs: dict[str, MonotoneFlag]
    mono_types: dict[str, MonoType]
    interactions: list[list[str]] = field(default_factory=list)
    ranks: list[tuple[int, int]] = field(default_factory=list)
    fix_asc_zero: str | None = None
    learn_asc: bool = False  # linear ASC learner
    asc_learner: AscLearnerType = "linear"  # linear, or zspline
    group_id_col: Optional[int] = None  # used for group-split datasets
    hold_out_criterion: Optional[dict] = None  # used for setting hold-out test set
    alt_xs_knots: Optional[dict[str, list[int]]] = None  # used for customize Grid numbers for each attribute
    z_mono_dirs: Optional[dict[str, int]] = None  # used for setting monotone direction for z
    z_spline_configs: Optional[dict[str, dict]] = None  # used for customize ZSplineHead config for each z
    
    share_taste_p: bool = True                   # if False, split selected x_names by alt
    taste_split_attrs: Optional[list[str]] = None # which x_names to split; None => split all
    k_gpi_cap: float | dict[str, list[float]] = 1.0  # cap value of k_gpi, defaults to 1.0. can be a dict of alt_name -> list of float capping each feature.

    def __post_init__(self) -> None:
        assert len(self.alt_names) == len(self.alt_av_cols) == len(self.alt_xs) == len(self.alt_label_cols)
        for alt_name, xs in self.alt_xs.items():
            assert alt_name in self.alt_names, "alt_xs key not in alt_names"
            assert set(xs).issubset(set(self.x_names)), "alt_xs must be subset of x_names"
        assert set(self.mono_dirs.keys()).issubset(set(self.x_names)), "mono_dirs keys must be subset of x_names"
        assert set(self.mono_types.keys()).issubset(set(self.x_names)), "mono_types keys must be subset of x_names"
        if self.z_cols is not None:
            assert len(self.z_names) == len(self.z_cols), "z_names and z_cols must match"
        if self.fix_asc_zero is not None:
            assert self.fix_asc_zero in self.alt_names, "fix_asc_zero must be an alternative"
        self.group_id_col = self.group_id_col if self.group_id_col is not None else self.ID_col
        assert self.group_id_col is not None

        if self.z_spline_configs is None:
            self.z_spline_configs = dict(
                MC_DISCRETE_CONFIG={"k": 0, "n_distinct_values_max": 8},
                MC_CONTINUOUS_CONFIG={"k": 2, "num_knots": 12},
            )

        # cap k_gpi
        if isinstance(self.k_gpi_cap, float):
            self.k_gpi_cap = {alt_name: [self.k_gpi_cap] * len(self.alt_xs[alt_name]) for alt_name in self.alt_names}
        elif isinstance(self.k_gpi_cap, dict):
            for alt_name in self.alt_names:
                assert alt_name in self.k_gpi_cap, f"k_gpi_cap must have key {alt_name}"
                assert len(self.k_gpi_cap[alt_name]) == len(self.alt_xs[alt_name]), (
                    f"k_gpi_cap {alt_name} must match alt_xs {self.alt_xs[alt_name]}"
                )
        else:
            raise ValueError(f"k_gpi_cap must be float or dict, but got {type(self.k_gpi_cap)}")
        for alt_name, caps in self.k_gpi_cap.items():
            assert all(float(c) >= 0.0 for c in caps), f"k_gpi_cap[{alt_name}] must be >= 0, got {caps}"
        # (the <= 1 bound for the I-spline variant is enforced where heter_target is known: SplineUtility)

        # Perform taste split BEFORE validating share_mc knot consistency, so names are final.
        if not self.share_taste_p:
            self._expand_x_by_alt()

        # Re-check that alt_xs entries exist in (possibly expanded) x_names
        for alt_name, xs in self.alt_xs.items():
            assert set(xs).issubset(set(self.x_names)), "after taste split, alt_xs must be subset of x_names"

        if self.alt_xs_knots is not None:
            for alt_name, knots in self.alt_xs_knots.items():
                assert alt_name in self.alt_names, f"alt_xs_knots key {alt_name} not in alt_names"
                assert len(knots) == len(self.alt_xs[alt_name]), (
                    f"alt_xs_knots {alt_name} must match alt_xs {self.alt_xs[alt_name]}"
                )
            # Only enforce cross-alt equal knots when STILL sharing MonoCoef **and** taste keys
            if self.share_mc and self.share_taste_p:
                _x_name_knots: dict[str, int] = {}
                for alt_name, x_names in self.alt_xs.items():
                    for x_name, n_knot in zip(x_names, self.alt_xs_knots[alt_name]):
                        assert n_knot == _x_name_knots.setdefault(x_name, n_knot), (
                            f"Inconsistent number of knots for attribute '{x_name}' across alternatives "
                            "when share_mc is True and share_taste_p is True"
                        )

    # --- helper: expand chosen x_names to per-alt keys ---
    def _expand_x_by_alt(self) -> None:
        """
        When share_taste_p=False, rename selected x_names to '{x}-{alt}' and
        broadcast dependent dicts (alt_xs, mono_dirs, mono_types).
        """
        orig_x_names = list(self.x_names)
        split_targets = set(orig_x_names) if self.taste_split_attrs is None else set(self.taste_split_attrs)

        # map[x][alt] -> new_name
        name_map: dict[str, dict[str, str]] = {}
        new_x_names: list[str] = []

        for x in orig_x_names:
            if x in split_targets:
                name_map[x] = {}
                for alt in self.alt_names:
                    if x not in self.alt_xs[alt]:  # skip for non-split attrs
                        continue
                    xn = f"{x}-{alt}"
                    name_map[x][alt] = xn
                    new_x_names.append(xn)
            else:
                # unchanged name is mapped to itself for all alts (for uniform replacement below)
                name_map[x] = {alt: x for alt in self.alt_names}
                new_x_names.append(x)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for xn in new_x_names:
            if xn not in seen:
                seen.add(xn)
                deduped.append(xn)
        self.x_names = deduped

        # alt_xs: replace each x with the alt-specific key for that alt
        self.alt_xs = {
            alt: [name_map[x][alt] for x in xs]
            for alt, xs in self.alt_xs.items()
        }

        # mono_dirs/types: copy entries for new per-alt names where split applies
        def _broadcast_dict(d: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for x, v in d.items():
                if x in split_targets:
                    for alt in self.alt_names:
                        if f"{x}-{alt}" in self.x_names:
                            out[f"{x}-{alt}"] = v
                else:
                    out[x] = v
            return out

        self.mono_dirs  = _broadcast_dict(self.mono_dirs)
        self.mono_types = _broadcast_dict(self.mono_types)
        # alt_x_cols and alt_xs_knots remain aligned (lengths unchanged). No change needed.


# ==============================================================================
# DcmDataset
# ==============================================================================
from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit



class DcmDataset(Dataset):
    def __init__(
        self,
        config: DcmConfig,
        device: torch.device | str | None = None,
        materialize_on_device: bool = True,
        float_type: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = len(self.config.alt_names)
        self.data: pd.DataFrame = pd.read_csv(f"./data/{self.config.file_name}")
        self.materialized_on_device: bool = materialize_on_device

        self.device = torch.device(device) if device is not None else get_device()
        if float_type is None:
            self._float_type = torch.float64 if self.device.type == "cpu" else torch.float32
        else:
            self._float_type = float_type

        self._coerce_numeric_columns()

        target_device = self.device if materialize_on_device else torch.device("cpu")

        self._ID = torch.tensor(
            self.data.iloc[:, self.config.ID_col].values,
            dtype=torch.int64,
            device=target_device,
        )
        assert self.config.group_id_col is not None
        self._group_id = torch.tensor(
            self.data.iloc[:, self.config.group_id_col].values,
            dtype=torch.int64,
            device=target_device,
        )
        self._av = torch.tensor(
            self.data.iloc[:, self.config.alt_av_cols].values,
            dtype=self._float_type,
            device=target_device,
        )
        self._label = torch.tensor(
            self.data.iloc[:, self.config.alt_label_cols].values,
            dtype=self._float_type,
            device=target_device,
        )

        self._x_per_alt: list[torch.Tensor] = []
        for alt in self.config.alt_names:
            cols = self.config.alt_x_cols[alt]
            t = torch.tensor(
                self.data.iloc[:, cols].values,
                dtype=self._float_type,
                device=target_device,
            )
            self._x_per_alt.append(t)

        if self.config.z_cols is not None:
            z_vals = self.data.iloc[:, self.config.z_cols].values.astype("float")
        else:
            z_vals = self.data[self.config.z_names].values.astype("float")
        self._z = torch.tensor(z_vals, dtype=self._float_type, device=target_device)

        self.materialized_on_device = target_device.type != "cpu"

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        res: dict[str, Any] = {}
        res["idx"] = torch.tensor([idx], dtype=torch.int64, device=self.device)
        res["ID"] = self._ID[idx]
        res["group_id"] = self._group_id[idx]
        res["av"] = self._av[idx]
        res["label"] = self._label[idx]
        res["x"] = [x[idx] for x in self._x_per_alt]
        res["z"] = self._z[idx]
        return res

    def _coerce_numeric_columns(self) -> None:
        id_col_name = self.data.columns[self.config.ID_col]
        self.data[id_col_name] = (
            pd.to_numeric(self.data[id_col_name], errors="coerce")
            .fillna(-1)
            .astype("int64")
        )
        assert self.config.group_id_col is not None
        group_id_col_name = self.data.columns[self.config.group_id_col]
        self.data[group_id_col_name] = (
            pd.to_numeric(self.data[group_id_col_name], errors="coerce")
            .fillna(-1)
            .astype("int64")
        )

        av_names = list(self.data.columns[self.config.alt_av_cols])
        self.data[av_names] = (
            self.data[av_names].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        label_names = list(self.data.columns[self.config.alt_label_cols])
        self.data[label_names] = (
            self.data[label_names].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        all_x_idx: list[int] = []
        for alt in self.config.alt_names:
            all_x_idx.extend(self.config.alt_x_cols[alt])
        all_x_idx = sorted(set(all_x_idx))
        x_names = list(self.data.columns[all_x_idx])
        self.data[x_names] = self.data[x_names].apply(pd.to_numeric, errors="coerce")

        if self.config.z_cols is not None:
            z_names = list(self.data.columns[self.config.z_cols])
        else:
            z_names = self.config.z_names
        self.data[z_names] = self.data[z_names].apply(pd.to_numeric, errors="coerce")

    def make_batch_from_indices(self, indices: torch.Tensor | np.ndarray | list[int],
                                device: torch.device | str | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Slice dataset by row indices and return a batch dict consistent with __getitem__.
        All tensors are moved to `device` if provided; otherwise keep on dataset's device.
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices = indices.to(dtype=torch.long, device=self.device)

        batch: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        batch["idx"] = indices
        batch["ID"] = self._ID[indices]
        batch["group_id"] = self._group_id[indices]
        batch["av"] = self._av[indices]
        batch["label"] = self._label[indices]
        batch["x"] = [x[indices] for x in self._x_per_alt]
        batch["z"] = self._z[indices]

        # move to target device if specified (e.g., trainer.device)
        if device is not None:
            dev = torch.device(device)
            batch["idx"] = batch["idx"].to(dev)
            batch["ID"] = batch["ID"].to(dev)
            batch["group_id"] = batch["group_id"].to(dev)
            batch["av"] = batch["av"].to(dev)
            batch["label"] = batch["label"].to(dev)
            batch["x"] = [t.to(dev) for t in batch["x"]]  # type: ignore
            batch["z"] = batch["z"].to(dev)
        return batch
    
    def make_batch_from_criterion(self, criterion: str, n_samples: int | None = None, device: torch.device | str | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Slice dataset by a criterion (e.g., "age <= 60 and female == 1", "and/or") and return a batch with that subset of data.
        
        note:
        
            - The criterion should be a valid pandas expression (e.g., "age <= 60", "gender == 1", "and/or").
            - The criterion should be a boolean expression that can be evaluated by pandas.
            - The column names should be in self.config.z_names, or self.data.columns.
        
        All tensors are moved to `device` if provided; otherwise keep on dataset's device.
        """
        if n_samples is None:
            n_samples = len(self)
        mask = self.data.eval(criterion, local_dict=self.data)
        indices = mask.index[mask].tolist()  # type: ignore
        if len(indices) > n_samples:
            indices = np.random.choice(indices, size=n_samples, replace=False)
        else:
            print(f"Warning: only {len(indices)} samples satisfy criterion '{criterion}', which is less than n_samples={n_samples}.")
        if len(indices) == 0:
            raise ValueError(f"Empty batch after applying criterion '{criterion}'. Please check the criterion and the dataset.")
        return self.make_batch_from_indices(indices, device=device)


def _two_stage_shuffle_split(
    n_samples: int,
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample-wise two-stage split using ShuffleSplit.
    Stage 1: carve test
    Stage 2: carve val from remaining
    """
    assert 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0
    assert val_ratio + test_ratio < 1.0

    all_idx = np.arange(n_samples)

    # First: train+val vs test
    if test_ratio == 0.0:
        trainval_idx = np.arange(n_samples)
        test_idx = np.array([], dtype=int)
    else:
        # guard tiny splits by using an integer test_size
        n_test = int(round(n_samples * test_ratio))
        n_test = max(1, min(n_test, n_samples - 1))
        s1 = ShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        (trainval_idx, test_idx) = next(s1.split(all_idx))
        
    # Second: train vs val inside remaining
    remain = all_idx[trainval_idx]
    if val_ratio == 0.0:
        train_idx = remain
        val_idx = np.array([], dtype=int)
    else:
        # relative val ratio wrt remaining pool
        val_ratio_rel = val_ratio / (1.0 - test_ratio) if (1.0 - test_ratio) > 0 else 0.0
        # use integer count to avoid sklearn's "test_size=0.0" error on tiny sets
        n_val = int(round(remain.size * val_ratio_rel))
        n_val = max(1, min(n_val, max(remain.size - 1, 1)))  # ensure 1..(n-1)
        if remain.size <= 1:
            # degenerate: no room for a validation split
            train_idx = remain
            val_idx = np.array([], dtype=int)
        else:
            s2 = ShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
            (train_rel, val_rel) = next(s2.split(remain))
            train_idx = remain[train_rel]
            val_idx = remain[val_rel]
    test_idx = all_idx[test_idx]

    # Sort to keep deterministic order across loaders if you prefer
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _two_stage_group_shuffle_split(
    ids: np.ndarray,
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group-wise two-stage split using GroupShuffleSplit.
    Stage 1: carve test groups
    Stage 2: carve val groups from remaining groups
    """
    assert 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0
    assert val_ratio + test_ratio < 1.0

    all_idx = np.arange(len(ids))

    # First: train+val groups vs test groups
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    (trainval_mask, test_mask) = next(gss1.split(all_idx, groups=ids))
    trainval_idx = all_idx[trainval_mask]
    test_idx = all_idx[test_mask]

    # Second: train groups vs val groups inside remaining
    ids_remain = ids[trainval_idx]
    val_ratio_rel = val_ratio / (1.0 - test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_rel, random_state=seed)
    (train_mask_r, val_mask_r) = next(gss2.split(trainval_idx, groups=ids_remain))

    train_idx = trainval_idx[train_mask_r]
    val_idx = trainval_idx[val_mask_r]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
