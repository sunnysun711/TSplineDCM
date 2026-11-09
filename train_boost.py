# train_boost.py — gradient-boosting baselines on the shared splits: RUMBoost (Salvadé & Hillel 2025) and a plain
# LightGBM multiclass classifier. Both plug into BaseTrainer (fit / predict_proba), so metrics.json, predictions.npz,
# analyze.py and tune.py treat them exactly like the torch models.
"""
RUMBoost (backend: rumboost)
    One LightGBM binary ensemble per alternative, trained jointly with a softmax cross-entropy objective. The
    official specification (rumboost/models.py SwissMetro / LPMC) is reproduced from the dataset yaml:
      * each alternative's ensemble sees its OWN attributes (configs/datasets alt_xs) with the yaml's monotone
        direction (mono_dirs) as a LightGBM monotone constraint, plus the individual characteristics z without
        constraint (rum_config.socio: utility);
      * interaction constraints let every tree split on ONE feature only, so each ensemble is a sum of
        one-dimensional step functions ("GBUV" in the paper): one stair per attribute, one stair per z.
        There is no taste heterogeneity — the same attribute utility for everyone; z only shifts the level.
    Variants (rum_config):
      socio: utility | none | functional_effect   z inside each utility (official) / no z / a separate z ensemble
                                                  per alternative whose trees may split on several z ("functional
                                                  effect" model of the paper)
      monotone: true | false                       drop the monotone constraints (ablation)
      smooth: true                                 PCUF: replace each attribute stair by a monotone PCHIP through
                                                  the stair means; the number of knots per attribute is chosen
                                                  greedily on the VALIDATION set (the package optimises on the test
                                                  set, we never touch it); prediction uses the smoothed utilities.
    Availability: rumboost's softmax ignores av; all three datasets have full availability, and predictions are
    masked anyway (metrics.probs_from_utilities).
    Every run writes rumboost_model.json (the boosters at the best iteration) and rumboost_utility.json — an
    explicit, lightgbm-free description of the fitted utilities (stairs: thresholds + values anchored at x = 0;
    PCUF knots), which plots/extract/rumboost_curves.py reads.

LightGBM multiclass (backend: lightgbm, model gbdt)
    Flat feature vector (every alternative's attributes + z) -> multiclass softmax, early stopping on the
    validation log-loss. No utility structure: a pure predictive reference, probe-styled in figures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dcm_data import DcmConfig
from metrics import probs_from_utilities
from trainer_base import BaseTrainer

PCUF_CANDIDATES = (0, 4, 6, 8, 12, 16, 24, 32)  # knots per attribute tried by the greedy validation search; 0 = keep stairs
DEFAULT_THREADS = 8


def _shim_biogeme_logging() -> None:
    """rumboost 1.0.0 imports biogeme.logging (renamed biogeme_logging in biogeme >= 3.2.13). Alias it back."""
    if "biogeme.logging" in sys.modules:
        return
    try:
        import biogeme.logging  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import biogeme.biogeme_logging as _bl
    except ModuleNotFoundError:
        return
    sys.modules["biogeme.logging"] = _bl


# --------------------------------------------------------------------------- #
# explicit one-dimensional utility pieces (what rumboost_utility.json stores)
# --------------------------------------------------------------------------- #
def _tree_thresholds(tree: dict, out: dict[int, set]) -> None:
    if "split_feature" not in tree:
        return
    out.setdefault(int(tree["split_feature"]), set()).add(float(tree["threshold"]))
    _tree_thresholds(tree["left_child"], out)
    _tree_thresholds(tree["right_child"], out)


def _feature_thresholds(booster, num_iteration: int | None) -> dict[str, np.ndarray]:
    """feature name -> sorted unique split thresholds over all trees of the booster."""
    dump = booster.dump_model(num_iteration=num_iteration)
    names = dump["feature_names"]
    acc: dict[int, set] = {}
    for t in dump["tree_info"]:
        _tree_thresholds(t["tree_structure"], acc)
    return {names[i]: np.array(sorted(s), dtype=np.float64) for i, s in acc.items()}


def _stairs_from_booster(booster, cols: list[str], num_iteration: int | None) -> tuple[dict[str, dict], float]:
    """Exact decomposition of a no-interaction booster: V(x) = const + sum_f stair_f(x_f), each stair anchored at
    stair_f(0) = 0. Uses raw predictions with every other feature at 0, so any tree depth is handled."""
    thr = _feature_thresholds(booster, num_iteration)
    zero = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    const = float(booster.predict(zero, raw_score=True, num_iteration=num_iteration)[0])
    stairs: dict[str, dict] = {}
    for f in cols:
        t = thr.get(f)
        if t is None or t.size == 0:                       # feature never used
            stairs[f] = {"thresholds": [], "values": [0.0]}
            continue
        # one representative point per interval (-inf, t1], (t1, t2], ..., (tm, inf)
        reps = np.concatenate([[t[0] - 1.0], (t[:-1] + t[1:]) / 2.0, [t[-1] + 1.0]])
        X = pd.DataFrame(np.zeros((len(reps), len(cols))), columns=cols)
        X[f] = reps
        v = booster.predict(X, raw_score=True, num_iteration=num_iteration) - const
        v0 = v[int(np.searchsorted(t, 0.0, side="left"))]          # value of the interval that contains x = 0
        stairs[f] = {"thresholds": t.tolist(), "values": (v - v0).tolist()}
        const += v0
    return stairs, const


def eval_stairs(piece: dict, x: np.ndarray) -> np.ndarray:
    t = np.asarray(piece["thresholds"], dtype=np.float64)
    v = np.asarray(piece["values"], dtype=np.float64)
    if t.size == 0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return v[np.searchsorted(t, np.asarray(x, dtype=np.float64), side="left")]   # x <= t_i -> left, as LightGBM


def eval_pchip(piece: dict, x: np.ndarray) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    f = PchipInterpolator(np.asarray(piece["x_knots"], dtype=np.float64), np.asarray(piece["y_knots"], dtype=np.float64),
                          extrapolate=True)
    return f(np.asarray(x, dtype=np.float64))


def eval_piece(piece: dict, x: np.ndarray) -> np.ndarray:
    return eval_pchip(piece, x) if "x_knots" in piece else eval_stairs(piece, x)


def stair_means(piece: dict, x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean x of the training data inside each stair interval, stair value there) for non-empty intervals."""
    t = np.asarray(piece["thresholds"], dtype=np.float64)
    v = np.asarray(piece["values"], dtype=np.float64)
    idx = np.searchsorted(t, x_train, side="left")
    xm, ym = [], []
    for i in range(len(v)):
        sel = idx == i
        if sel.any():
            xm.append(float(x_train[sel].mean()))
            ym.append(float(v[i]))
    return np.asarray(xm), np.asarray(ym)


def pchip_piece(piece: dict, x_train: np.ndarray, n_knots: int, x_lo: float = 0.0, x_hi: float | None = None) -> dict:
    """PCUF piece: knots on the stair means (mean x of the training rows inside a stair, stair value there), PCHIP
    through them -> monotone wherever the stairs are. End knots at x_lo (= 0, the AS-UCF anchor: the stairs are
    flat left of the first threshold, so the spline is flat there too) and x_hi (the column's global maximum, so
    the spline is never extrapolated inside the prediction / plotting range).
    Knot candidates sit at training-data QUANTILES (as rumboost.utility_smoothing.optimal_knots_position starts
    them), not equally spaced over the range: equal spacing puts most knots where there is no data and the
    interpolant then drifts away from the stairs by whole utility units in the dense region."""
    xm, ym = stair_means(piece, x_train)
    x_lo = float(min(x_lo, np.min(x_train)))
    x_hi = float(max(np.max(x_train), x_hi if x_hi is not None else -np.inf))
    if n_knots + 1 >= len(xm):
        xk, yk = xm, ym
    else:
        cand = np.quantile(x_train, np.linspace(0.0, 1.0, n_knots + 1))
        sel = np.unique(np.clip(np.searchsorted(xm, cand, side="left") - 1, 0, len(xm) - 1))
        xk, yk = xm[sel], ym[sel]
    if xk[0] != x_lo:
        xk, yk = np.insert(xk, 0, x_lo), np.insert(yk, 0, eval_stairs(piece, np.array([x_lo]))[0])
    if xk[-1] != x_hi:
        xk, yk = np.append(xk, x_hi), np.append(yk, eval_stairs(piece, np.array([x_hi]))[0])
    keep = np.concatenate([[True], np.diff(xk) > 0])                  # PCHIP needs strictly increasing knots
    return {"x_knots": xk[keep].tolist(), "y_knots": yk[keep].tolist(), "n_knots": int(n_knots)}


def utilities_from_pieces(util: dict, X: pd.DataFrame, boosters: list | None = None,
                          num_iteration: int | None = None) -> np.ndarray:
    """V [N, I] from rumboost_utility.json content (+ live boosters for the non-decomposable ensembles)."""
    I = len(util["alt_names"])
    V = np.zeros((len(X), I), dtype=np.float64)
    for ens in util["ensembles"]:
        j = int(ens["alt_index"])
        if ens["kind"] == "pieces":
            v = np.full(len(X), float(ens["const"]))
            for f, piece in ens["pieces"].items():
                v = v + eval_piece(piece, X[f].to_numpy(dtype=np.float64))
        else:                                                           # trees with interactions: ask lightgbm
            assert boosters is not None, "non-decomposable ensemble needs the boosters"
            v = boosters[int(ens["booster"])].predict(X[ens["columns"]], raw_score=True, num_iteration=num_iteration)
        V[:, j] += v
    return V


# --------------------------------------------------------------------------- #
# shared feature frame
# --------------------------------------------------------------------------- #
class _BoostBase(BaseTrainer):
    def __init__(self, dataset_name: str, dcm_config: DcmConfig, model_name: str, spec: dict[str, Any],
                 **base_kwargs: Any) -> None:
        super().__init__(dataset_name=dataset_name, dcm_config=dcm_config, model_name=model_name, **base_kwargs)
        self.spec = dict(spec)
        cols = list(self.df.columns)
        d = self.dcfg
        self.x_cols = {alt: [cols[i] for i in d.alt_x_cols[alt]] for alt in d.alt_names}
        self.z_cols = [cols[i] for i in d.z_cols] if d.z_cols is not None else list(d.z_names)
        feats: list[str] = []
        for alt in d.alt_names:
            feats += [c for c in self.x_cols[alt] if c not in feats]
        feats += [c for c in self.z_cols if c not in feats]
        self.features = feats
        self.X = self.df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float64)
        self.choice = self.y.argmax(axis=1).astype(int)
        # yaml x name (e.g. TT-CAR) per alternative, aligned with x_cols
        self.x_names = {alt: list(d.alt_xs[alt]) for alt in d.alt_names}
        tc = self.spec.get("train_config") or {}
        self.round_cap = int(tc["epochs"]) if tc.get("epochs") else None    # tune.py --max-epochs lands here

    def _threads(self) -> int:
        return int((self.spec.get("boost_config") or {}).get("num_threads", DEFAULT_THREADS))


# --------------------------------------------------------------------------- #
# RUMBoost
# --------------------------------------------------------------------------- #
class RUMBoostTrainer(_BoostBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        rc = dict(self.spec.get("rum_config") or {})
        self.socio = str(rc.get("socio", "utility"))
        if self.socio not in ("utility", "none", "functional_effect"):
            raise ValueError(f"rum_config.socio must be utility | none | functional_effect, got {self.socio!r}")
        self.monotone = bool(rc.get("monotone", True))
        self.smooth = bool(rc.get("smooth", False))
        self.pcuf_candidates = tuple(int(c) for c in rc.get("pcuf_candidates", PCUF_CANDIDATES))
        self.rum_structure = self._structure()
        self.model = None
        self.util: dict | None = None
        self.best_iteration: int | None = None

    def _structure(self) -> list[dict[str, Any]]:
        d = self.dcfg
        structure = []
        for alt in d.alt_names:
            cols = list(self.x_cols[alt])
            mono = [int(d.mono_dirs.get(x, 0)) if self.monotone else 0 for x in self.x_names[alt]]
            if self.socio == "utility":
                cols += self.z_cols
                mono += [0] * len(self.z_cols)
            structure.append({"columns": cols, "monotone_constraints": mono,
                              "interaction_constraints": [[i] for i in range(len(cols))], "categorical_feature": []})
            if self.socio == "functional_effect":
                structure.append({"columns": list(self.z_cols), "monotone_constraints": [0] * len(self.z_cols),
                                  "interaction_constraints": [list(range(len(self.z_cols)))], "categorical_feature": []})
        return structure

    def _params(self, seed: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
        bc = dict(self.spec.get("boost_config") or {})
        p = {"objective": "multiclass", "num_classes": len(self.dcfg.alt_names), "verbosity": -1, "metric": "None",
             "num_threads": self._threads(), "seed": int(seed),
             "learning_rate": float(bc.get("learning_rate", 0.1)), "num_leaves": int(bc.get("num_leaves", 7)),
             "min_data_in_leaf": int(bc.get("min_data_in_leaf", 20)), "lambda_l2": float(bc.get("lambda_l2", 0.0)),
             "lambda_l1": float(bc.get("lambda_l1", 0.0)), "max_bin": int(bc.get("max_bin", 255)),
             "min_gain_to_split": float(bc.get("min_gain_to_split", 0.0)),
             "early_stopping_round": int(bc.get("early_stopping_round", 100))}
        p_fe = None
        if self.socio == "functional_effect":
            p_fe = {**p, "num_leaves": int(bc.get("num_leaves_fe", 15)), "learning_rate": float(bc.get("learning_rate_fe", p["learning_rate"]))}
        return p, p_fe

    def _fit(self, train_idx, val_idx, test_idx, init_seed, run_dir: Path) -> dict[str, Any]:
        _shim_biogeme_logging()
        import warnings

        from lightgbm import Dataset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rumboost.rumboost import rum_train

        params, params_fe = self._params(init_seed)
        rounds = int((self.spec.get("boost_config") or {}).get("num_boost_round", 3000))
        if self.round_cap:
            rounds = min(rounds, self.round_cap)
        Xtr, Xva = self.X.iloc[train_idx].reset_index(drop=True), self.X.iloc[val_idx].reset_index(drop=True)
        train_set = Dataset(Xtr, label=self.choice[train_idx], free_raw_data=False, params={"verbosity": -1})
        valid_set = Dataset(Xva, label=self.choice[val_idx], free_raw_data=False, params={"verbosity": -1})
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = rum_train(params, train_set, rum_structure=self.rum_structure, num_boost_round=rounds,
                                   valid_sets=[valid_set], params_fe=params_fe)
        self.best_iteration = int(self.model.best_iteration) or None
        boost_seconds = time.time() - t0
        self.model.save_model(run_dir / "rumboost_model.json", num_iteration=self.best_iteration)

        # explicit utilities (+ PCUF smoothing on the validation set)
        self.util = self._explicit_utilities()
        pcuf_info = self._pcuf(train_idx, val_idx) if self.smooth else {}
        (run_dir / "rumboost_utility.json").write_text(json.dumps(self.util), encoding="utf-8")

        n_leaves = sum(sum(t["num_leaves"] for t in b.dump_model(num_iteration=self.best_iteration)["tree_info"])
                       for b in self.model.boosters)
        n_trees = [int(b.num_trees()) if self.best_iteration is None else int(self.best_iteration) for b in self.model.boosters]
        return {"n_params": int(n_leaves), "n_leaves": int(n_leaves), "n_trees_per_ensemble": n_trees,
                "best_iteration": self.best_iteration, "best_val_ce": float(self.model.best_score),
                "boost_seconds": round(boost_seconds, 2), "socio": self.socio, "monotone": self.monotone,
                "smooth": self.smooth, "rounds_cap": rounds, "params": params, **pcuf_info,
                "note": "n_params = total number of leaves over all trees at the best iteration"}

    def _explicit_utilities(self) -> dict:
        ens = []
        for k, (booster, struct) in enumerate(zip(self.model.boosters, self.rum_structure)):
            j = k // 2 if self.socio == "functional_effect" else k
            decomposable = all(len(g) == 1 for g in struct["interaction_constraints"])
            if decomposable:
                stairs, const = _stairs_from_booster(booster, struct["columns"], self.best_iteration)
                ens.append({"booster": k, "alt_index": j, "alt": self.dcfg.alt_names[j], "kind": "pieces",
                            "columns": struct["columns"], "const": const, "pieces": stairs,
                            "attrs": {c: x for c, x in zip(self.x_cols[self.dcfg.alt_names[j]], self.x_names[self.dcfg.alt_names[j]])
                                      if c in struct["columns"]}})
            else:
                ens.append({"booster": k, "alt_index": j, "alt": self.dcfg.alt_names[j], "kind": "trees",
                            "columns": struct["columns"]})
        return {"alt_names": list(self.dcfg.alt_names), "features": self.features, "z_cols": self.z_cols,
                "x_cols": self.x_cols, "socio": self.socio, "monotone": self.monotone, "smooth": self.smooth,
                "best_iteration": self.best_iteration, "ensembles": ens}

    def _val_nll(self, X: pd.DataFrame, y: np.ndarray, av: np.ndarray) -> float:
        V = utilities_from_pieces(self.util, X, self.model.boosters, self.best_iteration)
        P = probs_from_utilities(V, av)
        return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())

    def _pcuf(self, train_idx, val_idx) -> dict:
        """Greedy per-attribute knot count on the validation NLL (candidates PCUF_CANDIDATES; 0 keeps the stairs)."""
        Xtr, Xva = self.X.iloc[train_idx], self.X.iloc[val_idx]
        yva, avva = self.choice[val_idx], self.av[val_idx]
        base = self._val_nll(Xva, yva, avva)
        chosen: dict[str, int] = {}
        for ens in self.util["ensembles"]:
            if ens["kind"] != "pieces":
                continue
            for col in ens["attrs"]:                                # own attributes only; z stairs stay stairs
                stair = ens["pieces"][col]
                if len(stair["thresholds"]) < 2:
                    continue
                xtr = Xtr[col].to_numpy(dtype=np.float64)
                best_piece, best_nll, best_n = stair, base, 0
                for n in self.pcuf_candidates:
                    if n == 0:
                        continue
                    cand = pchip_piece(stair, xtr, n, x_lo=0.0, x_hi=float(self.X[col].max()))
                    ens["pieces"][col] = cand
                    nll = self._val_nll(Xva, yva, avva)
                    if nll < best_nll - 1e-9:
                        best_piece, best_nll, best_n = cand, nll, n
                ens["pieces"][col] = best_piece
                base = best_nll
                chosen[f"{ens['alt']}:{col}"] = best_n
                if best_n:
                    best_piece["stairs"] = {"thresholds": stair["thresholds"], "values": stair["values"]}
        return {"pcuf_knots": chosen, "val_nll_after_pcuf": base}

    def _predict_proba(self, idx: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.util is not None, "_predict_proba called before _fit"
        V = utilities_from_pieces(self.util, self.X.iloc[idx], self.model.boosters, self.best_iteration)
        return probs_from_utilities(V, self.av[idx])

    def _config_snapshot(self) -> dict[str, Any]:
        return {"backend": "rumboost", "boost_config": self.spec.get("boost_config") or {},
                "rum_config": {"socio": self.socio, "monotone": self.monotone, "smooth": self.smooth},
                "train_config": self.spec.get("train_config") or {}}


# --------------------------------------------------------------------------- #
# plain LightGBM multiclass
# --------------------------------------------------------------------------- #
class LightGBMTrainer(_BoostBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.best_iteration: int | None = None

    def _fit(self, train_idx, val_idx, test_idx, init_seed, run_dir: Path) -> dict[str, Any]:
        import lightgbm as lgb

        bc = dict(self.spec.get("boost_config") or {})
        rounds = int(bc.get("num_boost_round", 3000))
        if self.round_cap:
            rounds = min(rounds, self.round_cap)
        params = {"objective": "multiclass", "num_class": len(self.dcfg.alt_names), "metric": "multi_logloss",
                  "verbosity": -1, "num_threads": self._threads(), "seed": int(init_seed),
                  "learning_rate": float(bc.get("learning_rate", 0.05)), "num_leaves": int(bc.get("num_leaves", 31)),
                  "min_data_in_leaf": int(bc.get("min_data_in_leaf", 20)), "lambda_l2": float(bc.get("lambda_l2", 0.0)),
                  "feature_fraction": float(bc.get("feature_fraction", 1.0)), "bagging_fraction": float(bc.get("bagging_fraction", 1.0)),
                  "bagging_freq": int(bc.get("bagging_freq", 0)), "max_bin": int(bc.get("max_bin", 255))}
        tr = lgb.Dataset(self.X.iloc[train_idx], label=self.choice[train_idx], params={"verbosity": -1})
        va = lgb.Dataset(self.X.iloc[val_idx], label=self.choice[val_idx], reference=tr, params={"verbosity": -1})
        self.model = lgb.train(params, tr, num_boost_round=rounds, valid_sets=[va], valid_names=["val"],
                               callbacks=[lgb.early_stopping(int(bc.get("early_stopping_round", 100)), verbose=False)])
        self.best_iteration = int(self.model.best_iteration) or None
        self.model.save_model(run_dir / "lightgbm_model.txt", num_iteration=self.best_iteration)
        n_leaves = sum(t["num_leaves"] for t in self.model.dump_model(num_iteration=self.best_iteration)["tree_info"])
        return {"n_params": int(n_leaves), "n_leaves": int(n_leaves), "best_iteration": self.best_iteration,
                "best_val_logloss": float(self.model.best_score["val"]["multi_logloss"]), "rounds_cap": rounds,
                "params": params, "note": "n_params = total number of leaves at the best iteration"}

    def _predict_proba(self, idx: np.ndarray) -> np.ndarray:
        assert self.model is not None, "_predict_proba called before _fit"
        P = self.model.predict(self.X.iloc[idx], num_iteration=self.best_iteration)
        P = P * (self.av[idx] > 0)                                       # no mass on unavailable alternatives
        return P / P.sum(axis=1, keepdims=True)

    def _config_snapshot(self) -> dict[str, Any]:
        return {"backend": "lightgbm", "boost_config": self.spec.get("boost_config") or {},
                "train_config": self.spec.get("train_config") or {}}
