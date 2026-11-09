"""train_boost.py: the explicit utility description written by the RUMBoost trainer must reproduce the boosters'
raw predictions exactly (that json is what the figures are drawn from), PCUF must stay monotone and predictive,
and every variant / the plain LightGBM baseline must run end to end on one split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import experiment_from_args, load_experiment  # noqa: E402
from train_boost import eval_piece, utilities_from_pieces  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists(),
                                reason="Swissmetro csv not present")
pytest.importorskip("lightgbm")


def _run(tmp_path, model_name, sets, rounds=60):
    exp = experiment_from_args("SM", "rumboost" if model_name != "gbdt" else "gbdt", model_name, splits="0")
    exp["overrides"] = {"boost_config": {"num_boost_round": rounds, "early_stopping_round": 20, "num_threads": 4}, **sets}
    plan = load_experiment(exp, out_root=tmp_path)
    tt = plan["trainer"]
    rec = tt.run(split_id=0, init_seed=42, overwrite=True)
    return tt, rec


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("rb"), "rumboost", {})


def test_pieces_reproduce_booster_predictions(official):
    tt, rec = official
    X = tt.X.iloc[:400]
    V = utilities_from_pieces(tt.util, X, tt.model.boosters, tt.best_iteration)
    for k, (b, s) in enumerate(zip(tt.model.boosters, tt.rum_structure)):
        raw = b.predict(X[s["columns"]], raw_score=True, num_iteration=tt.best_iteration)
        assert np.abs(V[:, k] - raw).max() < 1e-9, k
    assert rec["fit_info"]["n_params"] > 0 and (tt.run_dir(0, 42) / "rumboost_utility.json").exists()


def test_stairs_are_anchored_and_monotone(official):
    tt, _ = official
    for ens in tt.util["ensembles"]:
        alt = ens["alt"]
        for col, xname in ens["attrs"].items():
            piece = ens["pieces"][col]
            assert eval_piece(piece, np.array([0.0]))[0] == 0.0                      # anchored at x = 0
            d = int(tt.dcfg.mono_dirs.get(xname, 0))
            v = np.asarray(piece["values"])
            if d < 0 and len(v) > 1:
                assert (np.diff(v) <= 1e-12).all(), (alt, col)
            if d > 0 and len(v) > 1:
                assert (np.diff(v) >= -1e-12).all(), (alt, col)


def test_metrics_and_probabilities(official):
    tt, rec = official
    P = tt._predict_proba(np.arange(50))
    assert np.allclose(P.sum(1), 1.0) and (P >= 0).all()
    assert 0.5 < rec["test"]["nll"] < 1.1


def test_pcuf_smooths_and_stays_monotone(tmp_path):
    tt, rec = _run(tmp_path, "rumboost_pcuf", {"rum_config": {"smooth": True, "pcuf_candidates": [0, 5, 12]}})
    u = json.loads((tt.run_dir(0, 42) / "rumboost_utility.json").read_text(encoding="utf-8"))
    n_smoothed = 0
    for ens in u["ensembles"]:
        for col, xname in ens["attrs"].items():
            piece = ens["pieces"][col]
            if "x_knots" in piece:
                n_smoothed += 1
                xs = np.linspace(piece["x_knots"][0], piece["x_knots"][-1], 300)
                ys = eval_piece(piece, xs)
                if int(tt.dcfg.mono_dirs.get(xname, 0)) < 0:
                    assert (np.diff(ys) <= 1e-9).all(), (ens["alt"], col)
    assert "pcuf_knots" in rec["fit_info"] and rec["fit_info"]["val_nll_after_pcuf"] <= rec["fit_info"]["best_val_ce"] + 1e-9
    assert np.isfinite(rec["test"]["nll"])


@pytest.mark.parametrize("name,sets", [
    ("rumboost_nomono", {"rum_config": {"monotone": False}}),
    ("rumboost_noz", {"rum_config": {"socio": "none"}}),
    ("rumboost_fe", {"rum_config": {"socio": "functional_effect"}}),
])
def test_variants_run(tmp_path, name, sets):
    tt, rec = _run(tmp_path, name, sets, rounds=30)
    assert np.isfinite(rec["test"]["nll"])
    kinds = [e["kind"] for e in tt.util["ensembles"]]
    if name == "rumboost_fe":
        assert kinds.count("trees") == len(tt.dcfg.alt_names) and len(kinds) == 2 * len(tt.dcfg.alt_names)
        X = tt.X.iloc[:100]
        V = utilities_from_pieces(tt.util, X, tt.model.boosters, tt.best_iteration)
        raw = sum(b.predict(X[s["columns"]], raw_score=True, num_iteration=tt.best_iteration)
                  for b, s in zip(tt.model.boosters[:2], tt.rum_structure[:2]))
        assert np.abs(V[:, 0] - raw).max() < 1e-9
    if name == "rumboost_noz":
        assert all(set(e["columns"]).isdisjoint(tt.z_cols) for e in tt.util["ensembles"])


def test_gbdt_runs_and_masks(tmp_path):
    tt, rec = _run(tmp_path, "gbdt", {}, rounds=40)
    P = tt._predict_proba(np.arange(20))
    assert np.allclose(P.sum(1), 1.0) and np.isfinite(rec["test"]["nll"]) and rec["fit_info"]["n_params"] > 0
