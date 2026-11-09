"""plots/extract/rumboost_curves.py must turn a run's rumboost_utility.json into the csv layout the fig scripts read."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import experiment_from_args, load_experiment  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists(),
                                reason="Swissmetro csv not present")
pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rb")
    exps = {}
    for name, over in (("rumboost", {}), ("rumboost_pcuf", {"rum_config": {"smooth": True, "pcuf_candidates": [0, 6, 12]}}),
                       ("rumboost_fe", {"rum_config": {"socio": "functional_effect"}})):
        exp = experiment_from_args("SM", "rumboost", name, splits="0")
        exp["overrides"] = {"boost_config": {"num_boost_round": 40, "early_stopping_round": 10, "num_threads": 4}, **over}
        load_experiment(exp, out_root=root)["trainer"].run(0, 42, overwrite=True)
        exps[name] = exp
    return root, exps


def test_extract_writes_every_table(runs, tmp_path):
    from plots.extract.rumboost_curves import extract

    root, exps = runs
    out = tmp_path / "fd"
    for name, exp in exps.items():
        paths = extract(exp, [0], results_root=root, out_dir=out, n_samples=300, n_points=50)
        assert {"asu", "asu_agg"} <= set(paths) and (("z_offset" in paths) == (name != "rumboost_fe"))
    agg = pd.read_csv(out / "asu_curves_agg.csv")
    assert set(agg.model_name) == set(exps) and {"x", "mean", "lo", "hi", "general", "n_splits"} <= set(agg.columns)
    assert np.allclose(agg["mean"], agg["general"])                         # no heterogeneity: mean == general
    assert (agg.groupby(["model_name", "alt", "attr"])["mean"].first() == 0).all()   # anchored at x = 0
    mu = pd.read_csv(out / "mu_curves_agg.csv")
    assert set(mu.model_name) == {"rumboost_pcuf"}                          # derivative only for the smoothed variant
    zo = pd.read_csv(out / "z_offset.csv")
    assert set(zo.param) == {"ASC", "MALE", "FIRST", "AGE_1", "AGE_2"} | {f"PURPOSE_{i}" for i in range(1, 9)}
    assert set(zo.model_name) == {"rumboost", "rumboost_pcuf"} and (zo.groupby("model_name").size() == 3 * 13).all()
    vot = pd.read_csv(out / "vot_individual.csv.gz")
    assert set(vot.model_name) == {"rumboost_pcuf"} and {"mu_time", "mu_cost", "vot", "idx"} <= set(vot.columns)
    assert (out / "attr_samples.csv").exists() and (out / "individuals.csv.gz").exists()
    ind = pd.read_csv(out / "individuals.csv.gz")
    assert len(ind) == 300 and {"idx", "ID", "MALE", "CAR__TT-CAR", "av"} <= set(ind.columns)
