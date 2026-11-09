"""synthetic/generate.py: the I-spline DGP must produce monotone curves for EVERY individual, reproduce its
theory shapes, be seed-reproducible, and round-trip through syn_truth.json (which is what the figures read)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from synthetic.generate import CONFIG, LPMC_CSV, TRUTH_JSON, SynDGP, covariate_frame, theory_curve

pytestmark = pytest.mark.skipif(not LPMC_CSV.exists(), reason="LPMC csv not present (data/ is not in git)")


@pytest.fixture(scope="module")
def dgp():
    df = covariate_frame(LPMC_CSV, repeat=1)
    return SynDGP(CONFIG).fit_from_df(df), df


def _grid(d, name, n=400):
    return np.linspace(d.ranges[name][0], d.ranges[name][1], n)


def test_population_curve_matches_theory(dgp):
    d, _ = dgp
    for name in d.attr_map:
        x = _grid(d, name)
        err = np.abs(d.general_curve(name, x) - theory_curve(d.cfg["SHAPES"][name], x)).max()
        tol = 0.15 if name == "CAR_TT" else 0.05          # CAR: degree-1 fit of a kinked curve on fixed knots
        assert err < tol, (name, err)


def test_increments_anchored_and_nonpositive(dgp):
    d, _ = dgp
    for name in d.attr_map:
        a = d.incr[name]
        assert a[0] == 0.0
        assert (a <= 0.0).all(), name
        assert d.kcoef[name][0] == 0.0 and (0.0 <= d.kcoef[name]).all() and (d.kcoef[name] <= 1.0).all()


def test_every_individual_curve_is_monotone(dgp):
    d, _ = dgp
    tau = np.random.default_rng(0).uniform(-0.999, 0.999, 500)
    for name in d.attr_map:
        Y = d.individual_curves(name, _grid(d, name), tau)
        assert (np.diff(Y, axis=1) <= 1e-9).all(), name
        assert np.allclose(Y[:, 0], 0.0)                   # f(0) = 0 for everyone


def test_tau_zero_is_the_population_curve(dgp):
    d, _ = dgp
    for name in d.attr_map:
        x = _grid(d, name)
        assert np.allclose(d.individual_curves(name, x, np.zeros(3)), d.general_curve(name, x)[None, :])


def test_heterogeneity_changes_the_slope_where_k_is_positive(dgp):
    d, _ = dgp
    for name in d.attr_map:
        x = _grid(d, name)
        hi, lo = d.individual_curves(name, x, np.array([0.9, -0.9]))
        if d.kcoef[name].max() > 0:
            assert (hi <= lo + 1e-12).all() and hi.min() < lo.min() - 1e-3      # tau > 0 -> steeper (more negative)
        else:
            assert np.allclose(hi, lo)                                          # PT_TT: kmax = 0 -> everyone equal


def test_truth_roundtrip(dgp):
    d, _ = dgp
    d2 = SynDGP.from_truth(json.loads(json.dumps(d.truth_dict())))
    tau = np.linspace(-0.8, 0.8, 5)
    for name in d.attr_map:
        x = _grid(d, name)
        assert np.allclose(d2.individual_curves(name, x, tau), d.individual_curves(name, x, tau))
        assert np.allclose(d2.greville(name), d.greville(name))


def test_simulation_is_seeded(dgp):
    d, df = dgp
    sub = df.head(3000)
    s1 = d.simulate(sub, np.random.default_rng(7))
    s2 = d.simulate(sub, np.random.default_rng(7))
    s3 = d.simulate(sub, np.random.default_rng(8))
    assert (s1["choice_sim"].to_numpy() == s2["choice_sim"].to_numpy()).all()
    assert (s1["choice_sim"].to_numpy() != s3["choice_sim"].to_numpy()).any()
    assert np.allclose(s1[["P_WALK", "P_PT", "P_CAR", "P_CYCLE"]].sum(axis=1), 1.0)
    assert np.allclose(d.utilities(sub), s1[["V_WALK", "V_PT", "V_CAR", "V_CYCLE"]].to_numpy())


@pytest.mark.skipif(not TRUTH_JSON.exists(), reason="synthetic/syn_truth.json not generated")
def test_committed_truth_file_matches_the_code():
    """Re-fitting with the committed config on the same covariates must give the committed knots / a / k:
    guards against silently changing the DGP without regenerating syn_truth.json (and the csv)."""
    t = json.loads(TRUTH_JSON.read_text(encoding="utf-8"))
    df = covariate_frame(LPMC_CSV, repeat=t["repeat"])
    d = SynDGP(t["config"], kmax_scale=1.0).fit_from_df(df)
    for name, spec in t["attrs"].items():
        assert np.allclose(d.knots[name], spec["knots"]), name
        assert np.allclose(d.incr[name], spec["a"], atol=1e-8), name
        assert np.allclose(d.kcoef[name], spec["k"], atol=1e-10), name
    for z, st in t["z_standardisation"].items():
        assert np.isclose(d.z_stats[z][0], st["mean"]) and np.isclose(d.z_stats[z][1], st["std"])
