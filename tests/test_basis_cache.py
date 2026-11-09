"""The row basis cache (models/tspline/basis_cache.py) must not change the forward pass.

For SM (linear ASC head) and LPMC (z-spline ASC head) we build the paper model on CPU, take a batch of real
rows, and compare model(batch) with the caches on against the plain evaluation from x. The splines gather the
very basis they used to evaluate, so SM and LPMC (default, per-z head) must be torch.equal (bit for bit). With
model_config.zhead_fused the LPMC z-spline head is one matmul over all z (the sum no longer runs z by z), so
there we allow float32 rounding (1e-5). Then the same after a grid update (the caches must notice the new grids
and rebuild) and after a state-dict load (the other way a grid can change). Batches without "idx" must still
work (the asu_* helpers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists() or not (ROOT / "data" / "LPMC" / "lpmc_rumboost_version" / "lpmc_processed_rumboost.csv").exists()
                                or not (ROOT / "splits" / "SM.npz").exists() or not (ROOT / "splits" / "LPMC_officialtest.npz").exists(),
                                reason="data csv / splits not present (data/ and splits/ are not in git)")
EXPS = {"SM": ROOT / "configs" / "exp" / "SM_tspline_tuned.yaml",
        "LPMC": ROOT / "configs" / "exp" / "LPMC_tspline_tuned.yaml"}


def _caches_off(model):
    """Detach every cache (so forward evaluates from x) and return what to put back."""
    saved = {}
    for name, util in model.sp_utils.items():
        saved[name] = util._basis_cache
        util._basis_cache = None
    head = model.asc_learner
    if hasattr(head, "_basis_caches"):
        saved["__head__"] = (head._basis_caches, head._fused)
        head._basis_caches = None
        head._fused = None
    return saved


def _caches_on(model, saved):
    for name, util in model.sp_utils.items():
        util._basis_cache = saved[name]
    if "__head__" in saved:
        model.asc_learner._basis_caches, model.asc_learner._fused = saved["__head__"]


def _outputs(model, batch, train: bool):
    model.train(train)
    with torch.no_grad():
        out = model(batch, sample=train)
    return {k: out[k].detach().clone() for k in ("V", "P", "kld") if k in out}


def _assert_same(a, b, what, exact: bool):
    for k in a:
        diff = (a[k] - b[k]).abs().max().item()
        if exact:
            assert torch.equal(a[k], b[k]), f"{what}: {k} differs (max |diff| {diff:.3e})"
        else:
            assert torch.allclose(a[k], b[k], atol=1e-5, rtol=1e-6), f"{what}: {k} differs (max |diff| {diff:.3e})"


@pytest.mark.parametrize("dataset, zhead_fused", [("SM", False), ("LPMC", False), ("LPMC", True)])
def test_cached_forward_is_bit_exact(dataset, zhead_fused, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from run import load_experiment

    trainer = load_experiment(str(EXPS[dataset]), out_root=tmp_path)["trainer"]
    trainer.mcfg.zhead_fused = zhead_fused
    trainer.prepare(seed=0, device="cpu", float_type=torch.float32)   # build_model builds the caches; float32 like the GPU runs
    model, ds = trainer.model, trainer.ds
    assert model.basis_cache_bytes() > 0
    exact = not zhead_fused      # the fused z-spline head sums in another order
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=512, replace=False)
    batch = ds.make_batch_from_indices(idx)

    for train in (True, False):
        with_cache = _outputs(model, batch, train)
        saved = _caches_off(model)
        plain = _outputs(model, batch, train)
        _caches_on(model, saved)
        _assert_same(with_cache, plain, f"{dataset} train={train}", exact)

    # a batch without row ids takes the plain path
    no_idx = {k: v for k, v in batch.items() if k != "idx"}
    if model.mcfg.TR_per_sample:
        no_idx["idx"] = batch["idx"]           # the eps bank needs it; the splines get idx=None through data.get
        model.eval()
        with torch.no_grad():
            v_no_idx = model(no_idx, sample=False)["V"]
        _assert_same({"V": v_no_idx}, {"V": _outputs(model, batch, False)["V"]}, f"{dataset} batch without idx", exact)

    # grid update -> caches rebuild themselves and still agree
    gu = {"enabled": True, "every": 0, "warmup": 1, "epoch_cap": 100, "sample_per_x": len(ds)}
    train_idx = rng.choice(len(ds), size=min(len(ds), 20000), replace=False)
    trainer._maybe_update_grid(1, train_idx, gu)
    with_cache = _outputs(model, batch, True)
    saved = _caches_off(model)
    plain = _outputs(model, batch, True)
    _caches_on(model, saved)
    _assert_same(with_cache, plain, f"{dataset} after grid update", exact)

    # state-dict load (another way a grid can change) -> versions bump, rebuild, still equal
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    with_cache = _outputs(model, batch, False)
    saved = _caches_off(model)
    plain = _outputs(model, batch, False)
    _caches_on(model, saved)
    _assert_same(with_cache, plain, f"{dataset} after state-dict load", exact)
