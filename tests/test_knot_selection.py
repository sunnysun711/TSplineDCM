"""Knot-selection plumbing: configurable strict floor (0 = increments may vanish), degree-1 splines, and the two
structural penalties. Defaults must leave the old models bit-exact (floor 0.01, penalties off -> no "penalty" key)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.config import ModelConfig  # noqa: E402
from models.tspline.mono_coef import STRICT_DELTA_FLOOR, MonoCoef, reverse_cumulative_monotonic  # noqa: E402
from run import load_dataset_config  # noqa: E402


def test_defaults_are_the_old_behaviour():
    m = ModelConfig()
    assert m.MC_delta_floor == STRICT_DELTA_FLOOR == 0.01 and m.SP_l2_incr == 0.0 and m.SP_tv_slope == 0.0
    mc = MonoCoef(num=6, k=3, mono=-1, mono_type="strict")
    assert mc.delta_floor == 0.01
    a = -torch.diff(mc())
    assert (a >= 0.01 - 1e-6).all()                       # every increment carries the floor


@pytest.mark.parametrize("k", [1, 3])
def test_zero_floor_lets_increments_vanish_and_stays_monotone(k):
    torch.manual_seed(0)
    mc = MonoCoef(num=60, k=k, mono=-1, mono_type="strict", delta_floor=0.0)
    with torch.no_grad():
        mc.coef_raw[10:40] = -30.0                        # softplus(-30) ~ 1e-13: these bases carry no drop
    c = mc()
    a = -torch.diff(c)
    assert (a >= -1e-9).all() and a[10:39].max() < 1e-6 and a[:9].min() > 1e-3
    # round trip with floor 0
    rec = mc.reverse(c.detach())
    assert torch.allclose(mc.reverse(mc().detach()), rec) and torch.isfinite(rec).all()


def test_strict_reverse_clamps_instead_of_nan():
    target = torch.tensor([0.0, -0.5, -0.4, -1.0])        # locally NON-monotone target (as a grid refit can produce)
    raw = reverse_cumulative_monotonic(target, mono=-1, mono_type="strict", delta_floor=0.0)
    assert torch.isfinite(raw).all()
    raw2 = reverse_cumulative_monotonic(target, mono=-1, mono_type="strict")          # default floor 0.01
    assert torch.isfinite(raw2).all()


def _model(**over):
    from models.tspline.model import TSplineDCM

    dcfg = load_dataset_config("SM")
    dcfg.alt_xs_knots = {"TRAIN": [40, 40, 5], "SM": [40, 40, 5], "CAR": [40, 40]}
    mcfg = ModelConfig(T_vae_layers=[16], **over)
    return TSplineDCM(mcfg, dcfg, device="cpu")


def _batch(model, n=32):
    torch.manual_seed(1)
    I, Dz = len(model.dcfg.alt_names), len(model.dcfg.z_names)
    x = [torch.rand(n, len(model.dcfg.alt_xs[a])) * 3 for a in model.dcfg.alt_names]
    return {"idx": torch.arange(n), "ID": torch.arange(n), "av": torch.ones(n, I), "x": x, "z": torch.rand(n, Dz),
            "label": torch.nn.functional.one_hot(torch.randint(0, I, (n,)), I).float()}


def test_penalty_only_in_train_mode_and_only_when_weighted():
    m0 = _model(MC_k=1, MC_delta_floor=0.0)
    out = m0.train()(_batch(m0), sample=False)
    assert "penalty" not in out                            # weights 0 -> loss unchanged
    m1 = _model(MC_k=1, MC_delta_floor=0.0, SP_l2_incr=1e-3, SP_tv_slope=1e-3)
    b = _batch(m1)
    out_tr = m1.train()(b, sample=False)
    assert "penalty" in out_tr and torch.isfinite(out_tr["penalty"]) and out_tr["penalty"] > 0
    out_ev = m1.eval()(b, sample=False)
    assert "penalty" not in out_ev                         # validation / test NLL never contain it
    out_tr["penalty"].backward()                           # gradients flow into the spline coefficients
    assert any(mc.coef_raw.grad is not None and mc.coef_raw.grad.abs().sum() > 0
               for mc in m1.modules() if isinstance(mc, MonoCoef))


def test_degree_one_model_runs_and_is_monotone_per_individual():
    m = _model(MC_k=1, MC_delta_floor=0.0)
    m.eval()
    b = _batch(m, 8)
    xs = torch.linspace(0.0, 0.99, 50)                   # inside the initial [0, 1] grid (update_grid not called)
    y = m.asu_individual_curves(batch=b, alt_name="CAR", x_name="TT-CAR", x_eval=xs, sample_on=False)
    assert y.shape == (8, 50) and (torch.diff(y, dim=1) <= 1e-6).all()
