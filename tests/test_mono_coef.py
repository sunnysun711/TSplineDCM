# tests/test_mono_coef.py — the three manual checks that used to sit at the end of mono_coef.py, as pytest.
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tspline.mono_coef import MonoCoef  # noqa: E402


@pytest.mark.parametrize("mono_type", ["strict", "soft", "window"])
def test_gradient_flows_through_every_monotone_type(mono_type):
    torch.manual_seed(0)
    mc = MonoCoef(num=6, k=2, mono=1, mono_type=mono_type, win_size=3)
    coef = mc()
    assert coef.requires_grad
    coef.sum().backward()
    assert mc.coef_raw.grad is not None and mc.coef_raw.grad.norm().item() > 1e-8


@pytest.mark.parametrize("mono_type,kwargs", [("strict", {"mono": -1}), ("soft", {"mono": -1, "elu_alpha": 0.01}),
                                              ("window", {"mono": 1, "win_size": 3})])
def test_reverse_recovers_raw_coefficients(mono_type, kwargs):
    torch.manual_seed(0)
    mc = MonoCoef(num=6, k=2, mono_type=mono_type, noise_scale=0.05, **kwargs)
    constrained = mc().detach()
    recovered = mc.reverse(constrained)
    with torch.no_grad():
        mc.coef_raw.data = recovered
    assert torch.allclose(mc(), constrained, atol=1e-4)


@pytest.mark.parametrize("init_method", ["random", "smart"])
def test_strict_monotone_after_init(init_method):
    torch.manual_seed(0)
    mc = MonoCoef(num=6, k=3, mono=1, mono_type="strict", init_method=init_method)
    coef = mc().detach()
    assert coef.shape == mc.coef_raw.shape
    assert torch.all(coef[1:] >= coef[:-1])
