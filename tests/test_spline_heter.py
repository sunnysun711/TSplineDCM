"""Unit tests for SplineUtility heter_target ("level" vs "increment").

Runs standalone (no pytest required):
    python -m tests.test_spline_heter
or with pytest:
    pytest tests/test_spline_heter.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from models.config import ModelConfig
from models.tspline.mono_coef import MonoCoef
from models.tspline.spline_utility import SplineUtility

K = 3


def make_su(monos=(1, -1), Gs=(6, 6), mono_type="strict", heter_mode="mul",
            heter_target="increment", k_fix=False, keep_zero=False, cap=1.0, seed=0):
    torch.manual_seed(seed)
    mcs = [MonoCoef(num=g, k=K, mono=m, mono_type=mono_type, taste_name=f"f{i}")
           for i, (m, g) in enumerate(zip(monos, Gs))]
    su = SplineUtility(mono_coefs=mcs, heter_mode=heter_mode, heter_target=heter_target,
                       k_fix_across_g=k_fix, keep_fist_c_zero=keep_zero, k_gpi_cap=cap)
    for p in su.k_gpi_raw_list:  # randomize k_gpi (sigmoid -> roughly (0.12, 0.88))
        p.data.uniform_(-2.0, 2.0)
    return su


def population_coef(su):
    """Padded population coefficients, shape (in_dim, G_max+k)."""
    with torch.no_grad():
        return torch.stack([
            torch.cat([mc(), torch.zeros(su.G_max + su.k - mc.G - mc.k)])
            for mc in su.mono_coefs_typed
        ])


def check_mono(coef, su, atol=1e-6):
    """coef: (B, in_dim, 1, G_max+k); per-individual ordering on the real region."""
    for d, mono in enumerate(su.mono_dirs):
        if mono == 0:
            continue
        rows = coef[:, d, 0, : su.G_list[d] + su.k]
        diffs = torch.diff(rows, dim=-1)
        ok = (diffs >= -atol) if mono == 1 else (diffs <= atol)
        assert bool(ok.all()), (
            f"dim {d} (mono={mono}) violated: diff range "
            f"[{diffs.min().item():.4g}, {diffs.max().item():.4g}]"
        )


def assert_raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_config_default_is_increment():
    assert ModelConfig().SP_heter_target == "increment"


def test_increment_strict_mono():
    # random k and tau, both mono directions, free dim included -> per-individual monotone
    su = make_su(monos=(1, -1, 0), Gs=(8, 8, 5))
    tau = torch.rand(64, 3) * 2 - 1
    check_mono(su.forward_coef(64, taste_heter=tau), su)
    # production setting: keep_first_c_zero=True -> anchor exact + monotone
    su2 = make_su(monos=(1, -1), Gs=(8, 8), keep_zero=True)
    coef2 = su2.forward_coef(64, taste_heter=torch.rand(64, 2) * 2 - 1)
    assert bool((coef2[..., 0] == 0).all())
    check_mono(coef2, su2)


def test_level_mode_regression():
    # heter_target="level" must reproduce the legacy formula bit-for-bit
    su = make_su(heter_target="level")
    tau = torch.rand(16, 2) * 2 - 1
    coef = su.forward_coef(16, taste_heter=tau)
    with torch.no_grad():
        pop = population_coef(su)
        k_gpi = su.forward_k_gpi()
        expect = pop[None, :, None, :] * (1 + k_gpi[None, :, None, :] * tau[:, :, None, None])
    assert torch.equal(coef, expect)


def test_uniform_k_level_equals_increment():
    # constant factor across g commutes with cumsum -> the two targets coincide
    tau = torch.rand(8, 2) * 2 - 1
    out = {}
    for tgt in ("level", "increment"):
        su = make_su(Gs=(6, 6), heter_target=tgt, k_fix=True, seed=7)  # same seed -> same params
        out[tgt] = su.forward_coef(8, taste_heter=tau.clone())
    assert torch.allclose(out["level"], out["increment"], atol=1e-5)


def test_tau_zero_equals_population():
    su = make_su(monos=(1, -1, 0), Gs=(6, 6, 6))
    coef = su.forward_coef(4, taste_heter=torch.zeros(4, 3))
    pop = population_coef(su)
    assert torch.allclose(coef, pop[None, :, None, :].expand_as(coef), atol=1e-5)


def test_mixed_grid_padding_zero():
    # increment-mode cumsum must not leak into the padded tail (valid mask)
    su = make_su(monos=(1, -1), Gs=(10, 4))
    coef = su.forward_coef(32, taste_heter=torch.rand(32, 2) * 2 - 1)
    assert bool((coef[:, 1, :, 4 + K:] == 0).all())


def test_mul_lognorm_increment_mono():
    su = make_su(heter_mode="mul_lognorm")
    B = 64
    tau = torch.zeros(B, 2)  # required arg; unused by the lognorm path
    z = torch.randn(B, 2)
    s = torch.rand(B, 2) * 2.0
    check_mono(su.forward_coef(B, taste_heter=tau, taste_gauss=z, taste_sigma=s), su)


def test_add_increment_raises():
    assert_raises(ValueError, make_su, heter_mode="add", heter_target="increment")


def test_counterexample_transform_math():
    # TEMP-doc counterexample c=[0,1,1.1], k=[0,1,0], tau=1:
    # level-scaling breaks the ordering, increment-scaling preserves it
    c = torch.tensor([0.0, 1.0, 1.1])
    kg = torch.tensor([0.0, 1.0, 0.0])
    factor = 1 + kg * 1.0
    level = c * factor
    inc = torch.cumsum(torch.diff(c, prepend=torch.zeros(1)) * factor, dim=-1)
    assert not bool((torch.diff(level) >= 0).all())  # [0, 2.0, 1.1] -> violated
    assert bool((torch.diff(inc) >= 0).all())        # [0, 2.0, 2.1] -> preserved


def test_increment_gradients_flow():
    su = make_su(monos=(1, -1), Gs=(6, 6))
    tau = (torch.rand(8, 2) * 2 - 1).requires_grad_(True)
    su.forward_coef(8, taste_heter=tau).sum().backward()
    assert tau.grad is not None and bool(torch.isfinite(tau.grad).all())
    raw_grads = [mc.coef_raw.grad for mc in su.mono_coefs_typed]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in raw_grads)
    assert all(p.grad is not None for p in su.k_gpi_raw_list)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
