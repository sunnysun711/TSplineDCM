# tests/test_bspline_np.py — the numpy B-spline used by the illustration figures matches the torch one.
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))

from _bspline_np import b_batch, coef2curve, curve2coef, extend_grid, greville  # noqa: E402
from models.tspline import b_spline as tb  # noqa: E402


def _cases():
    rng = np.random.default_rng(0)
    for k in (0, 1, 2, 3):
        for fix_left, fix_right in ((False, False), (True, False), (True, True)):
            core = np.array([0.0, 0.15, 0.2, 0.35, 0.4, 0.5, 0.8, 0.9, 1.0])
            x = np.concatenate([rng.uniform(0, 1, 300), core])      # include knots and both ends
            yield k, fix_left, fix_right, core, x


def test_extend_grid_and_basis_match_torch():
    for k, fl, fr, core, x in _cases():
        ext_np = extend_grid(core, k, fl, fr)
        ext_t = tb.extend_grid(torch.tensor(core, dtype=torch.float64).unsqueeze(0), k_extend=k, fix_left=fl, fix_right=fr)
        assert np.allclose(ext_np, ext_t.squeeze(0).numpy()), (k, fl, fr)
        B_np = b_batch(x, ext_np, k)
        B_t = tb.B_batch(torch.tensor(x, dtype=torch.float64), ext_t.squeeze(0), k=k).numpy()
        assert B_np.shape == B_t.shape
        assert np.allclose(B_np, B_t, atol=1e-12), (k, fl, fr, np.abs(B_np - B_t).max())


def test_curve_roundtrip_and_greville():
    core = np.linspace(0, 1, 9)
    for k in (1, 2, 3):
        ext = extend_grid(core, k, True, False)
        n = ext.size - k - 1
        coef = np.random.default_rng(1).normal(size=n)
        x = np.linspace(0, 1, 500)
        y = coef2curve(x, ext, coef, k)
        y_t = tb.coef2curve(torch.tensor(x, dtype=torch.float64).unsqueeze(1), torch.tensor(ext, dtype=torch.float64).unsqueeze(0),
                            torch.tensor(coef, dtype=torch.float64).view(1, 1, -1), k=k)[:, 0, 0].numpy()
        assert np.allclose(y, y_t, atol=1e-6)          # torch casts the basis to float32
        assert np.allclose(curve2coef(x, y, ext, k, ridge=1e-12), coef, atol=1e-5)
        assert greville(ext, k).shape == (n,)
