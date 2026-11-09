# plots/_bspline_np.py — numpy port of models/tspline/b_spline.py for the torch-free illustration figures.
"""Same conventions as the torch code (tests/test_bspline_np.py checks them numerically):
    b_batch(x, grid, k)      -> (m, n_knots - k - 1) Cox-de Boor bases, right end closed (x == t_last -> last basis)
    extend_grid(core, k, fix_left, fix_right) -> knots extended k times per side: by the uniform step h, or repeated
    coef2curve(x, ext, coef, k) -> curve values;  curve2coef(x, y, ext, k, ridge) -> least-squares coefficients
    greville(ext, k)         -> Greville abscissae, one per basis
"""
from __future__ import annotations

import numpy as np


def b_batch(x: np.ndarray, grid: np.ndarray, k: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    g = np.asarray(grid, dtype=np.float64).reshape(1, -1)
    if k == 0:
        v = ((x >= g[:, :-1]) & (x < g[:, 1:])).astype(np.float64)
    else:
        b_km1 = b_batch(x[:, 0], grid, k - 1)
        d1 = g[:, k:-1] - g[:, :-(k + 1)]
        w1 = np.where(d1 != 0, (x - g[:, :-(k + 1)]) / np.where(d1 != 0, d1, 1.0), 0.0)
        d2 = g[:, k + 1:] - g[:, 1:-k]
        w2 = np.where(d2 != 0, (g[:, k + 1:] - x) / np.where(d2 != 0, d2, 1.0), 0.0)
        v = w1 * b_km1[:, :-1] + w2 * b_km1[:, 1:]
    v[:, -1] = np.where(x[:, 0] == g[0, -1], 1.0, v[:, -1])
    return np.nan_to_num(v)


def extend_grid(core: np.ndarray, k: int, fix_left: bool = False, fix_right: bool = False) -> np.ndarray:
    grid = np.asarray(core, dtype=np.float64).copy()
    h = (grid[-1] - grid[0]) / (len(grid) - 1)
    for _ in range(k):
        grid = np.concatenate([[grid[0] if fix_left else grid[0] - h], grid, [grid[-1] if fix_right else grid[-1] + h]])
    return grid


def coef2curve(x: np.ndarray, ext: np.ndarray, coef: np.ndarray, k: int) -> np.ndarray:
    return b_batch(x, ext, k) @ np.asarray(coef, dtype=np.float64)


def curve2coef(x: np.ndarray, y: np.ndarray, ext: np.ndarray, k: int, ridge: float = 1e-8) -> np.ndarray:
    B = b_batch(x, ext, k)
    A = B.T @ B + ridge * np.eye(B.shape[1])
    return np.linalg.solve(A, B.T @ np.asarray(y, dtype=np.float64))


def greville(ext: np.ndarray, k: int) -> np.ndarray:
    t = np.asarray(ext, dtype=np.float64)
    n = t.size - k - 1
    if k > 0:
        return np.array([t[i + 1: i + k + 1].mean() for i in range(n)])
    return 0.5 * (t[:-1] + t[1:])
