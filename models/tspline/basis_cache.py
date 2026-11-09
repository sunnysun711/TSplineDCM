# models/tspline/basis_cache.py — B-spline basis of every dataset row, computed once per grid.
"""Why: a training step used to re-evaluate the B-spline basis (the k-level recursion in B_batch) for every
row of the batch, for every attribute spline and for every (alternative, z) pair of the z-spline ASC head.
Those values depend only on the row's x (or z) and on the knot grid, and the grid changes only at a grid
update or a checkpoint load. So we evaluate the basis for ALL rows once per grid, keep it on the device and
gather the batch rows by index. The contraction with the coefficients (curve_from_basis) is unchanged, so the
utilities are byte-for-byte the same as the uncached path (tests/test_basis_cache.py).

Staleness: every MonoCoef carries `grid_version`, bumped by update_grid() and by load_state_dict(). A cache
remembers the versions it was built from and rebuilds itself (from the dataset tensor it was given) the first
time it is asked for rows under new versions. Memory: N x in_dim x (G_max+k) float32 per utility — LPMC
(81k rows, 13 splines) ~270 MB, SYN (548k rows, 4 splines) ~550 MB; fine on the A6000, reported at build."""
from __future__ import annotations

import torch
from torch import Tensor

from .b_spline import basis_matrix

CHUNK_ROWS = 65536   # B_batch keeps (rows, in_dim, G+2k) temporaries per recursion level; chunk to bound them


class RowBasisCache:
    """Basis values of every row of `x_all` on one extended grid, gathered by row index."""

    def __init__(self) -> None:
        self.values: Tensor | None = None       # (N, in_dim, G+k) float32
        self.key: tuple | None = None            # grid versions (+ grid shape) the values were built from
        self.x_all: Tensor | None = None         # dataset tensor (N, in_dim), kept by reference for rebuilds

    def build(self, x_all: Tensor, grid: Tensor, k: int, key: tuple) -> None:
        self.x_all = x_all
        parts = [basis_matrix(x_all[s: s + CHUNK_ROWS], grid, k=k) for s in range(0, x_all.shape[0], CHUNK_ROWS)]
        self.values = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        self.key = key

    def valid_for(self, key: tuple) -> bool:
        return self.values is not None and self.key == key

    def rows(self, idx: Tensor) -> Tensor:
        assert self.values is not None
        return self.values.index_select(0, idx.to(device=self.values.device, dtype=torch.long))

    @property
    def nbytes(self) -> int:
        return 0 if self.values is None else self.values.numel() * self.values.element_size()
