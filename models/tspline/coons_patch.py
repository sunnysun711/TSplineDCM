# tspline/core/coons_patch.py
import torch
from torch import Tensor, nn


from .b_spline import B_batch
from .mono_coef import MonoCoef


def count_parameters(m) -> int:
    return sum(v.numel() for v in m.parameters() if v.requires_grad)


def _build_proj_1d(
    old_grid: Tensor, new_grid: Tensor, k: int, nsamples: int = 512, ridge: float = 1e-8
) -> Tensor:
    """P maps old 1D coefs to new: c_new = P @ c_old."""
    device = old_grid.device
    lo = float(torch.minimum(old_grid.min(), new_grid.min()))
    hi = float(torch.maximum(old_grid.max(), new_grid.max()))
    x = torch.linspace(lo, hi, steps=nsamples, device=device)  # (S,)

    B_old = B_batch(x, old_grid, k)  # (S, nb_old)
    B_new = B_batch(x, new_grid, k)  # (S, nb_new)
    nb_new = B_new.shape[1]
    I = torch.eye(nb_new, device=device)
    # P = (B_new^T B_new + λI)^(-1) B_new^T B_old
    P = torch.linalg.solve(B_new.T @ B_new + ridge * I, B_new.T @ B_old)
    return P  # (nb_new, nb_old)


class CoonsPatch2D(nn.Module):
    """
    2D interaction surface that is zero on rectangle boundaries in (x,y),
    built from tensor-product B-spline basis on shared grids.
    """

    def __init__(
        self,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        k_x: int,
        k_y: int,
        num_x: int,
        num_y: int,
        rank: tuple[int, int] | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.grid_x: Tensor
        self.grid_y: Tensor
        self.register_buffer("grid_x", grid_x.data.clone())
        self.register_buffer("grid_y", grid_y.data.clone())
        self.k_x, self.k_y = k_x, k_y
        self.num_x, self.num_y = num_x, num_y
        self.rank = rank
        self.device = device

        # full 2D coefficient or low-rank factors
        gx = num_x + k_x  # number of basis funcs in x (depending on your impl)
        gy = num_y + k_y  # number of basis funcs in y
        if self.rank is None:
            self.W = nn.Parameter(torch.zeros(gx, gy))
        else:
            rx, ry = self.rank
            self.A = nn.Parameter(torch.zeros(gx, rx))
            self.B = nn.Parameter(torch.zeros(ry, gy))
            self.C = nn.Parameter(torch.zeros(rx, ry))

        # small init near zero to avoid early dominance
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=1e-3)

    @staticmethod
    def _normalize01(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param Tensor x: (B,) or (B,1) values for attribute p
        :param Tensor y: (B,) or (B,1) values for attribute q
        :return: (B,) interaction value

        Example:
        --------

        >>> coef1, coef2 = MonoCoef(), MonoCoef()
        >>> coons = CoonsPatch2D(coef1.grid, coef2.grid, coef1.k, coef2.k, coef1.G, coef2.G)
        >>> x = torch.randn(10,)
        >>> y = torch.randn(10,)
        >>> coons(x, y)  # doctest: +SKIP
        tensor([ 3.5140e-06, -2.4049e-05, -2.0711e-08,  2.0818e-06,  6.5508e-06,
                -2.9065e-05, -4.6344e-08,  0.0000e+00,  7.9389e-06, -4.5510e-05],
               grad_fn=<MulBackward0>)


        Examples
        --------
        Basic usage with MonoCoef grids (shape check)::

            >>> mcx = MonoCoef(num=5, k=2)
            >>> mcy = MonoCoef(num=6, k=2)
            >>> coons = get_coons_from_mc(mcx, mcy)
            >>> x = torch.linspace(-0.5, 0.5, steps=7)
            >>> y = torch.linspace(-0.25, 0.25, steps=7)
            >>> out = coons(x, y)
            >>> out.shape
            torch.Size([7])

        Boundary gate zeros the interaction on edges::

            >>> xb = mcx.grid.min() * torch.ones(4)
            >>> yb = torch.linspace(0.0, 0.1, steps=4)
            >>> edge = coons(xb, yb)
            >>> torch.allclose(edge, torch.zeros_like(edge), atol=1e-6)
            True

        Grid sync (Option A fast path, basis size unchanged)::

            >>> new_gx = mcx.grid + 0.1
            >>> new_gy = mcy.grid - 0.2
            >>> coons.update_grids(new_gx, new_gy)
            >>> torch.max(torch.abs(coons.grid_x - new_gx)).item() == 0.0
            True
            >>> torch.max(torch.abs(coons.grid_y - new_gy)).item() == 0.0
            True

        (Optional) Reproject when basis size changes (safe if your
        update_grids_reproject replaces buffers rather than in-place copy)::

            >>> mcy2 = MonoCoef(num=7, k=2)
            >>> coons.update_grids_reproject(new_gx, mcy2.grid) 
            >>> out2 = coons(x, y)
            >>> out2.shape
            torch.Size([7])

        """
        x = x.view(-1)
        y = y.view(-1)

        # 1) 1D B-spline bases on shared grids
        Bx = B_batch(x, self.grid_x, self.k_x)  # (B, Gx)
        By = B_batch(y, self.grid_y, self.k_y)  # (B, Gy)

        # 2) Interior gating to zero out edges (Coons-like residual)
        tx = self._normalize01(x, float(self.grid_x.min()), float(self.grid_x.max()))
        ty = self._normalize01(y, float(self.grid_y.min()), float(self.grid_y.max()))
        gate = (tx * (1.0 - tx) * ty * (1.0 - ty)).view(-1, 1)  # (B,1)

        # 3) Tensor-product surface value
        if hasattr(self, "W"):
            # (B,) via (B,Gx) @ (Gx,Gy) @ (Gy,) per-sample => do as batched matmul
            surface = (Bx @ self.W) * By  # (B, Gy)
            val = surface.sum(dim=1)  # (B,)
        else:
            # low rank: (Bx A) C (B^T By^T) efficiently
            U = Bx @ self.A  # (B, rx)
            V = By @ self.B.t()  # (B, ry)
            mid = U @ self.C  # (B, ry)
            val = (mid * V).sum(dim=1)  # (B,)

        return gate.view(-1) * val

    @torch.no_grad()
    def update_grids(self, new_grid_x: Tensor, new_grid_y: Tensor) -> None:
        """
        Fast path when basis sizes do not change.
        """
        if (self.grid_x.shape[0] != new_grid_x.shape[0]) or (
            self.grid_y.shape[0] != new_grid_y.shape[0]
        ):
            raise ValueError("Grid size changed; use update_grids_reproject(...)")

        self.grid_x.copy_(new_grid_x.detach())
        self.grid_y.copy_(new_grid_y.detach())

    @torch.no_grad()
    def update_grids_reproject(
        self,
        new_grid_x: Tensor,
        new_grid_y: Tensor,
        nsamples: int = 512,
        ridge: float = 1e-8,
        refactor_rank: tuple[int, int] | None = None,
    ) -> None:
        """
        Safe path when basis sizes change. Projects old coefficients to new basis.
        """
        # materialize W_old regardless of storage
        if hasattr(self, "W"):
            W_old = self.W.data.clone()  # (nbx_old, nby_old)
        else:
            W_old = (self.A @ self.C) @ self.B.T  # (nbx_old, nby_old)

        P_x = _build_proj_1d(self.grid_x, new_grid_x, self.k_x, nsamples, ridge)
        P_y = _build_proj_1d(self.grid_y, new_grid_y, self.k_y, nsamples, ridge)
        W_new = P_x @ W_old @ P_y.T  # (nbx_new, nby_new)

        # commit new grids
        self.grid_x.copy_(new_grid_x.detach())
        self.grid_y.copy_(new_grid_y.detach())

        # store as full or refactor
        if hasattr(self, "W"):
            self.W = nn.Parameter(W_new, requires_grad=True)
        else:
            u, s, vT = torch.linalg.svd(W_new, full_matrices=False)
            if refactor_rank is None:
                r = s.numel()
            else:
                r = min(refactor_rank[0], refactor_rank[1], s.numel())
            self.A = nn.Parameter(u[:, :r], requires_grad=True)
            self.C = nn.Parameter(torch.diag(s[:r]), requires_grad=True)
            self.B = nn.Parameter(vT[:r, :].T, requires_grad=True)

    
    def extra_repr(self) -> str:
        s = super().extra_repr()
        if self.rank is None:
            s += f"rank=None, full W shape={tuple(self.W.shape)}"
        else:
            s += f"rank={self.rank}, A shape={tuple(self.A.shape)}, B shape={tuple(self.B.shape)}, C shape={tuple(self.C.shape)}"
        s += f", num params={count_parameters(self)}"
        return s

def get_coons_from_mc(
    mono_coef_x: MonoCoef,
    mono_coef_y: MonoCoef,
    rank: tuple[int, int] | None = None,
) -> CoonsPatch2D:

    coons_patch = CoonsPatch2D(
        grid_x=mono_coef_x.grid,
        grid_y=mono_coef_y.grid,
        k_x=mono_coef_x.k,
        k_y=mono_coef_y.k,
        num_x=mono_coef_x.G,
        num_y=mono_coef_y.G,
        rank=rank,
        device=mono_coef_x.device,
    )
    return coons_patch


if __name__ == "__main__":
    cp = CoonsPatch2D(
        grid_x=torch.linspace(0, 1, 10),
        grid_y=torch.linspace(0, 1, 10),
        k_x=3,
        k_y=3,
        num_x=10,
        num_y=10,
        # rank=(2, 2),
    )
    print(cp)
    pass