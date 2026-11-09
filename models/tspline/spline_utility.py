# tspline/core/spline_utility.py
import math
from typing import Any, Sequence, cast

import numpy as np
import torch
from torch import Tensor, nn

from models.config import HeterType, HeterTarget

# from ..config import MonotoneFlag, MonoType
from .b_spline import coef2curve, curve_from_basis
from .basis_cache import RowBasisCache
from .mono_coef import MonoCoef
from .coons_patch import CoonsPatch2D


def _pad_row_right(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Right-pad a 1D tensor with zeros to target_len.
    """
    if x.numel() == target_len:
        return x
    if x.numel() > target_len:
        # should not happen if G_max is computed correctly
        return x[:target_len]
    pad = torch.zeros(target_len - x.numel(), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


class SplineUtility(nn.Module):
    def __init__(
        self,
        mono_coefs: list[MonoCoef],
        interaction_constraints: list[list[int]] | None = None,
        k_gpi_start: float=0.5,
        heter_mode: HeterType = "mul",
        heter_target: HeterTarget = "increment",
        alt_name: str | None = None,
        k_fix_across_g: bool = False,
        device: str | torch.device = "cpu",
        k_gpi_cap: float | list[float] = 1.0,
        keep_fist_c_zero: bool = False,
    ):
        """Utility function with spline representation of each feature.
        
        This is one utility for one alternative.

        :param list[MonoCoef] mono_coefs: monotonic coefficients.
        :param list[list[int]] | None interaction_constraints: interaction constraints, defaults to None. 
        :param float k_gpi_start: start value of k_gpi, defaults to 0.5, which means setting k_gpi_raw to zeros.
        :param HeterType heter_mode: heterogeneity mode, defaults to "mul", can also be "add".
        :param HeterTarget heter_target: what the heterogeneity factor multiplies. Default (and the
            paper's method) is "increment": scale the I-spline increments and re-accumulate, which
            structurally preserves each individual's monotonicity; only valid with multiplicative heter
            modes ("mul", "mul_lognorm"). "level" is the legacy B-spline variant (scale the cumsum-ed
            coefficient levels; individual monotonicity NOT guaranteed) — kept for ablation only, must be
            requested explicitly.
        :param str | None alt_name: alternative name, defaults to None.
        :param bool k_fix_across_g: whether to fix k_gpi across all grid points, defaults to False.
        :param str | torch.device device: device, defaults to "cpu"
        :param float | list[float] k_gpi_cap: cap value of k_gpi, defaults to 1.0. can be a list of float capping each feature.
        :param bool keep_first_c_zero: whether to keep the first of the coefficients for all attributes to be zero, 
            which guarantees that all attribute-specific utility contribution function starts at zero at the left end point.
        
        Examples
        --------
        Basic initialization::
            
            >>> num, k = 10, 3
            >>> c_vott = MonoCoef(num=num, k=k, mono=-1, mono_type="strict", taste_name="TT")  # value of travel time
            >>> c_vohw = MonoCoef(num=num, k=k, mono=-1, mono_type="soft", taste_name="HE")  # value of headway
            >>> c_cost = MonoCoef(num=num, k=k, mono=-1, mono_type="strict", taste_name="CO")  # value of cost
            >>> util1 = SplineUtility(mono_coefs=[c_vott, c_vohw, c_cost], alt_name="PT", )
            >>> util2 = SplineUtility(mono_coefs=[c_vott, c_cost], alt_name="CAR", )
            >>> util1.mono_coefs[2] == util2.mono_coefs[1]  # shared c_cost
            True
            >>> all(util1.mono_coefs[0].coef_raw.data == util2.mono_coefs[0].coef_raw.data)  # shared data of c_vott
            True
            
        """
        super().__init__()
        # === Processing default inputs ===
        interaction_constraints = [] if interaction_constraints is None else interaction_constraints
        assert interaction_constraints is None or all(len(constraint) >= 2 for constraint in interaction_constraints), "Each interaction constraint must contain two indices."
        assert all(0 <= idx < len(mono_coefs) for constraint in interaction_constraints for idx in constraint), "Interaction constraints must be within the input dimension range."
        k0 = mono_coefs[0].k
        assert all(mc.k == k0 for mc in mono_coefs), "All MonoCoefs must share the same spline order k."
            
        self.alt_name:str = alt_name if alt_name is not None else ""
        attribute_names = [f"{self.alt_name}.{mc.taste_name}" for mc in mono_coefs]  # CAR.CO, PT.TT, ...
        self.in_dim: int = len(mono_coefs)
        self.mono_dirs: list[int] = [mc.mono for mc in mono_coefs]
        self.mono_coefs: nn.ModuleList = nn.ModuleList(mono_coefs)
        self.k = k0
        self.G_list = [mc.G for mc in self.mono_coefs_typed]
        self.G_max = max(self.G_list)
        self.mono_types: list[str] = [mc.mono_type for mc in self.mono_coefs_typed]
        self.interaction_constraints: list[list[int]] = interaction_constraints
        self.attribute_names: list[str] = attribute_names
        self.heter_mode: HeterType = heter_mode
        self.heter_target: HeterTarget = heter_target
        if heter_target not in ("level", "increment"):
            raise ValueError(f"heter_target {heter_target} is not supported.")
        if heter_target == "increment" and heter_mode == "add":
            raise ValueError(
                "heter_target='increment' requires a multiplicative heter_mode ('mul' or "
                "'mul_lognorm'); additive shifts on increments do not preserve monotonicity."
            )
        self.k_gpi_raw_activate = nn.Sigmoid() if heter_mode in ["mul", "mul_lognorm"] else nn.ReLU()
        # self.k_gpi_raw_activate = nn.Identity()
        self.k_fix_across_g: bool = k_fix_across_g
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.k_gpi_cap: list[float] = k_gpi_cap if isinstance(k_gpi_cap, list) else [k_gpi_cap] * self.in_dim
        # I-spline with the multiplicative factor (1 + k_gpi * tau), tau in (-1, 1): individual monotonicity
        # needs the factor >= 0, i.e. k_gpi <= 1. k_gpi is bounded by the cap, so the cap must be <= 1.
        # ("mul_lognorm" uses exp(.) > 0 and is safe for any cap; "level" never guaranteed it anyway.)
        if self.heter_target == "increment" and self.heter_mode == "mul" and any(float(c) > 1.0 for c in self.k_gpi_cap):
            raise ValueError(
                f"k_gpi_cap must be <= 1.0 for heter_target='increment' with heter_mode='mul' (got {self.k_gpi_cap}); "
                "a cap > 1 lets (1 + k_gpi*tau) go negative and breaks individual monotonicity."
            )
        self.keep_first_c_zero: bool = keep_fist_c_zero

        # build interactions
        self.interaction_pairs: list[tuple[int, int]] = []
        if interaction_constraints:
            for pair in interaction_constraints:
                p, q = pair[:2]
                assert 0 <= p < self.in_dim and 0 <= q < self.in_dim and p != q
                self.interaction_pairs.append((p, q))

        self.interactions = nn.ModuleDict()
        for (p, q) in self.interaction_pairs:
            mc_p, mc_q = self.mono_coefs_typed[p], self.mono_coefs_typed[q]
            self.interactions[f"{p}_{q}"] = CoonsPatch2D(
                grid_x=mc_p.grid, grid_y=mc_q.grid,
                k_x=mc_p.k, k_y=mc_q.k,
                num_x=mc_p.G, num_y=mc_q.G,
                rank=None,  # or (rx, ry) if you prefer low-rank
                device=self.device,
            )

        # Heterogeneity control parameters: init k_gpi_raw with start
        if self.heter_mode == "add":
            start = k_gpi_start
        elif self.heter_mode in ["mul", "mul_lognorm"]:   # compute logit of start value
            start = math.log(k_gpi_start / (1 - k_gpi_start + 1e-8))
        else:
            raise ValueError(f"heter_mode {self.heter_mode} is not supported.")
        
        if not self.k_fix_across_g:  # learn k_gpi in every grid
            self.k_gpi_raw_list = nn.ParameterList([
                nn.Parameter(
                    torch.full(
                        (mc.G + mc.k,), float(start), 
                        dtype=torch.float32, device=self.device, requires_grad=True
                    )
                )
                for mc in self.mono_coefs_typed
            ])
        else:  # learn one k_gpi across all grid points
            self.k_gpi_raw_list = nn.ParameterList([
                nn.Parameter(
                    torch.tensor([float(start)], dtype=torch.float32, device=self.device, requires_grad=True)
                )
                for _ in self.mono_coefs_typed
            ])
        
        # validity mask over the padded coefficient axis: 1 where j < G_i + k, else 0.
        # increment-target cumsum smears the last real level into the zero padding;
        # multiplying by this mask restores the zero tail. persistent=False -> no state_dict key.
        positions = torch.arange(self.G_max + self.k)
        valid = torch.stack([(positions < g + self.k) for g in self.G_list]).to(torch.float32)
        self.register_buffer("coef_valid_mask", valid, persistent=False)  # (in_dim, G_max+k)

        # basis of every dataset row (build_basis_cache); rebuilt by itself when a grid changes
        self._basis_cache: RowBasisCache | None = None

        self.to(self.device)

    @property
    def mono_coefs_typed(self) -> Sequence[MonoCoef]:
        """Only used for type hinting."""
        return cast(Sequence[MonoCoef], self.mono_coefs)
    
    @property
    def grid(self) -> torch.Tensor:
        """
        Padded view: stack each MonoCoef's extended_grid row, right-pad by repeating
        the last element so shapes match (in_dim, G_max + 2k + 1).
        """
        rows = []
        target_len = self.G_max + 2 * self.k + 1
        for mc in self.mono_coefs_typed:
            g = mc.grid  # shape (1, Gi + 2k + 1)
            if g.size(1) < target_len:
                pad_n = target_len - g.size(1)
                pad_row = torch.cat([g, g[:, -1:].expand(1, pad_n)], dim=1)
                rows.append(pad_row)
            else:
                rows.append(g)
        return torch.cat(rows, dim=0)  # (in_dim, G_max + 2k + 1)
    
    
    def forward_k_gpi(self) -> torch.Tensor:
        if not self.k_fix_across_g:
            k_gpi_list = [_pad_row_right(self.k_gpi_raw_activate(k_gpi_raw), self.G_max+self.k) for k_gpi_raw in self.k_gpi_raw_list]
            k_gpi = torch.stack(k_gpi_list, dim=0)  # shape (in_dim, G_max+k)
        else:
            k_gpi_list = [
                _pad_row_right(self.k_gpi_raw_activate(k_gpi_raw).view(1).expand(mc.G+mc.k), self.G_max+self.k) 
                for k_gpi_raw, mc in zip(self.k_gpi_raw_list, self.mono_coefs_typed)
                ]
            k_gpi = torch.stack(k_gpi_list, dim=0)  # shape (in_dim, G_max+k)
        k_gpi *= torch.tensor(self.k_gpi_cap, dtype=torch.float32, device=self.device).unsqueeze(1)
        if self.keep_first_c_zero:
            k_gpi[:, :1] = 0.0
        return k_gpi
    
    def forward_coef(
        self, 
        batch_size:int, 
        taste_heter: torch.Tensor | None = None, 
        taste_gauss: torch.Tensor | None = None, 
        taste_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the coefficients with taste heterogeneity. (including monotonic constraints)
        
        :param int batch_size: batch size
        :param Tensor taste_heter: 2D torch.Tensor with shape (batch_size, in_dim), 
            should be in range [-1, 1], only used in `mul` or `add` heter_type.
        :param Tensor taste_gauss: 2D torch.Tensor with shape (batch_size, in_dim), 
            follows Normal distribution N(0, 1), only used in `mul_lognorm` heter_type
        :param Tensor taste_sigma: 2D torch.Tensor with shape (batch_size, in_dim), 
            representing the variance of the tastes, only used with `taste_gauss` with 
            `mul_lognorm` heter_type.
        
        :return: mono-constraint and heterogeneous coefficients with shape (bs, in_dim, out_dim, G+k)
        :rtype: Tensor
        """
        # Get constrained coefficients from each MonoCoef
        coef_list = []
        for dim in range(self.in_dim):
            coef_dim = self.mono_coefs[dim]()  # Shape: (G_i+k,)
            if self.keep_first_c_zero and self.k == 0:
                # degree-0 basis 0 spans the whole first interval, so hard-zeroing c0
                # (the k>0 point anchor) would null the utility there; anchor by
                # shifting all coefficients instead, which preserves the step shape
                coef_dim = coef_dim - coef_dim[:1]
            coef_dim = _pad_row_right(coef_dim, self.G_max + self.k)  # (G_max+k)
            coef_dim = coef_dim.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, G_max+k)
            coef_list.append(coef_dim)
        coef = torch.cat(coef_list, dim=0)  # (in_dim, out_dim, G_max+k)
        coef = coef.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()  # (B, in_dim, 1, G_max+k)
        
        # Add taste heterogeneity terms
        if taste_heter is not None:
            assert taste_heter.shape == (batch_size, self.in_dim), f"taste_heter shape must be ({batch_size}, {self.in_dim})"
            # Expand coefficients for batch processing: (bs, in_dim, out_dim, n_coef)
            k_gpi = self.forward_k_gpi()
            if self.heter_mode == "add":
                coef = coef + (k_gpi[None, :, None, :] * taste_heter[:, :, None, None])
            else:
                # multiplicative factor, shared by both heter targets
                if self.heter_mode == "mul":
                    factor = 1 + k_gpi[None, :, None, :] * taste_heter[:, :, None, None]
                elif self.heter_mode == "mul_lognorm":
                    assert taste_gauss is not None and taste_sigma is not None, "taste_gauss and taste_sigma must be provided with mul_lognorm heter_type."
                    k = k_gpi[None, :, None, :]  # (1, in_dim, 1, G+k)
                    z = taste_gauss[:, :, None, None]  # (B, in_dim, 1, 1) ~ N(0,1)
                    s = taste_sigma[:, :, None, None]  # (B, in_dim, 1, 1) >= 0

                    # clamps for robustness
                    k = torch.clamp(k, 0.0, 0.7)
                    z = torch.clamp(z, -4.0, 4.0)

                    # scale = exp( k * (s*z) - 0.5 * (k^2) * (s^2))  => E[scale] = 1 exactly
                    factor = torch.exp(k * (s*z) - 0.5 * (k**2) * (s**2))
                else:
                    raise ValueError(f"heter_mode {self.heter_mode} is not supported.")

                if self.heter_target == "level":
                    coef = coef * factor
                else:  # "increment"
                    # Scale I-spline increments instead of levels, then re-accumulate.
                    # factor >= 0 preserves each increment's sign, so every individual's
                    # coefficient sequence stays monotone wherever the population's is.
                    delta = torch.diff(coef, dim=-1, prepend=torch.zeros_like(coef[..., :1]))
                    coef = torch.cumsum(delta * factor, dim=-1)
                    coef = coef * self.coef_valid_mask[None, :, None, :]  # re-zero the padded tail

        if self.keep_first_c_zero:
            coef[:, :, :, :1] = 0.0  # set first coefficient to be zero
        
        return coef

    # ---- basis cache: B-spline basis of every dataset row, gathered by row index ----

    def _basis_key(self) -> tuple:
        return tuple(mc.grid_version for mc in self.mono_coefs_typed) + (self.G_max, self.k)

    def build_basis_cache(self, x_all: torch.Tensor) -> None:
        """x_all: (N, in_dim) — this alternative's attribute columns for EVERY dataset row (DcmDataset._x_per_alt[i]).
        forward(idx=...) then gathers rows instead of re-evaluating B_batch. Staleness is handled inside."""
        assert x_all.dim() == 2 and x_all.size(1) == self.in_dim, f"x_all must be (N, {self.in_dim}), got {tuple(x_all.shape)}"
        if self._basis_cache is None:
            self._basis_cache = RowBasisCache()
        self._basis_cache.build(x_all, self.grid, self.k, self._basis_key())

    def _cached_basis_rows(self, idx: torch.Tensor) -> torch.Tensor | None:
        """Basis rows for `idx`, or None when no cache was ever built (plain coef2curve then)."""
        c = self._basis_cache
        if c is None:
            return None
        if not c.valid_for(self._basis_key()):      # grid changed (update_grid / checkpoint): rebuild from the kept tensor
            assert c.x_all is not None
            c.build(c.x_all, self.grid, self.k, self._basis_key())
        return c.rows(idx)

    def forward(
        self, 
        x: torch.Tensor, 
        taste_heter: torch.Tensor | None = None, 
        taste_gauss: torch.Tensor | None = None, 
        taste_sigma: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the SplineUtility with configurable monotonicity constraints.

        :param Tensor x: 2D torch.Tensor with shape (batch_size, in_dim)
        :param Tensor | None idx: dataset row index of every batch row, (batch_size,). With a basis cache built
            (build_basis_cache) the basis is gathered by these rows instead of evaluated from x; the result is
            identical. Without it (or idx=None, e.g. the asu_* curve helpers) x is evaluated as before.
        :param Tensor taste_heter: 2D torch.Tensor with shape (batch_size, in_dim), 
            should be in range [-1, 1], only used in `mul` or `add` heter_type.
        :param Tensor taste_gauss: 2D torch.Tensor with shape (batch_size, in_dim), 
            follows Normal distribution N(0, 1), only used in `mul_lognorm` heter_type
        :param Tensor taste_sigma: 2D torch.Tensor with shape (batch_size, in_dim), 
            representing the variance of the tastes, only used with `taste_gauss` with 
            `mul_lognorm` heter_type.
        
        :return: outputs 2D torch.Tensor with shape (batch_size, out_dim)
        :rtype: Tensor
        
        Examples
        --------
        Basic forward pass returns correct output shape::

            >>> c1 = MonoCoef(num=10, k=3, mono=-1, taste_name="feat1")
            >>> c2 = MonoCoef(num=4, k=3, mono=0, taste_name="feat2")
            >>> c3 = MonoCoef(num=8, k=3, mono=-1, mono_type="strict", taste_name="feat3")
            >>> su = SplineUtility(mono_coefs=[c1, c2, c3], interaction_constraints=[[0,1]], k_gpi_start=0.1, alt_name="Choice1")
            >>> x = torch.randn(8, 3)
            >>> su(x).shape
            torch.Size([8, 1])

        Forward pass with taste heterogeneity produces different results::

            >>> su2 = SplineUtility(mono_coefs=[c1,c2], alt_name="su2")
            >>> x = torch.zeros(3, 2)  # batch_size=3, n_attr=2
            >>> taste_heter = torch.tensor([[0.5, 0.0], [-0.3, 0.2], [0.0, -0.5]])
            >>> y1 = su2(x)
            >>> y2 = su2(x, taste_heter=taste_heter)
            >>> y1.shape == y2.shape
            True
            >>> torch.allclose(y1, y2)
            False

        Monotonicity constraint (strict increasing) is enforced::

            >>> x = torch.zeros(10, 2)
            >>> x[:, 0] = torch.linspace(-1, 1, 10)  # Vary first feature -> mono-decrease
            >>> y = su2(x)
            >>> diffs = torch.diff(y.squeeze())
            >>> torch.all(diffs <= 1e-6)  # Non-increasing (allowing numerical precision)
            tensor(True)

        """
        batch_size = x.shape[0]
        assert x.shape[1] == self.in_dim, f"Input dimension mismatch: expected {self.in_dim}, got {x.shape[1]}"
        
        coef = self.forward_coef(batch_size, taste_heter, taste_gauss, taste_sigma)
        
        # Evaluate splines: (bs, in_dim, out_dim)
        b = self._cached_basis_rows(idx) if idx is not None else None
        y = curve_from_basis(b, coef) if b is not None else coef2curve(x, self.grid, coef=coef, k=self.k)
        
        # Sum across input dimensions: (bs, out_dim)
        y = torch.sum(y, dim=1)  # (bs, 1)
        
        # add interactions if any
        if len(self.interactions) > 0:
            inter_sum = torch.Tensor([0.0]).to(x.device)
            for name, mod in self.interactions.items():
                p, q = map(int, name.split("_"))
                inter_sum = inter_sum + mod(x[:, p], x[:, q])  # (bs,)
            y = y + inter_sum.view(-1, 1)
            
        return y
    
    @torch.no_grad()
    def update_grid(self, x: Tensor, margin: float = 0.0, grid_eps: float = 0.02):
        """Update the grid points based on input samples for better coverage and adaptivity.

        :param Tensor x: Input samples, shape (batch, in_dim)
        :param float margin: Extra margin added to grid range, default 0.0
        :param float grid_eps: Mixing factor between adaptive and uniform grid, default 0.02
        
        Examples
        --------
        Basic grid update with normally distributed data::

            >>> c1 = MonoCoef(num=10, k=3, mono=-1, taste_name="feat1")
            >>> c2 = MonoCoef(num=4, k=3, mono=0, taste_name="feat2")  # accepts different G for each feature
            >>> su = SplineUtility(mono_coefs=[c1, c2], alt_name="test")
            >>> original_grid_shape = su.grid.shape
            >>> original_grid_shape  # (num_attr, G_max + 2 * k + 1)
            torch.Size([2, 17])
            >>> x = torch.randn(50, 2)
            >>> su.update_grid(x)
            >>> su.grid.shape == original_grid_shape
            True

        Grid update adapts to data distribution::

            >>> original_grid = su.grid.clone()
            >>> # Create skewed data - concentrated around 0.5
            >>> x = torch.normal(0.5, 0.1, (100, 2)).clamp(-1, 1)
            >>> su.update_grid(x, margin=0.1)
            >>> # Grid should have changed
            >>> torch.allclose(original_grid, su.grid)
            False
        
        A clearer example on grid adaptation::
        
            >>> c1 = MonoCoef(num=4, k=2, mono=-1, taste_name="feat1")
            >>> su = SplineUtility(mono_coefs=[c1], alt_name="test")
            >>> su.grid  # Original uniform grid
            tensor([[-2.0000, -1.5000, -1.0000, -0.5000,  0.0000,  0.5000,  1.0000,  1.5000,
                      2.0000]])
            >>> x = torch.cat([torch.randn(100).unsqueeze(1), torch.tensor([[2.8], [3.1], [2.9], [3.2]])], dim=0).clamp(min=-0.5, max=4)
            >>> su.update_grid(x)
            >>> su.grid  # doctest: +SKIP
            tensor([[-2.3500, -1.4250, -0.5000, -0.4815,  0.0269,  1.0767,  3.2000,  4.1250,
                      5.0500]])
                      
        """
        assert x.dim() == 2 and x.size(1) == self.in_dim
        for d, mc in enumerate(self.mono_coefs_typed):
            mc.update_grid(x[:, d:d+1], margin=margin, grid_eps=grid_eps)
            
        # 2) 2D interactions: push new grids; if basis size changed, reproject
        for name, mod in self.interactions.items():
            p, q = map(int, name.split("_"))
            gx_new = self.mono_coefs[p].grid
            gy_new = self.mono_coefs[q].grid
            try:
                mod.update_grids(gx_new, gy_new)  # fast path  # type: ignore
            except ValueError:
                # basis size changed, do robust reprojection
                mod.update_grids_reproject(gx_new, gy_new, nsamples=512, ridge=1e-8)  # type: ignore

    @torch.no_grad()
    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f"attribute_names={self.attribute_names}"
        s += f", taste heter mode=\"{self.heter_mode}\""
        s += f", taste heter target=\"{self.heter_target}\""
        # s += f", anchor_x0={self.anchor_x0.cpu().numpy()}"
        np.set_printoptions(precision=4, suppress=True, floatmode="fixed", linewidth=500)
        # s += f"\nb0 (padded) {(*self.b0.shape,)}=\n{self.b0.cpu().numpy()}"
        k_gpi = self.forward_k_gpi().cpu().numpy()
        s += f"\nk_gpi (activated, padded, and gated) {(*k_gpi.shape,)}=\n{k_gpi}"
        grid = self.grid.cpu().numpy()
        s += f"\ngrid (padded) {(*grid.shape,)}=\n{grid}"
        return s
    

def main():
    coef1, coef2 = MonoCoef(num=4, k=2), MonoCoef(num=4, k=2)
    coons = CoonsPatch2D(coef1.grid, coef2.grid, coef1.k, coef2.k, coef1.G, coef2.G)
    x1 = torch.randn(100, 1)
    x2 = torch.randn(100, 1)
    y = coons(x1, x2)
    print("y: ", y)
    
    c1 = MonoCoef(num=10, k=3, mono=-1, taste_name="feat1")
    c2 = MonoCoef(num=4, k=3, mono=0, taste_name="feat2")
    c3 = MonoCoef(num=8, k=3, mono=-1, mono_type="strict", taste_name="feat3")
    
    sp = SplineUtility(mono_coefs=[c1, c2, c3], interaction_constraints=[[0,1]], k_gpi_start=0.1, alt_name="Choice1")
    print(sp)
    sp.update_grid(torch.randn(100, 3))
    print("After grid update:")
    print(sp)


if __name__ == '__main__':
    main()
    
    pass