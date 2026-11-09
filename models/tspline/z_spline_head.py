# tspline/models/z_spline_head.py
import pandas as pd
import torch
from torch import nn

from dcm_data import DcmConfig
from .b_spline import coef2curve, curve_from_basis
from .basis_cache import RowBasisCache
from .mono_coef import MonoCoef


GRID_EPS = 1e-4



class ZSplineHead(nn.Module):
    
    def __init__(
        self, 
        num_z: int, 
        num_alt: int,
        dcfg: DcmConfig,
        z_df: pd.DataFrame | None = None,  # dataframe with z_names as columns, used to determine grid knots for continuous z, must have the same order in columns as z_names
        z_ranges: list[tuple[float, float]] = [],  # if empty, will determine from z_df
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        
        z_names = dcfg.z_names
        z_mono_dirs = dcfg.z_mono_dirs
        disc_config = dcfg.z_spline_configs["MC_DISCRETE_CONFIG"]  # type: ignore
        cont_config = dcfg.z_spline_configs["MC_CONTINUOUS_CONFIG"]  # type: ignore
        # basis of every dataset row per z feature (build_basis_cache); one cache per z, shared by the alternatives
        # whose grids are equal (they are built identically, so normally all of them)
        self._basis_caches: list[list[RowBasisCache]] | None = None
        self._basis_z_all: torch.Tensor | None = None
        # fused tables (built with the caches when every alternative shares the grid of each z, the normal case):
        # all z bases of a row side by side, so the whole head is one gather + one matmul per step
        self._fused: dict | None = None
        self.fused_enabled: bool = False  # TSplineDCM sets it from model_config.zhead_fused before the cache is built
        if z_df is None:
            self.z_df = None  # type: ignore
        else:
            self._post_init(
                num_z, num_alt, z_names, disc_config, cont_config, z_df, z_ranges, z_mono_dirs, device
            )
        
    def _post_init(
        self, 
        num_z: int, 
        num_alt: int,
        z_names: list[str],
        dics_config: dict, 
        cont_config: dict, 
        z_df: pd.DataFrame,  # dataframe with z_names as columns, used to determine grid knots for continuous z, must have the same order in columns as z_names
        z_ranges: list[tuple[float, float]] = [],  # if empty, will determine from z_df
        z_mono_dirs: dict[str, int] | None = None,   # -1: descending, 1: ascending, 0: not-constrained (default)
        device: str | torch.device = "cpu",
    ):
        # preprocess
        assert len(z_names) == num_z, "z_names must have the same length as num_z"
        assert z_df.shape[1] == num_z, "z_df must have the same number of columns as num_z"
        if z_mono_dirs is None:
            z_mono_dirs = {}
        assert all(v in [-1, 1, 0] for v in z_mono_dirs.values()), "z_mono_dirs values must be -1, 1, or 0"
        if z_ranges:
            assert len(z_ranges) == num_z, "z_ranges must have the same length as num_z"
        else:
            z_ranges = [(z_df.iloc[:, i].min() - GRID_EPS, z_df.iloc[:, i].max() + GRID_EPS) for i in range(num_z)]
        z_knots: list[int] = []
        z_ks: list[int] = []
        z_is_continuous: list[bool] = []
        for i, z_name in enumerate(z_names):
            # mono_dirs
            if z_name not in z_mono_dirs:
                z_mono_dirs[z_name] = 0
            # is continuous or discrete based on the number of distinct values
            if z_df.iloc[:, i].nunique() <= dics_config["n_distinct_values_max"]:
                z_is_continuous.append(False)
                z_knots.append(z_df.iloc[:, i].nunique())
                z_ks.append(dics_config["k"])
            else:
                z_is_continuous.append(True)
                z_knots.append(cont_config["num_knots"])
                z_ks.append(cont_config["k"])
        
        # store attributes
        self.num_z = num_z
        self.num_alt = num_alt
        self.z_names = z_names
        self.z_ranges: list[tuple[float, float]] = z_ranges
        self.z_df: pd.DataFrame = z_df
        self.mono_coefs = nn.ModuleList([
            nn.ModuleList([
            MonoCoef(
                num=z_knots[i],  # not (knots - 1) because we need each grid for one distinct value
                k=z_ks[i],
                grid_range=z_ranges[i],
                mono=z_mono_dirs[z_names[i]],  # type: ignore
                mono_type="strict",
                win_size=1,
                taste_name=z_names[i],
                device=device,
            )
            for i in range(num_z)
            ]) 
            for j in range(num_alt)
        ])
        for j in range(num_alt):
            for i in range(num_z):
                mc: MonoCoef = self.mono_coefs[j][i]  # type: ignore
                
                if z_is_continuous[i]:
                    mc.update_grid(
                        x=torch.tensor(z_df.iloc[:, i], dtype=mc.grid.dtype).unsqueeze(1),
                        clamp_left_to=z_ranges[i][0],  # set left clamp to min value, make left end always start at zero
                        pin_right_core_to=z_ranges[i][1],  # set right clamp to max value
                    )
                else:
                    # if discrete type, update grid knots to distinct values (midpoints of each interval)
                    distinct_values = torch.unique(torch.tensor(z_df.iloc[:, i], dtype=mc.grid.dtype))
                    mid_points = (distinct_values[:-1] + distinct_values[1:]) / 2
                    for _i, mid_pt in enumerate(mid_points):
                        mc.grid.data[0, _i + mc.k + 1] = mid_pt
    
    # ---- basis cache: B-spline basis of every dataset row per (alternative, z), gathered by row index ----

    def build_basis_cache(self, z_all: torch.Tensor) -> None:
        """z_all: (N, num_z) — the z columns of EVERY dataset row (DcmDataset._z). forward(idx=...) then gathers
        rows instead of re-evaluating B_batch for each of the num_alt x num_z splines (the bulk of a step)."""
        assert z_all.dim() == 2 and z_all.size(1) == self.num_z, f"z_all must be (N, {self.num_z}), got {tuple(z_all.shape)}"
        self._basis_z_all = z_all
        caches: list[list[RowBasisCache]] = [[RowBasisCache() for _ in range(self.num_z)] for _ in range(self.num_alt)]
        for q in range(self.num_z):
            for j in range(self.num_alt):
                mc: MonoCoef = self.mono_coefs[j][q]  # type: ignore
                shared = None
                for jj in range(j):   # same grid as an earlier alternative -> share its values
                    mc2: MonoCoef = self.mono_coefs[jj][q]  # type: ignore
                    if mc2.k == mc.k and mc2.grid.shape == mc.grid.shape and torch.equal(mc2.grid, mc.grid):
                        shared = caches[jj][q]
                        break
                if shared is not None:
                    caches[j][q] = shared
                else:
                    caches[j][q].build(z_all[:, q:q + 1].to(device=mc.grid.device, dtype=mc.grid.dtype), mc.grid, mc.k,
                                       self._basis_key(j, q))
        self._basis_caches = caches
        self._fused = None
        if self.fused_enabled and all(caches[j][q] is caches[0][q] for q in range(self.num_z) for j in range(self.num_alt)):
            self._build_fused(caches)

    def _build_fused(self, caches: list[list[RowBasisCache]]) -> None:
        """Side-by-side basis of all z (N, W) with W = sum_q (G_q + k_q), plus the index maps that turn the
        num_alt x num_z coefficient vectors into one (W, num_alt) matrix with each segment anchored at 0."""
        vals = [caches[0][q].values for q in range(self.num_z)]
        assert all(v is not None for v in vals)
        widths = [v.shape[-1] for v in vals]  # type: ignore[union-attr]
        B_all = torch.cat([v[:, 0, :] for v in vals], dim=1)  # type: ignore[index]  (N, W)
        dev = B_all.device
        starts = torch.tensor([sum(widths[:q]) for q in range(self.num_z)], device=dev)
        W = int(sum(widths))
        seg_first = torch.repeat_interleave(starts, torch.tensor(widths, device=dev))         # (W,) first position of each segment
        not_first = torch.ones(W, device=dev, dtype=torch.float32)
        not_first[starts] = 0.0
        self._fused = {"B": B_all, "seg_first": seg_first, "not_first": not_first, "W": W,
                       "key": tuple(self._basis_key(j, q) for j in range(self.num_alt) for q in range(self.num_z))}
        for q in range(self.num_z):          # the per-z copies are not needed any more (a rebuild starts from z_all)
            caches[0][q].values = None

    def basis_cache_bytes(self) -> int:
        if self._fused is not None:
            return int(self._fused["B"].numel() * self._fused["B"].element_size())
        if self._basis_caches is None:
            return 0
        seen: dict[int, int] = {}
        for row in self._basis_caches:
            for c in row:
                seen[id(c)] = c.nbytes
        return sum(seen.values())

    def _forward_fused(self, idx: torch.Tensor) -> torch.Tensor:
        """Same quantity as the per-(alternative, z) loop below, as one matmul: U[b, j] = sum_q sum_g B[b, q, g] * c[j, q, g]
        (the sum runs in one go instead of z by z, so the last bits may differ from the loop)."""
        f = self._fused
        assert f is not None
        flat = torch.cat([self.mono_coefs[j][q]() for j in range(self.num_alt) for q in range(self.num_z)])  # (num_alt * W,)
        W = f["W"]
        c0 = flat.detach().view(self.num_alt, W).index_select(1, f["seg_first"]).reshape(-1)   # each segment's first level
        C = ((flat - c0).view(self.num_alt, W) * f["not_first"]).t()                             # (W, num_alt), anchored at 0
        return f["B"].index_select(0, idx.to(device=f["B"].device, dtype=torch.long)) @ C.to(f["B"].dtype)

    def _basis_key(self, j: int, q: int) -> tuple:
        mc: MonoCoef = self.mono_coefs[j][q]  # type: ignore
        return (mc.grid_version, tuple(mc.grid.shape), mc.k)

    def _cached_basis_rows(self, j: int, q: int, idx: torch.Tensor) -> torch.Tensor | None:
        if self._basis_caches is None:
            return None
        c = self._basis_caches[j][q]
        if not c.valid_for(self._basis_key(j, q)):
            # a grid changed under us (checkpoint load): rebuild the whole head from the kept z tensor, so that
            # which alternatives share a cache is decided again from the actual grids
            assert self._basis_z_all is not None
            self.build_basis_cache(self._basis_z_all)
            c = self._basis_caches[j][q]
        return c.rows(idx)

    def forward(self, z: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        z: (batch_size, num_z)
        idx: (batch_size,) dataset row index of every batch row; with a basis cache built the basis is gathered
            by these rows instead of evaluated from z (identical result). None -> evaluate from z.
        """
        if self.z_df is None:
            raise ValueError("z_df must be provided to initialize the model")
        assert z.dim() == 2, f"z must be 2D (batch_size, num_z), got shape {tuple(z.shape)}"
        assert z.shape[1] == self.num_z, f"z must have the same number of columns as num_z, got {z.shape[1]}"

        if idx is not None and self._fused is not None:
            if self._fused["key"] != tuple(self._basis_key(j, q) for j in range(self.num_alt) for q in range(self.num_z)):
                assert self._basis_z_all is not None
                self.build_basis_cache(self._basis_z_all)      # a grid changed (checkpoint load): rebuild everything
            if self._fused is not None:
                return self._forward_fused(idx)

        batch_size = z.shape[0]
        outputs = []

        for j in range(self.num_alt):
            # Sum over features for alternative j
            s_j = None
            for q in range(self.num_z):
                mc: MonoCoef = self.mono_coefs[j][q]  # type: ignore (per-alt, per-feature)
                coef = mc().unsqueeze(0).unsqueeze(0)
                
                # Ensure left anchor at 0 for splines by shifting
                c0 = coef[..., :1].detach()
                coef = coef - c0
                coef[..., :1] = 0.0
                
                b = self._cached_basis_rows(j, q, idx) if idx is not None else None
                if b is not None:
                    v_i = curve_from_basis(b, coef)  # (bs, 1, out_dim), same contraction as coef2curve
                else:
                    z_i = z[:, q:q+1].to(device=mc.grid.device, dtype=mc.grid.dtype)
                    v_i = coef2curve(z_i, mc.grid, coef=coef, k=mc.k)  # (bs, in_dim, out_dim)
                v_i = v_i[:, :, 0]  # (bs, 1)
                s_j = v_i if s_j is None else (s_j + v_i)
            # s_j: (batch_size, 1)
            outputs.append(s_j)

        # Concatenate along alternative dimension: (batch_size, num_alt)
        U = torch.cat(outputs, dim=1)
        return U