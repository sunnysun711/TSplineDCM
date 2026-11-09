# tspline/models/tspline_dcm.py
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

from models.config import ModelConfig, TrainConfig, get_device
from dcm_data import DcmDataset, DcmConfig
from .b_spline import B_batch
from .mono_coef import MonoCoef
from .spline_utility import SplineUtility
from .taste_encoder import TasteEncoder, reparameterize
from .z_spline_head import ZSplineHead


def set_mc_num(dcm_config: DcmConfig, MC_num:int) -> None:
    """Set MC_num for all features in dcm_config.alt_xs_knots.
    """
    if dcm_config.alt_xs_knots is None:  # use model_config.MC_num to set for all features
        dcm_config.alt_xs_knots = {
            alt: [MC_num for _ in x_names] for alt, x_names in dcm_config.alt_xs.items()
        }
    return


class TSplineDCM(nn.Module):
    """TSplineDCM, can be interpreted as Taste-Spline- or Tau-Spline-Discrete-Choice-Model
    
    :param ModelConfig model_config: the ML model configuration
    :param DcmConfig dcm_config: the DCM data model configuration
    :param float kld_weight: the weight of the KLD loss, defaults to 0.001
    :param int random_seed: the random seed to use, defaults to 42
    :param torch.device | str device: the device to use, defaults to get_device()
    
    Example:
    --------
    
        >>> from TauSpline.configs import get_sm_dcfg
        >>> mcfg = ModelConfig()
        >>> dcfg = get_sm_dcfg()
        >>> model = TSplineDCM(mcfg, dcfg, kld_weight=0.001, device="cpu")
        >>> model.taste_encoder
        TasteEncoder(
          (layers): ModuleList(
            (0): Linear(in_features=25, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=32, bias=True)
            (3): ReLU()
          )
          (encoder_mu): Linear(in_features=32, out_features=4, bias=True)
          (encoder_logvar): Linear(in_features=32, out_features=4, bias=True)
        )
        >>> model.sp_utils  # doctest: +SKIP
        >>> model.x_name2MonoCoef  # should be empty if share_mc is False
        ModuleDict()
    """

    def __init__(
        self,
        model_config: ModelConfig,
        dcm_config: DcmConfig,
        *,
        z_df: pd.DataFrame | None = None,
        kld_weight: float = 0.001,
        random_seed: int = 42,
        device: torch.device | str = get_device(),
    ):
        super().__init__()
        self.mcfg: ModelConfig = model_config
        self.dcfg: DcmConfig = dcm_config
        self.random_seed: int = random_seed
        self.I = len(self.dcfg.alt_names)
        self.Dz = len(self.dcfg.z_names)
        
        # ---  Taste Encoder  ---
        assert self.mcfg.T_vae_layers is not None, "T_vae_layers must be specified"
        taste_encoder_layers = [len(self.dcfg.z_names)] + self.mcfg.T_vae_layers
        if self.mcfg.T_activation.lower() == "relu":
            activation = torch.nn.ReLU()
        elif self.mcfg.T_activation.lower() == "tanh":
            activation = torch.nn.Tanh()
        elif self.mcfg.T_activation.lower() == "sigmoid":
            activation = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Activation {self.mcfg.T_activation} not supported.")
        
        # ---  Taste Encoder: from z_qn to taste heterogeneity  ---
        self.taste_encoder = TasteEncoder(
            layers=taste_encoder_layers, 
            latent_dim=len(self.dcfg.x_names), 
            kld_weight=kld_weight,
            kld_loss_type=self.mcfg.T_kld_loss_type,
            beta=self.mcfg.T_beta,
            gamma=self.mcfg.T_gamma,
            C_max=self.mcfg.T_C_max,
            max_iter=self.mcfg.T_max_iter,
            activation=activation,
            use_target_sigma=self.mcfg.T_use_target_sigma,
            device=device,
        )
        
        # ---  Monotonic coefficients for attributes  ---
        self.x_name2MonoCoef = nn.ModuleDict()
        if self.dcfg.alt_xs_knots is None:
            set_mc_num(self.dcfg, self.mcfg.MC_num)
            assert self.dcfg.alt_xs_knots is not None, "alt_xs_knots must be specified"
        if getattr(self.dcfg, "share_mc", True):
            x_names_knots: dict[str, int] = {}  # x_name -> num_knots
            alt_xs_knots = self.dcfg.alt_xs_knots
            for (alt, x_names), (alt, knots) in zip(self.dcfg.alt_xs.items(), alt_xs_knots.items()):
                for x_name, n_knots in zip(x_names, knots):
                    if x_name not in x_names_knots:
                        x_names_knots[x_name] = n_knots
                    else:
                        assert x_names_knots[x_name] == n_knots, f"x_name {x_name} has different num_knots {n_knots} in alt {alt}."
            for x_name, n_knots in x_names_knots.items():
                self.x_name2MonoCoef[x_name] = MonoCoef(
                    num=n_knots, 
                    k=self.mcfg.MC_k, 
                    grid_range=self.mcfg.MC_grid_range, 
                    mono=self.dcfg.mono_dirs[x_name],
                    mono_type=self.dcfg.mono_types[x_name],
                    elu_alpha=self.mcfg.MC_elu_alpha,
                    win_size=self.mcfg.MC_win_size,
                    noise_scale=self.mcfg.MC_noise_scale,
                    init_method=self.mcfg.MC_init_method,  # type: ignore
                    taste_name=x_name,
                    device=device,
                    delta_floor=getattr(self.mcfg, "MC_delta_floor", 0.01),
                    reverse_eps=getattr(self.mcfg, "MC_reverse_eps", 1e-8),
                )
            
        # ---  Spline utilities  ---
        self.sp_utils = nn.ModuleDict()
        for alt_name in self.dcfg.alt_names:
            x_names_this_alt = self.dcfg.alt_xs[alt_name]  # [x_name1, x_name2, ...]
            x_knots_this_alt = self.dcfg.alt_xs_knots[alt_name]  # [x1_n_knots, x2_n_knots, ...]
            assert len(x_names_this_alt) == len(x_knots_this_alt), f"alt {alt_name} has different num of x_names ({len(x_names_this_alt)}) and num_knots ({len(x_knots_this_alt)})."
            
            if getattr(self.dcfg, "share_mc", True):
                self.sp_utils[alt_name] = SplineUtility(
                    mono_coefs=[self.x_name2MonoCoef[x_name] for x_name in x_names_this_alt],  # type: ignore
                    interaction_constraints=[[x_names_this_alt.index(a), x_names_this_alt.index(b)] for [a, b] in self.dcfg.interactions],
                    k_gpi_start=self.mcfg.SP_k_gpi_start,
                    heter_mode=self.mcfg.SP_heter_mode,
                    heter_target=getattr(self.mcfg, "SP_heter_target", "increment"),  # I-spline unless told otherwise
                    k_fix_across_g=getattr(self.mcfg, "SP_k_gpi_fix_across_g", False),
                    device=device,
                    k_gpi_cap=self.dcfg.k_gpi_cap[alt_name],  # type: ignore
                    keep_fist_c_zero=True,
                )
            else:
                mc_list: list[MonoCoef] = []
                for x_name, n_knots in zip(x_names_this_alt, x_knots_this_alt):
                    mc_list.append(MonoCoef(
                        num=n_knots,
                        k=self.mcfg.MC_k,
                        grid_range=self.mcfg.MC_grid_range,
                        mono=self.dcfg.mono_dirs[x_name],
                        mono_type=self.dcfg.mono_types[x_name],
                        elu_alpha=self.mcfg.MC_elu_alpha,
                        win_size=self.mcfg.MC_win_size,
                        noise_scale=self.mcfg.MC_noise_scale,
                        init_method=self.mcfg.MC_init_method,  # type: ignore
                        taste_name=x_name,
                        device=device,
                        delta_floor=getattr(self.mcfg, "MC_delta_floor", 0.01),
                        reverse_eps=getattr(self.mcfg, "MC_reverse_eps", 1e-8),
                    ))
                self.sp_utils[alt_name] = SplineUtility(
                    mono_coefs=mc_list,
                    interaction_constraints=[[x_names_this_alt.index(a), x_names_this_alt.index(b)] for [a, b] in self.dcfg.interactions],
                    k_gpi_start=self.mcfg.SP_k_gpi_start,
                    heter_mode=self.mcfg.SP_heter_mode,  # could be either mul or add
                    heter_target=getattr(self.mcfg, "SP_heter_target", "increment"),  # I-spline unless told otherwise
                    alt_name=alt_name,
                    k_fix_across_g=getattr(self.mcfg, "SP_k_gpi_fix_across_g", False),
                    device=device,
                    k_gpi_cap=self.dcfg.k_gpi_cap[alt_name],  # type: ignore
                    keep_fist_c_zero=True
                )
    
        # --- ASC Learner / Param & Mask ---
        if not getattr(self.dcfg, "learn_asc", True):
            self.asc_learner = None
        elif self.dcfg.asc_learner == "linear":
            self.asc_learner = nn.Linear(self.Dz, self.I, bias=False)
        elif self.dcfg.asc_learner == "zspline":
            self.asc_learner = ZSplineHead(
                num_z = self.Dz,
                num_alt=self.I,
                dcfg = self.dcfg,
                z_df = z_df,
                device=device,
            )

        self.asc_param = nn.Parameter(torch.zeros(self.I, dtype=torch.float32), requires_grad=True)

        mask = torch.ones(self.I, dtype=torch.float32)
        if self.dcfg.fix_asc_zero is not None:
            fixed_alt = self.dcfg.fix_asc_zero  # str
            assert fixed_alt in self.dcfg.alt_names, \
                f"fix_asc_zero={fixed_alt} not in alt_names={self.dcfg.alt_names}"
            mask[self.dcfg.alt_names.index(fixed_alt)] = 0.0
        self.asc_mask: Tensor
        self.register_buffer("asc_mask", mask.view(1, self.I))
        
        # cache: alt -> LongTensor of column indices into tau/x_names
        self._alt_cols: dict[str, Tensor] = {}
        for alt in self.dcfg.alt_names:
            cols = [self.dcfg.x_names.index(name) for name in self.dcfg.alt_xs[alt]]
            self._alt_cols[alt] = torch.tensor(cols, device=device, dtype=torch.long)
    
        # ---  Initialize epsilon cache pool for reparmeterization trick  ---
        self.reset_eps_cache()
        
        # --- center tau for all samples with EMA (exponetial moving average) ---
        self.tau_mean_ema: torch.Tensor
        self.register_buffer("tau_mean_ema", torch.zeros(len(self.dcfg.x_names)))  # shape [P]
        self.ema_momentum = getattr(self.mcfg, "ema_momentum", 0.99)
        self.kld_on_centered_mu = getattr(self.mcfg, "kld_on_centered_mu", False)

        self.device = device
        self.to(device)
        
    # def reset_eps_cache(self):
    #     # called once per training batch by the trainer
    #     self.eps_cache_pool: dict[int, Tensor] = {}  # key: ID, value: tensor of epsilon sampled from torch.randn
    
    def reset_eps_cache(self):
        # persistent eps bank indexed by key = idx or ID (0..max_key)
        self._eps_bank = None          # Tensor or None
        self._eps_filled = None        # BoolTensor or None
        self._eps_bank_shape = None    # tuple or None

    def _ensure_eps_bank(self, n_rows: int, I: int, P: int, dtype: torch.dtype) -> None:
        want_shape = (n_rows, I, P) if self.mcfg.TR_per_alt else (n_rows, P)
        if (self._eps_bank is None) or (self._eps_bank_shape != want_shape):
            self._eps_bank = torch.empty(want_shape, device=self.device, dtype=dtype)
            self._eps_bank[:] = float("nan")
            self._eps_filled = torch.zeros(want_shape[0], device=self.device, dtype=torch.bool)
            self._eps_bank_shape = want_shape

    def _get_eps(self, keys: Tensor, I: int, P: int, dtype: torch.dtype, sample: bool | float) -> Tensor:
        if keys.ndim != 1:
            keys = keys.reshape(-1)
        if keys.dtype != torch.long:
            keys = keys.to(torch.long)
        
        B = keys.shape[0]
        if isinstance(sample, float):
            shape = (B, I, P) if self.mcfg.TR_per_alt else (B, P)
            return torch.full(shape, sample, device=self.device, dtype=dtype)
        if isinstance(sample, bool) and sample is False:
            shape = (B, I, P) if self.mcfg.TR_per_alt else (B, P)
            return torch.zeros(shape, device=self.device, dtype=dtype)

        # sample == True
        if self.mcfg.TR_per_sample:
            # keys are dataset row ids, unique within a batch. The bank below re-allocates (and forgets) whenever the
            # batch's max key changes, i.e. for practically every training batch, so what it actually did was: draw
            # randn(B, ...) from the generator restarted at random_seed and hand the rows out in ascending key order.
            # Do exactly that, without the three device->host reads the bank needed (keys.max(), .any(), unique()).
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.random_seed))
            draw_shape = (B, I, P) if self.mcfg.TR_per_alt else (B, P)
            noise = torch.randn(draw_shape, device=self.device, dtype=dtype, generator=gen)
            return noise.index_select(0, keys.argsort().argsort())      # row b gets the draw of its rank among the keys

        max_key = int(keys.max().item())
        if max_key < 0:
            raise ValueError("keys must be non-negative.")
        self._ensure_eps_bank(max_key + 1, I, P, dtype)
        assert self._eps_bank is not None and self._eps_filled is not None, "eps bank not initialized"
        missing_mask = ~self._eps_filled[keys]
        if missing_mask.any():
            uniq_missing = keys[missing_mask].unique()
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.random_seed))
            draw_shape = (uniq_missing.numel(), I, P) if self.mcfg.TR_per_alt else (uniq_missing.numel(), P)
            new_noise = torch.randn(draw_shape, device=self.device, dtype=dtype, generator=gen)
            if self.mcfg.TR_per_alt:
                self._eps_bank[uniq_missing, :, :] = new_noise
            else:
                self._eps_bank[uniq_missing, :] = new_noise
            self._eps_filled[uniq_missing] = True
        return self._eps_bank.index_select(0, keys)

    
    def get_left_margin_from_full_ds(self, full_ds) -> dict[str, float]:
        """
        Get the left margin (min value) for each feature x_name from the full dataset.

        Parameters
        ----------
        full_ds
            The full dataset instance containing train, val, and test sets.

        Returns
        -------
        dict[str, float]
            A dictionary mapping f"{alt_name}.{x_name}" their minimum values (left margin).
        """
        left_margins: dict[str, float] = {}
        for alt_name in self.dcfg.alt_names:
            for x_name in self.dcfg.alt_xs[alt_name]:
                left_margins[f"{alt_name}.{x_name}"] = self.x_bounds_from_dataset(full_ds, alt_name=alt_name, x_name=x_name)[0]

        return left_margins
    
    def get_right_margin_from_full_ds(self, full_ds) -> dict[str, float]:
        """
        Get the right margin (max value) for each feature x_name from the full dataset.

        Parameters
        ----------
        full_ds : DcmDataset
            The full dataset instance containing train, val, and test sets.

        Returns
        -------
        dict[str, float]
            A dictionary mapping f"{alt_name}.{x_name}" their maximum values (right margin).
        """
        right_margins: dict[str, float] = {}
        for alt_name in self.dcfg.alt_names:
            for x_name in self.dcfg.alt_xs[alt_name]:
                right_margins[f"{alt_name}.{x_name}"] = self.x_bounds_from_dataset(full_ds, alt_name=alt_name, x_name=x_name)[1]

        return right_margins
    
    
    @torch.no_grad()
    def update_grid(
        self,
        df: pd.DataFrame,
        *,
        use_availability: bool = True,
        sample_per_x: int | None = 50000,
        clamp_left_tos: dict[str, float] | None = None,
        pin_right_core_tos: dict[str, float] | None = None
    ) -> None:
        """
        Update each MonoCoef's knot grid from raw DataFrame slices.

        Semantics
        ---------
        - clamp_left_tos: clamp the *left boundary* (open-clamped) to a value.
        This repeats the left boundary knot (k+1) times inside MonoCoef.update_grid.
        - pin_right_core_tos: pin *only the last core knot* to a value; the k
        right extension knots remain strictly increasing (extrapolated) and are
        NOT clamped. This preserves tail freedom.

        Parameters
        ----------
        df : pd.DataFrame
            Tabular data subset used for grid updates (typically training-only).
        use_availability : bool, optional
            If True, only rows with availability==1 for an alternative contribute
            to that alternative's feature values. Default True.
        sample_per_x : int | None, optional
            If not None and values for a given feature exceed this count, randomly
            downsample to this size (speed/memory). Default 50_000.
        clamp_left_tos : dict[str, float] | None, optional
            Optional per-(alt,x) overrides for the left boundary:
            keys are "ALT.X". If provided, the left boundary is clamped to this value.
            If not provided, the left boundary defaults to min(values) observed for
            the feature (share_mc=True uses the pooled min).
        pin_right_core_tos : dict[str, float] | None, optional
            Optional per-(alt,x) overrides for the *last core knot*:
            keys are "ALT.X". If provided, the last core is pinned to this value.
            If not provided, it defaults to max(values) observed for the feature
            (share_mc=True uses the pooled max).

        Notes
        -----
        - Aggregates values by feature name across alternatives when share_mc=True.
        - Availability columns follow dataset.config.alt_av_cols ordered like alt_names.
        - To avoid unintended leakage in CV, call this only with training splits.
        """
        cfg = self.dcfg
        assert cfg.alt_names and cfg.alt_xs and cfg.alt_x_cols, "DcmConfig is missing alt mappings."

        # -------- collect values per feature (pooled when share_mc=True) --------
        xbuf: dict[str, list[torch.Tensor]] = {x: [] for x in cfg.x_names}

        if getattr(self.dcfg, "share_mc", True):
            # One shared MonoCoef per feature: pool values across alternatives (with availability if requested)
            for alt_idx, alt_name in enumerate(cfg.alt_names):
                av_mask = None
                if use_availability:
                    av_col_idx = cfg.alt_av_cols[alt_idx]
                    # vectorized boolean mask
                    av_mask = (df.iloc[:, av_col_idx].astype("float").to_numpy() > 0.5)

                x_names_this_alt = cfg.alt_xs[alt_name]
                x_colidx_this_alt = cfg.alt_x_cols[alt_name]

                for j, x_name in enumerate(x_names_this_alt):
                    col_idx = x_colidx_this_alt[j]
                    col = df.iloc[:, col_idx].to_numpy(dtype="float32")
                    if use_availability:
                        col = col[av_mask]
                    # drop NaN
                    if col.size == 0:
                        continue
                    col = col[~np.isnan(col)]
                    if col.size == 0:
                        continue
                    xbuf[x_name].append(torch.from_numpy(col))

            # Update each shared MonoCoef once
            g = torch.Generator().manual_seed(self.random_seed)
            for x_name, chunks in xbuf.items():
                if not chunks:
                    continue
                vals = torch.cat(chunks, dim=0)
                if sample_per_x is not None and vals.numel() > sample_per_x:
                    idx = torch.randperm(vals.numel(), generator=g)[:sample_per_x]
                    vals = vals.index_select(0, idx)

                # defaults from data range
                clamp_left_to = float(vals.min().item())
                pin_right_core_to = float(vals.max().item())

                # allow any per-(alt,x) overrides to widen/tighten bounds
                # (for shared MC, use min of provided lefts and max of provided rights across ALTs)
                if clamp_left_tos:
                    for alt_name in cfg.alt_names:
                        key = f"{alt_name}.{x_name}"
                        if key in clamp_left_tos:
                            clamp_left_to = min(clamp_left_to, float(clamp_left_tos[key]))
                if pin_right_core_tos:
                    for alt_name in cfg.alt_names:
                        key = f"{alt_name}.{x_name}"
                        if key in pin_right_core_tos:
                            pin_right_core_to = max(pin_right_core_to, float(pin_right_core_tos[key]))

                mc: MonoCoef = self.x_name2MonoCoef[x_name]  # type: ignore
                mc.update_grid(
                    vals.view(-1, 1),
                    clamp_left_to=clamp_left_to,
                    pin_right_core_to=pin_right_core_to,
                    grid_eps=self.mcfg.GU_blend_eps,
                )

        else:
            # Separate MonoCoef per (alt, x): do not pool; update each utility's MonoCoef
            g = torch.Generator().manual_seed(self.random_seed)

            for alt_idx, alt_name in enumerate(cfg.alt_names):
                util: SplineUtility = self.sp_utils[alt_name]  # type: ignore
                x_names_this_alt = cfg.alt_xs[alt_name]
                x_colidx_this_alt = cfg.alt_x_cols[alt_name]

                av_mask = None
                if use_availability:
                    av_col_idx = cfg.alt_av_cols[alt_idx]
                    av_mask = (df.iloc[:, av_col_idx].astype("float").to_numpy() > 0.5)

                for j, x_name in enumerate(x_names_this_alt):
                    col_idx = x_colidx_this_alt[j]
                    col = df.iloc[:, col_idx].to_numpy(dtype="float32")
                    if use_availability:
                        col = col[av_mask]
                    if col.size == 0:
                        continue
                    col = col[~np.isnan(col)]
                    if col.size == 0:
                        continue

                    vals = torch.from_numpy(col)
                    if sample_per_x is not None and vals.numel() > sample_per_x:
                        idx = torch.randperm(vals.numel(), generator=g)[:sample_per_x]
                        vals = vals.index_select(0, idx)

                    # defaults from this alt-feature range
                    clamp_left_to = float(vals.min().item())
                    pin_right_core_to = float(vals.max().item())

                    # per-(alt,x) overrides
                    if clamp_left_tos:
                        key = f"{alt_name}.{x_name}"
                        if key in clamp_left_tos:
                            clamp_left_to = float(clamp_left_tos[key])

                    if pin_right_core_tos:
                        key = f"{alt_name}.{x_name}"
                        if key in pin_right_core_tos:
                            pin_right_core_to = float(pin_right_core_tos[key])

                    util.mono_coefs_typed[j].update_grid(
                        vals.view(-1, 1),
                        clamp_left_to=clamp_left_to,
                        pin_right_core_to=pin_right_core_to,
                        grid_eps=self.mcfg.GU_blend_eps,
                    )
    
    
    def build_basis_cache(self, ds: DcmDataset) -> None:
        """Evaluate the B-spline basis of every dataset row once (attribute splines and the z-spline ASC head) so
        that forward() gathers rows by data["idx"] instead of re-evaluating it per batch. Same numbers, far fewer
        kernels. Call after building the model; the caches rebuild themselves after a grid update
        or a checkpoint load. Batches without "idx" (the asu_* helpers) keep the plain evaluation."""
        for i, alt in enumerate(self.dcfg.alt_names):
            self.sp_utils[alt].build_basis_cache(ds._x_per_alt[i])  # type: ignore[attr-defined]
        if isinstance(self.asc_learner, ZSplineHead):
            self.asc_learner.fused_enabled = bool(getattr(self.mcfg, "zhead_fused", False))
            self.asc_learner.build_basis_cache(ds._z)

    def basis_cache_bytes(self) -> int:
        n = 0
        for util in self.sp_utils.values():
            c = getattr(util, "_basis_cache", None)
            n += c.nbytes if c is not None else 0
        if isinstance(self.asc_learner, ZSplineHead):
            n += self.asc_learner.basis_cache_bytes()
        return n

    def forward(self, data: dict[str, Tensor], sample: bool | float = True) -> dict[str, Tensor]:
        """
        :param dict[str, Tensor] data: the input data with:
            {
                "idx": (B,),                                        # index of the sample
                "ID": (B,),                                         # individual id
                "av": (B, I),                                       # available alternatives, 0-1
                "x": list of length I; x[i] is (B, D_i) in the same order as dcfg.alt_xs[alt_names[i]],
                "z": (B, D_z),                                      # individual-specific attributes
            }
        :param bool sample: whether to sample from the latent distribution, default is True.
            Can be switched off for evaluation, or a float directly passed as epsilon.
        
        :return: the output data with:
            {
                "V": V,              # (B, I)                # utilities
                "P": P,              # (B, I)                # probabilities
                "kld": kld,          # ()                    # KL divergence loss
                "mu": mu,            # (B, P)                # mean of the latent distribution
                "logvar": logvar,    # (B, P)                # log variance of the latent distribution
                "tau_raw": tau_raw,  # (B, I, P) or (B, P)   # raw taste preferences  # (B, P) if TR_per_alt is False
                "tau": tau,          # (B, I, P) or (B, P)   # bounded taste preferences  # (B, P) if TR_per_alt is False
                "eps": eps,          # (B, I, P) or (B, P)   # epsilon for reparameterization trick (saved in eps_cache_pool)  # (B, P) if TR_per_alt is False
            }
        """
        # === 1) Get batch size and number of alternatives ===
        B = data["z"].shape[0]
        I = len(self.dcfg.alt_names)
        P = len(self.dcfg.x_names)
        
        # === 2) Get individual-specific taste preferences ===
        mu, logvar = self.taste_encoder(data["z"])  # (B, P)
        
        # === 2.1) EMA-centering of the latent mean (by dataset) ===
        # Do not backprop through EMA statistics
        with torch.no_grad():
            batch_mu = mu.detach().mean(dim=0)  # [P]
            self.tau_mean_ema.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * batch_mu)
            if not getattr(self.mcfg, "center_mu", False):
                self.tau_mean_ema = torch.zeros_like(self.tau_mean_ema)

        # Use centered mean for sampling, so E[tau] ≈ 0 across the dataset
        mu_centered = mu - self.tau_mean_ema  # [B, P]
        
        # === 3) get eps (vectorized bank) ===
        keys = (data["idx"] if self.mcfg.TR_per_sample else data["ID"]).to(torch.long)
        eps = self._get_eps(keys, I, P, dtype=mu.dtype, sample=sample)  # (B, I, P) or (B, P)

        
        # === 4) Reparameterize and bound to [-1, 1] ===
        if self.mcfg.TR_per_alt:
            mu_e = mu_centered[:, None, :].expand(B, I, P)
            lv_e = logvar[:, None, :].expand(B, I, P)
            tau_raw = reparameterize(mu_e, lv_e, eps=eps)    # (B, I, P)
        else:
            tau_raw = reparameterize(mu_centered, logvar, eps=eps)    # (B, P)
        tau = torch.tanh(tau_raw)  # restrain to (-1,1) range
        # tau = nn.Identity()(tau_raw)
        
        # === 5-1) Per-alt spline utilities ===
        idx = data.get("idx")  # dataset row ids: lets the splines gather their cached basis (None -> evaluate from x)
        V_parts = []
        for i, alt in enumerate(self.dcfg.alt_names):
            cols = self._alt_cols[alt]
            Xi = data["x"][i]                                                                                    # (B, D_i)
            # (sigma is recomputed per alternative on purpose: hoisting it changes how the gradient accumulates into
            #  logvar and the run no longer reproduces the paper runs bit for bit)
            if self.mcfg.TR_per_alt:
                tau_i   = tau[:, i, :].index_select(-1, cols)                                                    # (B, D_i) in [-1, 1]
                zstd_i  = eps[:, i, :].index_select(-1, cols)                                                    # (B, D_i) ~ N(0, 1)
                sigma_i = logvar[:, :].exp().sqrt()[:, None, :].repeat(1, I, 1)[:, i, :].index_select(-1, cols)  # (B, D_i)
            else:
                tau_i   = tau.index_select(-1, cols)                                                             # (B, D_i) in [-1, 1]
                zstd_i  = eps.index_select(-1, cols)                                                             # (B, D_i) ~ N(0, 1)
                sigma_i = logvar.exp().sqrt().index_select(-1, cols)                                             # (B, D_i)
            Vi = self.sp_utils[alt](Xi, taste_heter=tau_i, taste_gauss=zstd_i, taste_sigma=sigma_i, idx=idx)     # (B, 1)
            V_parts.append(Vi)
        V_sp = torch.cat(V_parts, dim=1)  # (B, I)

        # === 5-2) ASC ===
        asc_const = self.asc_param.view(1, I).expand(B, I)
        if isinstance(self.asc_learner, ZSplineHead):
            asc_z: Tensor = self.asc_learner(data["z"], idx=idx)
        elif self.asc_learner is not None:
            asc_z = self.asc_learner(data["z"])
        else:
            asc_z = torch.zeros((B, 1), device=self.device, dtype=self.asc_param.dtype).expand(B, I)
        asc_batch = (asc_const + asc_z) * self.asc_mask
        V = V_sp + asc_batch
        
        # === 6) Availability-masked softmax ===
        logits = V.clone()
        logits.masked_fill_(~data["av"].bool(), float("-inf"))
        P = torch.softmax(logits, dim=1)

        
        # === 7) KL ===
        if self.kld_on_centered_mu:
            kld = self.taste_encoder.cal_kld_loss(mu=mu_centered, logvar=logvar)
        else:
            kld = self.taste_encoder.cal_kld_loss()
        kld = self.taste_encoder.cal_weighted_kld_loss(kld_loss=kld)
        
        out = {
            "V": V,              # (B, I)
            "V_sp": V_sp,        # (B, I)
            "ASC": asc_batch,    # (B, I)
            "P": P,              # (B, I)
            "kld": kld,          # ()
            "mu": mu,            # (B, P)
            "logvar": logvar,    # (B, P)
            "tau_raw": tau_raw,  # (B, I, P) or (B, P)
            "tau": tau,          # (B, I, P) or (B, P)
            "eps": eps,          # (B, I, P) or (B, P)
        }
        if self.training:
            pen = self.structural_penalty()
            if pen is not None:
                out["penalty"] = pen
        return out

    def structural_penalty(self) -> Tensor | None:
        """Weighted structural penalties on the POPULATION increments of every spline (train loss only).
        l2_incr:  sum a_g^2                       -> shrinks each drop (RUMBoost's lambda_l2 on leaf values)
        tv_slope: sum |slope_{g+1} - slope_g|      -> l1 trend filtering on the control-polygon slopes
                  slope_g = a_g / (xi_{g+1} - xi_g), xi = Greville abscissae; for k = 1 this is exactly the
                  total variation of the piecewise-linear derivative, so few slope changes = few effective knots.
        Returns None when both weights are 0 (loss unchanged, bit-exact with the pre-penalty code)."""
        l2_w = float(getattr(self.mcfg, "SP_l2_incr", 0.0) or 0.0)
        tv_w = float(getattr(self.mcfg, "SP_tv_slope", 0.0) or 0.0)
        if l2_w <= 0.0 and tv_w <= 0.0:
            return None
        l2 = tv = None
        seen: set[int] = set()
        # ONLY the attribute splines (sp_utils). The z-spline ASC head also owns MonoCoefs, but those are offsets in
        # z, some with degree 0 (stairs for discrete z): the Greville window of a k = 0 spline is empty and its mean
        # is NaN, which poisoned every LPMC trial of the paper protocol.
        for util in self.sp_utils.values():
            for mc in util.mono_coefs_typed:
                if id(mc) in seen:
                    continue
                seen.add(id(mc))
                # mc() again rather than the tensor forward_coef just made: a separate graph branch keeps the gradient
                # accumulation order of the paper runs (reusing it changes the last bits and the trajectory drifts)
                c = mc()                                                   # (G+k,) constrained levels
                a = torch.diff(c)                                          # (G+k-1,) increments (sign = mono dir)
                l2_i = (a * a).sum()
                l2 = l2_i if l2 is None else l2 + l2_i
                if mc.k >= 1:
                    t = mc.grid[0]                                         # (G+2k+1,) extended knots
                    xi = t[1:-1].unfold(0, mc.k, 1).mean(-1)               # (G+k,) Greville abscissae
                    dxi = torch.diff(xi).clamp_min(1e-8)
                    tv_i = torch.diff(a / dxi).abs().sum()
                    tv = tv_i if tv is None else tv + tv_i
        if l2 is None:
            return None
        pen = l2_w * l2
        if tv_w > 0.0 and tv is not None:
            pen = pen + tv_w * tv
        return pen
    
    def resolve_name_to_ind(self, alt_name: str, x_name: str) -> tuple[int, int]:
        """
        Map (alt_name, x_name) -> (alt_idx, attr_idx_within_alt).
        Implement if you need custom mapping. By default, assumes:
            self.dcfg.alt_names and self.dcfg.alt_xs[alt] exist.
        """
        alt_idx = self.dcfg.alt_names.index(alt_name)
        attr_idx = self.dcfg.alt_xs[alt_name].index(x_name)
        return alt_idx, attr_idx

    def x_bounds_from_dataset(self, dataset: DcmDataset, alt_name: str, x_name: str) -> tuple[float, float]:
        alt_idx, attr_idx = self.resolve_name_to_ind(alt_name, x_name)
        x_ = dataset.data.iloc[:, self.dcfg.alt_x_cols[alt_name][attr_idx]]
        return min(x_), max(x_)
    
    def _resolve_sp(self, alt_name: str, x_name: str):
        sp_util: SplineUtility = self.sp_utils[alt_name]  # type: ignore
        attr_full = f"{alt_name}.{x_name}"
        attr_idx = sp_util.attribute_names.index(attr_full)
        mc = sp_util.mono_coefs_typed[attr_idx]
        return sp_util, attr_idx, mc
    
    def asu_individual_curves(
            self,
            batch: dict[str, Tensor],
            alt_name: str,
            x_name: str,
            x_eval: Tensor,
            sample_on: bool | float = False,
            use_general: bool = False,
        ) -> Tensor:
        """
        Vectorized version: no per-individual Python loop.
        - Caches nothing externally (safe drop-in).
        - Computes the B-spline basis once on x_eval, then one matmul for all individuals.
        - Keeps tensors on the model device until the very end; returns on CPU (to match previous behavior).
        """
        sp_util, attr_idx, mc = self._resolve_sp(alt_name, x_name)

        # ---- coerce inputs ----
        p0 = next(self.parameters())
        dev = p0.device
        dt  = p0.dtype
        z = batch["z"].to(device=dev, dtype=dt)                  # (B, Dz) for TasteEncoder
        x_eval = x_eval.to(device=dev, dtype=dt).view(-1, 1)     # (n_points, 1)

        # Taste heterogeneity tau from encoder over z:
        with torch.no_grad():
            mu, logvar = self.taste_encoder(z)                   # (B, P_total)
            if hasattr(self, "tau_mean_ema"):
                mu = mu - self.tau_mean_ema

        cols = [self.dcfg.x_names.index(nm) for nm in self.dcfg.alt_xs[alt_name]]
        alt_idx = self.dcfg.alt_names.index(alt_name)
        P_total = mu.shape[1]

        # ----- build tau (same semantics as before) -----
        # Select per-alt slices
        mu_sub     = mu[:, cols]                     # (B, D_i)
        logvar_sub = logvar[:, cols]                          # (B, D_i)
        sigma_sub  = (0.5 * logvar_sub).exp()                 # (B, D_i)  == torch.exp(0.5*logvar)

        if sample_on is False:
            tau = torch.tanh(mu[:, cols])                        # (B, D_i)
        else:
            keys = (batch["idx"] if self.mcfg.TR_per_sample else batch["ID"]).to(torch.long)

            if isinstance(sample_on, float):
                if self.mcfg.TR_per_alt:
                    eps_full = torch.full((z.size(0), self.I, P_total), float(sample_on), device=dev, dtype=dt)
                    eps_sub  = eps_full[:, alt_idx, :][:, cols]  # (B, D_i)
                else:
                    eps_full = torch.full((z.size(0), P_total), float(sample_on), device=dev, dtype=dt)
                    eps_sub  = eps_full[:, cols]                 # (B, D_i)
            else:
                eps_full = self._get_eps(keys, self.I, P_total, dtype=dt, sample=True)
                if self.mcfg.TR_per_alt:
                    eps_sub = eps_full[:, alt_idx, :][:, cols]   # (B, D_i)
                else:
                    eps_sub = eps_full[:, cols]                  # (B, D_i)

            tau_raw = reparameterize(mu[:, cols], logvar[:, cols], eps=eps_sub)  # (B, D_i)
            tau = torch.tanh(tau_raw)                                             # (B, D_i)

        # Prepare inputs for SplineUtility based on heter_mode
        taste_gauss = None
        taste_sigma = None

        if sp_util.heter_mode in ("mul", "add"):
            pass                                # previous semantics
        elif sp_util.heter_mode == "mul_lognorm":
            if sample_on is False:
                # Deterministic individual curve from posterior mean:
                # z = mu_centered, implement as zstd = mu_centered / (sigma+eps), s = sigma
                eps_num = 1e-8
                taste_gauss = mu_sub / (sigma_sub + eps_num) # (B, D_i)
                taste_sigma = sigma_sub                      # (B, D_i)
            else:
                # Sampled individual curve: z = mu_centered + sigma * eps
                # Implement as zstd = eps, s = sigma
                taste_gauss = eps_sub                        # (B, D_i), ~ N(0,1)  # type: ignore
                taste_sigma = sigma_sub                      # (B, D_i)
        else:
            raise ValueError(f"Unsupported heter_mode: {sp_util.heter_mode}")

        if use_general:
            tau = torch.zeros_like(tau)
        
        # ----- vectorized coef -> curves -----
        with torch.no_grad():
            # Keep on device; do NOT .cpu() here
            coef = sp_util.forward_coef(batch_size=z.shape[0], taste_heter=tau, taste_gauss=taste_gauss, taste_sigma=taste_sigma)   # shape depends on SplineUtility
            # Original indexing kept intact:
            coef_attr = coef[:, attr_idx, 0, :][:, : mc.G + mc.k]                 # [B, G_i + k]

            # Build basis on x_eval ONCE, then single matmul for everyone.
            # Local import avoids changing file-level imports.
            grid_full = mc.grid.squeeze(0).to(device=dev, dtype=dt)               # extended knot vector
            B = B_batch(x_eval, grid_full, k=mc.k).to(dtype=dt)                                 # [n_points, G_i + k] (or [n_points, 1, G_i + k])
            if B.dim() == 3:
                B = B.squeeze(1)                                                  # [n_points, G_i + k]

            # Curves: [B, n_points] = [B, G_i+k] @ [G_i+k, n_points]
            y = coef_attr @ B.t()                                                 # [B, n_points]

        # Match previous behavior: return on CPU for plotting
        return y.detach().cpu()


    def asu_general_trend(
        self,
        alt_name: str,
        x_name: str,
        x_eval: Tensor,
    ) -> Tensor:
        sp_util, attr_idx, mc = self._resolve_sp(alt_name, x_name)
        x_gen, y_gen = mc.get_general_asu(
            n_points=int(x_eval.shape[0]), x_eval=x_eval,
            keep_first_c_zero=sp_util.keep_first_c_zero,
        )
        return y_gen  # [n_points]
    
    def asu_has_general_trend(self) -> bool:
        return True

    def asu_heterogeneity_profile(
        self,
        alt_name: str,
        x_name: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the heterogeneity level profile for a given (alternative, attribute)
        pair, evaluated at the Greville abscissae of the corresponding B-spline basis.

        The function returns:
        - the Greville abscissae (`xi`): representative x-positions of each B-spline basis function;
        - the heterogeneity coefficients (`k_gpi`): per-basis scaling factors capturing
            the strength of taste heterogeneity for this attribute and alternative;
        - the effective internal knot vector (`grid_knots`): interior knots used to draw
            vertical reference lines in visualization.

        Mathematically, for each basis index *g*:
            xi_g = mean(T[g+1 : g+k+1])
            k_gpi = k_{gpi}  (heterogeneity weight at basis g)

        The returned values can be directly used for visualization in a heterogeneity
        panel, where `xi` defines the x-axis positions of scatter points, `k_gpi`
        the corresponding heterogeneity magnitudes, and `grid_knots` the knot
        locations for vertical reference lines.

        Parameters
        ----------
        alt_name : str
            Name of the alternative (e.g., "CAR", "TRAIN", "SM").
        x_name : str
            Name of the attribute (e.g., "CO", "TT", "HE").

        Returns
        -------
        xi : torch.Tensor, shape (Gi+k-2,)
            Greville abscissae (evaluation positions) for each B-spline basis,
            excluding padded boundary knots.
        k_gpi : torch.Tensor, shape (Gi+k-2,)
            Heterogeneity level values corresponding to each Greville point.
            Represents the per-basis strength of individual taste variation.
        grid_knots : torch.Tensor, shape (Gi+k-2,)
            Effective internal knot vector, i.e., `grid_full[k:-k]`, used for
            reference vertical lines in heterogeneity plots.

        Notes
        -----
        - The output excludes the boundary coefficients (`[1:-1]`) to avoid edge
        artifacts from spline padding.
        - This function is primarily used by visualization utilities (e.g.,
        `draw_heter_on_ax`) to render heterogeneity scatter plots with grid
        references.
        - Requires that the model implements `_resolve_sp()` and that the
        returned `sp_util` object exposes `forward_k_gpi()`.
        - Interpretation depends on `SP_heter_target`: with "level" (legacy),
        k_gpi is the relative heterogeneity of the utility *level* near each
        Greville point; with "increment", it is the relative heterogeneity of
        the local *slope* (marginal utility / I-spline increment) there.
        """

        sp_util, attr_idx, mc = self._resolve_sp(alt_name, x_name)
        with torch.no_grad():
            k_gpi = sp_util.forward_k_gpi()[attr_idx][: mc.G + mc.k]  # [Gi+k]
        grid_full = mc.grid.squeeze(0)  # Gi+2k+1
        k = mc.k
        
        # --- Greville abscissae for each B-spline basis on the given knot vector. ---
        T = grid_full
        n_coef = T.numel() - k - 1  # Gi+k
        if k > 0:
            xi = torch.stack([T[i+1:i+k+1].mean() for i in range(n_coef)], dim=0)
        else:
            xi = 0.5 * (T[:-1] + T[1:])
        # grid_full[0:-0] would be an empty slice; degree-0 grids have no padding to strip
        grid_knots = grid_full[k:-k] if k > 0 else grid_full
        return xi[1:-1], k_gpi[1:-1], grid_knots
    
    @torch.no_grad()
    def z_offset_params(self) -> dict:
        """Linear z head: V_offset[:, i] = (asc_param[i] + z @ W[i]) * asc_mask[i],
        where W == asc_learner.weight when asc_learner=="linear". For a non-linear
        z head (zspline) or none, weight is None and only the ASC consts are shown
        (the per-alt z-spline contributions are the separate ZSU plot)."""
        const = self.asc_param.detach().cpu()                       # (I,)
        mask = self.asc_mask.view(-1).detach().cpu().tolist()       # (I,)
        masked_alts = [a for a, m in zip(self.dcfg.alt_names, mask) if m == 0.0]
        weight = None
        note = ""
        if isinstance(self.asc_learner, nn.Linear):
            weight = self.asc_learner.weight.detach().cpu()         # (I, Dz)
        elif self.asc_learner is not None:
            note = (f"asc_learner={self.dcfg.asc_learner!r}: non-linear z head; "
                    "only ASC consts shown (see ZSU plot for the z-spline contributions)")
        return {
            "weight": weight,
            "const": const,
            "z_names": list(self.dcfg.z_names),
            "alt_names": list(self.dcfg.alt_names),
            "masked_alts": masked_alts,
            "note": note,
        }

    @torch.no_grad()
    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f"ASC_base: {(self.asc_param* self.asc_mask).cpu().numpy()}"
        return s
