# tspline/benchmarks/base_dcm.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch import nn, Tensor

from models.config import get_device

class BaseDCM(nn.Module, ABC):
    """
    A lightweight, trainer-compatible base class for Discrete Choice Models.

    Contract with the trainer:
    - __init__(model_config, dcm_config, device, random_seed)
    - __call__(batch) -> dict with keys:
        - "V": utilities [B, I]
        - optional "P": probabilities [B, I] (already masked if you compute it)
        - optional "kld": scalar KL loss term
    - reset_eps_cache()
    - update_grid(dataset, use_availability: bool, sample_per_x: int)
    """

    def __init__(
        self,
        model_config: Any,
        dcm_config: Any,
        z_df: pd.DataFrame | None = None,  # kept for compatibility with trainer
        device: torch.device | str = get_device(),
        random_seed: int = 42,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # Store configs
        self.mcfg = model_config  # not used, but kept for compatibility
        self.dcfg = dcm_config
        # per-model hyperparameters (configs/models/<m>.yaml: benchmark_config). Each subclass merges
        # them over its own *_DEFAULTS inside _build(); nothing is written to module-level state.
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.random_seed: int = random_seed
        self.I: int = len(self.dcfg.alt_names)
        self.Dz: int = len(self.dcfg.z_names)
        self.Xs: list[int] = [len(self.dcfg.alt_xs[alt_name]) for alt_name in self.dcfg.alt_names]

        # Resolve device
        self.device: torch.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        # Subclasses typically register parameters/layers here
        self._build()

        # Final device move for all registered parameters/buffers
        self.to(self.device)
    
    @abstractmethod
    def _build(self) -> None:
        """
        Subclasses should register parameters/layers here.
        """
        pass
    
    @abstractmethod
    def _utilities(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Subclasses should implement this to compute utilities.
        
        :param dict[str, Tensor] batch: A batch of data.
            {
                "idx": (B,),                                        # index of the sample
                "ID": (B,),                                         # individual id
                "av": (B, I),                                       # available alternatives, 0-1
                "x": list of length I; x[i] is (B, D_i) in the same order as dcfg.alt_xs[alt_names[i]],
                "z": (B, D_z),                                      # individual-specific attributes
            }
        :return: Utilities [B, I].
        :rtype: Tensor
        """
        pass
    
    def reset_eps_cache(self) -> None:
        """
        Optional per-epoch reset for reparam noise caches, etc.
        Default no-op so classical models do not need to implement it.
        """
        pass
    
    def update_grid(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Optional grid update interface for spline-based models.
        Default no-op; benchmarks that do not need it can ignore.
        """
        pass

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Numerically stable softmax over available alternatives only.
        logits: [B, I], mask: [B, I] in {0,1} or None
        returns P: [B, I] with zeros where mask == 0
        """
        if mask is None:
            return torch.softmax(logits, dim=-1)
        masked_logits = torch.where(mask > 0, logits, logits.new_full(logits.shape, -1e9))
        P = torch.softmax(masked_logits, dim=-1)
        # Hard zero on unavailable alts to mirror trainer's convention
        if (mask is not None) and (mask.dtype != torch.bool):
            mask_bool = mask > 0
        else:
            mask_bool = mask
        return torch.where(mask_bool, P, torch.zeros_like(P))

    def predict_choice(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Argmax prediction over utilities. Returns indices [B].
        """
        with torch.no_grad():
            V = self._utilities(batch)
            return V.argmax(dim=-1)
        
    @abstractmethod
    def forward(self, batch: Dict[str, Tensor], sample: bool | float = False) -> Dict[str, Tensor]:
        """
        Subclasses should implement this to compute utilities and probabilities.
        
        :param dict[str, Tensor] data: the input data with:
            {
                "idx": (B,),                                        # index of the sample
                "ID": (B,),                                         # individual id
                "av": (B, I),                                       # available alternatives, 0-1
                "x": list of length I; x[i] is (B, D_i) in the same order as dcfg.alt_xs[alt_names[i]],
                "z": (B, D_z),                                      # individual-specific attributes
            }
        :param bool | float sample: whether to sample from the latent distribution, default is False.
            Can be switched off for evaluation, or a float directly passed as epsilon.
            Optional for non-VAE models, default is False.
        
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
        pass
    
    
    # ---- ASU Explainability API ----
    # Plotter uses these to gather curves/lines & meta-information.

    def resolve_name_to_ind(self, alt_name: str, x_name: str) -> tuple[int, int]:
        """
        Map (alt_name, x_name) -> (alt_idx, attr_idx_within_alt).
        Implement if you need custom mapping. By default, assumes:
            self.dcfg.alt_names and self.dcfg.alt_xs[alt] exist.
        """
        alt_idx = self.dcfg.alt_names.index(alt_name)
        attr_idx = self.dcfg.alt_xs[alt_name].index(x_name)
        return alt_idx, attr_idx

    def x_bounds_from_dataset(
        self,
        dataset,  # TSplineTrainer.ds
        alt_name: str,
        x_name: str,
    ) -> tuple[float, float]:
        """
        Return (xmin, xmax) for the chosen attribute using the dataset.
        The default implementation looks up dataset examples and min/max that column.
        """
        alt_idx, attr_idx = self.resolve_name_to_ind(alt_name, x_name)
        x_data = dataset.data.iloc[:, self.dcfg.alt_x_cols[alt_name][attr_idx]]
        return min(x_data), max(x_data)

    def asu_individual_curves(
        self,
        batch: Dict[str, Tensor],
        alt_name: str,
        x_name: str,
        x_eval: Tensor,             # [n_points, 1] over the chosen attribute's domain
        sample_on: bool | float = False,  # API for compatibility with other models.
    ) -> Tensor:
        """
        Return per-individual curves as [N, n_points].
        """
        raise NotImplementedError

    @torch.no_grad()
    def asu_general_trend(
        self,
        alt_name: str,
        x_name: str,
        x_eval: Tensor,
    ) -> Optional[Tensor]:
        """
        Optional structural general trend curve y_gen(x): [n_points].
        Return None if not applicable.
        """
        return None

    @torch.no_grad()
    def asu_heterogeneity_profile(
        self,
        alt_name: str,
        x_name: str,
    ) -> Optional[tuple[Tensor, Tensor, Tensor]]:
        """
        Optional heterogeneity profile for the bottom panel.
        Return (xi, k_level), where both are 1D tensors with same length.
          - xi: evaluation points (e.g., Greville abscissae) [Gi+1]
          - k_level: heterogeneity level per point [Gi+1]
          - grid: the grid points used to evaluate the heterogeneity profile [Gi+1]
        Return None if not applicable.
        """
        return None
    
    @abstractmethod
    def asu_has_general_trend(self) -> bool:
        """
        Whether the model has a general trend.
        """
        return False

    # ---- z-head (individual-characteristics fixed offset) Explainability API ----
    @torch.no_grad()
    def z_offset_params(self) -> Optional[Dict[str, Any]]:
        """The linear individual-characteristics utility offset, if the model has one:

            V_offset[:, i] = const[i] + z @ weight[i]

        Returns a dict (or None when the model has no such head):
          - "weight":      Tensor (I, Dz) or None (None == no additive z-linear term,
                           e.g. TasteNet where z modulates tastes instead);
          - "const":       Tensor (I,)  — the constant ASC offset (effective, post-mask);
          - "z_names":     list[str] column order of `weight`;
          - "alt_names":   list[str] row order of `weight`/`const`;
          - "masked_alts": list[str] alts whose ASC is pinned to 0 (dead params);
          - "note":        str  optional human note for the figure/legend.

        Tensors are returned detached on CPU. Default: None (no z head).
        """
        return None
    
    def get_right_margin_from_full_ds(self, full_ds) -> dict[str, float]:
        """
        Get the right margin (max value) for each feature x_name from the full dataset.

        Parameters
        ----------
        full_ds
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