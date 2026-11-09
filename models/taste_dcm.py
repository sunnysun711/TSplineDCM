# tspline/benchmarks/taste_dcm.py
import torch
from torch import nn, Tensor

from models._base import BaseDCM


# defaults; override per experiment via benchmark_config -> self.cfg
T_DCM_DEFAULTS: dict = {
    "encoder_layers": [128],
    "k_p_learnable": True,
    "k_p": [1.0],  # if k_p_learnable is False, this should be a list of floats
}

class TasteDCM(BaseDCM):
    """
    Taste-heterogeneous DCM with per-alternative linear utilities scaled
    by an individual-specific taste factor.

    The deterministic utility for individual b and alternative i is:

        V_{bi} = v0_i + s_{bi} * (x_{bi} @ v_i)

    where:
        - v0_i is the ASC for alternative i.
        - v_i ∈ R^{P_i} is the linear coefficient vector for alternative i.
        - x_{bi} ∈ R^{P_i} is the corresponding feature vector.

    The individual-specific taste scale s_{bi} is:

        s_{bi} = 1 + τ_{b, J_i}^T k_{J_i}

    where:
        - τ_b = Encoder(z_b) ∈ [-1, 1]^J
            (J equals len(x_names); encoder ends with Tanh)
        - k ∈ [0,1]^J
            (obtained via sigmoid(k_raw) if learnable)
        - J_i is the index set of taste dimensions relevant to alt i
          (from `tau_index_per_alt[i]`)

    Thus, only the taste dimensions associated with alternative i contribute
    to its scaling.

    Notes:
    - No z_b–specific term is added directly to utility; z_b enters only via τ_b.
    - Availability mask av ∈ {0,1}^{B×I} is handled in the softmax.

    Expected batch keys:
        - "z": Tensor [B, Dz]
        - "x": list of I tensors; x[j] is [B, P_j]
        - "av": [B, I] availability mask
    """

    def _build(self) -> None:
        cfg = {**T_DCM_DEFAULTS, **self.cfg}
        self.betas: nn.ModuleList = nn.ModuleList([nn.Linear(self.Xs[i], 1, bias=False) for i in range(self.I)])
        self.asc: nn.Parameter = nn.Parameter(torch.zeros(self.I))

        # Taste index mapping per alternative: each entry is a list of idx into [0..J-1]
        x_names = getattr(self.dcfg, "x_names", None)
        assert x_names is not None, "x_names must be provided."
        tau_index_per_alt = [
            [x_names.index(x_name) for x_name in alt_x_names]
            for alt_x_names in self.dcfg.alt_xs.values()
        ]
        self.tau_index_per_alt: list[list[int]] = tau_index_per_alt

        # Encoder architecture
        self.encoder_layers: list[int] = list(cfg["encoder_layers"])

        # Determine taste dimension J
        if not cfg["k_p_learnable"]:
            k_p_fixed = cfg["k_p"]
            assert isinstance(k_p_fixed, list), "k_p must be a list of floats."
            assert len(k_p_fixed) == len(x_names), "k_p must have the same length as x_names."
            assert all(0 <= k <= 1 for k in k_p_fixed), "k_p must be in [0,1]."
            self.k_raw = nn.Parameter(torch.tensor(k_p_fixed, dtype=torch.float32, device=self.device), requires_grad=False)
            self.k_act = nn.Identity()
        else:
            self.k_raw = nn.Parameter(torch.randn(len(x_names), 1), requires_grad=True)
            self.k_act = nn.Sigmoid()  # keep between [0,1]
        
        # -------------------------
        # Encoder over z -> tau in [-1,1]^J
        # -------------------------
        layers: list[nn.Module] = []
        in_dim = self.Dz
        for h in self.encoder_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, len(x_names)))
        layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*layers)


    def get_k(self) -> Tensor:
        return self.k_act(self.k_raw)  # [J, 1]


    def _utilities(self, batch: dict[str, Tensor]) -> Tensor:
        # Inputs
        z: Tensor = batch["z"]                                       # [B, Dz]
        x_list: list[Tensor] = [xi for xi in batch["x"]]             # list of [B, Pj]
        B = z.shape[0]
        I = self.I

        # Encode tastes: tau in [-1,1], shape [B, J]
        tau: Tensor = self.encoder(z)                                # [B, J]
        k: Tensor = self.get_k()                                     # [J, 1]

        # Per-alt linear x term: xb[:, j] = (Linear_j x_j).squeeze(-1)
        # self.betas[j]: Linear(Pj, 1, bias=False)
        xb_terms = [self.betas[j](x_list[j]).squeeze(-1) for j in range(I)]  # list of [B]
        xb = torch.stack(xb_terms, dim=1)                            # [B, I]

        # Per-alt scalar taste scale: s_bj = 1 + tau_b[idx] @ k[idx]
        s_list: list[Tensor] = []
        for j in range(I):
            idx = self.tau_index_per_alt[j]
            if len(idx) == 0:
                s_list.append(torch.ones(B, device=self.device))
            else:
                tj = tau[:, idx]                                     # [B, len(idx)]
                kj = k[idx, :]                                       # [len(idx), 1]
                sj = 1.0 + (tj @ kj).squeeze(-1)                     # [B]
                s_list.append(sj)
        s = torch.stack(s_list, dim=1)                               # [B, I]

        # Intercept + scaled linear term
        V = self.asc.unsqueeze(0) + s * xb                           # [B, I]

        return V

    def forward(self, batch: dict[str, Tensor], sample: bool | float = False) -> dict[str, Tensor]:
        # 'sample' kept for signature compatibility; unused here.
        V = self._utilities(batch)
        out: dict[str, Tensor] = {"V": V}
        out["P"] = self._masked_softmax(V, batch.get("av"))
        return out
    
    @torch.no_grad()
    def asu_individual_curves(
        self,
        batch: dict[str, Tensor],
        alt_name: str,
        x_name: str,
        x_eval: Tensor,
        sample_on: bool | float = False,
    ) -> Tensor:
        """Get ASU curves for each individual.

        :param dict[str, Tensor] batch: batch data
        :param str alt_name: alternative name
        :param str x_name: attribute name
        :param Tensor x_eval: evaluation points for the attribute
        :param bool | float sample_on: API for compatibility with other models. Ignored for TasteDCM.
        :return Tensor: ASU curves for each individual. [N, n_points]
        """
        alt_idx, attr_idx = self.resolve_name_to_ind(alt_name, x_name)
        # device & dtype from model parameters
        p0 = next(self.parameters())
        dev = p0.device
        dt  = p0.dtype

        # coerce z and every x_i
        z = batch["z"].to(device=dev, dtype=dt)
        # x_list = [xi.to(device=dev, dtype=dt) for xi in batch["x"]]
        B = z.shape[0]

        tau = self.encoder(z)                     # [B, J]
        k = self.get_k()                          # [J, 1]
        idx = self.tau_index_per_alt[alt_idx]
        if len(idx) == 0:
            s_b = torch.ones(B, device=dev, dtype=dt)  # [B]
        else:
            s_b = (1.0 + (tau[:, idx] @ k[idx]).squeeze(-1))  # [B]

        beta_ip = self.betas[alt_idx].weight[0, attr_idx]      # type: ignore scalar

        xs = x_eval.squeeze(-1).to(device=dev, dtype=dt)       # [n_points]
        y_each = s_b.unsqueeze(1) * (beta_ip * xs.unsqueeze(0))  # [B, n_points]
        return y_each  # [B, n_points]
    
    @torch.no_grad
    def asu_general_trend(self, alt_name: str, x_name: str, x_eval: Tensor) -> Tensor:
        alt_idx, attr_idx = self.resolve_name_to_ind(alt_name, x_name)
        # device & dtype from model parameters
        p0 = next(self.parameters())
        dev = p0.device
        dt  = p0.dtype
        
        k = self.get_k()                          # [J, 1]
        beta_ip = self.betas[alt_idx].weight[0, attr_idx]      # type: ignore scalar
        y_eval = beta_ip * x_eval.squeeze(-1).to(device=dev, dtype=dt)  # [n_points]
        return y_eval
    
    def asu_has_general_trend(self) -> bool:
        """
        Whether the model has a general trend.
        """
        return True
