# tspline/benchmarks/tastenet.py
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional

from models._base import BaseDCM


# defaults = TasteNet paper setup for Swissmetro; override per experiment via benchmark_config -> self.cfg
TASTE_NET_DEFAULTS: dict = {
    "hidden_activation": "relu",       # "relu" | "tanh"
    "output_activation": "exp",   # "softplus" | "relu" | "exp"
    "hidden_layers": [110],
    "l2_reg": 0.0,
    "fix_CO": True,
    "set_ASC_zero": "CAR",  # or one of the alt_names
}


def _make_act(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported hidden_activation: {name}")


def _pos_activation(x: Tensor, kind: str) -> Tensor:
    k = kind.lower()
    if k == "relu":
        return F.relu(x)
    if k == "softplus":
        return F.softplus(x)  # smooth, strictly positive for x>0
    if k == "exp":
        return torch.exp(x)
    raise ValueError(f"Unsupported output_activation: {kind}")


class _TasteHead(nn.Module):
    """
    A small MLP: z -> beta_i(z) in R^{P_i} (pre-activation).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: list[int], hidden_activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(d, h), _make_act(hidden_activation)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)  # [B, P_i]


class TasteNet(BaseDCM):
    """
    TasteNet: betas are predicted from z by per-alternative heads, then
    passed through a direction-aware monotone activation:
        if mono_dirs[x_name] == +1: beta = +pos_act(raw)
        if mono_dirs[x_name] == -1: beta = -pos_act(raw)
        if mono_dirs[x_name] ==  0: beta = raw
    Utilities:
        V_bi = ASC_i + x_{bi} · beta_i(z_b)
    """

    def _build(self) -> None:
        # Configs
        cfg = {**TASTE_NET_DEFAULTS, **self.cfg}
        self.hidden_layers: list[int] = list(cfg.get("hidden_layers", []))
        self.hidden_activation: str = str(cfg.get("hidden_activation", "relu"))
        self.output_activation: str = str(cfg.get("output_activation", "softplus"))
        self.l2_reg: float = float(cfg.get("l2_reg", 0.0))

        # ASC per alternative
        self.asc: nn.Parameter = nn.Parameter(torch.zeros(self.I, dtype=torch.float32, device=self.device))

        # Per-alt heads: z -> beta_i(z) in R^{P_i}
        heads: list[_TasteHead] = []
        for i in range(self.I):
            heads.append(_TasteHead(self.Dz, self.Xs[i], self.hidden_layers, self.hidden_activation))
        self.heads = nn.ModuleList(heads)

        # Names and monotonic directions
        # dcfg.alt_xs: dict[alt_name -> List[x_name]]
        # dcfg.mono_dirs: dict[x_name -> {-1, 0, +1}]
        assert hasattr(self.dcfg, "alt_xs"), "dcfg.alt_xs must exist"
        assert hasattr(self.dcfg, "mono_dirs"), "dcfg.mono_dirs must exist"
        self.alt_names: list[str] = list(self.dcfg.alt_xs.keys())
        self.alt_xs: dict[str, list[str]] = self.dcfg.alt_xs
        self.mono_dirs: dict[str, int] = self.dcfg.mono_dirs

        self.fix_CO: bool = bool(cfg.get("fix_CO", False))
        self.set_ASC_zero: Optional[str] = str(cfg.get("set_ASC_zero", None))
        
        # --- build beta_fix_masks for CO per alternative ---
        # beta_fix_masks[i]: shape [P_i], True where beta must be fixed (CO)
        self.beta_fix_masks: list[Tensor] = []
        for i, alt_name in enumerate(self.alt_names):
            x_names = self.alt_xs[alt_name]
            mask = torch.zeros(len(x_names), dtype=torch.bool, device=self.device)
            if self.fix_CO and ("CO" in x_names):
                co_idx = x_names.index("CO")
                mask[co_idx] = True
            self.beta_fix_masks.append(mask)

        # --- ASC learning mask: 1 for learnable, 0 for pinned-to-zero ---
        # shape [I]
        asc_mask = torch.ones(self.I, dtype=torch.float32, device=self.device)
        if isinstance(self.set_ASC_zero, str):
            if self.set_ASC_zero not in self.alt_names:
                raise ValueError(f"set_ASC_zero={self.set_ASC_zero} not in alt_names {self.alt_names}")
            j = self.alt_names.index(self.set_ASC_zero)
            asc_mask[j] = 0.0
        self.register_buffer("asc_learn_mask", asc_mask)

    def _apply_mono_activation_one_alt(self, beta_raw: Tensor, alt_idx: int) -> Tensor:
        """
        Apply direction-aware monotone activation to beta_raw for alternative alt_idx.
        beta_raw: [B, P_i]
        """
        alt_name = self.alt_names[alt_idx]
        x_names = self.alt_xs[alt_name]
        out = []
        for j, x_name in enumerate(x_names):
            d = int(self.mono_dirs.get(x_name, 0))
            b = beta_raw[:, j]
            if d == 0:
                out.append(b)
            elif d > 0:
                out.append(_pos_activation(b, self.output_activation))
            else:
                out.append(-_pos_activation(b, self.output_activation))
        return torch.stack(out, dim=-1)  # [B, P_i]

    def _predict_betas(self, z: Tensor) -> list[Tensor]:
        """
        Return a list of betas per alternative after monotone activation,
        then apply fixed-beta masks (e.g., CO = -1).
        Each element has shape [B, P_i].
        """
        betas = []
        z = z.to(self.device)
        for i in range(self.I):
            raw = self.heads[i](z)
            beta_i = self._apply_mono_activation_one_alt(raw, i)  # [B, P_i]

            # Apply fixed-CO mask if any: beta := (-1) at masked positions, no gradient.
            mask = self.beta_fix_masks[i]  # [P_i] bool
            if mask.any():
                # Expand mask to [B, P_i]
                m = mask.unsqueeze(0).expand(beta_i.size(0), -1)
                # Detach trick: constant where masked; original where not masked.
                beta_i = beta_i * (~m) + m.to(beta_i.dtype).to(beta_i.device) * -1.0
                beta_i = beta_i.detach() * m + beta_i * (~m)  # cut grads on masked entries

            betas.append(beta_i)
        return betas


    def _utilities(self, batch: dict[str, Tensor]) -> Tensor:
        z: Tensor = batch["z"].to(self.device)
        x_list: list[Tensor] = [xi.to(self.device) for xi in batch["x"]]

        betas = self._predict_betas(z)                           # list of [B, P_i]
        xb = [(x_list[i] * betas[i]).sum(dim=-1) for i in range(self.I)]
        xb = torch.stack(xb, dim=-1)                             # [B, I]

        asc_eff = (self.asc * self.asc_learn_mask).unsqueeze(0)  # [1, I], pinned ASC is zero and no grad  # type: ignore
        V = asc_eff + xb                                         # [B, I]
        return V


    def forward(self, batch: dict[str, Tensor], sample: bool | float = False) -> dict[str, Tensor]:
        V = self._utilities(batch)
        P = self._masked_softmax(V, batch.get("av"))
        out: dict[str, Tensor] = {"V": V, "P": P}

        # Optional L2 penalty (only for the heads' weights); returned for the trainer to add if desired.
        if self.l2_reg > 0.0:
            l2 = torch.zeros((), device=self.device)
            for m in self.heads.modules():
                if isinstance(m, nn.Linear):
                    l2 = l2 + (m.weight.pow(2).sum())
            out["reg"] = self.l2_reg * l2
        return out

    # ---------- ASU Explainability API ----------
    @torch.no_grad()
    def asu_individual_curves(
        self,
        batch: dict[str, Tensor],
        alt_name: str,
        x_name: str,
        x_eval: Tensor,
        sample_on: bool | float = False,  # API for compatibility with other models. Ignored for TasteNet.
    ) -> Tensor:
        alt_idx, attr_idx = self.resolve_name_to_ind(alt_name, x_name)
        p0 = next(self.parameters())
        dev, dt = p0.device, p0.dtype

        z = batch["z"].to(device=dev, dtype=dt)
        betas = self._predict_betas(z)              # list of [B, P_i]
        beta_ip = betas[alt_idx][:, attr_idx]       # [B]

        xs = x_eval.squeeze(-1).to(device=dev, dtype=dt)  # [n_points]
        y_each = beta_ip.unsqueeze(1) * xs.unsqueeze(0)   # [B, n_points] (no ASC)
        return y_each

    def asu_general_trend(
        self,
        alt_name: str,
        x_name: str,
        x_eval: Tensor,
    ) -> Optional[Tensor]:
        return None

    def asu_has_general_trend(self) -> bool:
        return False

    @torch.no_grad()
    def z_offset_params(self) -> dict:
        """TasteNet has NO additive z-linear offset: z enters utility only through
        the per-alt taste heads (beta_i(z) * x). The only fixed offset is the ASC
        constant, so weight is None and the figure shows ASC consts alone."""
        const = (self.asc * self.asc_learn_mask).detach().cpu()  # (I,) effective (masked -> 0)
        masked = [self.set_ASC_zero] if isinstance(self.set_ASC_zero, str) else []
        return {
            "weight": None,
            "const": const,
            "z_names": list(self.dcfg.z_names),
            "alt_names": list(self.dcfg.alt_names),
            "masked_alts": masked,
            "note": "TasteNet: z modulates tastes (beta(z) * x); no additive z-offset, only ASC consts",
        }
