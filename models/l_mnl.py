# models/l_mnl.py — L-MNL (Sifringer, Lurkin & Alahi 2020): linear-in-attributes utility + a DNN term on z.
"""Learning-MNL: the "knowledge" part is a plain MNL over each alternative's own attributes (with ASC), the
"data-driven" part is a small DNN that maps the individual characteristics z (only z, never the attributes) to one
additive utility term per alternative:

    V[:, i] = ASC_i + x[i] @ beta_i + DNN(z)[:, i]

So the attribute utilities are straight lines shared by everyone (AS-UCF = beta_i * x, structural), and all the
individual-level flexibility sits in a non-linear ASC. Ported from DCM2 solver/l_mnl/model.py; hyperparameters
come from configs/models/l_mnl.yaml benchmark_config -> self.cfg.

Reference: Sifringer, B., Lurkin, V., Alahi, A. (2020). Enhancing discrete choice models with representation
learning. Transportation Research Part B, 140, 236-261.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models._base import BaseDCM

LMNL_DEFAULTS: dict[str, Any] = {"hidden_layers": [100, 100], "activation": "relu", "drop_out": 1e-3}
_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class LMNL(BaseDCM):
    def _build(self) -> None:
        cfg = {**LMNL_DEFAULTS, **self.cfg}
        self.hidden_layers: list[int] = list(cfg["hidden_layers"])
        act = _ACT.get(str(cfg["activation"]).lower())
        if act is None:
            raise ValueError(f"activation {cfg['activation']!r} not supported")
        self.drop_out = float(cfg["drop_out"])
        self.betas = nn.ModuleList([nn.Linear(self.Xs[i], 1, bias=False) for i in range(self.I)])
        self.asc = nn.Parameter(torch.zeros(self.I))
        layers: list[nn.Module] = []
        dims = [self.Dz] + self.hidden_layers + [self.I]
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act(), nn.Dropout(self.drop_out)]
        self.z_net = nn.Sequential(*layers[:-2])                     # no activation / dropout on the output

    def _utilities(self, batch: dict[str, Tensor]) -> Tensor:
        p0 = next(self.parameters())
        z = batch["z"].to(device=p0.device, dtype=p0.dtype)
        x = [xi.to(device=p0.device, dtype=p0.dtype) for xi in batch["x"]]
        V_lin = torch.stack([self.betas[i](x[i]).squeeze(-1) + self.asc[i] for i in range(self.I)], dim=-1)
        return V_lin + self.z_net(z)

    def forward(self, batch: dict[str, Tensor], sample: bool | float = False) -> dict[str, Tensor]:
        V = self._utilities(batch)
        return {"V": V, "P": self._masked_softmax(V, batch.get("av"))}

    # ---- explainability: the attribute part is linear, the z part is a non-linear ASC ----
    @torch.no_grad()
    def asu_individual_curves(self, batch: dict[str, Tensor], alt_name: str, x_name: str, x_eval: Tensor,
                              sample_on: bool | float = False) -> Tensor:
        i, p = self.resolve_name_to_ind(alt_name, x_name)
        beta = self.betas[i].weight[0, p]
        return beta * x_eval.view(1, -1).expand(batch["idx"].numel(), -1)

    @torch.no_grad()
    def asu_general_trend(self, alt_name: str, x_name: str, x_eval: Tensor) -> Tensor:
        return self.asu_individual_curves({"idx": torch.tensor([0])}, alt_name, x_name, x_eval)[0]

    def asu_has_general_trend(self) -> bool:
        return True

    @torch.no_grad()
    def z_offset_params(self) -> dict:
        """No linear z weight (the z term is a DNN). const = ASC + DNN(0): the offset at all-zero z, the same
        anchor as FDNN / TSplineDCM's zspline head. Nothing is pinned, so figures subtract the reference alt."""
        p0 = next(self.parameters())
        was_training = self.training
        self.eval()
        const = (self.asc + self.z_net(torch.zeros(1, self.Dz, device=p0.device, dtype=p0.dtype))[0]).detach().cpu()
        if was_training:
            self.train()
        return {"weight": None, "const": const, "z_names": list(self.dcfg.z_names), "alt_names": list(self.dcfg.alt_names),
                "masked_alts": [], "note": "L-MNL: ASC + DNN(z) at z = 0 (non-linear z term)"}
