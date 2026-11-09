# models/reslogit.py — ResLogit (Wong & Farooq 2021): linear MNL utilities passed through residual layers.
"""The utility vector starts as a linear MNL (ASC + own attributes + linear z, same as models/mnl.py) and then goes
through M residual blocks that mix the I alternatives' utilities:

    h_0 = V_lin(x, z)                                   [B, I]
    h_m = h_{m-1} - softplus(h_{m-1}) @ W_m^T           W_m in R^{I x I}, m = 1..M
    V   = h_M

W_m are initialised at zero, so the untrained model IS the MNL and training only adds what the residual layers
can explain. Because the blocks mix alternatives, the attribute utility of one alternative is no longer a
structural function of its own attribute: the asu_* curves are partial-dependence probes (drawn probe-style,
like FDNN). Hyperparameters: configs/models/reslogit.yaml benchmark_config -> self.cfg.

Reference: Wong, M., Farooq, B. (2021). ResLogit: A residual neural network logit model for data-driven choice
modelling. Transportation Research Part C, 126, 103050.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models._base import BaseDCM

RESLOGIT_DEFAULTS: dict[str, Any] = {"n_layers": 4}


class ResLogit(BaseDCM):
    def _build(self) -> None:
        cfg = {**RESLOGIT_DEFAULTS, **self.cfg}
        self.n_layers = int(cfg["n_layers"])
        self.betas = nn.ModuleList([nn.Linear(self.Xs[i], 1, bias=False) for i in range(self.I)])
        self.gamma = nn.Parameter(torch.zeros(self.I, self.Dz))
        self.asc = nn.Parameter(torch.zeros(self.I))
        self.res = nn.ModuleList([nn.Linear(self.I, self.I, bias=False) for _ in range(self.n_layers)])
        for lin in self.res:
            nn.init.zeros_(lin.weight)                      # start exactly at the MNL

    def _linear(self, batch: dict[str, Tensor]) -> Tensor:
        p0 = next(self.parameters())
        z = batch["z"].to(device=p0.device, dtype=p0.dtype)
        x = [xi.to(device=p0.device, dtype=p0.dtype) for xi in batch["x"]]
        return torch.stack([self.betas[i](x[i]).squeeze(-1) + z @ self.gamma[i] + self.asc[i] for i in range(self.I)], dim=-1)

    def _utilities(self, batch: dict[str, Tensor]) -> Tensor:
        h = self._linear(batch)
        for lin in self.res:
            h = h - lin(torch.nn.functional.softplus(h))
        return h

    def forward(self, batch: dict[str, Tensor], sample: bool | float = False) -> dict[str, Tensor]:
        V = self._utilities(batch)
        return {"V": V, "P": self._masked_softmax(V, batch.get("av"))}

    # ---- explainability: probe curves (the residual layers mix alternatives) ----
    @torch.no_grad()
    def asu_individual_curves(self, batch: dict[str, Tensor], alt_name: str, x_name: str, x_eval: Tensor,
                              sample_on: bool | float = False) -> Tensor:
        i, p = self.resolve_name_to_ind(alt_name, x_name)
        p0 = next(self.parameters())
        z = batch["z"].to(device=p0.device, dtype=p0.dtype)
        x_base = [xi.to(device=p0.device, dtype=p0.dtype).clone() for xi in batch["x"]]
        xe = x_eval.to(device=p0.device, dtype=p0.dtype).view(-1)
        N = z.shape[0]
        was_training = self.training
        self.eval()
        x0 = [xi.clone() for xi in x_base]
        x0[i][:, p] = 0.0
        V0 = self._utilities({"z": z, "x": x0})[:, i]
        ys = []
        for v in xe:
            xj = [xi.clone() for xi in x_base]
            xj[i][:, p] = v
            ys.append((self._utilities({"z": z, "x": xj})[:, i] - V0).unsqueeze(1))
        if was_training:
            self.train()
        return torch.cat(ys, dim=1)

    def asu_has_general_trend(self) -> bool:
        return False

    @torch.no_grad()
    def z_offset_params(self) -> dict:
        """ASC of the linear part plus what the residual layers add at an all-zero input (z = 0, x = 0)."""
        p0 = next(self.parameters())
        was_training = self.training
        self.eval()
        V = self._utilities({"z": torch.zeros(1, self.Dz, device=p0.device, dtype=p0.dtype),
                             "x": [torch.zeros(1, int(d), device=p0.device, dtype=p0.dtype) for d in self.Xs]})[0]
        if was_training:
            self.train()
        return {"weight": self.gamma.detach().cpu(), "const": V.detach().cpu(), "z_names": list(self.dcfg.z_names),
                "alt_names": list(self.dcfg.alt_names), "masked_alts": [],
                "note": "ResLogit: linear z weights of the input layer; const = output at an all-zero input"}
