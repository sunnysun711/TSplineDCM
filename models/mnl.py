# tspline/benchmarks/mnl.py
import torch
from torch import nn, Tensor

from models._base import BaseDCM



class MNL(BaseDCM):
    """
    Linear-utility Multinomial Logit benchmark.

    Expected batch:
      - "z": [B, Dz]                 individual features
      - "x": list of I tensors;      each x[i] is [B, D_i] (alt-specific features)
      - "av": [B, I] in {0,1}        availability mask

    Utility:
      V_bi = sum_p x_{bip} * beta_{ip} + z_b @ gamma_i + ASC_i

    Shapes at runtime:
      - beta_i: implemented as a Linear(D_i, 1, bias=False) per alternative
      - gamma:  [I, Dz]
      - asc:    [I]
    """
    
    def _build(self) -> None:
        self.betas: nn.ModuleList = nn.ModuleList([nn.Linear(self.Xs[i], 1, bias=False) for i in range(self.I)])
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(self.I, self.Dz))
        self.asc: nn.Parameter = nn.Parameter(torch.zeros(self.I))
        
    def _utilities(self, batch: dict[str, Tensor]) -> Tensor:
        z = batch["z"].to(self.device)            # [B, Dz]
        x_list: list[Tensor] = [xi.to(self.device) for xi in batch["x"]]
        V = torch.stack([self.betas[i](x_list[i]).squeeze(-1) + z @ self.gamma[i] + self.asc[i] for i in range(self.I)], dim=-1)  # [B, I]
        return V  # [B, I]
    
    def forward(self, batch: dict[str, Tensor], sample: bool | float = False) -> dict[str, Tensor]:
        # sample is ignored for plain DNN (needed only for VAE-like models)
        V = self._utilities(batch)
        P = self._masked_softmax(V, batch.get("av"))
        return {"V": V, "P": P}
    
    @torch.no_grad()
    def asu_individual_curves(self, batch: dict[str, Tensor], alt_name: str, x_name: str, x_eval: Tensor, sample_on: bool | float = False,) -> Tensor:
        """
        Compute ASU curve for a single individual.

        Args:
            batch: Input batch dictionary. Must contain "idx" key, the `numel()` 
                   of which is used as the number of individuals.
            alt_name: Name of the alternative.
            x_name: Name of the attribute.
            x_eval: Evaluation points for the attribute. [D_eval] 
            sample_on: API for compatibility with other models. Ignored for MNL.

        Returns:
            Tensor: ASU curve values. [N, D_eval]
        """
        alt_ind, attr_ind = self.resolve_name_to_ind(alt_name=alt_name, x_name=x_name)
        beta_ip = self.betas[alt_ind].weight[0, attr_ind]  # type: ignore
        N = batch['idx'].numel()  # number of individuals in the batch
        asu = beta_ip * x_eval.unsqueeze(0).expand(N, -1)  # [N, D_eval]
        return asu  # [N, D_eval]

    def asu_general_trend(self, alt_name: str, x_name: str, x_eval: Tensor) -> Tensor | None:
      """Compute the general trend of ASU for a single alternative.

      :param str alt_name: Alternative name.
      :param str x_name: Attribute name.
      :param Tensor x_eval: Evaluation points for the attribute. [D_eval] 
      :return Tensor | None: ASU curve values. [D_eval]
      """
      asu = self.asu_individual_curves(batch={"idx": torch.tensor([0])}, alt_name=alt_name, x_name=x_name, x_eval=x_eval)  # [1, D_eval]
      return asu[0]  # [D_eval]
    
    def asu_has_general_trend(self) -> bool:
        """
        Whether the model has a general trend.
        """
        return True

    @torch.no_grad()
    def z_offset_params(self) -> dict:
        """Linear z head: V_offset[:, i] = asc[i] + z @ gamma[i]. MNL fixes no ASC,
        so all rows are learned (masked_alts is empty)."""
        return {
            "weight": self.gamma.detach().cpu(),     # (I, Dz)
            "const": self.asc.detach().cpu(),        # (I,)
            "z_names": list(self.dcfg.z_names),
            "alt_names": list(self.dcfg.alt_names),
            "masked_alts": [],
            "note": "",
        }
