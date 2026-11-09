# tspline/benchmarks/fdnn.py
from typing import Any
import torch
from torch import nn, Tensor

from models._base import BaseDCM

# default hyperparameters; override per experiment via configs/models/FDNN.yaml benchmark_config -> self.cfg
FDNN_DEFAULTS: dict[str, Any] = {
    "hidden_layers": [250, 250],  # [250, 250]
    "activation": "relu",
    "drop_out": 1e-5,  # 1e-5
}

class FDNN(BaseDCM):
    """
    F-DNN (Fully Connected Feedforward DNN) Benchmark for DCM. 
    
    See Fig. 1 in Wang Shenhao's paper (10.1016/j.trc.2020.01.012)
    
    There is no specified utility function, as all inputs are correlated in the DNN structure.
    """
    def _build(self) -> None:
        cfg = {**FDNN_DEFAULTS, **self.cfg}
        self.hidden_layers: list[int] = list(cfg["hidden_layers"])
        activation_str: str = cfg["activation"]
        if activation_str.lower() == "relu":
            activation = nn.ReLU()
        elif activation_str.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_str.lower() == "sigmoid":
            activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation {activation_str} not supported.")
        self.activation: nn.Module = activation
        
        self.drop_out: float = float(cfg["drop_out"])
        
        self.layers = nn.ModuleList()
        layers = [self.Dz + sum(self.Xs)] + self.hidden_layers + [self.I]
        
        for layer1, layer2 in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(layer1, layer2))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(self.drop_out))
        self.layers = self.layers[:-2]  # remove final activation and dropout
    
    @staticmethod
    def _cat_input(batch: dict[str, Tensor]) -> Tensor:
        """
        Concatenate input features with z and x_list.
        """
        return torch.cat([batch["z"], *batch["x"]], dim=-1)
    
    def _utilities(self, batch: dict[str, Tensor]) -> Tensor:
        """Calculate utilities.

        :param dict[str, Tensor] batch: A batch of data.
        :return Tensor: Utilities [B, I].
        """
        # device & dtype from model parameters
        p0 = next(self.parameters())
        dev = p0.device
        dt  = p0.dtype

        # coerce z and every x_i
        z = batch["z"].to(device=dev, dtype=dt)
        x_list = [xi.to(device=dev, dtype=dt) for xi in batch["x"]]

        # your existing feature concatenation
        inpt = self._cat_input({"z": z, "x": x_list})  # type: ignore
        for layer in self.layers:
            inpt = layer(inpt)
        return inpt  # [B, I]
    
    def forward(self, batch: dict[str, Tensor], sample: bool | float = False) -> dict[str, Tensor]:
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
        :return: Utilities [B, I].
            {
                "V": V,              # (B, I)                # utilities
                "P": P,              # (B, I)                # probabilities
            }
        :return dict[str, Tensor]: A dictionary of outputs.
        """
        V = self._utilities(batch)
        P = self._masked_softmax(V, batch["av"])
        return {"V": V, "P": P}
        
    def asu_individual_curves(
        self,
        batch: dict[str, Tensor],
        alt_name: str,
        x_name: str,
        x_eval: Tensor,
        sample_on: bool | float = False,
        anchor_type: str = "zero",
    ) -> Tensor:
        """
        Return per-individual *contribution curves* for the selected attribute:
            y_i(x) = V_i(z, x_attr := x) - V_i(z, x_attr := x0)
        where x0 is an anchor (here: batch mean of that attribute).
        Output shape: [N, n_points].
        
        :param dict[str, Tensor] batch: A batch of data.
        :param str alt_name: The name of the alternative.
        :param str x_name: The name of the attribute.
        :param Tensor x_eval: The evaluation points.
        :param bool | float sample_on: API for compatibility with other models. Ignored for FDNN.
        :param str anchor_type: The type of anchor. Defaults to "mean". Should be "mean", "min", or "zero".
        :return Tensor: Contribution curves.
        """
        # ---- indices ----
        alt_idx, attr_idx = self.resolve_name_to_ind(alt_name, x_name)

        # ---- device & dtype normalization ----
        p0 = next(self.parameters())
        dev = p0.device
        dt  = p0.dtype
        z = batch["z"].to(device=dev, dtype=dt)
        x_list_base = [xi.to(device=dev, dtype=dt).clone() for xi in batch["x"]]
        x_eval = x_eval.to(device=dev, dtype=dt).view(-1, 1)  # [n_points, 1]

        N = z.shape[0]
        n_points = x_eval.shape[0]

        # ---- choose anchor x0 (everyone starts at (x0, 0)) ----
        # You can change this to median or a global constant if needed.
        if anchor_type == "mean":
            x0 = x_list_base[alt_idx][:, attr_idx].mean(0, keepdim=True)  # scalar tensor [1]
        elif anchor_type == "min":
            x0 = x_list_base[alt_idx][:, attr_idx].min(0, keepdim=True).values  # scalar tensor [1]
        elif anchor_type == "zero":
            x0 = torch.tensor([0.0], device=dev, dtype=dt)  # scalar tensor [1]
        else:
            raise ValueError(f"Unknown anchor_type: {anchor_type}")

        # ---- eval mode (disable dropout), restore afterwards ----
        was_training = self.training
        self.eval()

        # ---- baseline utilities at anchor ----
        x_list = [xi.clone() for xi in x_list_base]
        x_list[alt_idx][:, attr_idx] = x0.expand(N)
        V0 = self._utilities({"z": z, "x": x_list})[:, alt_idx]  # type: ignore [N]

        # ---- sweep over x_eval: replace only the target attribute ----
        ys = []
        for j in range(n_points):
            x_list = [xi.clone() for xi in x_list_base]
            x_list[alt_idx][:, attr_idx] = x_eval[j].expand(N)
            Vj = self._utilities({"z": z, "x": x_list})[:, alt_idx]  # type: ignore [N]
            ys.append((Vj - V0).unsqueeze(1))  # [N,1]

        # ---- restore train/eval state ----
        if was_training:
            self.train()

        return torch.cat(ys, dim=1)  # [N, n_points]
    
    def asu_has_general_trend(self) -> bool:
        """
        Whether the model has a general trend.
        """
        return False

    @torch.no_grad()
    def z_offset_params(self) -> dict:
        """FDNN has no structural ASC. Proxy: the pre-softmax outputs at an
        ALL-ZERO input (z = 0 and every attribute = 0) -- the same anchor the asu_* probe curves use --
        reported as per-alternative constants. No linear z weight (weight = None); nothing is masked, so
        figures/tables take differences to the reference alternative like for the torch MNL."""
        p0 = next(self.parameters())
        z = torch.zeros(1, self.Dz, device=p0.device, dtype=p0.dtype)
        x = [torch.zeros(1, int(d), device=p0.device, dtype=p0.dtype) for d in self.Xs]
        was_training = self.training
        self.eval()
        V = self._utilities({"z": z, "x": x})[0].detach().cpu()          # (I,)
        if was_training:
            self.train()
        return {
            "weight": None,
            "const": V,
            "z_names": list(self.dcfg.z_names),
            "alt_names": list(self.dcfg.alt_names),
            "masked_alts": [],
            "note": "FDNN: no ASC; pre-softmax output at an all-zero input (z = 0, x = 0)",
        }
