# tspline/models/taste_encoder.py
import numpy as np
import torch
from torch import Tensor, nn


def reparameterize(mu: Tensor, log_var: Tensor, eps: Tensor | None = None) -> Tensor:
    """Reparameterization trick to sample from N(mu, var) from N(0,1)
    
    :param Tensor mu: Mean of the normal distribution.
    :param Tensor log_var: Log variance of the normal distribution.
    :param Tensor eps: Random noise tensor. Defaults to None.
    
    :return: Reparameterized tensor.
    :rtype: Tensor
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std) if eps is None else eps
    z = mu + eps * std
    return z


def softclip(tensor: torch.Tensor, min_val: float | None = None) -> torch.Tensor:
    """
    Clips the tensor values to be above the specified min_val in a smooth way.
    
    :param Tensor tensor: The input tensor to be clipped.
    :param float | None min_val: The minimum value to clip the tensor values at.
    
    :return: The clipped tensor.
    :rtype: Tensor
    """
    if min_val is not None:
        tensor = min_val + nn.functional.softplus(tensor - min_val)

    return tensor


class TasteEncoder(nn.Module):
    """A Variational Autoencoder (VAE) encoder for modeling individual-specific taste heterogeneity in Discrete Choice Models (DCMs).

    This encoder transforms individual-specific attributes (z_{qn}) into latent taste preference representations (\tau_{np})
    that can be used to capture heterogeneous preferences in DCMs. The encoder uses a fully connected neural network
    architecture with configurable layers and latent dimensions.

    The VAE framework enables learning of complex, non-linear mappings from observed individual characteristics
    to latent taste parameters, while the KL divergence regularization ensures meaningful latent representations.

    Attributes:
        layers (nn.ModuleList): Sequential linear layers with activation functions forming the encoder network
        latent_dim (int): Dimension of the latent space for taste representations
        encoder_mu (nn.Linear): Linear layer mapping encoder output to latent mean
        encoder_logvar (nn.Linear): Linear layer mapping encoder output to latent log variance
        device (torch.device): Computation device for tensors
        kld_weight (float): Weight for KL divergence loss in the total loss computation
        kld_loss_type (str): Type of KL divergence weighting scheme ('B', 'H', or 'None')
        beta (float): Beta value for beta-VAE loss weighting
        gamma (float): Gamma value for gamma-VAE loss weighting
        C_max (float): Maximum capacity constraint for KL divergence
        max_iter (int): Maximum iterations for capacity annealing
        num_iter (int): Current iteration counter for loss computation
        target_sigma (nn.Parameter): Target standard deviation for the latent distribution, can be learned.

    Example:
        >>> encoder = TasteEncoder([10, 64, 32, 16], latent_dim=8, kld_weight=0.1)
        >>> # Encode individual attributes to taste preferences
        >>> individual_attrs = torch.randn(32, 10)  # batch of 32 individuals with 10 attributes
        >>> mu, logvar = encoder(individual_attrs)
        >>> # Calculate weighted KL divergence loss
        >>> kld_loss = encoder.cal_weighted_kld_loss()
    """

    # temporary attributes
    _mu: Tensor
    _logvar: Tensor
    # _z: Tensor
    _target_sigma: Tensor

    # number of interations for annealing KL Divergence loss calculation
    num_iter: int = 0

    def __init__(
        self,
        layers: list[int],
        latent_dim: int,
        kld_weight: float,
        kld_loss_type: str = "B",
        beta: float = 4.0,
        gamma: float = 1000.0,
        C_max: float = 5.0,
        max_iter: int = 10000,
        activation: nn.Module = nn.ReLU(),
        use_target_sigma: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(activation)
        self.latent_dim = latent_dim
        self.encoder_mu = nn.Linear(layers[-1], latent_dim, bias=False)
        self.encoder_logvar = nn.Linear(layers[-1], latent_dim, bias=False)
        self.device = device
        self.kld_weight = kld_weight
        self.kld_loss_type = kld_loss_type
        assert kld_loss_type in [
            "B",
            "H",
            "None",
        ], "kld_loss_type must be 'B' or 'H' or 'None'."
        self.beta = beta
        self.gamma = gamma
        self.C_max = C_max
        self.max_iter = max_iter
        if use_target_sigma:  # sigma is learned
            self.target_sigma = nn.Parameter(torch.ones(latent_dim), requires_grad=True)
        else:  # sigma is fixed as 1, indicating standard normal distribution
            self.target_sigma = nn.Parameter(torch.ones(latent_dim), requires_grad=False)
        self.to(device)
        
    def forward_sigma(self):
        sigma = self.target_sigma
        if self.target_sigma.requires_grad:
            sigma = softclip(sigma, min_val=1e-4)
            sigma = sigma.clamp(max=3.0)
        return sigma

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]: 
        for layer in self.layers:
            x = layer(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        # z = reparameterize(mu, logvar)

        # assign temporary attributes
        self._mu = mu
        self._logvar = logvar
        # self._z = z
        self._target_sigma = self.forward_sigma()
        return mu, logvar

    def cal_kld_loss(self, mu: Tensor | None = None, logvar: Tensor | None = None):
        mu = self._mu if mu is None else mu
        logvar = self._logvar if logvar is None else logvar
        assert (
            mu is not None and logvar is not None
        ), "mu and logvar must be set before loss computation."
        assert mu.shape == logvar.shape, "mu and logvar must have the same shape."

        if self.target_sigma.requires_grad:
            target_sigma = self._target_sigma
            kld_loss = torch.mean(
                0.5 * torch.sum(
                    2 * torch.log(target_sigma) - logvar + 
                    logvar.exp() / (target_sigma**2) + 
                    mu**2 - 1, 
                    dim=1
                ),
                dim=0
            )
        else:
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
            )
        return kld_loss

    def cal_weighted_kld_loss(self, kld_loss: Tensor | None = None):
        """Calculate weighted KL divergence loss based on the provided weight and loss type."""
        self.num_iter += 1
        kld_loss = self.cal_kld_loss() if kld_loss is None else kld_loss

        if self.kld_loss_type == "B":
            kld_loss = self.beta * kld_loss
        elif self.kld_loss_type == "H":
            C_max: Tensor = torch.tensor([self.C_max], device=self.device)
            C = torch.clamp(
                C_max / self.max_iter * self.num_iter, min=0, max=self.C_max
            )
            kld_loss = self.gamma * torch.clamp(kld_loss - C, min=0)
        elif self.kld_loss_type == "None":
            pass
        else:
            raise ValueError(
                f"kld_loss_type must be 'B' or 'H' or 'None', but got {self.kld_loss_type}."
            )

        return self.kld_weight * kld_loss
    
    @torch.no_grad()
    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f"kld_loss_type={self.kld_loss_type}, kld_weight={self.kld_weight}, "
        if self.kld_loss_type == "B":
            s += f"beta={self.beta}, "
        elif self.kld_loss_type == "H":
            s += f"gamma={self.gamma}, C_max={self.C_max}, max_iter={self.max_iter}, "
        
        use_target = self.target_sigma.requires_grad
        s += f"use_target_sigma={use_target}, "
        np.set_printoptions(precision=4, suppress=True, linewidth=500, floatmode="fixed")
        if use_target:
            sigma = self.forward_sigma()
            s += f"\ntarget_sigma {(*sigma.shape,)}={sigma.cpu().numpy()}, "
        return s
    
if __name__ == "__main__":
    te = TasteEncoder(
        layers=[10, 10, 10],
        latent_dim=16,
        kld_weight=1.0,
        kld_loss_type="B",
        beta=4.0,
        gamma=1000.0,
        C_max=5.0,
        max_iter=10000,
        activation=nn.ReLU(),
        use_target_sigma=True,
        device="cpu",
    )
    print(te)
