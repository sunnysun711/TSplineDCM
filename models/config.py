# models/config.py — ModelConfig, TrainConfig and the shared type aliases.
# Merged verbatim from v1 tspline/config/{custom_types,model,train}.py.
from typing import Literal
import torch

MonotoneFlag = Literal[-1, 0, 1]  # dec, free, inc
KldType = Literal["B", "H", "None"]
MonoType = Literal["strict", "window", "soft"]
HeterType = Literal["mul", "add", "mul_lognorm"]
HeterTarget = Literal["level", "increment"]  # what the heterogeneity factor multiplies: B-spline coefficient levels, or I-spline increments
EvalType = Literal["mu", "mc", "percentile"]
SchedulerType = Literal["none", "step", "cosine"]
InitMethods = Literal["random", "smart", "zero"]
AscLearnerType = Literal["linear", "zspline"]

_DEVICE: torch.device | None = None

def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")
    return _DEVICE

def sanitize_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(tag))
    return safe or None

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Taste encoder (beta-VAE)
    T_vae_layers: list[int] | None = None
    T_kld_loss_type: KldType = "B"
    T_kld_weight: float | None = None
    T_beta: float = 4.0
    T_gamma: float = 1000.0
    T_C_max: float = 5.0
    T_max_iter: int = 10000
    T_activation: str = "relu"
    T_use_target_sigma: bool = False

    # Reparameterize switches
    TR_per_sample: bool = False
    TR_per_alt: bool = False

    # MonoCoef defaults
    MC_num: int = 20
    MC_k: int = 3
    MC_grid_range: tuple[float, float] = (0.0, 1.0)
    MC_grid_fix_left_zero: bool = True  # fixing the left grid to be zero
    MC_elu_alpha: float = 0.01
    MC_win_size: int = 2
    MC_noise_scale: float = 0.05
    MC_init_method: InitMethods = "smart"
    MC_delta_floor: float = 0.01      # strict-mode increment floor; 0 -> an increment may vanish (knot selection)
    MC_reverse_eps: float = 1e-8      # clamp in the strict reverse after a grid refit: 1e-8 = a vanished basis is dead
                                      #   for good (raw ~ -18), 1e-4 = it can revive (raw ~ -9)

    # SplineUtility
    SP_k_gpi_start: float = 0.5
    SP_k_gpi_fix_across_g: bool = False
    SP_heter_mode: HeterType = "mul"  # or "add"
    # "increment": factor multiplies I-spline increments (per-individual monotone AS-UCFs);
    # "level": legacy B-spline level scaling (individual monotonicity not guaranteed).
    SP_heter_target: HeterTarget = "increment"
    # structural penalties on the population increments a_g (train loss only; 0 = off):
    SP_l2_incr: float = 0.0           # lambda * sum a_g^2            (RUMBoost's lambda_l2 on leaf values)
    SP_tv_slope: float = 0.0          # lambda * sum |slope_{g+1} - slope_g|, slope_g = a_g / (xi_{g+1} - xi_g)
                                      #   (l1 trend filtering: few slope changes = few effective knots)

    # Grid update (epoch, not step, and epoch starts at 1)
    basis_cache: bool = True  # evaluate every row's B-spline basis once per grid and gather by row (same numbers, fewer kernels)
    zhead_fused: bool = False  # with basis_cache: the z-spline ASC head as ONE matmul over all z (LPMC step 31 -> 15 ms). The sum
    #                            runs in another order, so after ~20 epochs a run drifts off the paper run's trajectory (split 0:
    #                            0.6692 vs 0.6700). Off = per-z einsums on the cached basis, bit-exact with the paper runs. Opt in
    #                            for new experiments where reproducing the paper runs to the bit does not matter.
    GU_every: int = 0  # if 0, disable grid update every GU_every epoch
    GU_warmup: int = 1  # GU at the start of GU_warmup epoch 
    GU_sample_per_x: int | None = None  # if not specified, use all data
    GU_epoch_cap: int = 100
    GU_blend_eps: float = 0.02  # 1.0 for uniform, 0.0 for adaptive
    
    # optional switches
    center_mu: bool = False  # whether to center mu by subtracting the mean
    ema_momentum: float = 0.99  # tasteEncoder mu centering momentum
    kld_on_centered_mu: bool = True 

    def __post_init__(self) -> None:
        self.T_vae_layers = self.T_vae_layers or [64, 32]
        assert self.MC_num >= 2, "invalid spline spec"

from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 16
    random_seed: int = 42
    # data splitting lives in splits.py (fixed npz per dataset)

    device: str | None = None

    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    scheduler: SchedulerType = "none"
    epochs: int = 50
    step_size: int = 20
    gamma: float = 0.5

    amp: bool = False
    grad_clip: float = 0.0
    deterministic: bool = False  # torch.use_deterministic_algorithms; bitwise repro at ~1.6x forward cost

    log_every: int = 50
    eval_every: int = 1
    early_stopping_patience: int | None = None
    compact_log: bool = False
    
    fused_adam: bool = False  # torch.optim.Adam(fused=True) on CUDA: one kernel for all parameters (-0.6 ms/step on LPMC) but its
    #                           rounding differs from the foreach Adam, so a run no longer reproduces the paper runs to 1e-8; opt in
    eval_mode: EvalType = "mu"  # mu, mc, percentile
    eval_mc_samples: int = 1  # number of Monte Carlo samples for epsilon
    eval_percentiles: list[float] | None = None
    eval_sample_ratio: float = 1.0  # ratio of samples to use for evaluation with sampling

    def __post_init__(self) -> None:
        self.device = self.device or get_device().type
        if self.device not in ("cuda", "cpu", "mps"):
            raise ValueError(f"Unsupported device type: {self.device}")
