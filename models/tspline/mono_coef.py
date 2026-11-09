# tspline/core/mono_coef.py
from typing import Iterable, Literal

import torch
from torch import nn, Tensor

from models.config import MonoType, MonotoneFlag, InitMethods
from .b_spline import B_batch, coef2curve, curve2coef, extend_grid

# strict mode: every increment is softplus(raw) + this floor, so no increment can vanish and the AS-UCF is
# strictly monotone. apply_* adds it and reverse_* must subtract it (fix: the reverse used to forget it,
# so every grid update steepened each increment by the floor).
STRICT_DELTA_FLOOR = 0.01


def stable_softplus_inverse(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable inverse of softplus function.

    eps: increments at or below it are clamped to it before the inverse. It decides how "dead" a basis whose refit
    increment vanished (grid update) becomes: 1e-8 -> raw about -18 (never revives), 1e-4 -> raw about -9 (can).
    
    Examples
    --------
    
    >>> v = torch.tensor([1e-9, 1.0, 20.0])
    >>> inv = stable_softplus_inverse(v)
    >>> torch.allclose(torch.nn.functional.softplus(inv), v, atol=1e-6)
    True
    """
    x = torch.clamp(x, min=float(eps))
    mask = x > 20
    result = torch.zeros_like(x)
    result[mask] = x[mask]
    result[~mask] = torch.log(torch.expm1(x[~mask]))
    return result

def stable_elu_inverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Numerically stable inverse of ELU function.

    For ELU, when y>=0, inverse is y; when y<0, inverse is ln(y/alpha + 1).

    Examples
    --------
    
    >>> alpha = 0.1
    >>> y = torch.tensor([-0.05, 0.0, 0.3])
    >>> x = stable_elu_inverse(y, alpha)
    >>> torch.allclose(torch.nn.functional.elu(x, alpha=alpha), y, atol=1e-6)
    True
    """
    mask = x >= 0
    result = torch.zeros_like(x)
    result[mask] = x[mask]  # For x >= 0, elu_inv(x) = x
    result[~mask] = torch.log(x[~mask] / alpha + 1)  # For x < 0
    return result

def normalize_constrained_coef(
    constrained: torch.Tensor,
    *,
    norm_type: str = "maxabs",    # "maxabs" or "l2"
    target: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize a *constrained* coefficient vector to a given scale and (optionally) 
    anchor-center it: <coef, B(x0)>=0.

    :param Tensor constrained: (G+k,) constrained coeffs (after monotone transform)
    :param str norm_type: "maxabs" -> max|c| = target; "l2" -> ||c||2 = target
    :param float target: target norm value
    :param float eps: numerical floor
    
    :return: constrained_normed (G+k,) normalized (and optionally centered) constrained coeffs
    :rtype: Tensor

    Examples
    --------
    
    >>> G, k = 5, 3
    >>> raw = torch.randn(G+k, requires_grad=True)
    >>> constrained = apply_cumulative_monotonic(raw, mono=1, mono_type="strict")
    >>> normalized = normalize_constrained_coef(constrained)
    >>> bool(torch.isclose(normalized.abs().max(), torch.tensor(1.0)))
    True
    """
    c = constrained

    if norm_type == "maxabs":
        s = c.abs().max().clamp_min(eps)
    elif norm_type == "l2":
        s = c.norm(p=2).clamp_min(eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    c = c * (target / s)

    return c

def apply_cumulative_monotonic(
    coef_raw: torch.Tensor,
    *,
    mono: MonotoneFlag,
    mono_type: MonoType,
    elu_alpha: float | None = None,
    win_size: int | None = None,
    delta_floor: float = STRICT_DELTA_FLOOR,
) -> Tensor:
    """Apply cumulative monotonicity to the coefficients.

    delta_floor (strict mode): every increment is softplus(raw) + delta_floor. The default 0.01 keeps the curve
    strictly monotone with a visible minimum slope; 0 lets an increment vanish (needed when many knots are
    offered and the data should decide which ones carry a drop).
    
    Examples
    --------
    Strict increasing::
    
        >>> _ = torch.manual_seed(0)
        >>> raw = torch.randn(6, requires_grad=True)
        >>> out = apply_cumulative_monotonic(raw, mono=1, mono_type="strict")
        >>> bool(torch.all(out[1:] >= out[:-1]))
        True

    Soft decreasing (more "soft", allowing small fluctuations but overall decreasing)::
    
        >>> out = apply_cumulative_monotonic(raw, mono=-1, mono_type="soft", elu_alpha=0.01)
        >>> (out[0] >= out[-1] - 1e-4).item()
        True

    Window-based increasing::
    
        >>> vals = torch.linspace(-0.2, 0.2, 9).requires_grad_()
        >>> out = apply_cumulative_monotonic(vals, mono=1, mono_type="window", win_size=3)
        >>> def win_means(t, w):
        ...     ms = []
        ...     for s in range(0, len(t), w):
        ...         e = min(s+w, len(t))
        ...         ms.append(t[s:e].mean())
        ...     return torch.stack(ms)
        >>> m = win_means(out, 3)
        >>> bool(torch.all(m[1:] >= m[:-1]))
        True
    """
    if mono == 0:  # no monotonic constraint
        return coef_raw  
    
    # if mono_type == "strict":
    #     delta = torch.nn.functional.softplus(coef_raw)
    if mono_type == "strict":
        delta = torch.nn.functional.softplus(coef_raw) + delta_floor   # floor keeps every increment > 0 (0 = may vanish)
        # Cumulative sum to enforce monotonicity
        delta = torch.cumsum(delta, dim=-1)
        if mono == -1:
            delta = -delta  # reverse for decreasing monotonicity
        return delta
    elif mono_type == "soft":
        assert elu_alpha is not None and 0 <= elu_alpha <= 0.5, f"alpha for ELU() should be a float between 0 and 0.5, got {elu_alpha}"
        delta = torch.nn.functional.elu(coef_raw, alpha=elu_alpha)
        # Cumulative sum to enforce monotonicity
        delta = torch.cumsum(delta, dim=-1)
        if mono == -1:
            delta = -delta  # reverse for decreasing monotonicity
        return delta
    elif mono_type == "window":
        assert win_size is not None and 0 < win_size < coef_raw.numel(), f"win_size must be provided for window-based monotonicity, got {win_size}"
        # Window-based monotonic constraints
        n_coef = coef_raw.numel()
        n_windows = (coef_raw.numel() + win_size - 1) // win_size  # ceil division

        # first n_windows coef as window means raw parameters
        window_means_raw = coef_raw[:n_windows]
        variations = coef_raw[n_windows:]
        window_means = torch.cumsum(torch.nn.functional.softplus(window_means_raw), dim=-1)
        if mono == -1:
            window_means = -window_means
        
        # Create coefficient tensor using vectorized operations
        result_parts = []
        variation_idx = 0
        
        for w in range(n_windows):
            start_global = w * win_size
            end_global = min((w + 1) * win_size, n_coef)
            current_win_size = end_global - start_global
            
            if current_win_size == 1:
                # Single element window
                result_parts.append(window_means[w:w+1])
            else:
                # Multi-element window: use vectorized construction
                current_variations = variations[variation_idx:variation_idx + current_win_size - 1]
                variation_idx += current_win_size - 1
                
                # Gradient-safe: Use torch.cat instead of indexing
                window_base = window_means[w].repeat(current_win_size - 1)
                elements = window_base + current_variations
                
                # Last element ensures window constraint: sum = win_size * mean
                last_element = window_means[w] * current_win_size - elements.sum()
                
                # Use torch.cat for gradient preservation
                window_coef = torch.cat([elements, last_element.unsqueeze(0)])
                result_parts.append(window_coef)
        
        return torch.cat(result_parts)
    else:
        raise ValueError(f"Unknown mono_type: {mono_type}")

def reverse_cumulative_monotonic(
    coef_constrained: torch.Tensor,
    *,
    mono: MonotoneFlag,
    mono_type: MonoType,
    elu_alpha: float | None = None,
    win_size: int | None = None,
    delta_floor: float = STRICT_DELTA_FLOOR,
    reverse_eps: float = 1e-8,
) -> Tensor:
    """
    Reverse the cumulative monotonic transformation to recover raw coefficients.

    Strict mode: an increment at or below delta_floor (a least-squares refit onto a new grid can produce a
    locally non-monotone target) is clamped to 1e-8 inside stable_softplus_inverse, i.e. projected onto the
    monotone cone (raw about -18: that basis carries no drop) instead of turning into NaN.
    
    This function performs the inverse operation of apply_cumulative_monotonic.
    
    Args:
        coef_constrained: Monotonicity-constrained coefficients (1D tensor)
        mono: Monotonic direction (-1, 0, 1)
        mono_type: Type of monotonic constraint
        elu_alpha: Alpha parameter for ELU (required for soft mode)
        win_size: Window size (required for window mode)
    
    Returns:
        Raw coefficients that would produce the constrained coefficients
        
    Note:
        The reverse transformation follows:
        
        For strict mode (softplus):
        1. constrained = direction * cumsum(softplus(raw))
        2. diff(constrained) = direction * softplus(raw[1:])
        3. raw[1:] = softplus_inverse(direction * diff(constrained))
        
        For soft mode (ELU):
        1. constrained = direction * cumsum(elu(raw, alpha))
        2. diff(constrained) = direction * elu(raw[1:], alpha)
        3. raw[1:] = elu_inverse(direction * diff(constrained), alpha)
        
        For window mode:
        Reconstructs window means and variations from constrained coefficients
        
    Examples
    --------
    
    Round-trip (strict)::
    
        >>> _ = torch.manual_seed(0)
        >>> raw = torch.randn(5)
        >>> constrained = apply_cumulative_monotonic(raw, mono=1, mono_type="strict")
        >>> rec = reverse_cumulative_monotonic(constrained, mono=1, mono_type="strict")
        >>> torch.allclose(rec, raw, atol=1e-5)
        True

    Round-trip (soft)::
    
        >>> constrained = apply_cumulative_monotonic(raw, mono=-1, mono_type="soft", elu_alpha=0.1)
        >>> rec = reverse_cumulative_monotonic(constrained, mono=-1, mono_type="soft", elu_alpha=0.1)
        >>> torch.allclose(rec, raw, atol=1e-5)
        True

    Round-trip (window)::
    
        >>> vals = torch.linspace(-0.2, 0.2, 7)
        >>> constrained = apply_cumulative_monotonic(vals, mono=1, mono_type="window", win_size=3)
        >>> rec = reverse_cumulative_monotonic(constrained, mono=1, mono_type="window", win_size=3)
        >>> torch.allclose(rec, torch.cat([rec[:3], rec[3:]]), atol=1e-6)  # smoke check
        True
    """
    if mono == 0:  # No constraint
        return coef_constrained.clone()
    
    n_coef = coef_constrained.numel()
    constrained_seq = coef_constrained.clone()
    
    if mono_type in ['strict', 'soft']:
        # Apply direction correction
        if mono == -1:
            constrained_seq = -constrained_seq
        
        # Compute differences (reverse of cumsum)
        if n_coef > 1:
            differences = torch.diff(constrained_seq)
            
            # Reconstruct raw coefficients
            raw_coef = torch.zeros_like(constrained_seq)
            
            # Apply inverse activation based on constraint type
            if mono_type == 'strict':
                raw_coef[0] = stable_softplus_inverse(constrained_seq[0] - delta_floor, reverse_eps)
                raw_coef[1:] = stable_softplus_inverse(differences - delta_floor, reverse_eps)
            else:  # soft mode
                assert elu_alpha is not None, "elu_alpha must be provided for soft mode"
                raw_coef[0] = stable_elu_inverse(constrained_seq[0], elu_alpha)
                raw_coef[1:] = stable_elu_inverse(differences, elu_alpha)
                
            return raw_coef
        else:
            # Single element case
            if mono_type == 'strict':
                return stable_softplus_inverse(constrained_seq - delta_floor, reverse_eps)
            else:
                assert elu_alpha is not None, "elu_alpha must be provided for soft mode"
                return stable_elu_inverse(constrained_seq, elu_alpha)
    
    elif mono_type == "window":
        assert win_size is not None, "win_size must be provided for window mode"
        
        # Window-based reverse transformation
        n_windows = (n_coef + win_size - 1) // win_size
        
        # Extract window means from constrained coefficients
        window_means = torch.zeros(n_windows, device=constrained_seq.device)
        variations = []
        
        for w in range(n_windows):
            start_idx = w * win_size
            end_idx = min((w + 1) * win_size, n_coef)
            window_coefs = constrained_seq[start_idx:end_idx]
            window_len = end_idx - start_idx
            window_means[w] = torch.mean(window_coefs)
            
            # Extract variations (except for single-element windows)
            if window_len > 1:
                for i in range(window_len - 1):
                    variations.append(window_coefs[i] - window_means[w])
        
        # Apply direction correction and inverse cumsum for window means
        if mono == -1:
            window_means = -window_means
        
        # Reconstruct raw window means
        if n_windows > 1:
            mean_diffs = torch.diff(window_means)
            raw_means = torch.zeros_like(window_means)
            raw_means[0] = stable_softplus_inverse(window_means[0])
            raw_means[1:] = stable_softplus_inverse(mean_diffs)
        else:
            raw_means = stable_softplus_inverse(window_means)
        
        # Reconstruct raw coefficients
        raw_coef = torch.zeros_like(constrained_seq)
        raw_coef[:n_windows] = raw_means
        
        if variations:
            variation_tensor = torch.stack(variations)
            raw_coef[n_windows:n_windows + len(variations)] = variation_tensor
        
        return raw_coef
    
    else:
        raise ValueError(f"Unknown mono_type: {mono_type}")


def generate_initial_curve(n_coef: int, mono: MonotoneFlag, noise_scale: float = 0.5, curve_scale: float = 1.0) -> torch.Tensor:
    """
    Generate initial curve values with appropriate trend for monotonicity.
    
    :param int n_coef: Number of coefficients of the curve
    :param int mono: Monotonic direction (-1, 0, 1)
    :param float noise_scale: Scale of random noise
    :param float curve_scale: Scale of the curve values, should be greater than 0
    
    :return: 1D tensor representing curve values
    :rtype: torch.Tensor
    
    Examples
    --------
        
    Increasing curve::
    
        >>> c_inc = generate_initial_curve(5, mono=1, noise_scale=0.0)
        >>> c_dec = generate_initial_curve(5, mono=-1, noise_scale=0.0)
        >>> bool(torch.all(c_inc[1:] >= c_inc[:-1]) and torch.all(c_dec[1:] <= c_dec[:-1]))
        True
    """
    assert 0 <= curve_scale, "curve_scale should be greater than 0"
    # Create base trend
    if mono == 1:  # Increasing
        base_trend = torch.linspace(0, 1, n_coef) * curve_scale
    elif mono == -1:  # Decreasing
        base_trend = torch.linspace(0, -1, n_coef) * curve_scale
    else:  # No constraint
        # base_trend = torch.zeros(n_coef)
        base_trend = torch.rand(n_coef) * curve_scale
    
    # Add controlled noise
    noise = (torch.rand(n_coef) - 0.5) * noise_scale * curve_scale / n_coef
    noise[0] = 0.0  # Keep the first point fixed to avoid large shifts
    
    return base_trend + noise


class MonoCoef(nn.Module):
    """Monotonic Coefficients for one spline."""

    def __init__(
        self,
        num: int = 5,
        k: int = 3,
        grid_range: tuple[float, float] | list[float] = (-1.0, 1.0),
        mono: MonotoneFlag = 0,
        mono_type: MonoType = "strict",
        elu_alpha: float = 0.01,
        win_size: int = 2,
        noise_scale: float = 0.05,
        init_method: InitMethods = "smart",
        taste_name: str = "",
        device: str | torch.device = "cpu",
        delta_floor: float = STRICT_DELTA_FLOOR,
        reverse_eps: float = 1e-8,
    ):
        """Monotonic Coefficients for one spline.

        :param int num: Number of grid points, defaults to 5
        :param int k: Spline order, defaults to 3
        :param list[float] grid_range: Range of the grid, defaults to [-1.0, 1.0]
        :param MONO_DIRS mono: Monotonic direction, defaults to 0
        :param MONO_TYPES mono_type: Monotonic type, defaults to "strict"
        :param float elu_alpha: Alpha for ELU activation, defaults to 0.01
        :param int win_size: Window size for window monotonic type, defaults to 2
        :param float noise_scale: Scale of random noise, defaults to 0.05
        :param INIT_METHODS init_method: Initialization method, defaults to "smart"
        :param str taste_name: Taste name, defaults to ""
        :param str | torch.device device: Device, defaults to "cpu"
        
        Example:
        -------
        Initialize a spline with 4 grid points, 2nd order, and increasing monotonicity::
        
            >>> mc = MonoCoef(num=4, k=2, grid_range=[-1, 1], mono=1, mono_type="strict", init_method="smart")
            >>> mc.grid
            tensor([[-1.0000, -1.0000, -1.0000, -0.5000,  0.0000,  0.5000,  1.0000,  1.5000,
                      2.0000]])
            >>> mc.coef_raw  # doctest: +SKIP
            tensor([ 0.0013, -0.0041,  0.4983,  0.5057,  0.9998,  0.9980],
                   requires_grad=True)
            >>> mc()  # doctest: +SKIP
            tensor([0.6938, 1.3849, 2.3579, 3.3355, 4.6487, 5.9605],
                   grad_fn=<CumsumBackward0>)
            >>> all(mc()[:-1] <= mc()[-1])  # increasing
            True
            
        """
        super().__init__()
        assert mono in [-1, 0, 1], f"mono must be -1, 0, or 1. Got {mono}."
        assert mono_type in ["strict", "soft", "window"], f"mono_type must be 'strict', 'soft', or 'window'. Got {mono_type}."
        assert init_method in InitMethods.__args__, f"init_method must be one of {InitMethods.__args__}. Got {init_method}."
        assert win_size < num + k, f"Window size must be less than the number of grid intervals plus spline order. Got {win_size} >= {num + k}."

        self.G: int = num                             # Grid size
        self.k: int = k                               # Spline order
        self.mono: MonotoneFlag = mono                   # monotonic direction
        self.mono_type: MonoType = mono_type        # monotonic type
        self.elu_alpha: float = elu_alpha             # alpha for ELU activation
        self.win_size: int = win_size                 # window size for window monotonic type
        self.noise_scale: float = noise_scale         # scale of random noise
        self.init_method: InitMethods = init_method  # initialization method
        self.taste_name: str = taste_name             # taste name, "VOTT", "VOHW", "COST"
        self.delta_floor: float = float(delta_floor)  # strict-mode increment floor (0 = increments may vanish)
        self.reverse_eps: float = float(reverse_eps)  # clamp inside the strict reverse (how dead a vanished basis gets)
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1).repeat(1, 1)
        grid = extend_grid(grid, k_extend=self.k, fix_left=True)
        self.grid: Tensor
        self.register_buffer("grid", grid)  # 2D tensor of size (1, G+2k+1)
        self.grid_version: int = 0  # bumped whenever `grid` changes (update_grid, state-dict load): basis caches key on it
        self.device: torch.device = torch.device(device)

        self.coef_raw = nn.Parameter(
            (torch.rand(self.G + self.k) - 1 / 2) * noise_scale / num
        )  # 1D tensor of size G+k
        if init_method == "smart":
            self._smart_initialization()
        elif init_method == "zero":
            self._zero_initialization()
        
        self.to(self.device)

    def _smart_initialization(self):
        """Smart initialization following the pattern from the original code."""
        # Generate initial curve values (more points for better curve representation)
        initial_curve = generate_initial_curve(self.G + 1, self.mono, self.noise_scale)

        # Convert curve values to B-spline coefficients
        initial_coef = curve2coef(
                self.grid[:, self.k : self.G+self.k+1].permute(1, 0), initial_curve[:, None, None], self.grid, self.k
            ).squeeze().float()  # (G+k, )
        
        if self.mono != 0:
            with torch.no_grad():
                temp_param = nn.Parameter(initial_coef)
                raw_coef = reverse_cumulative_monotonic(
                    temp_param,
                    mono=self.mono,
                    mono_type=self.mono_type,
                    elu_alpha=self.elu_alpha,
                    win_size=self.win_size,
                    delta_floor=self.delta_floor,
                    reverse_eps=self.reverse_eps,
                )
                self.coef_raw.data = raw_coef
        else:
            normalized_coef = normalize_constrained_coef(
                constrained=initial_coef,
                norm_type="maxabs",
                target=1., 
                eps=1e-8,
            )
            self.coef_raw.data = normalized_coef
            
    
    def _zero_initialization(self):
        """Zero initialization."""
        eps = 1e-6
        initial_curve = torch.zeros(self.G + 1) + eps * self.mono
        initial_curve[0] = 0.0  # ensure the first point is zero
        
        initial_coef = curve2coef(
                self.grid[:, self.k : self.G+self.k+1].permute(1, 0), initial_curve[:, None, None], self.grid, self.k
            ).squeeze().float()  # (G+k, )
        raw_coef = reverse_cumulative_monotonic(
                nn.Parameter(initial_coef),
                mono=self.mono,
                mono_type=self.mono_type,
                elu_alpha=self.elu_alpha,
                win_size=self.win_size,
            )
        self.coef_raw.data = raw_coef
        pass

    def forward(self) -> torch.Tensor:
        return apply_cumulative_monotonic(
            self.coef_raw,
            mono=self.mono,
            mono_type=self.mono_type,
            elu_alpha=self.elu_alpha if self.mono_type == "soft" else None,
            win_size=self.win_size if self.mono_type == "window" else None,
            delta_floor=self.delta_floor,
        )

    @torch.no_grad()
    def extra_repr(self) -> str:
        import numpy as np
        s = super().extra_repr()
        s += f"taste_name={self.taste_name}, mono={self.mono}"
        if self.mono != 0:
            s += f", mono_type={self.mono_type}"
            if self.mono_type == "window":
                s += f", win_size={self.win_size}"
            elif self.mono_type == "soft":
                s += f", elu_alpha={self.elu_alpha}"
        # s += f", noise_scale={self.noise_scale}, init_method={self.init_method}"
        s += f"\nG={self.G}, k={self.k}"
        np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=500)
        s += f"\ngrid {(*self.grid.shape,)}={self.grid.cpu().numpy()}"
        s += f"\ncoef_raw {(*self.coef_raw.shape,)}={self.coef_raw.cpu().numpy()}"
        coef_processed = apply_cumulative_monotonic(self.coef_raw, mono=self.mono, mono_type=self.mono_type, elu_alpha=self.elu_alpha,
                                                    win_size=self.win_size, delta_floor=self.delta_floor)
        s += f"\ncoef_processed {(*coef_processed.shape,)}={coef_processed.cpu().numpy()}"
        np.set_printoptions()  # reset print options
        return s

    def reverse(self, coef_constrained: Tensor) -> Tensor:
        """Reverse transform constrained coefficients to raw parameters.
        
        Example:
        --------

            >>> _ = torch.manual_seed(0)
            >>> mc = MonoCoef(num=6, k=2, mono=-1, mono_type="soft", init_method="smart", elu_alpha=0.01, win_size=3)
            >>> y = mc()
            >>> raw_rec = mc.reverse(y)
            >>> torch.allclose(raw_rec, mc.coef_raw.detach(), atol=1e-4)
            True
        """
        return reverse_cumulative_monotonic(
            coef_constrained,
            mono=self.mono,
            mono_type=self.mono_type,
            elu_alpha=self.elu_alpha if self.mono_type == "soft" else None,
            win_size=self.win_size if self.mono_type == "window" else None,
            delta_floor=self.delta_floor,
            reverse_eps=self.reverse_eps,
        )

    @torch.no_grad()
    def update_grid(
        self,
        x: Tensor,
        *,
        margin: float = 0.0,
        grid_eps: float = 0.02,
        clamp_left_to: float | None = 0.0,
        pin_right_core_to: float | None = None,
    ):
        """Update grid based on input data.

        Semantics:
            - Left side (clamp_left_to): if not None, the *left boundary* is open-clamped,
            i.e., the first (k+1) knots are set to this value.
            - Right side (pin_right_core_to): if not None, only the *last core knot*
            is pinned to this value; the k right extension knots remain strictly
            increasing (extrapolated using the last core span). If None, the last
            core knot is at least max(x)+margin.

        Args:
            x: Input data, shape (N, 1). Must be training data only.
            margin: Additive buffer for the uniform grid endpoints.
            grid_eps: Blend factor in [0,1] between uniform (1.0) and adaptive (0.0).
            clamp_left_to: If not None, force the left boundary to this value
                with multiplicity (k+1).
            pin_right_core_to: If not None, force the *last core knot* to this value
                (must be >= leftmost core). The right extension knots are NOT clamped;
                they are extrapolated to remain strictly increasing.

        Example:
            >>> mc = MonoCoef(num=4, k=2, grid_range=[-1, 1], mono=1, mono_type="strict", init_method="smart")
            >>> x = torch.cat([torch.randn(100).unsqueeze(1), torch.tensor([[2.8],[3.1],[2.9],[3.2]])], dim=0).clamp(min=-0.5, max=4)
            >>> mc.update_grid(x, clamp_left_to=0.0)
            >>> # left boundary is open-clamped at 0.0 (k+1 repeats), core[0] == 0.0
            >>> mc.update_grid(x, clamp_left_to=0.0, pin_right_core_to=4.0)
            >>> # core[-1] == 4.0, but right extension knots remain strictly increasing (not clamped)

        Notes:
            - The valid B-spline evaluation domain is [T[k], T[m-k]], where T is the extended grid
            (length m+1). With this policy, T[m-k] > core[-1] because the right side is extended.
            If you want to avoid endpoint artifacts in evaluation, clamp x_eval ≤ core[-1].
            - Coefficients are transported via curve2coef to preserve f(x) on the provided x.
        """
        assert x.dim() == 2 and x.size(1) == 1, f"x must be (N,1), got {tuple(x.shape)}"
        device = self.grid.device
        x = x.to(device)

        # === 1) Compute new (G+1) knot positions (core grid) ===
        N = x.size(0)
        x_sorted = torch.sort(x[:, 0])[0]  # (N,)
        # adaptive quantiles
        idx = torch.linspace(0, N - 1, self.G + 1, dtype=torch.long, device=device)
        grid_adaptive = x_sorted[idx]  # (G+1,)

        # uniform over [min-margin, max+margin]
        x_min = x_sorted[0] - margin
        x_max = x_sorted[-1] + margin
        grid_uniform = torch.linspace(x_min, x_max, steps=self.G + 1, device=device)

        # blend
        grid_new_core = grid_eps * grid_uniform + (1.0 - grid_eps) * grid_adaptive  # (G+1,)

        # ----- Left boundary policy: optional clamp of left boundary (k+1 repeats) -----
        if clamp_left_to is not None:
            grid_new_core = grid_new_core.clone()
            grid_new_core[0] = float(clamp_left_to)
            # keep nondecreasing order
            eps = 1e-8
            for g in range(1, self.G + 1):
                if grid_new_core[g] < grid_new_core[g - 1]:
                    grid_new_core[g] = grid_new_core[g - 1] + eps

        # ----- Right core policy: pin ONLY the last core knot, not the whole boundary -----
        if pin_right_core_to is not None:
            right_target = torch.as_tensor(float(pin_right_core_to), device=device)
            if right_target < grid_new_core[0]:
                raise ValueError(
                    f"pin_right_core_to ({float(right_target)}) must be >= left core ({float(grid_new_core[0])})."
                )
            grid_new_core[-1] = right_target
        else:
            # Ensure coverage: last core knot covers at least max(x)+margin
            if grid_new_core[-1] < x_max:
                grid_new_core[-1] = x_max

        # ===== 2) Extend to B-spline grid =====
        core_2d = grid_new_core.unsqueeze(0)  # (1, G+1)
        grid_new_ext = extend_grid(core_2d, k_extend=self.k, fix_left=True)

        # Left open-clamp (k+1 repeats) if requested
        if clamp_left_to is not None:
            grid_new_ext[:, : self.k + 1] = float(clamp_left_to)

        # ----- Right side: do NOT clamp the whole boundary -----
        # Only enforce the last core knot value; keep right extension strictly increasing by extrapolation.
        i_last_core = grid_new_ext.size(1) - (self.k + 1)  # index of the last core knot in the extended grid
        right_core = grid_new_core[-1]
        grid_new_ext[:, i_last_core] = right_core.clone()

        ## --- Option 1: Extrapolate right extension knots using the last core span Δ = core[-1] - core[-2] ---
        # delta = torch.clamp(grid_new_core[-1] - grid_new_core[-2], min=1e-8)
        
        ## --- Option 2: Extrapolate right extension knots using the median of the last 4 core spans ---
        tail_spans = torch.diff(grid_new_core[-min(4, self.G+1):])
        delta = torch.clamp(tail_spans.median(), min=1e-8)
        for j in range(1, self.k + 1):
            grid_new_ext[:, i_last_core + j] = right_core + delta * j

        # ===== 2b) Prepare target curve before swapping grids =====
        coef_constrained_old = self()  # (G+k,)
        coef_old = coef_constrained_old.unsqueeze(0).unsqueeze(0)  # (1,1,G+k)
        y_target = coef2curve(x, self.grid, coef=coef_old, k=self.k)  # (N,1,1)

        # ===== 3) Swap grids in-place =====
        self.grid.data.copy_(grid_new_ext)
        self.grid_version += 1

        # ===== 4) Transport coefficients to the new grid =====
        new_coef = curve2coef(
            x, y_target, self.grid, self.k, prefer_cpu_on_mps=True
        ).squeeze(0).squeeze(0)  # (G+k,)
        raw_coef = self.reverse(new_coef)  # (G+k,)
        self.coef_raw.data.copy_(raw_coef)

    
    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.grid_version += 1  # a checkpoint may carry another grid: invalidate any basis cache built on this one

    @torch.no_grad()
    def get_general_asu(self, n_points: int = 1000, x_eval: Tensor | None = None,
                        keep_first_c_zero: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get general Alternative-Specific Utility (ASU) curve for this MonoCoef (single attribute).
        With keep_first_c_zero=True, the first coefficient is anchored exactly as in
        SplineUtility.forward_coef (hard zero for k>0, c0-shift for k=0), so the general
        curve shares the mean/individual curves' left anchor f(x_left) = 0.
        Returns (x, y) with shapes (n_points,), (n_points,).
        """
        # 1) Get monotone-processed coefficients: (G+k,)
        coef = apply_cumulative_monotonic(
            self.coef_raw,
            mono=self.mono,
            mono_type=self.mono_type,
            elu_alpha=self.elu_alpha if self.mono_type == "soft" else None,
            win_size=self.win_size if self.mono_type == "window" else None,
            delta_floor=self.delta_floor,
        )  # shape: (G+k,)

        # 2) Anchor like the model's forward path (SplineUtility.forward_coef).
        #    The strict-mono increments carry a +0.01 floor, so c0 is never 0 on its
        #    own — without this the general curve misses the mean curves' anchor.
        if keep_first_c_zero:
            if self.k == 0:
                # degree-0 basis 0 spans the whole first interval: shift, don't zero
                coef = coef - coef[:1]
            else:
                coef = torch.cat([torch.zeros_like(coef[:1]), coef[1:]])

        # 4) Evaluate the curve on a dense grid
        grid_cpu = self.grid.squeeze(0).cpu()              # (G+2k+1,)
        effective = grid_cpu[self.k:self.G+self.k+1]       # (G,)
        if x_eval is not None:
            if x_eval.dim() == 1:
                x_eval = x_eval.unsqueeze(1)  # (n_points, 1)
            x = x_eval.cpu()                # (n_points, 1) — plotting path runs on cpu, matching grid below
        else:
            x = torch.linspace(effective[0], effective[-1], steps=n_points).unsqueeze(1)  # (n_points, 1)

        # coef2curve expects coef shaped as (1, 1, G+k); output (n_points, 1, 1)
        y = coef2curve(x, self.grid.cpu(), coef=coef.cpu().view(1, 1, -1), k=self.k)
        y = y.squeeze(1).squeeze(1)                      # (n_points,)
        x = x.squeeze(1)                                 # (n_points,)

        return x, y
