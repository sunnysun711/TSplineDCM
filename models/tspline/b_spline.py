# tspline/core/b_spline.py
import torch



def B_batch(x: torch.Tensor, grid: torch.Tensor, k: int = 0) -> torch.Tensor:
    """
    Evaluate x on B-spline bases.

    :param torch.Tensor x: inputs, shape (B, n_splines) or (B,) treated as (B, 1)
    :param torch.Tensor grid: grids, shape (n_splines, G+1) or (G+1,) treated as (1, G+1)
    :param int k: the piecewise polynomial order of splines, defaults to 0

    :return: spline values, shape (B, n_splines, G-k).
        B is batch size, G is the number of grid intervals, k is the spline order.
        If x is (B,), grid is (G+1,), then return (B, G-k).
    :rtype: torch.Tensor

    Example::

        >>> m=100; n=5; k=3; G=20
        >>> x = torch.rand(m, n)  # (m, n)
        >>> grid = torch.linspace(-1,1,steps=G+1).repeat(n, 1)  # (n, G+1)
        >>> B_batch(x, grid, k=3).shape  # (m, n, G-k)
        torch.Size([100, 5, 17])
        
        >>> x = torch.rand(m,)  # (m, )
        >>> grid = torch.linspace(-1, 1, steps=G+1)  # (G+1, )
        >>> B_batch(x, grid, k=3).shape  # m, G-k
        torch.Size([100, 17])
    """

    squeeze_later = False
    if x.dim() == 1:
        x = x.unsqueeze(1)      # (m,1)
        squeeze_later = True
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)  # (1, G+1)
    
    x = x.unsqueeze(dim=2)        # (m, n, 1)
    grid = grid.unsqueeze(dim=0)  # (1, n, G+1)

    # if k == 0:
    #     value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    # else:
    #     B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)

    #     value = (x - grid[:, :, : -(k + 1)]) / (
    #         grid[:, :, k:-1] - grid[:, :, : -(k + 1)]
    #     ) * B_km1[:, :, :-1] + (grid[:, :, k + 1 :] - x) / (
    #         grid[:, :, k + 1 :] - grid[:, :, 1:(-k)]
    #     ) * B_km1[
    #         :, :, 1:
    #     ]
    
    if k == 0:
        # zero-degree: left-closed, right-open on each interval [t_i, t_{i+1}),
        # and ONLY here include the very last knot x==t_m in the last basis.
        value = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
        right_eq = (x == grid[:, :, [-1]]).squeeze(-1)
        value[..., -1] = torch.where(right_eq, torch.ones_like(value[..., -1]), value[..., -1])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        denom1 = (grid[:, :, k:-1] - grid[:, :, :-(k + 1)])
        w1 = torch.where(denom1 != 0,
                        (x - grid[:, :, :-(k + 1)]) / denom1,
                        torch.zeros_like(denom1))
        denom2 = (grid[:, :, k + 1:] - grid[:, :, 1:(-k)])
        w2 = torch.where(denom2 != 0,
                        (grid[:, :, k + 1:] - x) / denom2,
                        torch.zeros_like(denom2))
        value = w1 * B_km1[:, :, :-1] + w2 * B_km1[:, :, 1:]


    # include the very last knot for any degree: put x==t_m into the last basis
    right_eq_all = (x == grid[:, :, [-1]]).squeeze(-1)
    value[..., -1] = torch.where(right_eq_all, torch.ones_like(value[..., -1]), value[..., -1])

    # in case grid is degenerate
    value = torch.nan_to_num(value)
    if squeeze_later:
        value = value.squeeze(dim=1)  # reduce to (m, G-k)
    return value


def coef2curve(
    x_eval: torch.Tensor, extended_grid: torch.Tensor, coef: torch.Tensor, k: int = 0
) -> torch.Tensor:
    """
    Converting B-spline coefficients to B-spline curves.
    Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    :param torch.Tensor x_eval: 2D torch.tensor shape (batch, in_dim)
    :param torch.Tensor extended_grid: 2D torch.tensor shape (in_dim, G+2k+1). G is num grid intervals; k is spline order.
    :param torch.Tensor coef: 4D torch.tensor shape (batch, in_dim, out_dim, G+k) if per-sample coefficients are used, otherwise 3D torch.tensor shape (in_dim, out_dim, G+k)
    :param int k: the piecewise polynomial order of splines, defaults to 0

    :return: y_eval, shape (batch, in_dim, out_dim)
    :rtype: torch.Tensor
    """

    b_splines = basis_matrix(x_eval, extended_grid, k=k)
    return curve_from_basis(b_splines, coef)


def basis_matrix(x_eval: torch.Tensor, extended_grid: torch.Tensor, k: int = 0) -> torch.Tensor:
    """The basis half of coef2curve: B_batch in float32, shape (batch, in_dim, G+k).
    Row i depends only on x_eval[i] and the grid, so it can be computed once per grid for every
    dataset row and gathered by row index (models/tspline/basis_cache.py)."""
    return B_batch(x_eval, extended_grid, k=k).float()


def curve_from_basis(b_splines: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """The contraction half of coef2curve (kept byte-for-byte: a cached basis gives the same output)."""
    if coef.dim() == 4:
        # Per-sample coefficients: coef shape (batch, in_dim, out_dim, G+k)
        y_eval = torch.einsum("ijk,ijlk->ijl", b_splines, coef.to(b_splines.device))
    else:
        # Shared coefficients: coef shape (in_dim, out_dim, G+k)
        y_eval = torch.einsum("ijk,jlk->ijl", b_splines.to(coef.dtype), coef.to(b_splines.device))

    return y_eval

def _solve_on_cpu(A: torch.Tensor,
                  b: torch.Tensor,
                  target_device: torch.device,
                  ridge_lambda: float = 1e-6) -> torch.Tensor:
    """CPU solve: try SVD lstsq, then ridge-normal-equations."""
    A_cpu = A.detach().to("cpu", dtype=torch.float32)
    b_cpu = b.detach().to("cpu", dtype=torch.float32)
    try:
        sol_cpu = torch.linalg.lstsq(A_cpu, b_cpu, driver="gelsd").solution  # (..., n_coef, 1)
        coef = sol_cpu[..., 0]
        return coef.to(target_device)
    except Exception:
        AT = A_cpu.transpose(-1, -2)                 # (..., n_coef, batch)
        ATA = AT @ A_cpu                              # (..., n_coef, n_coef)
        ATb = AT @ b_cpu                              # (..., n_coef, 1)
        I = torch.eye(ATA.shape[-1], device="cpu").expand_as(ATA)
        coef_cpu = torch.linalg.solve(ATA + ridge_lambda * I, ATb)[..., 0]
        return coef_cpu.to(target_device)


def _ill_conditioned(mat64: torch.Tensor, max_cond: float = 1e8) -> bool:
    m, n = mat64.shape[-2], mat64.shape[-1]
    if m < n:
        return True
    view = mat64.reshape(-1, m, n)
    for idx in torch.linspace(0, view.shape[0]-1, steps=min(3, view.shape[0])).round().long():
        s = torch.linalg.svdvals(view[idx])
        if s.numel() == 0 or s[-1] <= 0 or float(s[0]/s[-1]) > max_cond:
            return True
    return False

def _ridge(mat64: torch.Tensor, rhs64: torch.Tensor, lam: float = 1e-6) -> torch.Tensor:
    AT = mat64.transpose(-1, -2)
    ATA = AT @ mat64
    ATy = AT @ rhs64
    I = torch.eye(ATA.shape[-1], device=ATA.device, dtype=ATA.dtype).expand_as(ATA)
    A = ATA + lam * I
    try:
        L = torch.linalg.cholesky(A)
        sol = torch.cholesky_solve(ATy, L)
    except Exception:
        sol = torch.linalg.solve(A, ATy)
    return sol[..., 0]


def curve2coef(
    x_eval: torch.Tensor, 
    y_eval: torch.Tensor, 
    extended_grid: torch.Tensor, 
    k: int = 0,
    prefer_cpu_on_mps: bool = False,
    ridge_lambda: float = 1e-6,
    fix_left: bool | None = None,
) -> torch.Tensor:
    """
    Converting B-spline curves to B-spline coefficients using least squares.
    
    Auto-detect clamped left endpoint and decide whether to fix c0.

    :param torch.Tensor x_eval: 2D torch.tensor shape (batch, in_dim)
    :param torch.Tensor y_eval: 3D torch.tensor shape (batch, in_dim, out_dim)
    :param torch.Tensor extended_grid: 2D torch.tensor shape (in_dim, G+2k+1)
    :param int k: the piecewise polynomial order of splines, defaults to 0
    :param bool prefer_cpu_on_mps: whether to prefer CPU solve on MPS device, defaults to False
    :param float ridge_lambda: ridge regression lambda, defaults to 1e-6
    :param bool | None fix_left: whether to fix the left endpoint coefficient to 0, defaults to None

    :raise RuntimeError: if least squares fails

    :return: coef, shape (in_dim, out_dim, G+k)
    :rtype: torch.Tensor

    Example::

        >>> m=100; n=5; k=3; G=10; out_dim=2
        >>> x_eval = torch.rand(m, n)
        >>> y_eval = torch.rand(m, n, out_dim)
        >>> grid = torch.linspace(-1, 1, steps=G+1).repeat(n, 1)
        >>> extended_grid = extend_grid(grid, k_extend=k)  # (n, G+2k+1)
        >>> coef = curve2coef(x_eval, y_eval, extended_grid, k=k)
        >>> coef.shape  # in_dim, out_dim, G+k
        torch.Size([5, 2, 13])
    """
    # print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = extended_grid.shape[1] - k - 1
    mat = B_batch(x_eval, extended_grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    # print('mat', mat.shape)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    # print('y_eval', y_eval.shape)
    device = mat.device
    
    # ---- Auto-detect clamped left endpoint and decide whether to fix c0 ----
    # If fix_left is explicitly given, it wins; otherwise detect:
    if fix_left is None:
        if k > 0:
            same_left = extended_grid[:, : (k + 1)].eq(extended_grid[:, [0]]).all(dim=1)
            fix_left_eff = bool(same_left.all().item())
        else:
            fix_left_eff = False
    else:
        fix_left_eff = bool(fix_left)


    # --- Enforce c0 = 0 by removing the first column if needed ---
    if fix_left_eff:
        mat_used = mat[..., 1:]     # (..., batch, n_coef-1)
        n_coef_used = n_coef - 1
    else:
        mat_used = mat
        n_coef_used = n_coef
    
    # If tensors live on MPS or CUDA, do a silent CPU solve to avoid MPS/CUDA fallback warning.
    if device.type in ["mps", "cuda"] and prefer_cpu_on_mps:
        coef_used = _solve_on_cpu(mat_used, y_eval, target_device=device, ridge_lambda=ridge_lambda)
    else:
        mat64 = mat_used.to(dtype=torch.float64)
        rhs64 = y_eval.to(dtype=torch.float64)
        need_ridge = (batch < n_coef_used) or _ill_conditioned(mat64)
        if need_ridge:
            coef_used64 = _ridge(mat64, rhs64, ridge_lambda)
        else:
            try:
                sol = torch.linalg.lstsq(mat64, rhs64).solution   # (..., n_coef_used, 1)
                coef_used64 = sol[..., 0]
                if not torch.isfinite(coef_used64).all():
                    coef_used64 = _ridge(mat64, rhs64, lam=ridge_lambda)
            except (RuntimeError, NotImplementedError):
                coef_used64 = _ridge(mat64, rhs64, lam=ridge_lambda)
        coef_used = coef_used64.to(dtype=mat.dtype)
        
    # Stitch back the fixed zero coefficient if we dropped it
    if fix_left_eff:
        zeros = torch.zeros(in_dim, out_dim, 1, device=coef_used.device, dtype=coef_used.dtype)
        coef = torch.cat([zeros, coef_used], dim=-1)   # (in_dim, out_dim, n_coef)
    else:
        coef = coef_used

    return coef.to(dtype=y_eval.dtype)

def extend_grid(grid: torch.Tensor, k_extend: int = 0, fix_left:bool=False, fix_right:bool=False) -> torch.Tensor:
    """
    Extend grid by k points on both ends

    :param torch.Tensor grid: 2D torch.tensor, shape (in_dim, G+1), where G is the number of grid intervals
    :param int k_extend: number of points to extend on both ends, defaults to 0
    :param bool fix_left: whether to fix the left endpoint, defaults to False
    :param bool fix_right: whether to fix the right endpoint, defaults to False

    :return: extended grid, shape (in_dim, G+2k+1)
    :rtype: torch.Tensor

    Examples
    --------

    >>> G=10; k=3; in_dim=2;  # number of grid intervals, spline order, input dimension (number of splines)
    >>> grid = torch.linspace(-1, 1, steps=G+1).repeat(in_dim, 1)
    >>> extended_grid = extend_grid(grid, k_extend=k)
    >>> extended_grid.shape  # in_dim, G+2k+1
    torch.Size([2, 17])
    
    >>> G, k, in_dim = 5,2,1
    >>> grid = torch.linspace(-1, 1, steps=G+1).repeat(in_dim, 1)
    >>> extended_grid = extend_grid(grid, k_extend=k, fix_left=True, fix_right=True)
    >>> extended_grid.shape  # in_dim, G+2k+1
    torch.Size([1, 10])
    >>> grid
    tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
    >>> extended_grid
    tensor([[-1.0000, -1.0000, -1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000,
              1.0000,  1.0000]])
    >>> extend_grid(grid, k_extend=k, fix_left=True, fix_right=False)
    tensor([[-1.0000, -1.0000, -1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000,
              1.4000,  1.8000]])
        
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        if fix_left:
            grid = torch.cat([grid[:, [0]], grid], dim=1)
        else:
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        if fix_right:
            grid = torch.cat([grid, grid[:, [-1]]], dim=1)
        else:
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid


if __name__ == "__main__":
    
    import doctest
    doctest.testmod(verbose=True)
    pass