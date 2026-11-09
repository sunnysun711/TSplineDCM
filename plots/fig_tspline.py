# plots/fig_tspline.py — illustration: one TSpline AS-UCF, its individual variants for several tau, the marginal
# utility of the same curves, the k_g profile, and the basis the increments act on.
"""Paper figure tspline.pdf, redrawn for the I-spline model. Pure numpy; nothing to extract.
Top:    general curve (tau = 0) + individual curves for TAU_LEVELS, knots.
Middle: marginal utility df/dx of the same curves; the taste index multiplies the slope locally by (1 + k_g tau),
        so the fan opens exactly where k_g is large. k_g is drawn here at the Greville abscissae (right axis).
Bottom: the basis, chosen by BASIS_PANEL:
        "M"  M-splines M_{g,l} = l / (t_{g+l} - t_g) B_{g,l-1}: the basis of the increments a_g in the MARGINAL
             utility, f'(x) = sum_g a_g M_{g,l}(x). Localised (support [t_g, t_{g+l})); the fan of the middle panel
             at x is sum_g M_g(x) a_g k_g tau, so this is the panel that matches the middle one. Default.
        "I"  I-splines I_{g,l} = sum_{j >= g} B_{j,l}: the basis of the increments in the utility itself,
             f(x) = sum_g a_g I_{g,l}(x); cumulative, from 0 to 1.
        "B"  B-splines B_{g,l}: the basis of the coefficient LEVELS c_g, which the increments are differences of
             (the old look of this figure; it is the basis the code evaluates, not the one k_g acts on).
        The colour of basis g is the colour of the k_g dot with the same index. Index 0 is the anchor (c_0 = 0 is
        fixed, so B_0 / I_0 contribute nothing and M_0 has an empty support): neither its basis nor its dot is drawn.
        The leftmost basis drawn is index 1, the FIRST INCREMENT a_1 = c_1, which sets the slope at x_min and is free. The dots stay at the Greville
        abscissae of the degree-l B-splines, the convention of every data figure in the paper.
SHOW_MU = False gives the old two-panel layout (k_g on the top panel's right axis).

HETER_TARGET = "increment" (the paper's model): individual increments a_g * (1 + k_g * tau), re-accumulated ->
every individual curve is monotone whenever the general one is.
HETER_TARGET = "level" (legacy B-spline ablation): coefficients c_g * (1 + k_g * tau) -> can break monotonicity.

Edit the parameter block, then:   python plots/fig_tspline.py
"""
from __future__ import annotations

# ============================== parameters ==================================
HETER_TARGET = "increment"          # "increment" | "level"
DEGREE = 3
KNOTS = [0.0, 0.15, 0.20, 0.35, 0.40, 0.50, 0.80, 0.90, 1.0]   # core knots (G+1 values); None = G uniform
G = 8
FIX_LEFT, FIX_RIGHT = True, False
K_G = [0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.05, 0.02, 0.1, 0.3, 0.4]  # one per basis (G + DEGREE); None = auto bump
TAU_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]
COEF_ALPHA = 0.15                   # general coefficients c_g = -exp(alpha * g) / max, c_0 = 0
SHOW_MU = True                      # middle panel with the marginal utility df/dx (k_g dots move there)
MU_YLIM = (-2.2, 0.15)              # y-range of the middle panel; None = auto (the steep start then dominates)
BASIS_PANEL = "M"                   # "M" (derivative basis, matches the middle panel) | "I" | "B"
BASIS_SCALE = "exact"                # "unit" = every basis scaled to maximum 1 | "exact"
N_EVAL = 600
FIG_SIZE = (6.0, 4.4)               # (6.0, 3.0) for the two-panel layout
HEIGHT_RATIOS = [0.42, 0.33, 0.25]  # top / middle / bottom (middle ignored when SHOW_MU = False)
OUT_DIR = None                      # None -> figures/illustration/
FMT = "pdf"
FONT_SIZE = 8
# ============================================================================

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bspline_np import b_batch, coef2curve, extend_grid, greville  # noqa: E402
from style import latexify  # noqa: E402

OUT = Path(OUT_DIR) if OUT_DIR else HERE.parent / "figures" / "illustration"
TAU_COLORS = ["#3EBCD3", "#3490DB", "#0961DF", "#009D70", "#A3D0B3", "#E39B4F", "#C44E52"]
BASIS_COLORS = ["#43BC97", "#57B1A9", "#35A155", "#FBE5B3", "#F2D086", "#E2CD89", "#C3BC6C", "#B1C07A", "#95BE7E",
                "#7CBC8A", "#8C8C8C"]


def individual_coef(c, k_g, tau):
    if HETER_TARGET == "level":
        return (1.0 + k_g * tau) * c
    delta = np.diff(c, prepend=0.0)
    return np.cumsum(delta * (1.0 + k_g * tau))


def m_basis(x, ext, k):
    """M-splines on the extended knots: M_{g,k} = k / (t_{g+k} - t_g) * B_{g,k-1}; one column per basis index g
    (the same indexing as the degree-k B-splines and the coefficients, with one extra column at the end)."""
    Bm = b_batch(x, ext, k - 1)
    width = ext[k:k + Bm.shape[1]] - ext[:Bm.shape[1]]
    scale = np.where(width > 0, k / np.where(width > 0, width, 1.0), 0.0)
    return Bm * scale[None, :]


def i_basis(x, ext, k):
    """I-splines I_{g,k} = sum_{j >= g} B_{j,k} (partial sums from the right)."""
    B = b_batch(x, ext, k)
    return np.cumsum(B[:, ::-1], axis=1)[:, ::-1]


def _draw_k(ax_k, xi, k_g, n_coef, first=1):
    """k_g dots at the Greville abscissae on a right-hand axis of ax_k."""
    ax2 = ax_k.twinx()
    for i in range(first, n_coef - 1):
        ax2.scatter(xi[i], k_g[i], s=28, color=BASIS_COLORS[i % len(BASIS_COLORS)], zorder=4,
                    label=r"$k_{gpi}$ at Greville abscissae" if i == n_coef - 2 else None)
    ax2.set_ylabel(r"Heter. level ($k_{gpi}$)")
    ax2.set_ylim(-0.1, 1.1)
    return ax2


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    core = np.asarray(KNOTS, dtype=float) if KNOTS is not None else np.linspace(0.0, 1.0, G + 1)
    k = DEGREE
    ext = extend_grid(core, k, fix_left=FIX_LEFT, fix_right=FIX_RIGHT)
    n_coef = ext.size - k - 1
    c = -np.exp(COEF_ALPHA * np.arange(n_coef))
    c = c / np.abs(c).max()
    c[0] = 0.0
    if K_G is None:
        k_g = np.zeros(n_coef)
        band = max(1, n_coef // 3)
        start = max(0, n_coef // 2 - band // 2)
        k_g[start: start + band] = np.linspace(0.2, 0.8, band)
    else:
        k_g = np.asarray(K_G, dtype=float)
        assert k_g.size == n_coef, f"K_G needs {n_coef} values (G + degree), got {k_g.size}"
    x = np.linspace(ext[0], core[-1], N_EVAL)
    xi = greville(ext, k)

    # the derivative identity the middle and bottom panels rely on: f'(x) = sum_g a_g M_{g,k}(x)
    a = np.diff(c, prepend=0.0)
    M = m_basis(x, ext, k)
    f_prime = M[:, :n_coef] @ a
    err = np.nanmax(np.abs(f_prime[5:-5] - np.gradient(coef2curve(x, ext, c, k), x)[5:-5]))
    assert err < 0.05, f"derivative identity broken: max |sum a_g M_g - df/dx| = {err:.3g}"

    if SHOW_MU:
        fig, (ax, ax_mu, ax_b) = plt.subplots(3, 1, sharex=True, figsize=FIG_SIZE, height_ratios=HEIGHT_RATIOS)
    else:
        fig, (ax, ax_b) = plt.subplots(2, 1, sharex=True, figsize=FIG_SIZE,
                                       height_ratios=[HEIGHT_RATIOS[0], HEIGHT_RATIOS[2]])
        ax_mu = None

    # ---- top: AS-UCF ----
    y_gen = coef2curve(x, ext, c, k)
    ax.plot(x, y_gen, color="#E6F2F1", linewidth=6, label="general")
    curves = []
    for i, t in enumerate(TAU_LEVELS):
        y = coef2curve(x, ext, individual_coef(c, k_g, t), k)
        curves.append((t, y, TAU_COLORS[i % len(TAU_COLORS)]))
        ax.plot(x, y, color=curves[-1][2], linewidth=1.0, label=r"$\tau_{pin}$=" + f"{t:+.1f}")
    for gx in core[:-1]:
        ax.axvline(gx, linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axvline(core[-1], linestyle=":", linewidth=0.6, alpha=0.4, label="knot")
    ax.set_ylabel(r"AS-UCF $f_{pin}(x)$")
    which = "I-spline increments" if HETER_TARGET == "increment" else "B-spline levels"
    ax.set_title(fr"TSpline, $l$={k}, heterogeneity on {which} (left {'fixed' if FIX_LEFT else 'extended'}, "
                 fr"right {'fixed' if FIX_RIGHT else 'extended'}, knots {'non-uniform' if KNOTS else 'uniform'})")

    # ---- middle: marginal utility df/dx (numerical derivative on the dense grid) ----
    if ax_mu is not None:
        ax_mu.plot(x, np.gradient(y_gen, x), color="#E6F2F1", linewidth=6)
        for t, y, col in curves:
            ax_mu.plot(x, np.gradient(y, x), color=col, linewidth=1.0)
        ax_mu.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        for gx in core:
            ax_mu.axvline(gx, linestyle=":", linewidth=0.6, alpha=0.4)
        ax_mu.set_ylabel(r"Marginal utility $\partial f_{pin}/\partial x$")
        if MU_YLIM is not None:
            ax_mu.set_ylim(*MU_YLIM)
        ax_k = _draw_k(ax_mu, xi, k_g, n_coef)
    else:
        ax_k = _draw_k(ax, xi, k_g, n_coef)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_k.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=3, frameon=False, loc="lower left", fontsize=FONT_SIZE)

    # ---- bottom: the basis the increments act on (M), the utility basis of the increments (I), or the levels (B)
    if BASIS_PANEL == "M":
        Bp, first, ylab = M, 1, r"$M_{g,l}(x)$"          # a_0 = 0 (anchor): no M_0 term
    elif BASIS_PANEL == "I":
        Bp, first, ylab = i_basis(x, ext, k), 1, r"$I_{g,l}(x)$"
    elif BASIS_PANEL == "B":
        Bp, first, ylab = b_batch(x, ext, k), 1, r"$B_{g,l}(x)$"       # B_0 carries c_0 = 0 (anchor): not drawn
    else:
        raise SystemExit(f"BASIS_PANEL must be M, I or B, got {BASIS_PANEL!r}")
    for j in range(first, n_coef - 1):
        col = BASIS_COLORS[j % len(BASIS_COLORS)]
        yb = Bp[:, j]
        if BASIS_SCALE == "unit" and np.nanmax(yb) > 0:
            yb = yb / np.nanmax(yb)
        ax_b.plot(x, yb, color=col, linewidth=0.5)
        ax_b.fill_between(x, 0, yb, color=col, alpha=0.1)
    for gx in core:
        ax_b.axvline(gx, linestyle=":", linewidth=0.6, alpha=0.4)
    ax_b.set_xlabel(r"$x$")
    ax_b.set_ylabel("Basis " + ylab + ("" if BASIS_SCALE == "exact" else " (unit max)"))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"tspline_{HETER_TARGET}.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_tspline] {p}  (basis panel {BASIS_PANEL}, derivative identity error {err:.2e})")


if __name__ == "__main__":
    main()
