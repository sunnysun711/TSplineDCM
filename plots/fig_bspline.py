# plots/fig_bspline.py — illustration: B-spline fits of one curve for degrees 0/1/2 and a fixed-left non-uniform knot case.
"""Paper figure bspline.pdf. Pure numpy (plots/_bspline_np.py); nothing to extract.
Top row: least-squares B-spline fit vs the true curve, with the core knots; bottom row: the basis functions.

Edit the parameter block, then:   python plots/fig_bspline.py
"""
from __future__ import annotations

# ============================== parameters ==================================
DEGREES = [0, 1, 2, 2]
FIX_LEFT = [False, False, False, True]
KNOTS = [None, None, None, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]]   # None = G uniform intervals on [X_LEFT, X_RIGHT]
G = 5
X_LEFT, X_RIGHT = 0.0, 1.0
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
N_FIT, N_PLOT = 400, 600
FIG_SIZE = (9, 3.3)
OUT_DIR = None                      # None -> figures/illustration/
FMT = "pdf"
FONT_SIZE = 8
# ============================================================================

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bspline_np import b_batch, coef2curve, curve2coef, extend_grid  # noqa: E402
from style import latexify  # noqa: E402

OUT = Path(OUT_DIR) if OUT_DIR else HERE.parent / "figures" / "illustration"
BASIS_COLORS = ["#43BC97", "#57B1A9", "#FBE5B3", "#F2D086", "#C3BC6C", "#B1C07A", "#95BE7E", "#7CBC8A",
                "#35A155", "#E2CD89"]


def true_function(x):
    return 1.2 * x**3 - 2.3 * x**2 + 0.5 * x + 0.1 * np.sin(4 * np.pi * x)


def panel(ax_top, ax_bot, k, core, fix_left, label):
    ext = extend_grid(core, k, fix_left=fix_left, fix_right=False)
    x_fit = np.linspace(core[0], core[-1], N_FIT)
    coef = curve2coef(x_fit, true_function(x_fit), ext, k, ridge=1e-8)
    x = np.linspace(core[0], core[-1], N_PLOT)
    ax_top.plot(x, coef2curve(x, ext, coef, k), linewidth=1, color="#0E5FBB", label=rf"B-spline fit ($l=${k})")
    ax_top.plot(x, true_function(x), linewidth=1, linestyle="--", color="#DD3125", alpha=0.9, label="True")
    for gx in core[:-1]:
        ax_top.axvline(gx, linestyle=":", linewidth=0.7, alpha=0.5)
    ax_top.axvline(core[-1], linestyle=":", linewidth=0.7, alpha=0.5, label="Knots")
    ax_top.set_ylabel(r"$f(x)$")
    ax_top.legend(frameon=True, loc="lower left")
    B = b_batch(x, ext, k)
    for j in range(B.shape[1]):
        c = BASIS_COLORS[j % len(BASIS_COLORS)]
        ax_bot.plot(x, B[:, j], linewidth=0.5, color=c)
        ax_bot.fill_between(x, 0, B[:, j], color=c, alpha=0.1)
    for gx in core:
        ax_bot.axvline(gx, linestyle=":", linewidth=0.7, alpha=0.5)
    kind = {0: "PW constant", 1: "PW linear", 2: "PW quadratic"}.get(k, "cubic")
    extra = " (fix left, adjust knots)" if fix_left else f" ({kind})"
    ax_bot.set_xlabel(f"{label} degree $l$ = {k}{extra}")
    ax_bot.set_ylabel(r"Basis func. $B_{g,l}(x)$")


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    n = len(DEGREES)
    fig, axes = plt.subplots(2, n, figsize=FIG_SIZE, sharex="col", sharey="row", height_ratios=[1.0, 0.5])
    for i in range(n):
        core = np.asarray(KNOTS[i], dtype=float) if KNOTS[i] is not None else np.linspace(X_LEFT, X_RIGHT, G + 1)
        panel(axes[0, i], axes[1, i], DEGREES[i], core, FIX_LEFT[i], PANEL_LABELS[i])
    for ax in axes.ravel():
        ax.label_outer()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"bspline.{FMT}"
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig_bspline] {p}")


if __name__ == "__main__":
    main()
