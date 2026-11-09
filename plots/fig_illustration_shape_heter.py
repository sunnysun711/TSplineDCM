# plots/fig_illustration_shape_heter.py — Figure 1 (a)/(b): the SHAPE of an AS-UCF per ML-enhanced model, then its HETEROGENEITY.
"""Palette: red/pink truth, blue TSplineDCM, orange TasteNet,
green RUMBoost, grey DNN) plus teal for DCM-LN. Truth and TSplineDCM come from the paper's own model (I-spline increments
x (1 + k tau)), so their individual curves are monotone by construction. Pure numpy; nothing to extract.

The comparison is among ML-enhanced DCMs only (2026-11-02: the mixed logit and MNL panels were dropped; the paper's core
contrast is with the ML-enhanced models, the mixed logit is handled in the text and in sec:limitations).

Outputs (figures/illustration/):
  shape.pdf       (Fig. 1a) one population curve per model against a truth with a threshold (decline, plateau, steep
                  drop, plateau): TasteNet linear (least-squares fit, i.e. the specified form at its best), RUMBoost
                  steps, DCM-LN piecewise-linear, ASU-DNN probe (turns up at the right end: not monotone), TSplineDCM smooth.
                  TasteNet, DCM-LN and ASU-DNN are dashed and listed last in the legend: their population curve is only
                  a post-hoc average over individuals (TasteNet averages beta(z_n), the other two average whole curves).
  heter_grid.pdf  (Fig. 1b) four panels in a 2 x 2 grid, one model each, every one with the true range of individuals
                  as the yardstick: TasteNet (lines at a few z), DCM-LN, ASU-DNN, TSplineDCM (population curve +
                  monotone fan). Every individual curve is dashed, every population curve solid. The two files are
                  placed side by side in the paper (subfloats, .48 textwidth each),
                  so they share FIG_SIZE and FONT_SIZE.

Edit the parameter block, then:   python plots/fig_illustration_shape_heter.py
"""
from __future__ import annotations

# ============================== parameters ==================================
RED, BLUE, ORANGE, GREEN, GREY, TEAL = "#DE3026", "#0A61DE", "#F08229", "#26AD61", "#737373", "#17BECF"
PINK_BAND, BLUE_BAND = "#E84294", "#0A85E3"
BAND_ALPHA = 0.25
LW_CURVE, LW_AXIS, LW_THIN, LW_MODEL = 2.0, 1.0, 1.1, 1.3   # LW_MODEL: every model line in (a) (the truth keeps LW_CURVE)
LS_POSTHOC = "--"                            # (a) models whose population curve is only a post-hoc average over individuals (TasteNet: beta(z_n) per person; DCM-LN, ASU-DNN: a curve per person)
LS_INDIV = "--"                              # (b) every individual curve; population curves are solid
# truth: a decreasing I-spline with a threshold — moderate decline, plateau, steep drop, plateau (like the synthetic
# CAR curve); k = where tastes diverge, peaked on the drop
KNOTS = [0.0, 0.08, 0.16, 0.26, 0.38, 0.52, 0.68, 0.84, 1.0]
INCR_TRUE = [0.0, 0.20, 0.12, 0.03, 0.02, 0.30, 0.26, 0.04, 0.02, 0.02, 0.02]
K_TRUE = [0.0, 0.02, 0.02, 0.03, 0.10, 0.34, 0.30, 0.08, 0.02, 0.01, 0.01]
INCR_FIT_JITTER = [0.0, -0.15, 0.30, 0.40, 0.30, -0.12, 0.10, 0.30, -0.20, 0.20, 0.30]   # TSplineDCM = truth x (1 + jitter)
K_FIT_JITTER = [0.0, 0.01, 0.0, 0.02, 0.03, -0.05, 0.04, 0.02, 0.0, 0.0, 0.0]
DEGREE = 3
TAU_INDIV = [-0.7, 0.7]                      # thin individual TSplineDCM curves drawn inside the band
# (a) shapes
N_STEPS = 7                                  # RUMBoost pieces
LN_SEGMENTS = 6                              # DCM-LN: uniform lattice breakpoints on [0, 1]
LN_JITTER = [0.0, 0.10, -0.05, 0.08, -0.04, 0.06, 0.0]   # relative error of the lattice values at the breakpoints
DNN_UPTURN = (0.85, 0.04, 0.16)              # DNN probe = truth + a logistic upturn (centre, width, height x total drop): not monotone
# (b) heterogeneity
TASTENET_SLOPES = [0.3, 0.6, 1.0, 1.5]       # TasteNet: beta(z) for a few values of the characteristics z -> individual lines
LN_INDIV_SCALES = [0.6, 1.0, 1.25]           # DCM-LN individual curves: the lattice curve rescaled
DNN_SEED, DNN_START, DNN_WIGGLE = 7, [0.06, -0.06, 0.12], 0.13   # DNN individual curves: wiggly, not monotone
PANEL_SUBTITLES = {"TasteNet": "linear", "DCM-LN": "piecewise linear", "ASU-DNN": "unconstrained", "TSplineDCM": "smooth"}
FIG_SIZE = (3.4, 3.1)                        # (a) and (b) share the size: side by side in the paper at .48 textwidth each
OUT_DIR = None
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _bspline_np import b_batch, extend_grid  # noqa: E402
from style import latexify  # noqa: E402

OUT = Path(OUT_DIR) if OUT_DIR else HERE.parent / "figures" / "illustration"


def ispline_curve(x, ext, incr, k):
    return -(b_batch(x, ext, k) @ np.cumsum(incr))


def individual(incr, kg, tau):
    return incr * (1.0 + kg * tau)


class World:
    """The truth, the TSplineDCM fit and the competitors' curves, all on x in [0, 1]."""

    def __init__(self) -> None:
        core = np.asarray(KNOTS, float)
        self.ext = extend_grid(core, DEGREE, fix_left=True, fix_right=False)
        self.a_true, self.k_true = np.asarray(INCR_TRUE, float), np.asarray(K_TRUE, float)
        self.a_fit = self.a_true * (1.0 + np.asarray(INCR_FIT_JITTER, float))
        self.k_fit = np.clip(self.k_true + np.asarray(K_FIT_JITTER, float), 0.0, 1.0)
        self.x = x = np.linspace(0.0, 1.0, 400)
        self.f_true = ispline_curve(x, self.ext, self.a_true, DEGREE)
        self.f_fit = ispline_curve(x, self.ext, self.a_fit, DEGREE)
        self.band_true = [ispline_curve(x, self.ext, individual(self.a_true, self.k_true, t), DEGREE) for t in (-1.0, 1.0)]
        self.band_fit = [ispline_curve(x, self.ext, individual(self.a_fit, self.k_fit, t), DEGREE) for t in (-1.0, 1.0)]
        self.total = -self.f_true[-1]

    # ---- competitors' population curves (the specified forms are least-squares fits through the anchor)
    def linear(self):
        c = (self.x @ self.f_true) / (self.x @ self.x)
        return c * self.x

    def rumboost(self):
        edges = np.linspace(0.0, 1.0, N_STEPS + 1)
        xs, ys = [0.0], [0.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (self.x >= lo) & (self.x <= hi)
            xs += [lo, hi]; ys += [self.f_true[m].mean()] * 2
        return np.asarray(xs), np.asarray(ys)

    def lattice(self, scale: float = 1.0):
        bp = np.linspace(0.0, 1.0, LN_SEGMENTS + 1)
        v = np.interp(bp, self.x, self.f_true) * scale * (1.0 + np.asarray(LN_JITTER, float))
        v = np.minimum.accumulate(v)                      # a lattice network keeps the monotone constraint
        return np.interp(self.x, bp, v)

    def dnn_probe(self):
        """Truth plus a logistic upturn at the right end: the curve rises again and stays up (not monotone)."""
        c, w, h = DNN_UPTURN
        return self.f_true * 1.03 + h * self.total / (1.0 + np.exp(-(self.x - c) / w))

    def dnn_individuals(self):
        rng = np.random.default_rng(DNN_SEED)
        out = []
        for j in range(len(DNN_START)):
            ph = rng.uniform(0, 2 * np.pi, size=2)
            wig = DNN_WIGGLE * self.total * (np.sin(2 * np.pi * 1.2 * self.x + ph[0]) + 0.5 * np.sin(2 * np.pi * 2.3 * self.x + ph[1])) * np.minimum(1, 4 * self.x)
            out.append(DNN_START[j] * self.total + self.f_true * (0.8 + 0.2 * j) + wig)
        return out

    def ours_individual(self, tau: float):
        return ispline_curve(self.x, self.ext, individual(self.a_fit, self.k_fit, tau), DEGREE)


# ------------------------------------------------------------------ drawing helpers
def bare_axes(ax, total, y_top=0.22, y_bot=-1.45, arrows=True, xlabel=True, ylabel=True, label_size=None):
    ax.set_xlim(-0.03, 1.08); ax.set_ylim(y_bot * total, y_top * total)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    if arrows:
        ax.annotate("", xy=(1.08, y_bot * total), xytext=(-0.03, y_bot * total),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=LW_AXIS, mutation_scale=12))
        ax.annotate("", xy=(-0.03, y_top * total), xytext=(-0.03, y_bot * total),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=LW_AXIS, mutation_scale=12))
    fs = FONT_SIZE if label_size is None else label_size
    if xlabel:
        ax.set_xlabel("Attribute", fontsize=fs)
    if ylabel:
        ax.set_ylabel("Utility contribution", fontsize=fs)


def panel_title(ax, name: str):
    """Model name in bold, its curve type on a second line (bold needs usetex; plain otherwise)."""
    import matplotlib as mpl
    sub = PANEL_SUBTITLES.get(name, "")
    if mpl.rcParams.get("text.usetex", False):
        ax.set_title(r"\textbf{" + name + "}" + "\n" + sub, fontsize=FONT_SIZE, pad=4, linespacing=1.1)
    else:
        ax.set_title(name + "\n" + sub, fontsize=FONT_SIZE, fontweight="bold", pad=4, linespacing=1.1)


# ------------------------------------------------------------------ the two figures
def draw_truth(ax, w: World, band=True):
    if band:
        ax.fill_between(w.x, w.band_true[0], w.band_true[1], color=PINK_BAND, alpha=BAND_ALPHA, linewidth=0, zorder=1)
    ax.plot(w.x, w.f_true, color=RED, linewidth=LW_CURVE, zorder=5)


def draw_ours(ax, w: World, band=True, indiv=True):
    if band:
        ax.fill_between(w.x, w.band_fit[0], w.band_fit[1], color=BLUE_BAND, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    if indiv:
        for t in TAU_INDIV:
            ax.plot(w.x, w.ours_individual(t), color=BLUE, linewidth=LW_MODEL, linestyle=LS_INDIV, alpha=0.7, zorder=5)
    ax.plot(w.x, w.f_fit, color=BLUE, linewidth=LW_MODEL, zorder=6)   # same width as in (a); linestyle tells population from individual


def fig_shape(w: World, plt):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    T = w.total
    draw_truth(ax, w, band=False)
    ax.plot(w.x, w.linear(), color=ORANGE, linewidth=LW_MODEL, linestyle=LS_POSTHOC, zorder=4)
    xs, ys = w.rumboost(); ax.plot(xs, ys, color=GREEN, linewidth=LW_MODEL, zorder=4)
    ax.plot(w.x, w.lattice(), color=TEAL, linewidth=LW_MODEL, linestyle=LS_POSTHOC, zorder=4)
    ax.plot(w.x, w.dnn_probe(), color=GREY, linewidth=LW_MODEL, linestyle=LS_POSTHOC, zorder=4)
    ax.plot(w.x, w.f_fit, color=BLUE, linewidth=LW_MODEL, zorder=6)
    bare_axes(ax, T, y_top=0.05, y_bot=-1.25)
    from matplotlib.lines import Line2D
    entries = [(RED, LW_CURVE, "-", "True"),
               (BLUE, LW_MODEL, "-", "TSplineDCM"),
               (GREEN, LW_MODEL, "-", "RUMBoost"),
               (ORANGE, LW_MODEL, LS_POSTHOC, "TasteNet"),
               (TEAL, LW_MODEL, LS_POSTHOC, "DCM-LN"),
               (GREY, LW_MODEL, LS_POSTHOC, "ASU-DNN")]
    handles = [Line2D([], [], color=c, linewidth=lw, linestyle=ls, label=t) for c, lw, ls, t in entries]
    ax.legend(handles=handles, loc="lower left", ncol=1, frameon=False, fontsize=FONT_SIZE - 1,
              handlelength=1.8, borderaxespad=0.2, labelspacing=0.35)
    fig.tight_layout()
    return fig


def draw_tastenet(ax, w: World):
    """beta_n = beta(z_n): a line per individual whose slope is a function of the characteristics."""
    for sl in TASTENET_SLOPES:
        ax.plot(w.x, -sl * w.total * w.x, color=ORANGE, linewidth=LW_THIN, linestyle=LS_INDIV, alpha=0.9, zorder=3)


def draw_ln_indiv(ax, w: World):
    for s in LN_INDIV_SCALES:
        ax.plot(w.x, w.lattice(s), color=TEAL, linewidth=LW_THIN, linestyle=LS_INDIV, alpha=0.9, zorder=4)


def draw_dnn_indiv(ax, w: World):
    for c in w.dnn_individuals():
        ax.plot(w.x, c, color=GREY, linewidth=LW_THIN, linestyle=LS_INDIV, alpha=0.9, zorder=3)


def fig_heter_grid(w: World, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    truth_h = [Line2D([], [], color=RED, linewidth=LW_CURVE, label="true population curve"),
               Patch(facecolor=PINK_BAND, alpha=BAND_ALPHA, label="true range of individuals")]
    panels = [
        ("TasteNet", lambda ax: draw_tastenet(ax, w),
         [Line2D([], [], color=ORANGE, linewidth=LW_THIN, linestyle=LS_INDIV, label="individual lines,\nslope $\\beta(z)$")]),
        ("DCM-LN", lambda ax: draw_ln_indiv(ax, w),
         [Line2D([], [], color=TEAL, linewidth=LW_THIN, linestyle=LS_INDIV, label="individual curves:\nmonotone")]),
        ("ASU-DNN", lambda ax: draw_dnn_indiv(ax, w),
         [Line2D([], [], color=GREY, linewidth=LW_THIN, linestyle=LS_INDIV, label="individual curves:\nnot monotone")]),
        ("TSplineDCM", lambda ax: draw_ours(ax, w, True, True),
         [Line2D([], [], color=BLUE, linewidth=LW_MODEL, label="population curve"),
          Line2D([], [], color=BLUE, linewidth=LW_MODEL, linestyle=LS_INDIV, alpha=0.7, label="individual curves"),
          Patch(facecolor=BLUE_BAND, alpha=BAND_ALPHA, label="range of individuals")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharey=True, sharex=True)
    axes = axes.ravel()
    T = w.total
    for i, (ax, (name, drawer, handles)) in enumerate(zip(axes, panels)):
        draw_truth(ax, w, band=True)
        drawer(ax)
        bare_axes(ax, T, y_top=0.15, y_bot=-1.30, arrows=False, xlabel=(i >= 2), ylabel=(i % 2 == 0), label_size=FONT_SIZE - 1)
        panel_title(ax, name)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(True); ax.spines[sp].set_linewidth(0.6)
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE - 3, frameon=False, handlelength=1.4,
                  borderpad=0.2, labelspacing=0.3, borderaxespad=0.2)
    fig.legend(handles=truth_h, loc="lower center", ncol=2, frameon=False, fontsize=FONT_SIZE - 1,
               bbox_to_anchor=(0.5, -0.03), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06, hspace=0.38)   # hspace leaves room for the two-line titles of the lower row
    return fig


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    w = World()
    OUT.mkdir(parents=True, exist_ok=True)
    for fig, name in ((fig_shape(w, plt), "shape"), (fig_heter_grid(w, plt), "heter_grid")):
        out = OUT / f"{name}.{FMT}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("[fig_illustration_shape_heter]", out)


if __name__ == "__main__":
    main()
