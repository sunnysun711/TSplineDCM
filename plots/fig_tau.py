# plots/fig_tau.py — taste index tau of one attribute across a socio-demographic group (e.g. trip purpose).
"""Paper figures SM_tau_TT-*.pdf: how the travel-time taste index tau = tanh(mu(z)) spreads over trip purposes.
Input: figures_data/<DATASET>/tau_individual.csv.gz (per individual x split x attribute) joined with individuals.csv.gz
(z columns). The group of an individual is the one-hot column in GROUP_COLS that equals 1.

Two modes, chosen by how many splits the csv holds:
    one split      -> per-individual points (jittered) + box per group
    several splits -> one point per (split, group) = that split's mean tau in the group, + box across splits
Optionally re-sample the posterior (tau = tanh(mu + sigma*eps), N_RESAMPLE draws per individual) to show
the within-person spread as in the old notebook.
Truth mode (TRUTH lists the dataset; the synthetic data): the DGP's true tau is drawn next to the learned one, and a
second figure per attribute (*_band) keeps only the individuals whose own attribute value lies where the true
heterogeneity profile k_g is positive, the only place where tau enters the utility. The panel title carries the
Spearman correlation between true and learned tau.

Edit the parameter block, then:   python plots/fig_tau.py
Only some figures (substrings of the output file name):   python plots/fig_tau.py --dataset SM --only TT-TRAIN
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASET = "SM"
MODELS = None                       # None = every model in tau_individual.csv.gz (TSplineDCM variants); or ["TSplineDCM-ispline"]
ATTRS = None                        # None = all attributes in tau_individual.csv; or ["TT", "CO"]
GROUP = None                        # None = pick the preset below by dataset name; or ("title", [one-hot cols], [labels])
GROUP_PRESETS = {                   # one-hot columns in individuals.csv -> readable labels (or one continuous column -> quantiles)
    "SM": ("Trip purpose", [f"PURPOSE_{i}" for i in range(1, 9)],
           ["Commute", "Shopping", "Business", "Leisure", "Return work", "Return shop.", "Return bus.", "Return leis."]),
    "LPMC": ("Trip purpose", ["purpose_HBW", "purpose_HBE", "purpose_HBO", "purpose_NHBO", "purpose_B"],
             ["Home-based\nwork", "Home-based\neducation", "Home-based\nother", "Non-home-\nbased other", "Business"]),
    # a continuous column instead of one-hot columns -> quantile bins, one label per bin
    "SYN": ("Departure time (quartiles)", "start_time_linear", ["Q1", "Q2", "Q3", "Q4"]),    # Q1 earliest (caption)
}
QUANTILE_CLIP = (0.0, 1.0)          # e.g. (0.1, 0.9) drops the tails per group before drawing (old notebook: 0.1-0.9)
N_RESAMPLE = 0                      # >0: draw eps per individual and plot the posterior samples instead of tanh(mu)
SEED = 42
FIG_SIZE = (2.3, 2.1)               # inches; the paper prints three per row (.33 textwidth = 2.1 in), so ~95 % scale
OUT_DIR = None
ONLY = None                         # None = every figure; or parts of the output file name, e.g. ["overlay_TRAIN_TT"]; "=name" = whole name
FMT = "pdf"
FONT_SIZE = 9
XTICK_ROTATION = 90                 # degrees of the group labels; 45 or 90 keep long labels apart
XTICK_WRAP = True                   # keep the line breaks of the preset labels (long LPMC labels); False joins them into one line
Y_LIM = (-1.08, 1.08)               # the same y range in every panel (tau = tanh(.) lies in (-1, 1)); None = auto per panel
Y_TICKS = (-1, -0.5, 0, 0.5, 1)
# truth mode (datasets with a known tau): dataset prefix -> csv with the DGP's tau per row (column tau_<ALT>_TT, idx = row)
TRUTH = {"SYN": "data/syn_data.csv"}
BAND_SHARE = 0.05                   # *_band figure: individuals whose x lies where the true k_g > this share of its maximum
LEGEND_ATTR = "TT-WALK"            # truth mode: the Truth / Learned legend goes into the all-individuals panel of this attribute
TRUTH_COLOR, LEARNED_COLOR = "#D32F2F", "#4C72B0"
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import PALETTE, cli, display, latexify, tex, wanted, x_unit  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / DATASET
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / DATASET


def _preset():
    if GROUP:
        return GROUP
    for key, preset in GROUP_PRESETS.items():
        if DATASET.startswith(key):
            return preset
    raise SystemExit(f"no group preset for {DATASET}; set GROUP = (title, cols, labels) in the parameter block")


GROUP_TITLE, GROUP_COLS, GROUP_LABELS = "", [], []      # resolved in main() so make_all can swap DATASET first
TRUTH_HETER = None                                      # truth mode: syn_truth_heter.csv (alt, xi, k)


def _group_of(ind: pd.DataFrame) -> pd.Series:
    if isinstance(GROUP_COLS, str):                                   # continuous column -> quantile bins
        v = ind[GROUP_COLS].to_numpy(dtype=float)
        edges = np.quantile(v, np.linspace(0.0, 1.0, len(GROUP_LABELS) + 1))
        g = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(GROUP_LABELS) - 1)
        return pd.Series(g, index=ind.index)
    onehot = ind[GROUP_COLS].to_numpy()
    g = np.full(len(ind), -1)
    hit = onehot.argmax(1)
    g[onehot.max(1) > 0.5] = hit[onehot.max(1) > 0.5]
    return pd.Series(g, index=ind.index)


def main() -> None:
    import matplotlib.pyplot as plt

    global GROUP_TITLE, GROUP_COLS, GROUP_LABELS
    GROUP_TITLE, GROUP_COLS, GROUP_LABELS = _preset()
    latexify(FONT_SIZE)
    tau_all = pd.read_csv(DATA / "tau_individual.csv.gz")
    ind = pd.read_csv(DATA / "individuals.csv.gz")
    ind["group"] = _group_of(ind)
    tau_all = tau_all.merge(ind[["idx", "group"]], on="idx", how="left")
    tau_all = tau_all[tau_all.group >= 0]
    truth_csv = next((v for k, v in TRUTH.items() if DATASET.startswith(k)), None)
    if truth_csv:                                       # true tau of each row + the individual's own attribute value
        global TRUTH_HETER
        syn = pd.read_csv(ROOT / truth_csv)
        TRUTH_HETER = pd.read_csv(DATA / "syn_truth_heter.csv")
        parts = []
        for attr, d in tau_all.groupby("attr"):
            alt = attr.split("-")[1]
            t = syn[[f"tau_{alt}_TT"]].rename(columns={f"tau_{alt}_TT": "truth"})
            t["idx"] = np.arange(len(syn))
            d = d.merge(t, on="idx", how="left").merge(ind[["idx", f"{alt}__{attr}"]].rename(columns={f"{alt}__{attr}": "x_own"}), on="idx", how="left")
            parts.append(d)
        tau_all = pd.concat(parts, ignore_index=True)
    models = list(MODELS) if MODELS else list(dict.fromkeys(tau_all.model_name))
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        _one_model(tau_all[tau_all.model_name == m], m)


def _points(d: pd.DataFrame, rng) -> tuple[pd.DataFrame, str]:
    """What to draw for one attribute: individuals (one split), split means (several splits) or posterior draws."""
    n_split = d["split"].nunique()
    if N_RESAMPLE > 0:
        eps = rng.standard_normal((len(d), N_RESAMPLE))
        vals = np.tanh(d["mu"].to_numpy()[:, None] + d["sigma"].to_numpy()[:, None] * eps)
        return pd.DataFrame({"group": np.repeat(d["group"].to_numpy(), N_RESAMPLE),
                             "split": np.repeat(d["split"].to_numpy(), N_RESAMPLE), "tau": vals.ravel()}), "posterior samples"
    if n_split > 1:
        cols = ["tau"] + (["truth"] if "truth" in d else [])
        return d.groupby(["split", "group"], as_index=False)[cols].mean(), f"split means, {n_split} splits"
    return d, "individuals"


def _draw(ax, points: pd.DataFrame, attr: str, rng, note: str | None = None, legend: bool = False) -> None:
    """Points + box per group on one axis. With a 'truth' column: the true tau (red) left of the learned one (blue)."""
    series = [("truth", -0.2, TRUTH_COLOR, "Truth"), ("tau", 0.2, LEARNED_COLOR, "Learned")] if "truth" in points else [("tau", 0.0, PALETTE[0], None)]
    two = len(series) == 2
    for gi in range(len(GROUP_LABELS)):
        for col, off, color, _ in series:
            v = points[points.group == gi][col].to_numpy()
            if v.size == 0:
                continue
            lo, hi = np.quantile(v, QUANTILE_CLIP[0]), np.quantile(v, QUANTILE_CLIP[1])
            v = v[(v >= lo) & (v <= hi)]
            jitter = rng.uniform(-0.12 if two else -0.22, 0.12 if two else 0.22, size=v.size)
            ax.scatter(gi + off + jitter, v, s=3 if two else (4 if v.size > 200 else 10),
                       alpha=0.08 if two else (0.15 if v.size > 200 else 0.6), color=color, linewidths=0, zorder=2)
            bp = ax.boxplot([v], positions=[gi + off], widths=0.32 if two else 0.45, patch_artist=True, showfliers=False,
                            zorder=3, medianprops=dict(color="black" if two else "darkred", linewidth=1.0 if two else 1.2))
            if two:
                bp["boxes"][0].set(facecolor=color, alpha=0.45, edgecolor=color)
                for k in ("whiskers", "caps"):
                    for a in bp[k]:
                        a.set(color=color)
            else:
                bp["boxes"][0].set_facecolor("lightskyblue")
                bp["boxes"][0].set_alpha(0.4)
    ax.axhline(0, color="0.4" if two else "red", linestyle="--", linewidth=0.8, alpha=0.5)
    if legend and two:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=c, alpha=0.45, label=l) for _, _, c, l in series], loc="lower right",
                  fontsize=FONT_SIZE - 2, frameon=False)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(Y_TICKS)
    ax.set_xticks(range(len(GROUP_LABELS)))
    labels = GROUP_LABELS if XTICK_WRAP else [l.replace("\n", " ") for l in GROUP_LABELS]
    upright = XTICK_ROTATION in (0, 90)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 2, rotation=XTICK_ROTATION, ha="center" if upright else "right",
                       rotation_mode="default" if upright else "anchor")
    ax.set_xlabel(GROUP_TITLE)
    title = tex(rf"$\tau$ ({attr})")                                # the attribute as panel title; no y label
    ax.set_title(title + ("\n" + note if note else ""), fontsize=FONT_SIZE - 1 if note else FONT_SIZE)


def _one_model(tau, MODEL):
    import matplotlib.pyplot as plt

    if tau.empty:
        print(f"[fig_tau] {MODEL} not in tau_individual.csv.gz; run plots/extract/tau_individual.py")
        return
    rng = np.random.default_rng(SEED)
    attrs = ATTRS or list(dict.fromkeys(tau.attr))
    for attr in attrs:
        d = tau[tau.attr == attr]
        variants = [("", d, None)]
        if "truth" in d:                                   # all individuals, then only those inside the k_g > 0 band
            rho = d["truth"].corr(d["tau"], method="spearman")
            variants = [("", d, rf"all individuals, $\rho_s = {rho:.2f}$")]
            k = TRUTH_HETER[TRUTH_HETER.alt == attr.split("-")[1]].sort_values("xi")
            band = k[k.k > BAND_SHARE * k.k.max()]
            if not band.empty:
                lo, hi = float(band.xi.min()), float(band.xi.max())
                inside = d[(d.x_own >= lo) & (d.x_own <= hi)]
                rho_in = inside["truth"].corr(inside["tau"], method="spearman")
                unit = x_unit(DATASET, attr).strip(" ()")
                variants.append(("_band", inside, rf"$x \in [{lo:.1f}, {hi:.1f}]$ {unit} ({len(inside) / len(d) * 100:.0f}\%), $\rho_s = {rho_in:.2f}$"))
        for suffix, dd, note in variants:
            stem = f"{DATASET}_tau_{MODEL}_{attr}{suffix}"
            if not wanted(stem, ONLY):
                continue
            points, what = _points(dd, rng)
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            _draw(ax, points, attr, rng, note=note, legend=(suffix == "" and attr == LEGEND_ATTR))
            fig.tight_layout()
            p = OUT / f"{stem}.{FMT}"
            fig.savefig(p)
            plt.close(fig)
            print(f"[fig_tau] {p}")


if __name__ == "__main__":
    cli(globals())                  # --dataset X  --only <name substring> ...  --out DIR (style.cli)
    main()
