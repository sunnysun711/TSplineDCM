# plots/fig_speedup.py — one training step before / after the speed-up (grid-stationary basis caching), per dataset.
"""Input: figures_data/timing/speedup_steps.csv, written by tools/profile_step.py --csv (one row per run:
experiment, label = old / cache / fused, step_ms and its sections, the forward split by module, kernel launches,
memcpy/sync events, cache size). All rows must come from the same idle machine and batch size.

Left panel: milliseconds per optimisation step, stacked by where the time goes (characteristic splines, attribute
splines, rest of the forward, backward, loss + optimiser + batch), one bar per mode, grouped by dataset; the
speed-up factor over `old` is written above each cached bar. Right panel: CUDA kernel launches per iteration, the
quantity the design actually reduces. Also writes figures/timing/speedup_table.tex with the same numbers.

Edit the parameter block, then:   python plots/fig_speedup.py
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASETS = {"SM_tspline_tuned": "SM", "LPMC_tspline_tuned": "LPMC", "SYN_tspline_tuned": "SYN"}   # experiment stem -> panel label
MODES = {"old": "no cache (basis every iteration)", "cache": "GSBC (default)", "fused": "GSBC + fused characteristic splines"}
PARTS = [("fwd_asc_ms", "characteristic splines", "#2A4D7F"), ("fwd_splines_ms", "attribute splines", "#4C72B0"),
         ("fwd_other_ms", "rest of forward", "#9DB8D9"), ("backward_ms", "backward", "#DD8452"),
         ("other_ms", "loss, optimiser, batch", "#BBBBBB")]
FIG_SIZE = (6.4, 2.6)
OUT_DIR = None
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import latexify  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / "timing" / "speedup_steps.csv"
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / "timing"


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d = d[d.experiment.isin(DATASETS)].copy()
    d["dataset"] = d.experiment.map(DATASETS)
    d["fwd_other_ms"] = d.forward_ms - d.fwd_asc_ms - d.fwd_splines_ms
    d["other_ms"] = d.step_ms - d.forward_ms - d.backward_ms
    d = d.drop_duplicates(["dataset", "label"], keep="last")
    return d


def figure(d: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [2.2, 1.0]})
    x, ticks, labels, gap = 0.0, [], [], 0.55
    for ds in DATASETS.values():
        sub = d[d.dataset == ds]
        modes = [m for m in MODES if m in set(sub.label)]
        old = float(sub[sub.label == "old"].step_ms.iloc[0]) if "old" in modes else np.nan
        for m in modes:
            r = sub[sub.label == m].iloc[0]
            bottom = 0.0
            for col, name, color in PARTS:
                v = float(r[col])
                ax.bar(x, v, bottom=bottom, width=0.8, color=color, edgecolor="white", linewidth=0.4,
                       label=name if (ds == list(DATASETS.values())[0] and m == modes[0]) else None)
                bottom += v
            ax2.bar(x, float(r.kernel_launches), width=0.8, color="#555555" if m == "old" else "#4C72B0")
            ax.text(x, bottom + 0.8, rf"$\times${old / r.step_ms:.1f}" if m != "old" and np.isfinite(old) else f"{r.step_ms:.0f} ms",
                    ha="center", va="bottom", fontsize=FONT_SIZE - 2)
            ax2.text(x, r.kernel_launches * 1.02, f"{r.kernel_launches:.0f}", ha="center", va="bottom", fontsize=FONT_SIZE - 3)
            ticks.append(x); labels.append({"old": "no cache", "cache": "GSBC", "fused": "+fused z"}[m])
            x += 1.0
        ax.text(np.mean(ticks[-len(modes):]), -0.33 * d.step_ms.max(), ds, ha="center", va="top", fontsize=FONT_SIZE)
        ax2.text(np.mean(ticks[-len(modes):]), -0.33 * d.kernel_launches.max(), ds, ha="center", va="top", fontsize=FONT_SIZE)
        x += gap
    for a in (ax, ax2):
        a.set_xticks(ticks)
        a.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE - 2)
        a.tick_params(axis="x", length=0)
    ax.set_ylabel("ms per iteration (batch 1024)")
    ax2.set_ylabel("kernel launches per iteration")
    ax.set_ylim(0, d.step_ms.max() * 1.18)
    ax2.set_ylim(0, d.kernel_launches.max() * 1.18)
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE - 3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"speedup_steps.{FMT}"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def table(d: pd.DataFrame) -> None:
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Dataset & Mode & step (ms) & z head & splines & rest fwd & backward & launches & memcpy & cache (MB) \\", r"\midrule"]
    for ds in DATASETS.values():
        sub = d[d.dataset == ds]
        for m in MODES:
            if m not in set(sub.label):
                continue
            r = sub[sub.label == m].iloc[0]
            lines.append(f"{ds} & {MODES[m]} & {r.step_ms:.1f} & {r.fwd_asc_ms:.1f} & {r.fwd_splines_ms:.1f} & {r.fwd_other_ms:.1f} & "
                         f"{r.backward_ms:.1f} & {r.kernel_launches:.0f} & {r.syncs_copies:.0f} & {r.cache_mb:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speedup_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "speedup_table.tex")


def main() -> None:
    d = load()
    figure(d)
    table(d)


if __name__ == "__main__":
    main()
