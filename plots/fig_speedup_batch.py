# plots/fig_speedup_batch.py — step time and epoch time against the batch size, before / after GSBC, per dataset.
"""Input: figures_data/timing/speedup_batch.csv (tools/profile_step.py --csv: one row per experiment x mode x
batch size with step_ms, steps_per_epoch, epoch_s, rows_per_s; idle machine, the real training loss).

Top row: milliseconds per optimisation step against the batch size (log-log). A flat line means the step is
dominated by per-step overhead (kernel launches), not by the amount of data; GSBC lowers that floor. Bottom
row: seconds per epoch of training = step time x steps per epoch, which is what a larger batch buys (at the
price of fewer parameter updates per epoch, i.e. a different optimisation problem — the paper runs use 1024).

Edit the parameter block, then:   python plots/fig_speedup_batch.py
"""
from __future__ import annotations

# ============================== parameters ==================================
DATASETS = {"SM_tspline_tuned": "SM", "LPMC_tspline_tuned": "LPMC", "SYN_tspline_tuned": "SYN"}
MODES = {"old": ("no cache (before)", "#555555", "o"), "cache": ("GSBC", "#4C72B0", "s"), "fused": ("GSBC + fused z head", "#DD8452", "^")}
PAPER_BATCH = 1024                  # vertical guide: the batch size of the paper runs
FIG_SIZE_PER_PANEL = (2.6, 2.2)
OUT_DIR = None
FMT = "pdf"
FONT_SIZE = 9
# ============================================================================

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import latexify  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "figures_data" / "timing" / "speedup_batch.csv"
OUT = Path(OUT_DIR) if OUT_DIR else ROOT / "figures" / "timing"


def main() -> None:
    import matplotlib.pyplot as plt

    latexify(FONT_SIZE)
    d = pd.read_csv(DATA)
    d = d[d.experiment.isin(DATASETS)].drop_duplicates(["experiment", "label", "batch_size"], keep="last")
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(FIG_SIZE_PER_PANEL[0] * len(DATASETS), FIG_SIZE_PER_PANEL[1] * 2),
                             sharex="col")
    for j, (exp, ds) in enumerate(DATASETS.items()):
        sub = d[d.experiment == exp]
        for m, (name, color, marker) in MODES.items():
            r = sub[sub.label == m].sort_values("batch_size")
            if r.empty:
                continue
            axes[0, j].plot(r.batch_size, r.step_ms, marker=marker, markersize=3.5, color=color, label=name)
            axes[1, j].plot(r.batch_size, r.epoch_s, marker=marker, markersize=3.5, color=color, label=name)
        n_train = int(sub.train_rows.iloc[0])
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.axvline(PAPER_BATCH, color="#999999", linewidth=0.6, linestyle=(0, (2, 2)))
            ax.grid(linewidth=0.3, alpha=0.5)
        axes[0, j].set_ylim(0, sub.step_ms.max() * 1.12)      # linear: a flat line = overhead-bound step
        axes[1, j].set_yscale("log")
        axes[0, j].set_title(f"{ds} ({n_train:,} training rows)")
        axes[1, j].set_xlabel("batch size")
    axes[0, 0].set_ylabel("ms per iteration")
    axes[1, 0].set_ylabel("s per epoch of training")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=mk, markersize=3.5, label=n) for n, c, mk in MODES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"speedup_batch.{FMT}"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
