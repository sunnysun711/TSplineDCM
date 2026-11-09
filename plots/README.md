# plots/

Figures are produced in two stages.

1. `plots/extract/*.py` is the only code that opens checkpoints and imports model classes. It writes the numbers to
   draw into `figures_data/<results name>/*.csv` (curves per split, heterogeneity parameters, per-individual values
   of time and taste shifts, z-offsets, timing).
2. `plots/fig_*.py` read those csv files and draw one figure type each into `figures/<results name>/`. Every script
   has its parameter block at the top; edit it and run `python plots/fig_<type>.py`. They never import a model.

`plots/make_all.py --dataset SM --exps ... --splits 0-99` runs both stages for a dataset; `--figs-only` and
`--extract-only` run one. `paper.py` calls it as its `figs` stage and writes `figures/<dataset>/MANIFEST.md`, which
lists for every figure the experiment yaml, the tuned winner, the results directory and the script that drew it.

The results name is the results directory name: the dataset (`SM`), the split file when the experiment pins one
(`LPMC_officialtest`), plus `_<tag>` for tagged experiments. `plots/style.py` holds the shared colours, units and
axis limits.

`synthetic/fig_syn_*.py` draw the synthetic-data figures against the truth in `synthetic/syn_truth.json`;
`tables/vot_table.py` and `synthetic/asc_table.py` write LaTeX tables from the same csv files.
