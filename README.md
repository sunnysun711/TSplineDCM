# TSplineDCM

Code for the paper *TSplineDCM: A semiparametric taste-spline utility model for heterogeneous discrete choices*
(Sun, Liu, Zhai, Zou, Shuai).

TSplineDCM replaces the linear-in-parameters utility of a logit model with monotone I-spline functions of each
attribute. Individual taste heterogeneity scales the spline **increments**, so every individual's utility curve is
monotone in the attribute, while the population curve and the heterogeneity profile stay interpretable
(marginal utilities, values of time, alternative-specific constants). This repository contains the model, the
baselines it is compared with, the experiment protocol (fixed splits, one hyperparameter-search budget for every
model, paired tests over 100 splits), and the scripts that produce every table and figure of the paper.

This is a publication snapshot. It ships code only: no data, no splits, no trained models, no results, no
extracted curves. Everything the paper reports can be regenerated from the two public datasets with the commands
below.

## Setup

```bash
conda create -n tsdcm python=3.10
conda activate tsdcm
pip install torch                  # 2.x, CUDA build recommended (https://pytorch.org)
pip install -r requirements.txt
export TSDCM_OUT=/path/with/space   # optional: root for results / HPO studies / logs; default ./results
```

Then put the two datasets under `data/` as described in [data/README.md](data/README.md) and generate the splits
and the synthetic dataset:

```bash
python splits.py                                 # splits/<dataset>.npz, 100 group-wise 70/15/15 splits, seed = split id
python synthetic/generate.py                     # data/syn_data.csv from the LPMC rows + synthetic/syn_truth.json
python -m pytest -q                              # unit tests (tests that need the csv files skip without them)
```

## Reproducing the paper

One command per dataset runs the whole protocol (tune every model with the same budget -> run all splits ->
summary tables -> extract figure data -> draw figures -> write a manifest of where every number came from):

```bash
python paper.py configs/paper/SM.yaml            # Swissmetro
python paper.py configs/paper/LPMC.yaml          # London Passenger Mode Choice, official test set held out
python paper.py configs/paper/SYN.yaml           # semi-synthetic data with a known heterogeneity structure
python paper.py configs/paper/SM.yaml --stage figs manifest     # only redraw / rewrite the manifest
```

Every stage is resumable: `tune.py` continues its optuna study, `run.py` skips finished splits.
The tuned configurations the paper used are frozen in `configs/exp/<dataset>_<model>_tuned.yaml`; the
`# HPO:` comment on top of each records the budget and the winning trial.

Tables land in `$TSDCM_OUT/<dataset>/_summary/` (`main_table.tex`, `ablation_table.tex`, `hpo_table.tex`,
`compute_table.tex`, `paired_tests.csv`, `per_split.csv`), figures in `figures/<dataset>/` together with
`MANIFEST.md`. `figures/SM/SM_vot_table.tex` and the synthetic `SYN_asc_table.tex` are written by
`tables/vot_table.py` and `synthetic/asc_table.py`.

## Running things by hand

```bash
python run.py configs/exp/SM_tspline_tuned.yaml                 # one experiment over splits 0-99
python run.py --dataset SM --model fdnn --splits 0-4            # dataset yaml + model yaml defaults, no experiment yaml
python run.py --dataset SM --model tspline --splits 0 --set model_config.T_beta=1.0   # override any config item
python tune.py --dataset SM --model tspline --trials 30 --tune-split 0-2              # optuna TPE on mean validation NLL of the tuning splits
python analyze.py --dataset SM --export                          # mean +/- SEM, paired tests, LaTeX tables
python plots/make_all.py --dataset SM --exps SM_tspline_tuned SM_fdnn_tuned --splits 0-99   # extract + all figures
python plots/fig_asucf.py                                        # one figure type; parameters at the top of the file
```

Configuration has three layers and one override order: `configs/datasets/<dataset>.yaml` (what each csv column is,
monotone directions, knots) < `configs/models/<model>.yaml` (defaults and the HPO search space) <
the experiment yaml's `overrides:` < `--set` on the command line. `model:` is the code name (registry in
`models/__init__.py`); `model_name:` is only the results directory, so variants such as `tspline_level` or
`tspline_homo` share one implementation and differ in overrides alone.

## Layout

```text
run.py  tune.py  analyze.py  splits.py  paper.py   entry points; run.py is also the config-loading library
dcm_data.py  metrics.py  trainer_base.py          dataset tensors; scoring; resumable batch-run skeleton
train_torch.py  train_biogeme.py  train_boost.py  training loops: torch models / biogeme MNL and mixed logit / RUMBoost and LightGBM
models/                                           one model per file; models/tspline/ holds TSplineDCM
                                                  (b_spline, mono_coef, spline_utility, taste_encoder, z_spline_head, basis_cache, model)
configs/datasets/  configs/models/  configs/exp/  the three config layers; configs/paper/ drives paper.py
plots/extract/                                    the only code that touches checkpoints; writes figures_data/<dataset>/*.csv
plots/fig_*.py  tables/  synthetic/fig_syn_*.py   figure and table scripts; read csv only, never import a model
synthetic/generate.py  synthetic/syn_truth.json   the semi-synthetic data generating process and its recorded truth
tests/                                            unit tests, incl. a golden forward pass of TSplineDCM on a fixed batch
tools/                                            apply_winner (ablations inherit a tuned winner), timing and profiling helpers
```

## Models

| registry name / trainer | model |
| --- | --- |
| `tspline` | TSplineDCM (this paper). `SP_heter_target: increment` is the default everywhere; `level` is the retired B-spline variant kept for the ablation |
| `mnl`, `l_mnl`, `reslogit` | multinomial logit, L-MNL (Sifringer et al.), ResLogit (Wong & Farooq) in torch |
| `fdnn`, `tastenet`, `taste_dcm` | fully-connected DNN, TasteNet (Han et al.), taste-parameter DCM |
| `train_biogeme.py` | MNL and mixed logit by maximum likelihood (biogeme) |
| `train_boost.py` | RUMBoost (Salvadé & Hillel) and its ablations, LightGBM multiclass |

## Citation

See [CITATION.cff](CITATION.cff). Code is released under the MIT License.
