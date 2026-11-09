# data/

The datasets are not distributed with this repository. Place the following files here; the dataset yaml files in
`configs/datasets/` name them and describe every column used.

```text
data/
├── Swissmetro/swissmetro_processed_rumboost.csv           Swissmetro, RUMBoost-paper preprocessing
├── LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv London Passenger Mode Choice, RUMBoost-paper preprocessing
└── syn_data.csv                                           written by python synthetic/generate.py (needs the LPMC csv)
```

**Swissmetro** (Bierlaire, Axhausen & Abay 2001): the raw `swissmetro.dat` is distributed with Biogeme
(biogeme.epfl.ch). **LPMC** (Hillel, Elshafie & Jin 2018): the raw data are available from the authors' dataset
page. Both csv files above follow the preprocessing of the RUMBoost paper (Salvadé & Hillel 2024), whose
repository provides the processing scripts and the processed files: rows with unavailable chosen alternatives are
removed and the attribute columns are kept in their original units. Use those processed files as they are; the
column layout the loaders expect is listed in `configs/datasets/SM.yaml` and `configs/datasets/LPMC.yaml`.

**Splits** are regenerated deterministically from the csv (`python splits.py`): 100 group-wise 70/15/15 splits
per dataset, grouped by individual (Swissmetro) or household (LPMC), with the random seed equal to the split id.
`LPMC_officialtest` pins the test set to the dataset's official test rows. Each `splits/<name>.npz` stores the md5 of
the csv it was built from, and the loaders refuse a csv whose md5 differs.

**Synthetic data** reuse the LPMC travel times and individual characteristics with synthetic utilities and
choices; `python synthetic/generate.py` writes `data/syn_data.csv` and records the truth (knots, population
coefficients, heterogeneity profiles, ASC, choice shares, csv md5) in `synthetic/syn_truth.json`.
