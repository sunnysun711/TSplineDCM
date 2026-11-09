"""Speed-up step 3 must not change what is computed: the on-device weighted F1 equals sklearn's, and the
bank-free eps draw for row-id keys is exactly what the bank handed out per batch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def test_weighted_f1_matches_sklearn():
    from train_torch import weighted_f1

    rng = np.random.default_rng(0)
    for n_classes, n in ((3, 500), (4, 37), (4, 2)):
        true = rng.integers(0, n_classes, size=n)
        pred = rng.integers(0, n_classes, size=n)
        if n_classes == 4:
            pred[pred == 3] = 0      # a class never predicted; and for n = 2 classes absent from `true` too
        ours = weighted_f1(torch.as_tensor(true), torch.as_tensor(pred), n_classes).item()
        ref = f1_score(true, pred, average="weighted", zero_division=0)
        assert abs(ours - ref) < 1e-12, (n_classes, n, ours, ref)


@pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists() or not (ROOT / "splits" / "SM.npz").exists(),
                    reason="data csv / splits not present (data/ and splits/ are not in git)")
@pytest.mark.parametrize("per_alt", [False, True])
def test_eps_per_sample_path_draws_what_the_bank_drew(per_alt, tmp_path, monkeypatch):
    """The bank re-allocated whenever a batch's max key changed (every training batch), so each batch got a
    fresh randn(B) handed out in ascending key order. The bank-free per-sample path must reproduce that exactly;
    the reference here is the bank code itself (TR_per_sample=False branch) with the bank reset before each batch."""
    monkeypatch.chdir(ROOT)
    from run import load_experiment

    trainer = load_experiment(str(ROOT / "configs" / "exp" / "SM_tspline_tuned.yaml"), out_root=tmp_path)["trainer"]
    trainer.mcfg.TR_per_alt = per_alt
    trainer.prepare(seed=0, device="cpu", float_type=torch.float32)
    model, ds = trainer.model, trainer.ds
    I, P = len(model.dcfg.alt_names), len(model.dcfg.x_names)
    rng = np.random.default_rng(1)
    batches = [torch.as_tensor(rng.choice(len(ds), size=256, replace=False)) for _ in range(3)]

    ref = []
    model.mcfg.TR_per_sample = False
    for b in batches:
        model.reset_eps_cache()
        ref.append(model._get_eps(b, I, P, dtype=torch.float32, sample=True).clone())
    model.mcfg.TR_per_sample = True
    model.reset_eps_cache()
    new = [model._get_eps(b, I, P, dtype=torch.float32, sample=True).clone() for b in batches]
    for a, b in zip(ref, new):
        assert a.shape == ((256, I, P) if per_alt else (256, P)) and torch.equal(a, b)
