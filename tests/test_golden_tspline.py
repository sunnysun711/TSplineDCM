"""Golden test: the copied TSplineDCM must reproduce v1's forward pass on a fixed batch.

The reference (tests/golden/tspline_sm_split0.npz) was exported by running v1's OWN
code (tests/golden/export_from_v1.py): split-0 checkpoint weights + the first 256
test rows + model(batch, sample=False) outputs. Here we rebuild the model from the
same experiment yaml in the NEW code base, load the stored weights, rebuild the
batch from the stored row indices, and require |V_new - V_v1| <= 1e-6 on CPU.

No training, no randomness: this is the hardest "did we copy the math correctly"
check, and the only guard when the training loop is later rewritten.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "tspline_sm_split0.npz"
EXP = ROOT / "configs" / "exp" / "SM_TSplineDCM-ispline.yaml"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden npz not exported yet")
@pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists() or not (ROOT / "splits" / "SM.npz").exists(),
                    reason="data csv / splits not present (data/ and splits/ are not in git)")
def test_tspline_forward_matches_v1(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # DcmDataset reads ./data/<file>
    from run import _load_yaml, build_trainer, load_model_spec

    g = np.load(GOLDEN)
    exp = _load_yaml(EXP)
    ov = exp.get("overrides") or {}
    trainer = build_trainer(
        dataset=exp["dataset"],
        model=exp["model"],
        model_spec=load_model_spec(exp["model"], ov),
        dcm_overrides=ov.get("dcm_config"),
        splits_name=exp.get("splits_name"),
        model_name=exp.get("model_name"),
        out_root=tmp_path,
    )
    trainer.tcfg = dataclasses.replace(trainer.tcfg_template, device="cpu")
    # same as the export: CPU, float32 data to match the float32 (mps-trained) weights
    trainer.prepare(seed=0, device="cpu", float_type=torch.float32)
    state = {k[len("sd__"):]: torch.as_tensor(g[k]) for k in g.files if k.startswith("sd__")}
    trainer.model.load_state_dict(state, strict=False, assign=True)
    trainer.model.eval()

    batch = trainer.ds.make_batch_from_indices(g["idx"], device="cpu")
    # 1) data pipeline identical: every stored batch tensor matches
    for k, v in batch.items():
        items = list(enumerate(v)) if isinstance(v, (list, tuple)) else [(None, v)]
        for i, t in items:
            if not torch.is_tensor(t):
                continue
            key = f"b__{k}" if i is None else f"b__{k}__{i}"
            assert key in g.files, f"batch key {key} missing from golden"
            np.testing.assert_allclose(t.detach().cpu().numpy(), g[key], atol=0, rtol=0,
                                       err_msg=f"batch tensor {key} differs")
    # 2) forward pass identical
    with torch.no_grad():
        out = trainer.model(batch, sample=False)
    np.testing.assert_allclose(out["V"].detach().cpu().double().numpy(), g["out__V"],
                               atol=1e-6, rtol=0, err_msg="V differs from v1")
    if "P" in out and "out__P" in g.files:
        np.testing.assert_allclose(out["P"].detach().cpu().double().numpy(), g["out__P"],
                                   atol=1e-6, rtol=0, err_msg="P differs from v1")
