"""I-spline ("increment") must be the default at EVERY level.

Config yaml, ModelConfig dataclass and the SplineUtility class default all say
"increment", and the model code has no getattr(..., "level") fallback, so a yaml
that forgets the key can never silently run the old B-spline level-scaling variant.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.config import ModelConfig  # noqa: E402
from models.tspline.mono_coef import MonoCoef  # noqa: E402
from models.tspline.spline_utility import SplineUtility  # noqa: E402


def test_modelconfig_default_is_increment():
    assert ModelConfig().SP_heter_target == "increment"


def test_spline_utility_default_is_increment():
    mcs = [MonoCoef(num=6, k=3, mono=-1, mono_type="strict", taste_name="TT")]
    assert SplineUtility(mono_coefs=mcs).heter_target == "increment"


def test_no_level_fallback_left_in_source():
    for rel in ("models/tspline/model.py", "models/tspline/spline_utility.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert 'SP_heter_target", "level"' not in src, f"{rel}: getattr fallback to 'level'"
        assert not re.search(r'heter_target:\s*HeterTarget\s*=\s*"level"', src), f"{rel}: class default 'level'"


def test_yaml_default_is_increment():
    import yaml

    spec = yaml.safe_load((ROOT / "configs" / "models" / "tspline.yaml").read_text(encoding="utf-8"))
    assert (spec.get("model_config") or {}).get("SP_heter_target") == "increment"


@pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists() or not (ROOT / "splits" / "SM.npz").exists(),
                    reason="data csv / splits not present (data/ and splits/ are not in git)")
def test_model_built_without_the_key_is_increment(monkeypatch):
    """Simulate a model yaml that forgot SP_heter_target: every SplineUtility must still be I-spline."""
    monkeypatch.chdir(ROOT)  # DcmDataset reads ./data/<file>
    from run import load_dataset_config, load_model_spec
    from dcm_data import DcmDataset
    from models import resolve_model_class

    spec = load_model_spec("tspline")
    mc = dict(spec.get("model_config") or {})
    mc.pop("SP_heter_target", None)
    dcfg = load_dataset_config("SM")
    ds = DcmDataset(config=dcfg, device="cpu", float_type=torch.float32)
    model = resolve_model_class("tspline")(
        model_config=ModelConfig(**mc), dcm_config=dcfg, device="cpu", random_seed=0,
        z_df=ds.data.iloc[:, dcfg.z_cols],
    )
    assert all(su.heter_target == "increment" for su in model.sp_utils.values())
