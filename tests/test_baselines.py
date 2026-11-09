"""Stage-5 baselines: registry, forward passes, the explainability interface, and the biogeme specs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import MODEL_REGISTRY, resolve_model_class  # noqa: E402
from models.config import ModelConfig  # noqa: E402
from models.l_mnl import LMNL  # noqa: E402
from models.mnl import MNL  # noqa: E402
from models.reslogit import ResLogit  # noqa: E402
from run import load_dataset_config, load_model_spec  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "Swissmetro" / "swissmetro_processed_rumboost.csv").exists(),
                                reason="Swissmetro csv not present")


def _batch(dcfg, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [torch.rand(n, len(dcfg.alt_xs[a]), generator=g) for a in dcfg.alt_names]
    z = (torch.rand(n, len(dcfg.z_names), generator=g) > 0.5).float()
    return {"idx": torch.arange(n), "x": x, "z": z, "av": torch.ones(n, len(dcfg.alt_names))}


def test_registry_and_specs_resolve():
    for name in ("l_mnl", "reslogit"):
        assert name in MODEL_REGISTRY and resolve_model_class(name) is not None
        assert load_model_spec(name)["backend"] == "torch" and load_model_spec(name)["search_space"]
    for name in ("rumboost", "gbdt", "mxl_biogeme"):
        assert load_model_spec(name)["backend"] in ("rumboost", "lightgbm", "biogeme")


def test_reslogit_starts_as_mnl():
    dcfg = load_dataset_config("SM")
    torch.manual_seed(0)
    r = ResLogit(ModelConfig(), dcfg, device="cpu", cfg={"n_layers": 3})
    m = MNL(ModelConfig(), dcfg, device="cpu")
    m.load_state_dict({k: v for k, v in r.state_dict().items() if not k.startswith("res.")}, strict=True)
    b = _batch(dcfg)
    assert torch.allclose(r(b)["P"], m(b)["P"], atol=1e-6)              # zero residual weights == the MNL
    with torch.no_grad():
        for lin in r.res:
            lin.weight.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(3))   # a uniform W would only shift all alts
    assert not torch.allclose(r(b)["P"], m(b)["P"], atol=1e-4)
    P = r(b)["P"]
    assert torch.allclose(P.sum(1), torch.ones(len(P)), atol=1e-6) and (P >= 0).all()


def test_lmnl_attribute_curves_are_linear_and_shared():
    dcfg = load_dataset_config("SM")
    torch.manual_seed(0)
    m = LMNL(ModelConfig(), dcfg, device="cpu", cfg={"hidden_layers": [8]})
    b = _batch(dcfg, n=5)
    xe = torch.linspace(0, 2, 7)
    y = m.asu_individual_curves(b, "CAR", "TT-CAR", xe)
    assert y.shape == (5, 7) and torch.allclose(y, y[:1].expand(5, -1))   # same line for everyone
    beta = m.betas[dcfg.alt_names.index("CAR")].weight[0, dcfg.alt_xs["CAR"].index("TT-CAR")]
    assert torch.allclose(y[0], beta * xe)
    zo = m.z_offset_params()
    assert zo["weight"] is None and zo["const"].shape == (3,)
    assert m.asu_has_general_trend() and not ResLogit(ModelConfig(), dcfg, device="cpu").asu_has_general_trend()


def test_reslogit_probe_curve_is_anchored():
    dcfg = load_dataset_config("SM")
    torch.manual_seed(1)
    r = ResLogit(ModelConfig(), dcfg, device="cpu", cfg={"n_layers": 2})
    with torch.no_grad():
        for lin in r.res:
            lin.weight.normal_(0, 0.2)
    b = _batch(dcfg, n=4)
    y = r.asu_individual_curves(b, "TRAIN", "CO-TRAIN", torch.tensor([0.0, 0.5, 1.0]))
    assert y.shape == (4, 3) and torch.allclose(y[:, 0], torch.zeros(4), atol=1e-6)


def test_biogeme_specs_build():
    from train_biogeme import BIOGEME_BUILDERS

    import re

    V, av, choice = BIOGEME_BUILDERS["SM_full_id"]()
    text = " ".join(str(e) for e in V.values())
    betas = set(re.findall(r"Beta\('([A-Z_0-9]+)'", text)) | set(re.findall(r"(ASC_[A-Z]+|B_[A-Z_0-9]+)", text))
    assert "B_PURPOSE_1" not in betas and "B_PURPOSE_2" in betas and "ASC_CAR" in betas
    V2, _, _ = BIOGEME_BUILDERS["SM_mxl"](random=["B_TIME"])
    text2 = " ".join(str(e) for e in V2.values())
    assert "B_TIME_S" in text2 and "B_TIME_RND" in text2 and "B_COST_S" not in text2
    for key in ("LPMC_mxl", "SYN_full", "LPMC_full"):
        V3, av3, _ = BIOGEME_BUILDERS[key]()
        assert set(V3) == set(av3) == {0, 1, 2, 3}
