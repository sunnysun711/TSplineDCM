"""Baseline hyperparameters travel with the instance (BaseDCM.cfg), never through a
module-level dict — so two configs in one process cannot contaminate each other
(v1 experiments/config_loader.apply_benchmark_config / DCM2 _BENCH_GLOBALS did exactly that).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import load_dataset_config  # noqa: E402
from models.config import ModelConfig  # noqa: E402
from models.fdnn import FDNN, FDNN_DEFAULTS  # noqa: E402
from models.taste_dcm import T_DCM_DEFAULTS, TasteDCM  # noqa: E402
from models.tastenet import TASTE_NET_DEFAULTS, TasteNet  # noqa: E402


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_two_fdnn_widths_do_not_leak():
    dcfg = load_dataset_config("SM")
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in FDNN_DEFAULTS.items()}
    a = FDNN(ModelConfig(), dcfg, device="cpu", cfg={"hidden_layers": [50, 50]})
    b = FDNN(ModelConfig(), dcfg, device="cpu", cfg={"hidden_layers": [200, 200]})
    c = FDNN(ModelConfig(), dcfg, device="cpu")
    assert _n(a) < _n(b) < _n(c)  # widths 50 < 200 < default 250
    assert a.hidden_layers == [50, 50] and b.hidden_layers == [200, 200]
    assert c.hidden_layers == before["hidden_layers"]
    assert FDNN_DEFAULTS == before, "module defaults were mutated"


def test_tastenet_and_tastedcm_read_cfg_not_globals():
    dcfg = load_dataset_config("SM")
    tn_before, td_before = dict(TASTE_NET_DEFAULTS), dict(T_DCM_DEFAULTS)
    tn = TasteNet(ModelConfig(), dcfg, device="cpu", cfg={"hidden_layers": [7]})
    td = TasteDCM(ModelConfig(), dcfg, device="cpu", cfg={"encoder_layers": [9]})
    assert tn.hidden_layers == [7] and td.encoder_layers == [9]
    assert TASTE_NET_DEFAULTS == tn_before and T_DCM_DEFAULTS == td_before
