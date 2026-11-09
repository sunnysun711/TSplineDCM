"""With the I-spline variant the heterogeneity factor is (1 + k*tau), tau in (-1, 1),
so k (hence its cap) must stay <= 1 or an individual's increments can flip sign. v1 had no guard;
the paper's old synthetic setting used cap 1.2, which the new code must refuse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.tspline.mono_coef import MonoCoef  # noqa: E402
from models.tspline.spline_utility import SplineUtility  # noqa: E402


def _mcs():
    return [
        MonoCoef(num=6, k=3, mono=-1, mono_type="strict", taste_name="TT"),
        MonoCoef(num=5, k=3, mono=-1, mono_type="strict", taste_name="CO"),
    ]


def test_cap_above_one_rejected_for_ispline():
    with pytest.raises(ValueError, match="k_gpi_cap must be <= 1.0"):
        SplineUtility(mono_coefs=_mcs(), heter_target="increment", heter_mode="mul", k_gpi_cap=1.2)
    with pytest.raises(ValueError, match="k_gpi_cap must be <= 1.0"):
        SplineUtility(mono_coefs=_mcs(), heter_target="increment", heter_mode="mul", k_gpi_cap=[1.0, 1.2])


def test_cap_of_exactly_one_is_fine():
    su = SplineUtility(mono_coefs=_mcs(), heter_target="increment", heter_mode="mul", k_gpi_cap=1.0)
    assert su.k_gpi_cap == [1.0, 1.0]


def test_legacy_level_variant_still_allows_large_cap():
    su = SplineUtility(mono_coefs=_mcs(), heter_target="level", heter_mode="mul", k_gpi_cap=1.2)
    assert su.heter_target == "level"


def test_dataset_config_rejects_negative_cap():
    from run import load_dataset_config

    with pytest.raises((AssertionError, ValueError)):
        load_dataset_config("SM", {"k_gpi_cap": -0.1})
