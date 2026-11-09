# synthetic/generate.py — the data generating process (DGP) of the synthetic experiment, I-spline version.
"""Semi-synthetic data: REAL LPMC travel times and individual characteristics (train rows of
data/LPMC/lpmc_rumboost_version/lpmc_processed_rumboost.csv, each repeated DATA_REPEAT times), SYNTHETIC utilities
and choices with a known taste-heterogeneity structure:

    V_i(p)      = ASC_i + f_i(x_ip ; tau_ip)                                   (one travel-time attribute per alternative)
    f_i(x; tau) = sum_g B_g(x) * c_g(tau),   c(tau) = cumsum( a * (1 + tau * k) )
                  a   = increments of the population curve (a_0 = 0 so f(0) = 0; a_g <= 0 -> decreasing),
                        an NNLS fit of the "theory" shape (log / quadratic / piecewise-linear / linear) on the knots,
                  k   = heterogeneity profile at the Greville points (K_PROFILES), 0 <= k_g <= 1,
    tau_ip      = tanh(a0 + a1 * age_z + a2 * start_time_z)                   (alt-specific coefficients, TAU block)
    choice      ~ Categorical(softmax(V))                                      (no extra Gumbel noise)

Because tau in (-1, 1) and k in [0, 1], every factor (1 + tau k) is positive, so EVERY individual's curve is
monotone — the same structure TSplineDCM's increment scaling assumes. An earlier DGP multiplied the coefficient
LEVELS instead, c_g (1 + tau k_g), which only keeps the population curve monotone; it is retired.

Everything else is fixed here: alternatives, ASC, theory shapes, knot construction, K_PROFILES,
the tau formula, DATA_REPEAT = 10, seed 42. The two research knobs are exposed on the command line
(--repeat, --kmax-scale) and recorded in the truth file.

    python synthetic/generate.py                     # -> data/syn_data.csv (not in git) + synthetic/syn_truth.json (in git)
    python synthetic/generate.py --repeat 1 --out data/syn_data_r1.csv --truth synthetic/syn_truth_r1.json

syn_truth.json holds everything needed to re-evaluate the truth WITHOUT the csv: the config as used, per attribute
the knots / degree / population coefficients c / increments a / k profile / Greville points / x range / tau
coefficients, the ASC, the z standardisation constants, the choice shares and the csv md5.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
LPMC_CSV = ROOT / "data" / "LPMC" / "lpmc_rumboost_version" / "lpmc_processed_rumboost.csv"
OUT_CSV = ROOT / "data" / "syn_data.csv"
TRUTH_JSON = Path(__file__).resolve().parent / "syn_truth.json"

# ============================== DGP parameters (v1 data/syn_config.json, tuples -> lists) ==============================
CONFIG: dict = {
    "SEED": 42,
    "DEFAULT_DEGREE": 3,
    "DEGREE_MAP": {"CAR_TT": 1, "CYCLE_TT": 1},                 # piecewise-linear truths use degree 1
    "INCLUDE_COST": False,                                        # cost attributes are not part of the paper's DGP
    "ALTS": ["WALK", "PT", "CAR", "CYCLE"],                       # csv label / availability column order
    "ASC": {"WALK": 0.6, "PT": 0.2, "CAR": 0.0, "CYCLE": 0.3},   # CAR is the reference (0)
    "ATTR_MAP": {"WALK_TT": "dur_walking", "PT_TT": "dur_pt", "CAR_TT": "dur_driving", "CYCLE_TT": "dur_cycling",
                 "PT_CO": "cost_transit", "CAR_CO": "cost_driving_fuel"},
    # knot construction per attribute (x range = [0, data max]); mode: uniform | mid_denser | band_denser | near0_denser | explicit
    "KNOTS": {
        "WALK_TT": {"mode": "mid_denser", "n_base": 5, "n_mid": 8, "mid_span": [0.05, 0.25]},
        "PT_TT": {"mode": "near0_denser", "n_near0": 6, "n_wide": 6, "near0_span": 0.3},
        "CAR_TT": {"mode": "explicit", "interior": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        "CYCLE_TT": {"mode": "near0_denser", "n_near0": 8, "n_wide": 6, "near0_span": 0.5},
        "PT_CO": {"mode": "near0_denser", "n_near0": 6, "n_wide": 6, "near0_span": 0.3},
        "CAR_CO": {"mode": "near0_denser", "n_near0": 6, "n_wide": 6, "near0_span": 0.3},
    },
    # theory (target) shapes the population curves are fitted to
    "SHAPES": {
        "WALK_TT": {"type": "walk_log", "scale": 1.0},                       # -log(1 + x)
        "PT_TT": {"type": "pt_quad", "scale": 1.0 / 3.0},                    # -x^2 / 3
        "CAR_TT": {"type": "car_pwlin", "x1": 0.5, "x2": 0.6, "slope_left": -0.8, "slope_right": -0.25, "drop_mid": -2.0},
        "CYCLE_TT": {"type": "cycle_linear", "scale": 1.0},                  # -x
        "PT_CO": {"type": "pt_cost_sqrt"},
        "CAR_CO": {"type": "car_cost_sqrt", "scale": 1.35},
    },
    # population-curve fit: increments a_g <= 0 by NNLS (ridge on a), optional value anchors (x, y, weight)
    "FIT": {
        "WALK_TT": {"ridge": 1e-6},
        "PT_TT": {"ridge": 1e-6},
        "CAR_TT": {"ridge": 1e-6, "anchor_breakpoints": True, "anchor_weight": 1e4},
        "CYCLE_TT": {"ridge": 1e-6},
        "PT_CO": {"ridge": 1e-6},
        "CAR_CO": {"ridge": 1e-6},
    },
    # heterogeneity profile k_g at the Greville points; type: peak_band | bump | left_right_small | scaled_left_right_small | increasing
    "K_PROFILES": {
        "WALK_TT": {"type": "peak_band", "band": [0.3, 1.5], "kmax": 0.8, "slope": 3.5},
        "PT_TT": {"type": "peak_band", "band": [0.25, 1.0], "kmax": 0.0, "slope": 2.0},      # PT: no heterogeneity
        "CAR_TT": {"type": "peak_band", "band": [0.35, 0.75], "kmax": 1.0, "slope": 4.0},
        "CYCLE_TT": {"type": "peak_band", "band": [0.4, 1.0], "kmax": 0.8, "slope": 3.5},
        "PT_CO": {"type": "scaled_left_right_small", "kmid": 0.6, "scale": 0.5, "mid_ratio": 0.4},
        "CAR_CO": {"type": "scaled_left_right_small", "kmid": 0.6, "scale": 0.5, "mid_ratio": 0.4},
    },
    # tau_time = tanh(a0 + a1 age_z + a2 st_z) per alternative ("default" fills missing keys)
    "TAU": {
        "eps": 1e-9,
        "time_by_alt": {
            "default": {"a0": 0.0, "a1": 0.3, "a2": 0.0},
            "WALK": {"a0": 0.0, "a1": 0.2, "a2": 1.0},      # late departure (st_z > 0) -> tau_WALK > 0
            "PT": {"a0": 0.0, "a1": 0.3, "a2": -0.2},
            "CAR": {"a0": 0.0, "a1": 0.3, "a2": -0.7},      # early departure -> tau_CAR > 0
            "CYCLE": {"a0": 0.0, "a1": 0.0, "a2": 1.0},
        },
        "cost_by_alt": {
            "default": {"b0": 0.0, "b1": 0.2, "b2": 0.8, "rho": 0.3, "sigma": 0.5, "scale": 1.0},
            "PT": {"b0": 0.0, "b1": 0.2, "b2": 0.8, "rho": 0.3, "sigma": 0.4, "scale": 0.9},
            "CAR": {"b0": 0.1, "b1": 0.1, "b2": 0.7, "rho": 0.2, "sigma": 0.6, "scale": 1.1},
        },
    },
    "NOISE": {"choice_mode": "softmax_sample", "softmax_temp": 1.0},
    "DATA_REPEAT": 10,                                            # every LPMC train row appears this many times
}
# ======================================================================================================================


# ---------------------------------------------------------------------------------------------------------------------
# B-spline pieces (clamped knot vectors: k+1 copies of each end)
# ---------------------------------------------------------------------------------------------------------------------
def cox_de_boor(x: np.ndarray, k: int, T: np.ndarray, i: int) -> np.ndarray:
    if k == 0:
        return ((x >= T[i]) & (x < T[i + 1])).astype(float)
    ld = (T[i + k] - T[i]) + 1e-12
    rd = (T[i + k + 1] - T[i + 1]) + 1e-12
    return ((x - T[i]) * cox_de_boor(x, k - 1, T, i) / ld + (T[i + k + 1] - x) * cox_de_boor(x, k - 1, T, i + 1) / rd)


def bspline_design(x: np.ndarray, T: np.ndarray, k: int) -> np.ndarray:
    """(len(x), G) basis matrix; the right end is closed (x == T[-1] -> last basis)."""
    x = np.asarray(x, dtype=float)
    n_coef = len(T) - k - 1
    B = np.stack([cox_de_boor(x, k, T, i) for i in range(n_coef)], axis=1)
    B[x >= T[-1], :] = 0.0
    B[x >= T[-1], -1] = 1.0
    return B


def greville_points(T: np.ndarray, k: int) -> np.ndarray:
    n_coef = len(T) - k - 1
    if k == 0:
        return 0.5 * (T[:-1] + T[1:])
    return np.array([T[i + 1: i + k + 1].mean() for i in range(n_coef)], dtype=float)


def make_knot_vector(xmin: float, xmax: float, interior: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k + 1, xmin), np.asarray(interior, float), np.full(k + 1, xmax)])


def build_knots(spec: dict, xmin: float, xmax: float, k: int) -> np.ndarray:
    mode = spec["mode"]
    if mode == "uniform":
        interior = np.linspace(xmin, xmax, int(spec["n"]))[1:-1]
    elif mode == "mid_denser":
        a, b = spec["mid_span"]
        interior = np.unique(np.r_[np.linspace(xmin, xmax, int(spec["n_base"])),
                                   np.linspace(a * (xmin + xmax), b * (xmin + xmax), int(spec["n_mid"]))])[1:-1]
    elif mode == "band_denser":
        b1, b2 = spec["band"]
        interior = np.unique(np.r_[np.linspace(xmin, xmax, int(spec["n_base"])),
                                   np.linspace(b1, b2, int(spec["n_band"]))])[1:-1]
    elif mode == "near0_denser":
        near0 = np.linspace(xmin, xmin + float(spec["near0_span"]) * (xmax - xmin), int(spec["n_near0"]))
        wide = np.linspace(xmin, xmax, int(spec["n_wide"]))
        interior = np.unique(np.r_[near0, wide])[1:-1]
    elif mode == "explicit":
        interior = np.array(spec["interior"], dtype=float)
    else:
        raise ValueError(f"unknown knots mode {mode!r}")
    return make_knot_vector(xmin, xmax, interior, k)


# ---------------------------------------------------------------------------------------------------------------------
# theory curves (all decreasing, f(0) = 0)
# ---------------------------------------------------------------------------------------------------------------------
def theory_curve(spec: dict, x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    t = spec["type"]
    if t == "walk_log":
        return -spec.get("scale", 1.0) * np.log(x + 1.0)
    if t == "pt_quad":
        return -spec.get("scale", 1.0 / 3.0) * x ** 2
    if t == "cycle_linear":
        return -spec.get("scale", 1.0) * x
    if t == "car_pwlin":
        x1, x2 = spec["x1"], spec["x2"]
        y1 = spec["slope_left"] * x1
        yL = spec["slope_left"] * x
        w = np.clip((x - x1) / max(x2 - x1, 1e-6), 0.0, 1.0)
        yM = (1 - w) * y1 + w * spec["drop_mid"]
        yR = spec["drop_mid"] + spec["slope_right"] * (x - x2)
        return np.where(x < x1, yL, np.where(x <= x2, yM, yR))
    if t == "pt_cost_sqrt":
        return -np.sqrt(x + 1e-6)
    if t == "car_cost_sqrt":
        return -spec.get("scale", 1.35) * np.sqrt(x + 1e-6)
    raise ValueError(f"unknown shape type {t!r}")


# ---------------------------------------------------------------------------------------------------------------------
# population-curve fit with monotone increments (this is what changed vs v1: coefficients, not just the curve, decrease)
# ---------------------------------------------------------------------------------------------------------------------
def fit_mono_increments(x_eval: np.ndarray, y: np.ndarray, T: np.ndarray, k: int, ridge: float = 1e-6,
                        anchors: list[tuple[float, float, float]] | None = None) -> np.ndarray:
    """Least squares  || B(x) cumsum(a) - y ||  with a_0 = 0 and a_g <= 0 (NNLS on -a), optional weighted value
    anchors [(x, y, w)]. Returns the INCREMENTS a (G,). Levels are c = cumsum(a), c_0 = 0."""
    B = bspline_design(x_eval, T, k)                           # (M, G)
    G = B.shape[1]
    L = np.tril(np.ones((G, G)))                               # c = L @ a
    D = (B @ L)[:, 1:]                                         # a_0 = 0 is fixed -> columns 1..G-1
    rows, rhs = [D], [np.asarray(y, dtype=float)]
    for xa, ya, wa in anchors or []:
        if T[0] <= xa <= T[-1]:
            Ba = bspline_design(np.array([xa]), T, k)
            rows.append(np.sqrt(wa) * (Ba @ L)[:, 1:])
            rhs.append(np.sqrt(wa) * np.array([ya]))
    if ridge > 0:
        rows.append(np.sqrt(ridge) * np.eye(G - 1))
        rhs.append(np.zeros(G - 1))
    M, yy = np.vstack(rows), np.concatenate(rhs)
    u, _ = nnls(-M, yy)                                        # a = -u, u >= 0  ->  decreasing coefficients
    return np.concatenate([[0.0], -u])


# ---------------------------------------------------------------------------------------------------------------------
# heterogeneity profiles k_g (evaluated at the Greville points)
# ---------------------------------------------------------------------------------------------------------------------
def k_profile(spec: dict, xi: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    t = spec["type"]
    if t == "peak_band":
        b1, b2 = spec["band"]
        ctr, rad = 0.5 * (b1 + b2), max(0.5 * (b2 - b1), 1e-6)
        return spec["kmax"] * np.exp(-spec.get("slope", 6.0) * ((xi - ctr) / rad) ** 2)
    if t == "bump":
        a, b = spec["mid_span"]
        mid1, mid2 = a * (xmin + xmax), b * (xmin + xmax)
        w1 = np.clip((xi - xmin) / max(mid1 - xmin, 1e-6), 0.0, 1.0)
        w2 = np.clip((xmax - xi) / max(xmax - mid2, 1e-6), 0.0, 1.0)
        return spec["kmax"] * np.minimum(w1, w2)
    if t in ("left_right_small", "scaled_left_right_small"):
        xr = spec.get("mid_ratio", 0.5)
        xmed = (1 - xr) * xmin + xr * xmax
        up = np.clip((xi - xmin) / max(xmed - xmin, 1e-6), 0.0, 1.0)
        down = np.clip((xmax - xi) / max(xmax - xmed, 1e-6), 0.0, 1.0)
        return spec.get("scale", 1.0) * spec["kmid"] * np.minimum(up, down)
    if t == "increasing":
        tt = (xi - xmin) / max(xmax - xmin, 1e-6)
        k0, kmax, s = spec.get("k0", 0.2), spec.get("kmax", 0.7), spec.get("s", 6.0)
        return k0 + (kmax - k0) / (1.0 + np.exp(-s * (tt - 0.5)))
    raise ValueError(f"unknown K_PROFILE type {t!r}")


# ---------------------------------------------------------------------------------------------------------------------
# the DGP
# ---------------------------------------------------------------------------------------------------------------------
class SynDGP:
    """Fit once on the covariate frame (knots from the data range, population increments from the theory curve,
    k profile), then simulate. `from_truth` rebuilds the same object from syn_truth.json (no fitting)."""

    def __init__(self, config: dict, kmax_scale: float = 1.0):
        self.cfg = copy.deepcopy(config)
        if kmax_scale != 1.0:
            for spec in self.cfg["K_PROFILES"].values():
                for key in ("kmax", "kmid"):
                    if key in spec:
                        spec[key] = float(spec[key]) * kmax_scale
        self.kmax_scale = float(kmax_scale)
        self.alts: list[str] = list(self.cfg["ALTS"])
        self.asc: dict[str, float] = {a: float(self.cfg["ASC"][a]) for a in self.alts}
        self.attr_map = {name: col for name, col in self.cfg["ATTR_MAP"].items()
                         if name.split("_")[0] in self.alts and (name.endswith("_TT") or self.cfg["INCLUDE_COST"])}
        self.degree = {n: int(self.cfg.get("DEGREE_MAP", {}).get(n, self.cfg["DEFAULT_DEGREE"])) for n in self.attr_map}
        self.knots: dict[str, np.ndarray] = {}
        self.incr: dict[str, np.ndarray] = {}         # a_g
        self.coef: dict[str, np.ndarray] = {}         # c_g = cumsum(a)
        self.kcoef: dict[str, np.ndarray] = {}        # k_g
        self.ranges: dict[str, tuple[float, float]] = {}
        self.z_stats: dict[str, tuple[float, float]] = {}

    # ----- fitting -----
    def fit_from_df(self, df: pd.DataFrame) -> "SynDGP":
        eps = float(self.cfg["TAU"]["eps"])
        for z in ("age", "start_time_linear"):
            v = df[z].to_numpy(dtype=float)
            self.z_stats[z] = (float(np.nanmean(v)), float(np.nanstd(v) + eps))
        for name, col in self.attr_map.items():
            k = self.degree[name]
            xmin, xmax = 0.0, float(df[col].max())            # left end forced to 0: f(0) = 0 anchors every curve
            self.ranges[name] = (xmin, xmax)
            T = build_knots(self.cfg["KNOTS"][name], xmin, xmax, k)
            self.knots[name] = T
            x_eval = np.linspace(xmin, xmax, 600)
            y = theory_curve(self.cfg["SHAPES"][name], x_eval)
            fit = self.cfg["FIT"][name]
            anchors = None
            shp = self.cfg["SHAPES"][name]
            if fit.get("anchor_breakpoints") and shp["type"] == "car_pwlin":
                w = float(fit.get("anchor_weight", 1e4))
                anchors = [(shp["x1"], shp["slope_left"] * shp["x1"], w), (shp["x2"], shp["drop_mid"], w)]
            a = fit_mono_increments(x_eval, y, T, k, ridge=float(fit.get("ridge", 1e-6)), anchors=anchors)
            self.incr[name] = a
            self.coef[name] = np.cumsum(a)
            xi = greville_points(T, k)
            kg = k_profile(self.cfg["K_PROFILES"][name], xi, xmin, xmax)
            kg[0] = 0.0                                        # a_0 = 0: the first basis carries no increment
            self.kcoef[name] = np.clip(kg, 0.0, 1.0)
        return self

    # ----- evaluation -----
    def greville(self, name: str) -> np.ndarray:
        return greville_points(self.knots[name], self.degree[name])

    def general_curve(self, name: str, x: np.ndarray) -> np.ndarray:
        return bspline_design(x, self.knots[name], self.degree[name]) @ self.coef[name]

    def individual_coef(self, name: str, tau: np.ndarray) -> np.ndarray:
        """(N, G) levels c(tau) = cumsum(a * (1 + tau k)); each row is monotone because 1 + tau k > 0."""
        a, kg = self.incr[name], self.kcoef[name]
        return np.cumsum(a[None, :] * (1.0 + np.outer(np.asarray(tau, dtype=float), kg)), axis=1)

    def individual_curves(self, name: str, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """(N, len(x)): every individual's curve on the grid x."""
        B = bspline_design(x, self.knots[name], self.degree[name])
        return self.individual_coef(name, tau) @ B.T

    def individual_values(self, name: str, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """(N,): each individual's curve at its OWN x (what enters the utility)."""
        B = bspline_design(x, self.knots[name], self.degree[name])
        return np.sum(B * self.individual_coef(name, tau), axis=1)

    def _zscore(self, df: pd.DataFrame, z: str) -> np.ndarray:
        m, s = self.z_stats[z]
        return (df[z].to_numpy(dtype=float) - m) / s

    def taus(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """tau per attribute name (alt-specific tanh index of age and start time); only time attributes here."""
        age_z, st_z = self._zscore(df, "age"), self._zscore(df, "start_time_linear")
        by_alt = self.cfg["TAU"]["time_by_alt"]
        out = {}
        for name in self.attr_map:
            alt = name.split("_")[0]
            p = {**by_alt.get("default", {}), **by_alt.get(alt, {})}
            out[name] = np.tanh(p["a0"] + p["a1"] * age_z + p["a2"] * st_z)
        return out

    def utilities(self, df: pd.DataFrame, taus: dict[str, np.ndarray] | None = None) -> np.ndarray:
        taus = taus if taus is not None else self.taus(df)
        V = np.zeros((len(df), len(self.alts)))
        for name, col in self.attr_map.items():
            j = self.alts.index(name.split("_")[0])
            V[:, j] += self.individual_values(name, df[col].to_numpy(dtype=float), taus[name])
        for j, a in enumerate(self.alts):
            V[:, j] += self.asc[a]
        return V

    def simulate(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        taus = self.taus(df)
        V = self.utilities(df, taus)
        temp = float(self.cfg["NOISE"].get("softmax_temp", 1.0))
        eV = np.exp((V - V.max(axis=1, keepdims=True)) / max(temp, 1e-8))
        P = eV / eV.sum(axis=1, keepdims=True)
        if self.cfg["NOISE"].get("choice_mode", "softmax_sample") == "argmax":
            y = np.argmax(V, axis=1)
        else:
            u = rng.random(len(df))
            y = np.minimum((P.cumsum(axis=1) < u[:, None]).sum(axis=1), len(self.alts) - 1)
        out = df.copy()
        for name, vec in taus.items():
            out[f"tau_{name}"] = vec
        for j, a in enumerate(self.alts):
            out[f"V_{a}"] = V[:, j]
        for j, a in enumerate(self.alts):
            out[f"P_{a}"] = P[:, j]
        out["choice_sim"] = y
        return out

    # ----- truth file -----
    def truth_dict(self) -> dict:
        attrs = {}
        for name in self.attr_map:
            alt, attr = name.split("_")
            p = {**self.cfg["TAU"]["time_by_alt"].get("default", {}), **self.cfg["TAU"]["time_by_alt"].get(alt, {})}
            attrs[name] = {
                "alt": alt, "attr": attr, "column": self.attr_map[name], "degree": self.degree[name],
                "x_range": list(self.ranges[name]), "knots": self.knots[name].tolist(),
                "greville": self.greville(name).tolist(),
                "c": self.coef[name].tolist(), "a": self.incr[name].tolist(), "k": self.kcoef[name].tolist(),
                "theory": self.cfg["SHAPES"][name], "fit": self.cfg["FIT"][name],
                "k_profile": self.cfg["K_PROFILES"][name], "tau": {"a0": p["a0"], "a1": p["a1"], "a2": p["a2"]},
            }
        return {
            "heterogeneity": "increment: c(tau) = cumsum(a * (1 + tau * k)), a_0 = 0, a_g <= 0",
            "tau": "tanh(a0 + a1 * age_z + a2 * start_time_z), z standardised with z_standardisation",
            "kmax_scale": self.kmax_scale,
            "alts": self.alts, "asc": self.asc,
            "z_standardisation": {z: {"mean": m, "std": s} for z, (m, s) in self.z_stats.items()},
            "attrs": attrs,
            "config": self.cfg,
        }

    @classmethod
    def from_truth(cls, truth: dict) -> "SynDGP":
        dgp = cls(truth["config"], kmax_scale=1.0)
        dgp.kmax_scale = float(truth.get("kmax_scale", 1.0))
        dgp.z_stats = {z: (d["mean"], d["std"]) for z, d in truth["z_standardisation"].items()}
        for name, t in truth["attrs"].items():
            dgp.degree[name] = int(t["degree"])
            dgp.ranges[name] = tuple(t["x_range"])
            dgp.knots[name] = np.asarray(t["knots"], dtype=float)
            dgp.coef[name] = np.asarray(t["c"], dtype=float)
            dgp.incr[name] = np.asarray(t["a"], dtype=float)
            dgp.kcoef[name] = np.asarray(t["k"], dtype=float)
        return dgp


# ---------------------------------------------------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------------------------------------------------
CSV_COLUMNS_BASE = ["household_id", "age", "distance", "start_time_linear",
                    "dur_walking", "dur_pt", "dur_driving", "cost_transit", "cost_driving_fuel", "dur_cycling"]


def covariate_frame(lpmc_csv: Path = LPMC_CSV, repeat: int = 10) -> pd.DataFrame:
    """LPMC train rows (official 'dataset' == train), total PT time summed, each row repeated `repeat` times."""
    df = pd.read_csv(lpmc_csv)
    df = df[df["dataset"] == "train"].copy()
    df["dur_pt"] = df["dur_pt_access"] + df["dur_pt_rail"] + df["dur_pt_bus"] + df["dur_pt_int_waiting"] + df["dur_pt_int_walking"]
    df = df[CSV_COLUMNS_BASE]
    if repeat > 1:
        df = pd.concat([df] * int(repeat), ignore_index=True)
    return df.reset_index(drop=True)


def to_syn_csv(sim: pd.DataFrame, alts: list[str]) -> pd.DataFrame:
    """Column layout of v1's syn_data.csv (configs/datasets/SYN.yaml counts on it):
    base covariates, tau_*, V_*, P_*, one-hot choice_sim_<j> (alt order), av_1..av_4 (all available)."""
    cols = CSV_COLUMNS_BASE + [f"tau_{a}_TT" for a in alts] + [f"V_{a}" for a in alts] + [f"P_{a}" for a in alts]
    out = sim[cols].copy()
    for j in range(len(alts)):
        out[f"choice_sim_{j}"] = (sim["choice_sim"].to_numpy() == j).astype(float)
    for j in range(len(alts)):
        out[f"av_{j + 1}"] = 1.0
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(lpmc_csv: Path = LPMC_CSV, out_csv: Path = OUT_CSV, truth_json: Path = TRUTH_JSON, repeat: int | None = None,
             seed: int | None = None, kmax_scale: float = 1.0, write_csv: bool = True) -> tuple[SynDGP, pd.DataFrame, dict]:
    cfg = copy.deepcopy(CONFIG)
    if repeat is not None:
        cfg["DATA_REPEAT"] = int(repeat)
    if seed is not None:
        cfg["SEED"] = int(seed)
    df = covariate_frame(lpmc_csv, cfg["DATA_REPEAT"])
    dgp = SynDGP(cfg, kmax_scale=kmax_scale).fit_from_df(df)
    sim = dgp.simulate(df, default_rng(cfg["SEED"]))
    syn = to_syn_csv(sim, dgp.alts)
    shares = {a: float((sim["choice_sim"] == j).mean()) for j, a in enumerate(dgp.alts)}
    truth = dgp.truth_dict()
    truth.update({
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": str(lpmc_csv.relative_to(ROOT)) if lpmc_csv.is_relative_to(ROOT) else str(lpmc_csv),
        "n_base_rows": int(len(df) // cfg["DATA_REPEAT"]), "n_rows": int(len(df)), "repeat": int(cfg["DATA_REPEAT"]),
        "seed": int(cfg["SEED"]), "choice_shares": shares, "columns": list(syn.columns),
    })
    if write_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        syn.to_csv(out_csv, index=False, lineterminator="\n")
        truth["csv"] = str(out_csv.relative_to(ROOT)) if out_csv.is_relative_to(ROOT) else str(out_csv)
        truth["csv_md5"] = _md5(out_csv)
        print(f"[generate] wrote {out_csv} ({len(syn)} rows, md5 {truth['csv_md5']})")
    truth_json.parent.mkdir(parents=True, exist_ok=True)
    truth_json.write_text(json.dumps(truth, indent=1), encoding="utf-8", newline="\n")
    print(f"[generate] wrote {truth_json}")
    print("[generate] choice shares:", {a: round(s, 4) for a, s in shares.items()})
    for name in dgp.attr_map:
        a = dgp.incr[name]
        print(f"[generate] {name}: degree {dgp.degree[name]}, {len(a)} bases, x in [0, {dgp.ranges[name][1]:.3f}], "
              f"a_g in [{a.min():.4f}, {a.max():.4f}], k_g max {dgp.kcoef[name].max():.3f}")
    return dgp, syn, truth


def main() -> None:
    ap = argparse.ArgumentParser(description="synthetic DGP -> data/syn_data.csv + synthetic/syn_truth.json")
    ap.add_argument("--lpmc", default=str(LPMC_CSV), help="LPMC csv the covariates come from")
    ap.add_argument("--out", default=str(OUT_CSV), help="output csv (not in git)")
    ap.add_argument("--truth", default=str(TRUTH_JSON), help="output truth json (in git)")
    ap.add_argument("--repeat", type=int, default=None, help=f"row repeats (default CONFIG DATA_REPEAT = {CONFIG['DATA_REPEAT']})")
    ap.add_argument("--seed", type=int, default=None, help=f"rng seed for the choices (default {CONFIG['SEED']})")
    ap.add_argument("--kmax-scale", type=float, default=1.0, help="multiply every K_PROFILES kmax/kmid (heterogeneity strength knob)")
    ap.add_argument("--no-csv", action="store_true", help="only write the truth json")
    a = ap.parse_args()
    generate(Path(a.lpmc), Path(a.out), Path(a.truth), repeat=a.repeat, seed=a.seed, kmax_scale=a.kmax_scale,
             write_csv=not a.no_csv)


if __name__ == "__main__":
    sys.exit(main())
