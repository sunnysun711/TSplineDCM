# train_biogeme.py — biogeme MLE models on the shared splits.
"""Biogeme utility specifications + the BaseTrainer adapter that estimates them.

Builders return (V, av, CHOICE) so one spec serves both estimation (models.loglogit)
and prediction (models.logit per alternative). Predictions flow through the same
dcm_metrics / predictions.npz currency as torch runs. Mixed logit goes here too.
"""
from __future__ import annotations

from biogeme.expressions import Beta, Variable


def lpmc_mnl_full():
    """Full LPMC MNL (walk is the socio-demographic reference).

    Alternatives: 0=WALK, 1=CYCLE, 2=PT, 3=CAR.
    """
    ASC_CYCLE = Beta('ASC_CYCLE', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    B_DUR_WALKING = Beta('B_DUR_WALKING', 0, None, None, 0)
    B_DUR_CYCLING = Beta('B_DUR_CYCLING', 0, None, None, 0)
    B_DUR_DRIVING = Beta('B_DUR_DRIVING', 0, None, None, 0)
    B_DUR_PT_ACCESS = Beta('B_DUR_PT_ACCESS', 0, None, None, 0)
    B_DUR_PT_RAIL = Beta('B_DUR_PT_RAIL', 0, None, None, 0)
    B_DUR_PT_BUS = Beta('B_DUR_PT_BUS', 0, None, None, 0)
    B_DUR_PT_INT_WAIT = Beta('B_DUR_PT_INT_WAIT', 0, None, None, 0)
    B_DUR_PT_INT_WALK = Beta('B_DUR_PT_INT_WALK', 0, None, None, 0)
    B_COST_TRANSIT = Beta('B_COST_TRANSIT', 0, None, None, 0)
    B_COST_DRIVING = Beta('B_COST_DRIVING', 0, None, None, 0)
    B_PT_INTERCHANGES = Beta('B_PT_INTERCHANGES', 0, None, None, 0)
    B_CONGESTION_CHARGE = Beta('B_CONGESTION_CHARGE', 0, None, None, 0)
    B_TRAFFIC_PERCENT = Beta('B_TRAFFIC_PERCENT', 0, None, None, 0)

    def socio(alt: str) -> list:
        return [
            Beta(f'B_DIST_{alt}', 0, None, None, 0),
            Beta(f'B_AGE_{alt}', 0, None, None, 0),
            Beta(f'B_FEMALE_{alt}', 0, None, None, 0),
            Beta(f'B_LICENSE_{alt}', 0, None, None, 0),
            Beta(f'B_CAR_OWN_{alt}', 0, None, None, 0),
            Beta(f'B_DAY_{alt}', 0, None, None, 0),
            Beta(f'B_START_{alt}', 0, None, None, 0),
            Beta(f'B_PURPOSE_HBE_{alt}', 0, None, None, 0),
            Beta(f'B_PURPOSE_HBO_{alt}', 0, None, None, 0),
            Beta(f'B_PURPOSE_B_{alt}', 0, None, None, 0),
            Beta(f'B_PURPOSE_NHBO_{alt}', 0, None, None, 0),
        ]

    SOCIO_VARS = [
        Variable('distance'), Variable('age'), Variable('female'),
        Variable('driving_license'), Variable('car_ownership'),
        Variable('day_of_week'), Variable('start_time_linear'),
        Variable('purpose_HBE'), Variable('purpose_HBO'),
        Variable('purpose_B'), Variable('purpose_NHBO'),
    ]

    def socio_effect(alt: str):
        return sum(b * v for b, v in zip(socio(alt), SOCIO_VARS))

    V_WALK = B_DUR_WALKING * Variable('dur_walking')
    V_CYCLE = ASC_CYCLE + B_DUR_CYCLING * Variable('dur_cycling') + socio_effect('CYCLE')
    V_PT = (
        ASC_PT
        + B_DUR_PT_ACCESS * Variable('dur_pt_access')
        + B_DUR_PT_RAIL * Variable('dur_pt_rail')
        + B_DUR_PT_BUS * Variable('dur_pt_bus')
        + B_DUR_PT_INT_WAIT * Variable('dur_pt_int_waiting')
        + B_DUR_PT_INT_WALK * Variable('dur_pt_int_walking')
        + B_COST_TRANSIT * Variable('cost_transit')
        + B_PT_INTERCHANGES * Variable('pt_n_interchanges')
        + socio_effect('PT')
    )
    V_CAR = (
        ASC_CAR
        + B_DUR_DRIVING * Variable('dur_driving')
        + B_COST_DRIVING * Variable('cost_driving_fuel')
        + B_CONGESTION_CHARGE * Variable('congestion_charge')
        + B_TRAFFIC_PERCENT * Variable('driving_traffic_percent')
        + Beta('B_FUELTYPE_DIESEL_CAR', 0, None, None, 0) * Variable('fueltype_Diesel')
        + Beta('B_FUELTYPE_HYBRID_CAR', 0, None, None, 0) * Variable('fueltype_Hybrid')
        + Beta('B_FUELTYPE_AVERAGE_CAR', 0, None, None, 0) * Variable('fueltype_Average')
        + socio_effect('CAR')
    )

    V = {0: V_WALK, 1: V_CYCLE, 2: V_PT, 3: V_CAR}
    av = {0: Variable('av_1'), 1: Variable('av_2'), 2: Variable('av_3'), 3: Variable('av_4')}
    return V, av, Variable('choice')


def _sm_terms(random: tuple[str, ...] = (), draw_type: str = "NORMAL_HALTON2"):
    """Shared pieces of the Swissmetro specs: betas (optionally random), variables, identified socio effect."""
    from biogeme.expressions import bioDraws

    def beta(name):
        b = Beta(name, 0, None, None, 0)
        if name in random:                                  # b + sigma * draw, one Halton sequence per coefficient
            sigma = Beta(f"{name}_S", 1, None, None, 0)
            return b + sigma * bioDraws(f"{name}_RND", draw_type)
        return b

    B_TIME, B_COST, B_HE = beta("B_TIME"), beta("B_COST"), beta("B_HE")
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    B_AGE_1, B_AGE_2 = Beta("B_AGE_1", 0, None, None, 0), Beta("B_AGE_2", 0, None, None, 0)
    B_MALE, B_FIRST = Beta("B_MALE", 0, None, None, 0), Beta("B_FIRST", 0, None, None, 0)
    # PURPOSE_1 (commute) is the reference: PURPOSE_1..8 sum to 1 on all but 9 rows, so a full set of purpose
    # dummies plus ASC_CAR is not identified
    B_PURPOSE = {i: Beta(f"B_PURPOSE_{i}", 0, None, None, 0) for i in range(2, 9)}
    # every Variable is created where it is used: biogeme's panel mode renames Variable objects per observation,
    # and an object shared by two utilities would be renamed twice (BiogemeError "No id defined ...__panel__01__panel__01")
    def socio():
        return (B_AGE_1 * Variable("AGE_1") + B_AGE_2 * Variable("AGE_2") + B_MALE * Variable("MALE")
                + sum(b * Variable(f"PURPOSE_{i}") for i, b in B_PURPOSE.items()))

    V = {
        0: (ASC_TRAIN + B_TIME * Variable("TRAIN_TT") + B_COST * Variable("TRAIN_COST") + B_HE * Variable("TRAIN_HE")
            + B_FIRST * Variable("FIRST") + socio()),
        1: B_TIME * Variable("SM_TT") + B_COST * Variable("SM_COST") + B_HE * Variable("SM_HE") + socio(),   # ASC_SM = 0
        2: ASC_CAR + B_TIME * Variable("CAR_TT") + B_COST * Variable("CAR_CO"),
    }
    av = {0: Variable("av_0"), 1: Variable("av_1"), 2: Variable("av_2")}
    return V, av, Variable("choice")


def swissmetro_mnl_identified():
    """Swissmetro MNL matching the RUMBoost paper specification, with PURPOSE_1 as the reference purpose (identified; 16 parameters)."""
    return _sm_terms()


def swissmetro_mxl(random=("B_TIME", "B_COST"), draw_type="NORMAL_HALTON2"):
    """Mixed logit on the identified SM spec: normally distributed B_TIME / B_COST (panel by ID, Halton draws)."""
    return _sm_terms(tuple(random), draw_type)


def _lpmc_terms(random: tuple[str, ...] = (), draw_type: str = "NORMAL_HALTON2"):
    from biogeme.expressions import bioDraws

    def beta(name):
        b = Beta(name, 0, None, None, 0)
        if name in random:
            sigma = Beta(f"{name}_S", 1, None, None, 0)
            return b + sigma * bioDraws(f"{name}_RND", draw_type)
        return b

    ASC_CYCLE, ASC_PT, ASC_CAR = (Beta(n, 0, None, None, 0) for n in ("ASC_CYCLE", "ASC_PT", "ASC_CAR"))
    B = {n: beta(n) for n in ("B_DUR_WALKING", "B_DUR_CYCLING", "B_DUR_DRIVING", "B_DUR_PT_ACCESS", "B_DUR_PT_RAIL",
                              "B_DUR_PT_BUS", "B_DUR_PT_INT_WAIT", "B_DUR_PT_INT_WALK", "B_COST_TRANSIT",
                              "B_COST_DRIVING", "B_PT_INTERCHANGES", "B_CONGESTION_CHARGE", "B_TRAFFIC_PERCENT")}
    SOCIO_VARS = ["distance", "age", "female", "driving_license", "car_ownership", "day_of_week",
                  "start_time_linear", "purpose_HBE", "purpose_HBO", "purpose_B", "purpose_NHBO"]
    SOCIO_TAGS = ["DIST", "AGE", "FEMALE", "LICENSE", "CAR_OWN", "DAY", "START", "PURPOSE_HBE", "PURPOSE_HBO",
                  "PURPOSE_B", "PURPOSE_NHBO"]

    def socio_effect(alt):          # fresh Variable objects per utility (panel mode renames them per observation)
        return sum(Beta(f"B_{t}_{alt}", 0, None, None, 0) * Variable(v) for t, v in zip(SOCIO_TAGS, SOCIO_VARS))

    V_WALK = B["B_DUR_WALKING"] * Variable("dur_walking")
    V_CYCLE = ASC_CYCLE + B["B_DUR_CYCLING"] * Variable("dur_cycling") + socio_effect("CYCLE")
    V_PT = (ASC_PT + B["B_DUR_PT_ACCESS"] * Variable("dur_pt_access") + B["B_DUR_PT_RAIL"] * Variable("dur_pt_rail")
            + B["B_DUR_PT_BUS"] * Variable("dur_pt_bus") + B["B_DUR_PT_INT_WAIT"] * Variable("dur_pt_int_waiting")
            + B["B_DUR_PT_INT_WALK"] * Variable("dur_pt_int_walking") + B["B_COST_TRANSIT"] * Variable("cost_transit")
            + B["B_PT_INTERCHANGES"] * Variable("pt_n_interchanges") + socio_effect("PT"))
    V_CAR = (ASC_CAR + B["B_DUR_DRIVING"] * Variable("dur_driving") + B["B_COST_DRIVING"] * Variable("cost_driving_fuel")
             + B["B_CONGESTION_CHARGE"] * Variable("congestion_charge") + B["B_TRAFFIC_PERCENT"] * Variable("driving_traffic_percent")
             + Beta("B_FUELTYPE_DIESEL_CAR", 0, None, None, 0) * Variable("fueltype_Diesel")
             + Beta("B_FUELTYPE_HYBRID_CAR", 0, None, None, 0) * Variable("fueltype_Hybrid")
             + Beta("B_FUELTYPE_AVERAGE_CAR", 0, None, None, 0) * Variable("fueltype_Average")
             + socio_effect("CAR"))
    V = {0: V_WALK, 1: V_CYCLE, 2: V_PT, 3: V_CAR}
    av = {0: Variable("av_1"), 1: Variable("av_2"), 2: Variable("av_3"), 3: Variable("av_4")}
    return V, av, Variable("choice")


def lpmc_mxl(random=("B_DUR_WALKING", "B_DUR_CYCLING", "B_DUR_DRIVING", "B_COST_TRANSIT", "B_COST_DRIVING"),
             draw_type="NORMAL_HALTON2"):
    """Mixed logit on the LPMC spec: normal random coefficients on the main time and cost attributes (panel by household)."""
    return _lpmc_terms(tuple(random), draw_type)


def syn_mnl():
    """Synthetic data: ASC + linear travel time per alternative + linear z (age, distance, start time), CAR reference."""
    ASC = {a: Beta(f"ASC_{a}", 0, None, None, 0) for a in ("WALK", "PT", "CYCLE")}
    B = {a: Beta(f"B_TT_{a}", 0, None, None, 0) for a in ("WALK", "PT", "CAR", "CYCLE")}
    Z = ("age", "distance", "start_time_linear")

    def socio(alt):
        return sum(Beta(f"B_{z}_{alt}", 0, None, None, 0) * Variable(v) for z, v in zip(("AGE", "DIST", "START"), Z))

    V = {0: ASC["WALK"] + B["WALK"] * Variable("dur_walking") + socio("WALK"),
         1: ASC["PT"] + B["PT"] * Variable("dur_pt") + socio("PT"),
         2: B["CAR"] * Variable("dur_driving"),
         3: ASC["CYCLE"] + B["CYCLE"] * Variable("dur_cycling") + socio("CYCLE")}
    av = {0: Variable("av_1"), 1: Variable("av_2"), 2: Variable("av_3"), 3: Variable("av_4")}
    return V, av, Variable("choice")


BIOGEME_BUILDERS = {
    "SM_full_id": swissmetro_mnl_identified,
    "SM_mxl": swissmetro_mxl,
    "LPMC_full": lpmc_mnl_full,
    "LPMC_mxl": lpmc_mxl,
    "SYN_full": syn_mnl,
}


# ==============================================================================
# trainer
# ==============================================================================
from pathlib import Path
from typing import Any

import numpy as np

from dcm_data import DcmConfig

from trainer_base import BaseTrainer


def _make_biogeme(database, formulas, **kw):
    """bio.BIOGEME(...) with output files off, tolerant to the biogeme 3.2 -> 3.3 kwarg rename.

    v1 ran on biogeme 3.2.x (``generate_pickle``); the dcm env has 3.3.2, where the
    switch is ``generate_yaml``. Estimation itself is unchanged. Extra kw (number_of_draws) pass through.
    """
    import biogeme.biogeme as bio

    try:
        return bio.BIOGEME(database, formulas, generate_html=False, generate_pickle=False, **kw)
    except Exception:  # biogeme >= 3.3: "Parameter [generate_pickle] does not exist"
        return bio.BIOGEME(database, formulas, generate_html=False, generate_yaml=False, **kw)


class BiogemeMNLTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_name: str,
        dcm_config: DcmConfig,
        model_name: str,
        spec: dict[str, Any],
        **base_kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dcm_config=dcm_config,
            model_name=model_name,
            **base_kwargs,
        )
        builder_key = spec.get("builder")
        if builder_key not in BIOGEME_BUILDERS:
            raise ValueError(
                f"biogeme_config.builder must be one of {sorted(BIOGEME_BUILDERS)}, "
                f"got {builder_key!r}"
            )
        self.builder = BIOGEME_BUILDERS[builder_key]
        self.spec = dict(spec)
        self.builder_kwargs = dict(spec.get("builder_kwargs") or {})
        self.mixed = bool(spec.get("mixed", False))          # Monte-Carlo integration over draws
        self.panel = bool(spec.get("panel", self.mixed))     # likelihood of the whole choice sequence per ID
        self.draws = int(spec.get("draws", 200))
        # "never" skips the Hessian after convergence (no standard errors, BFGS instead of Newton): on LPMC the
        # 54k-row x 100-draw analytical Hessian took longer than the estimation itself
        self.second_derivatives = spec.get("second_derivatives")      # None -> biogeme.toml default ("analytical")
        self.id_col = str(self.df.columns[self.dcfg.ID_col])
        # biogeme databases must be all-numeric and NaN-free
        self._bio_df = self.df.select_dtypes(include=[np.number]).copy()
        if "choice" not in self._bio_df:                     # SYN csv has only the one-hot label columns
            self._bio_df["choice"] = self.y.argmax(axis=1).astype(int)
        self.betas: dict[str, float] | None = None

    def _fit(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        init_seed: int,
        run_dir: Path,
    ) -> dict[str, Any]:
        import os

        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models

        from biogeme.expressions import MonteCarlo, PanelLikelihoodTrajectory, log

        V, av, choice = self.builder(**self.builder_kwargs)
        rows = self._bio_df.iloc[train_idx]
        if self.panel:
            rows = rows.sort_values(self.id_col, kind="stable")     # biogeme panels need contiguous IDs
        database = db.Database("train", rows.reset_index(drop=True))
        kw = {}
        if self.second_derivatives:
            kw["calculating_second_derivatives"] = str(self.second_derivatives)
        if self.mixed:
            kw["number_of_draws"] = self.draws
            if self.panel:
                database.panel(self.id_col)
                logprob = log(MonteCarlo(PanelLikelihoodTrajectory(models.logit(V, av, choice))))
            else:
                logprob = log(MonteCarlo(models.logit(V, av, choice)))
        else:
            logprob = models.loglogit(V, av, choice)

        b = _make_biogeme(database, logprob, **kw)
        b.modelName = self.model_name
        if hasattr(b, "calculate_null_loglikelihood"):
            # biogeme 3.2 reported the null LL by default; 3.3 only if asked
            try:
                b.calculate_null_loglikelihood(av)
            except Exception:
                pass
        # biogeme writes __{modelName}.iter / .toml into the cwd; keep that inside run_dir
        cwd = os.getcwd()
        os.chdir(run_dir)
        try:
            results = b.estimate()
        finally:
            os.chdir(cwd)
        self.betas = dict(results.get_beta_values())

        g = results.get_general_statistics()

        def stat(attr: str, *keys: str) -> float | None:
            # biogeme 3.3: full-precision attributes on the results object;
            # biogeme 3.2: get_general_statistics() -> {key: (value, fmt)};
            # some versions: {key: "formatted string"} (parse as fallback).
            try:
                v = getattr(results, attr, None)
            except Exception:  # biogeme 3.3 raises BiogemeError when a stat is unavailable
                v = None
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
            for key in keys:
                if key in g:
                    raw = g[key]
                    raw = raw[0] if isinstance(raw, (tuple, list)) else raw
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return None
            return None

        return {
            "n_params": len(self.betas),
            "mixed": self.mixed, "panel": self.panel, "draws": self.draws if self.mixed else 0,
            "second_derivatives": self.second_derivatives or "toml default",
            "n_train_ids": int(rows[self.id_col].nunique()) if self.panel else None,
            "final_ll": stat("final_log_likelihood", "Final log likelihood"),
            "null_ll": stat("null_log_likelihood", "Null log likelihood"),
            "rho_bar_squared": stat("rho_bar_square_null", "Rho-bar-square for the null model",
                                    "Rho-square-bar for the null model"),
            "akaike": stat("akaike_information_criterion", "Akaike Information Criterion"),
            "bayesian": stat("bayesian_information_criterion", "Bayesian Information Criterion"),
            "betas": self.betas,
            "note": "MLE is deterministic; init_seed has no effect",
        }

    def _predict_proba(self, idx: np.ndarray) -> np.ndarray:
        assert self.betas is not None, "_predict_proba called before _fit"
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models

        from biogeme.expressions import MonteCarlo

        V, av, _ = self.builder(**self.builder_kwargs)
        database = db.Database("sim", self._bio_df.iloc[idx].reset_index(drop=True))
        # mixed logit: the UNCONDITIONAL choice probability (integrated over the random coefficients) — the test
        # individuals / households are never in the training set (group-wise splits), so no posterior conditioning
        simulate = {f"P_{i}": (MonteCarlo(models.logit(V, av, i)) if self.mixed else models.logit(V, av, i)) for i in sorted(V)}
        b = _make_biogeme(database, simulate, **({"number_of_draws": self.draws} if self.mixed else {}))
        out = b.simulate(the_beta_values=self.betas)
        return out[[f"P_{i}" for i in sorted(V)]].to_numpy(dtype=np.float64)

    def _config_snapshot(self) -> dict[str, Any]:
        return {"backend": "biogeme", **self.spec}
