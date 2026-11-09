# experiments/evaluation.py
"""Model-agnostic evaluation: score choice probabilities, no torch required.

Any model that can produce a test-set probability matrix ``P [N, I]`` is
comparable through ``dcm_metrics`` — torch nets, biogeme MLE, boosting, etc.

Metric definitions follow tspline.tspline_trainer.default_dcm_loss, with two
deliberate deviations (documented so numbers can be reconciled):

1. ``nll`` here is the PURE negative log-likelihood. The trainer's
   ``evaluate()`` accumulates ``loss = nll + kld`` under the key "nll", so for
   VAE-type models the trainer's reported eval nll is inflated by the KLD term.
2. ``f1`` here is corpus-level weighted F1. The trainer computes F1 per batch
   and averages batch values (weighted by batch size), which is not the same
   estimator. Corpus-level is the standard definition.

``acc`` uses argmax over P. The trainer argmaxes raw utilities V without the
availability mask; the two differ only if a model ranks an unavailable
alternative highest, which a masked P already rules out.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

EPS = 1e-12
NEG_INF_LOGIT = -1e9  # same constant as tspline.tspline_trainer.masked_log_softmax


def probs_from_utilities(V: np.ndarray, av: np.ndarray | None = None) -> np.ndarray:
    """Masked softmax over utilities, identical to the trainer's masked_log_softmax.

    :param V: utilities [N, I]
    :param av: availability mask [N, I] in {0,1}, or None for all-available
    :return: probabilities [N, I]; unavailable alternatives get probability ~0
    """
    V = np.asarray(V, dtype=np.float64)
    if av is not None:
        V = np.where(np.asarray(av) > 0, V, NEG_INF_LOGIT)
    V = V - V.max(axis=1, keepdims=True)
    expV = np.exp(V)
    return expV / expV.sum(axis=1, keepdims=True)


def dcm_metrics(
    P: np.ndarray,
    y_onehot: np.ndarray,
    av: np.ndarray | None = None,
) -> dict[str, float]:
    """Score choice probabilities against one-hot labels.

    :param P: choice probabilities [N, I] (rows ~sum to 1; unavailable alts ~0)
    :param y_onehot: one-hot chosen alternative [N, I]
    :param av: availability mask [N, I]; consistency check + the rho^2 null model
    :return: {"nll", "acc", "f1", "brier", "gmpca", "rho2", "n"}

    gmpca = exp(-nll): geometric mean probability of the chosen alternative (what the
    RUMBoost paper reports). rho2 = McFadden 1 - LL_model/LL_null with the null =
    equal share among AVAILABLE alternatives (ported from DCM3 stats/metrics.py).
    nll/acc/f1/brier are unchanged from v1 experiments/evaluation.py.
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y_onehot, dtype=np.float64)
    assert P.shape == y.shape, f"P {P.shape} vs y {y.shape}"

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad = int((~np.isclose(row_sums, 1.0, atol=1e-3)).sum())
        raise ValueError(f"{bad} rows of P do not sum to 1 (max dev {np.abs(row_sums - 1).max():.2e})")
    if av is not None:
        leaked = float((P * (np.asarray(av) <= 0)).sum())
        if leaked > 1e-6 * len(P):
            raise ValueError(f"probability mass {leaked:.3e} on unavailable alternatives")

    logP = np.log(np.clip(P, EPS, None))
    nll = float(-(y * logP).sum(axis=1).mean())

    pred = P.argmax(axis=1)
    true = y.argmax(axis=1)
    acc = float((pred == true).mean())
    f1 = float(f1_score(true, pred, average="weighted"))
    brier = float(((P - y) ** 2).sum(axis=1).mean())

    gmpca = float(np.exp(-nll))
    if av is not None:
        navail = np.asarray(av, dtype=np.float64).sum(axis=1)
        navail[navail <= 0] = P.shape[1]
        ll_null = float(-np.log(navail).mean())
    else:
        ll_null = float(-np.log(P.shape[1]))
    rho2 = float(1.0 - (-nll) / ll_null) if ll_null != 0 else float("nan")

    return {"nll": nll, "acc": acc, "f1": f1, "brier": brier, "gmpca": gmpca, "rho2": rho2, "n": int(len(P))}
