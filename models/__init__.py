"""Model registry: one line per model, name -> "module:Class".

Lazy: the class is imported only when resolved, so an optional backend that is
not installed (lightgbm, rumboost, biogeme) can never break the other models.
"""
from importlib import import_module

MODEL_REGISTRY: dict[str, str] = {
    "tspline": "models.tspline.model:TSplineDCM",
    "fdnn": "models.fdnn:FDNN",
    "mnl": "models.mnl:MNL",
    "taste_dcm": "models.taste_dcm:TasteDCM",
    "tastenet": "models.tastenet:TasteNet",
    "l_mnl": "models.l_mnl:LMNL",
    "reslogit": "models.reslogit:ResLogit",
}


def resolve_model_class(name: str) -> type:
    try:
        target = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Model class {name!r} not in MODEL_REGISTRY (available: {sorted(MODEL_REGISTRY)})"
        ) from None
    mod, cls = target.split(":")
    return getattr(import_module(mod), cls)


def available() -> list[str]:
    """Registry names whose module actually imports on this machine."""
    out = []
    for n in MODEL_REGISTRY:
        try:
            resolve_model_class(n)
            out.append(n)
        except Exception:
            pass
    return out
