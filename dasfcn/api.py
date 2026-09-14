"""Unified minimize API over DA-SFCN family and baselines."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

from .algorithms import (
    DA_SFCN,
    Fed_DA_SFCN,
    GradientDescent,
    KrylovCRN,
    LBFGS,
    LK_Newton,
    RandomSubspaceCubic,
    SC_Newton,
    SaddleFreeNewton,
    VR_DA_SFCN,
    WQK_Newton,
)
from .core import Trace

MethodSpec = Union[str, Type, Any]

_REGISTRY: Dict[str, Type] = {
    "da-sfcn": DA_SFCN,
    "dasfcn": DA_SFCN,
    "krylov-crn": KrylovCRN,
    "krylov_crn": KrylovCRN,
    "sfn": SaddleFreeNewton,
    "gd": GradientDescent,
    "lbfgs": LBFGS,
    "l-bfgs": LBFGS,
    "vr-da-sfcn": VR_DA_SFCN,
    "vr_da_sfcn": VR_DA_SFCN,
    "lk-newton": LK_Newton,
    "lk_newton": LK_Newton,
    "wqk-newton": WQK_Newton,
    "wqk_newton": WQK_Newton,
    "fed-da-sfcn": Fed_DA_SFCN,
    "fed_da_sfcn": Fed_DA_SFCN,
    "sc-newton": SC_Newton,
    "sc_newton": SC_Newton,
    "sscn": RandomSubspaceCubic,
    "sscn-coord": RandomSubspaceCubic,
}


def available_methods():
    """Return sorted canonical method names."""
    return sorted(set(_REGISTRY.keys()))


def _resolve(method: MethodSpec):
    if isinstance(method, str):
        key = method.strip().lower()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown method {method!r}. Choose from: {', '.join(available_methods())}"
            )
        return _REGISTRY[key]
    if isinstance(method, type):
        return method
    # already an instance factory-like object with minimize
    if hasattr(method, "minimize"):
        return method
    raise TypeError("method must be a string name, a class, or an optimizer instance")


def minimize(
    problem,
    x0,
    method: MethodSpec = "da-sfcn",
    *,
    maxiter: int = 80,
    tol: float = 1e-8,
    return_optimizer: bool = False,
    **kwargs,
) -> Trace | tuple[Trace, Any]:
    """Run a named optimizer on ``problem`` starting at ``x0``.

    Parameters
    ----------
    problem :
        Object with ``f``, ``grad``, and usually ``hvp``.
    x0 :
        Initial point (array-like).
    method :
        Name such as ``\"da-sfcn\"``, ``\"lk-newton\"``, ``\"lbfgs\"``, or a class/instance.
    maxiter, tol :
        Passed to the runner when constructing from a name/class.
    return_optimizer :
        If True, also return the optimizer instance.
    **kwargs :
        Extra constructor arguments (e.g. ``m_min``, ``m_max``, ``alpha``).
    """
    resolved = _resolve(method)
    if isinstance(resolved, type):
        opt = resolved(maxiter=maxiter, tol=tol, **kwargs)
    else:
        opt = resolved
        for k, v in {"maxiter": maxiter, "tol": tol, **kwargs}.items():
            if hasattr(opt, k):
                setattr(opt, k, v)
    tr = opt.minimize(problem, x0)
    if return_optimizer:
        return tr, opt
    return tr
