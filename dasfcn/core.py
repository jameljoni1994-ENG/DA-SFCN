"""Lanczos, cubic subproblem, and shared utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import brentq


Array = np.ndarray
HVP = Callable[[Array], Array]


@dataclass
class Trace:
    f: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    m: List[int] = field(default_factory=list)
    n_hvp: List[int] = field(default_factory=list)
    n_grad: List[int] = field(default_factory=list)
    lam_min: List[float] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k in ("f", "grad_norm", "m", "n_hvp", "n_grad", "lam_min", "time"):
                getattr(self, k).append(v)
            else:
                self.extra.setdefault(k, []).append(v)


def lanczos(hvp: HVP, g: Array, m: int, reorth: bool = True, tol: float = 1e-14):
    """Lanczos on (H, g) with optional full reorthogonalization.

    Returns Q (n x m_eff), alpha, beta, beta0=||g||.
    """
    n = g.shape[0]
    m = int(max(1, min(m, n)))
    beta0 = float(np.linalg.norm(g))
    if beta0 < tol:
        return np.zeros((n, 0)), np.zeros(0), np.zeros(0), beta0

    Q = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(max(m - 1, 0))
    q = g / beta0
    Q[:, 0] = q
    w = hvp(q)
    alpha[0] = float(q @ w)
    w = w - alpha[0] * q
    k = 1
    while k < m:
        b = float(np.linalg.norm(w))
        if b < tol * max(1.0, abs(alpha[k - 1])):
            break
        beta[k - 1] = b
        q = w / b
        if reorth:
            q = q - Q[:, :k] @ (Q[:, :k].T @ q)
            nq = float(np.linalg.norm(q))
            if nq < tol:
                break
            q = q / nq
        Q[:, k] = q
        w = hvp(q)
        alpha[k] = float(q @ w)
        w = w - alpha[k] * q - b * Q[:, k - 1]
        if reorth:
            w = w - Q[:, : k + 1] @ (Q[:, : k + 1].T @ w)
        k += 1
    return Q[:, :k], alpha[:k], beta[: k - 1] if k > 1 else np.zeros(0), beta0


def tridiag(alpha: Array, beta: Array) -> Array:
    m = alpha.shape[0]
    T = np.diag(alpha)
    if m > 1:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    return T


def saddle_free(T: Array):
    """Replace T by |T| = U |Λ| U^T. Returns |T|, eigenvalues, eigenvectors."""
    w, U = np.linalg.eigh(T)
    absT = (U * np.abs(w)) @ U.T
    return absT, w, U


def solve_cubic_subspace(g_loc: Array, A: Array, M: float, eps: float = 1e-14) -> Array:
    """Minimize g^T y + 1/2 y^T A y + (M/6)||y||^3 with A ≽ 0."""
    m = A.shape[0]
    if m == 0:
        return np.zeros(0)
    evals, evecs = np.linalg.eigh(0.5 * (A + A.T))
    evals = np.maximum(evals, 0.0)
    ghat = evecs.T @ g_loc
    gn = float(np.linalg.norm(g_loc))
    if gn < eps:
        return np.zeros(m)
    if M <= eps:
        yhat = -ghat / np.maximum(evals, 1e-12)
        return evecs @ yhat

    def ynorm(lam: float) -> float:
        return float(np.sqrt(np.sum((ghat / (evals + lam + 1e-16)) ** 2)))

    def phi(lam: float) -> float:
        return 0.5 * M * ynorm(lam) - lam

    lam_hi = max(1.0, 0.5 * M * gn + 1e-8)
    for _ in range(60):
        try:
            if phi(lam_hi) <= 0:
                break
        except OverflowError:
            break
        lam_hi *= 2.0
        if lam_hi > 1e18:
            break
    try:
        if phi(1e-16) <= 0:
            lam = 0.0
        else:
            lam = float(brentq(phi, 1e-16, lam_hi, xtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError):
        lam = lam_hi
    yhat = -ghat / (evals + lam + 1e-16)
    return evecs @ yhat


def model_value(g_loc: Array, A: Array, y: Array, M: float) -> float:
    return float(g_loc @ y + 0.5 * y @ (A @ y) + (M / 6.0) * np.linalg.norm(y) ** 3)


def dynamic_m(g_norm: float, g0_norm: float, m_min: int, m_max: int, alpha: float) -> int:
    if g_norm <= 0:
        return int(m_max)
    val = m_min + int(np.ceil(alpha * np.log(1.0 + g0_norm / max(g_norm, 1e-30))))
    return int(min(m_max, max(m_min, val)))


def fd_hvp(grad: Callable[[Array], Array], x: Array, v: Array, eps: float = 1e-6) -> Array:
    nv = float(np.linalg.norm(v))
    if nv < 1e-16:
        return np.zeros_like(v)
    e = eps / nv
    return (grad(x + e * v) - grad(x - e * v)) / (2.0 * e)
