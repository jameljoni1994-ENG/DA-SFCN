"""P3 design hooks: optional L-BFGS-style diagonal preconditioner for Lanczos starts."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .core import Array


def lbfgs_diag_preconditioner(
    S: Sequence[Array],
    Y: Sequence[Array],
    eps: float = 1e-8,
) -> Optional[Array]:
    """Return a positive diagonal scaling approximating an inverse-Hessian action.

    This is a *lightweight hook* for future preconditioned Lanczos (WQK → DA-SFCN):
    use ``D^{-1/2}`` to rescale the gradient before building Krylov vectors.
    Heuristic only — not a proved rate replacement for exact Newton.
    """
    if not S or not Y:
        return None
    s, y = S[-1], Y[-1]
    ys = float(y @ s)
    if ys <= eps:
        return None
    # Barrett scaling: diag ~ (s∘s)/(s∘y) style damping
    num = s * s
    den = np.maximum(s * y, eps)
    d = np.maximum(num / den, eps)
    return d


def apply_diag_precond(v: Array, diag: Array) -> Array:
    return v / np.sqrt(np.maximum(diag, 1e-12))


# Theory status notes (also mirrored in docs/OPTIMIZER.md):
# - DA-SFCN global O(k^{-2/3}) and local quadratic under log m_k: stated in unified paper.
# - LK-Newton reuses (Q,|T|) for tau steps: inherits ARC decrease if reuse window short;
#   full local rate under client-scale drift is still open.
# - Fed-DA-SFCN: local cubic steps + FedAvg → neighborhood of stationarity; radius ~ drift.
# - SC-Newton certificate is a backward-error heuristic aligned with Dembo–Eisenstat–Steihaug
#   forcing terms when theta_m is small enough.
