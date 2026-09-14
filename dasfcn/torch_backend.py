"""Optional PyTorch CPU HVP helper (install: pip install dasfcn[torch])."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class TorchQuadratic:
    """Tiny strongly convex quadratic for CPU smoke tests with autograd HVP."""

    name = "torch_quadratic"

    def __init__(self, n: int = 16, seed: int = 0):
        if not torch_available():
            raise ImportError("PyTorch is not installed. pip install 'dasfcn[torch]'")
        import torch

        self.torch = torch
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, n))
        H = A.T @ A / n + 0.5 * np.eye(n)
        self.H = torch.tensor(H, dtype=torch.float64)
        self.xs = torch.tensor(rng.normal(size=n), dtype=torch.float64)
        self._n = n

    def dim(self) -> int:
        return self._n

    def x0(self, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.xs.numpy() + 0.3 * rng.normal(size=self._n)

    def _t(self, x: np.ndarray):
        return self.torch.tensor(np.asarray(x, dtype=float), dtype=self.torch.float64, requires_grad=True)

    def f(self, x: np.ndarray) -> float:
        xt = self._t(x)
        d = xt - self.xs
        val = 0.5 * d @ (self.H @ d)
        return float(val.detach().cpu())

    def grad(self, x: np.ndarray) -> np.ndarray:
        xt = self._t(x)
        d = xt - self.xs
        val = 0.5 * d @ (self.H @ d)
        (g,) = self.torch.autograd.grad(val, xt)
        return g.detach().cpu().numpy()

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product via autograd on the gradient."""
        xt = self._t(x)
        d = xt - self.xs
        val = 0.5 * d @ (self.H @ d)
        (g,) = self.torch.autograd.grad(val, xt, create_graph=True)
        vt = self.torch.tensor(np.asarray(v, dtype=float), dtype=self.torch.float64)
        (hv,) = self.torch.autograd.grad(g, xt, grad_outputs=vt, retain_graph=False)
        return hv.detach().cpu().numpy()


def make_torch_hvp(loss_fn: Callable, x_np: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``v |-> H(x) v`` for a scalar ``loss_fn(x_tensor)`` at NumPy point ``x_np``."""
    if not torch_available():
        raise ImportError("PyTorch is not installed. pip install 'dasfcn[torch]'")
    import torch

    x = torch.tensor(np.asarray(x_np, dtype=float), dtype=torch.float64, requires_grad=True)
    loss = loss_fn(x)
    (g,) = torch.autograd.grad(loss, x, create_graph=True)

    def hvp(v: np.ndarray) -> np.ndarray:
        vt = torch.tensor(np.asarray(v, dtype=float), dtype=torch.float64)
        (hv,) = torch.autograd.grad(g, x, grad_outputs=vt, retain_graph=True)
        return hv.detach().cpu().numpy()

    return hvp
