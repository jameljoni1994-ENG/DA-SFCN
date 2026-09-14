"""Optional torch backend smoke (skipped if torch missing)."""

import pytest

from dasfcn.torch_backend import torch_available


@pytest.mark.skipif(not torch_available(), reason="torch not installed")
def test_torch_quadratic_hvp_matches_dense():
    import numpy as np
    from dasfcn.torch_backend import TorchQuadratic

    prob = TorchQuadratic(n=12, seed=0)
    x = prob.x0(1)
    # dense Hessian from finite differences on grad
    g0 = prob.grad(x)
    n = x.size
    H = np.zeros((n, n))
    eps = 1e-6
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        H[:, i] = (prob.grad(x + e) - prob.grad(x - e)) / (2 * eps)
    v = np.random.default_rng(0).normal(size=n)
    hv = prob.hvp(x, v)
    assert np.allclose(hv, H @ v, rtol=1e-4, atol=1e-5)
    assert np.linalg.norm(g0) > 0
