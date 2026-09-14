"""Unit tests for Lanczos / cubic / saddle-free core (CPU, seconds)."""

import numpy as np
import pytest

from dasfcn.api import available_methods, minimize
from dasfcn.core import dynamic_m, lanczos, saddle_free, solve_cubic_subspace, tridiag
from dasfcn.problems import QuadraticSaddle, Rosenbrock


def test_lanczos_starts_with_gradient_direction():
    rng = np.random.default_rng(0)
    n = 20
    A = rng.normal(size=(n, n))
    H = 0.5 * (A + A.T) + np.eye(n)
    g = rng.normal(size=n)

    def hvp(v):
        return H @ v

    Q, al, be, beta0 = lanczos(hvp, g, m=8)
    assert Q.shape[1] >= 1
    assert abs(beta0 - np.linalg.norm(g)) < 1e-10
    q1 = Q[:, 0]
    assert abs(abs(q1 @ (g / beta0)) - 1.0) < 1e-8
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)


def test_saddle_free_flips_negative_eigs():
    T = np.diag([2.0, -3.0, 0.5])
    absT, w, _ = saddle_free(T)
    assert np.min(w) < 0
    ew = np.linalg.eigvalsh(absT)
    assert np.all(ew >= -1e-12)
    assert abs(np.max(ew) - 3.0) < 1e-10


def test_cubic_subspace_decrease():
    A = np.diag([1.0, 2.0, 3.0])
    g = np.array([1.0, -0.5, 0.25])
    y = solve_cubic_subspace(g, A, M=1.0)
    assert y.shape == (3,)
    # model at 0 is 0; model at y should be negative for descent direction
    from dasfcn.core import model_value

    assert model_value(g, A, y, 1.0) < 0


def test_dynamic_m_grows_with_accuracy():
    m0 = dynamic_m(1.0, 1.0, m_min=4, m_max=20, alpha=3.0)
    m1 = dynamic_m(1e-4, 1.0, m_min=4, m_max=20, alpha=3.0)
    assert m0 <= m1
    assert m1 <= 20


def test_minimize_api_da_sfcn_smoke():
    prob = QuadraticSaddle(n=30, n_neg=4, quartic=0.08)
    x0 = prob.x0(0)
    tr = minimize(
        prob, x0, method="da-sfcn", maxiter=50, tol=1e-8, m_min=3, m_max=12, alpha=2.5
    )
    assert len(tr.grad_norm) >= 2
    # Escape may temporarily increase ||g||; require eventual stationarity.
    assert tr.grad_norm[-1] < 1e-4
    assert tr.f[-1] < tr.f[0]
    assert "da-sfcn" in available_methods()


def test_minimize_api_unknown_method():
    prob = Rosenbrock(n=10)
    with pytest.raises(ValueError):
        minimize(prob, prob.x0(0), method="not-a-method")
