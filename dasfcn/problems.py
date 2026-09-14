"""Benchmark problems with gradients and Hessian-vector products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .core import fd_hvp


class Problem:
    name: str = "problem"

    def f(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return fd_hvp(self.grad, x, v)

    def dim(self) -> int:
        raise NotImplementedError


@dataclass
class Rosenbrock(Problem):
    n: int = 50
    name: str = "rosenbrock"

    def dim(self) -> int:
        return self.n

    def x0(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(scale=0.5, size=self.n)

    def f(self, x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        g = np.zeros(n)
        g[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
        for i in range(1, n - 1):
            g[i] = (
                200.0 * (x[i] - x[i - 1] ** 2)
                - 400.0 * x[i] * (x[i + 1] - x[i] ** 2)
                - 2.0 * (1.0 - x[i])
            )
        g[-1] = 200.0 * (x[-1] - x[-2] ** 2)
        return g


@dataclass
class QuadraticSaddle(Problem):
    """f(x) = 1/2 x^T D x with mixed-sign spectrum, plus a weak quartic well."""

    n: int = 80
    n_neg: int = 8
    quartic: float = 0.05
    name: str = "quadratic_saddle"

    def __post_init__(self):
        d = np.ones(self.n)
        d[: self.n_neg] = -np.linspace(0.5, 2.0, self.n_neg)
        d[self.n_neg :] = np.linspace(0.3, 4.0, self.n - self.n_neg)
        self.d = d

    def dim(self) -> int:
        return self.n

    def x0(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(scale=0.02, size=self.n)
        x[: self.n_neg] *= 0.1
        return x

    def f(self, x: np.ndarray) -> float:
        return float(0.5 * np.sum(self.d * x * x) + 0.25 * self.quartic * np.sum(x ** 4))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.d * x + self.quartic * x ** 3

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.d * v + 3.0 * self.quartic * (x ** 2) * v


@dataclass
class MonkeySaddle(Problem):
    name: str = "monkey_saddle"

    def dim(self) -> int:
        return 2

    def x0(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(scale=0.15, size=2)

    def f(self, x: np.ndarray) -> float:
        a, b = x
        return float(a ** 3 - 3.0 * a * b ** 2 + 0.15 * (a ** 2 + b ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        a, b = x
        return np.array(
            [3.0 * a ** 2 - 3.0 * b ** 2 + 0.3 * a, -6.0 * a * b + 0.3 * b],
            dtype=float,
        )

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        a, b = x
        H = np.array([[6.0 * a + 0.3, -6.0 * b], [-6.0 * b, -6.0 * a + 0.3]])
        return H @ v

    def grid(self, lim: float = 1.4, n: int = 180):
        t = np.linspace(-lim, lim, n)
        X, Y = np.meshgrid(t, t)
        Z = X ** 3 - 3.0 * X * Y ** 2 + 0.15 * (X ** 2 + Y ** 2)
        return X, Y, Z


@dataclass
class NonconvexLogistic(Problem):
    n_samples: int = 400
    n_features: int = 60
    alpha: float = 0.4
    seed: int = 0
    name: str = "nonconvex_logistic"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.X = rng.normal(size=(self.n_samples, self.n_features))
        self.X /= np.sqrt(self.n_features)
        w_true = rng.normal(size=self.n_features)
        w_true /= np.linalg.norm(w_true)
        logits = self.X @ w_true
        p = 1.0 / (1.0 + np.exp(-logits))
        self.y = (rng.random(self.n_samples) < p).astype(float)
        self._n = self.n_features

    def dim(self) -> int:
        return self._n

    def x0(self, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return 0.3 * rng.normal(size=self._n)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def _reg_grad(self, w: np.ndarray) -> np.ndarray:
        den = (1.0 + w ** 2) ** 2
        return self.alpha * w / den

    def _reg_hvp(self, w: np.ndarray, v: np.ndarray) -> np.ndarray:
        num = 1.0 - 3.0 * w ** 2
        den = (1.0 + w ** 2) ** 3
        return self.alpha * (num / den) * v

    def f(self, w: np.ndarray) -> float:
        z = self.X @ w
        z = np.clip(z, -40, 40)
        loss = np.mean(np.log1p(np.exp(z)) - self.y * z)
        reg = 0.5 * self.alpha * np.sum((w ** 2) / (1.0 + w ** 2))
        return float(loss + reg)

    def grad(self, w: np.ndarray) -> np.ndarray:
        p = self._sigmoid(self.X @ w)
        g = self.X.T @ (p - self.y) / self.n_samples
        return g + self._reg_grad(w)

    def hvp(self, w: np.ndarray, v: np.ndarray) -> np.ndarray:
        p = self._sigmoid(self.X @ w)
        d = p * (1.0 - p)
        return (self.X.T @ (d * (self.X @ v))) / self.n_samples + self._reg_hvp(w, v)

    def grad_i(self, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
        Xi = self.X[idx]
        yi = self.y[idx]
        p = self._sigmoid(Xi @ w)
        g = Xi.T @ (p - yi) / max(len(idx), 1)
        return g + self._reg_grad(w)

    def hvp_i(self, w: np.ndarray, v: np.ndarray, idx: np.ndarray) -> np.ndarray:
        Xi = self.X[idx]
        p = self._sigmoid(Xi @ w)
        d = p * (1.0 - p)
        return (Xi.T @ (d * (Xi @ v))) / max(len(idx), 1) + self._reg_hvp(w, v)

    def split_clients(self, n_clients: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_samples)
        chunks = np.array_split(perm, n_clients)
        return chunks


@dataclass
class MatrixFactorization(Problem):
    n_rows: int = 25
    n_cols: int = 20
    rank: int = 4
    seed: int = 0
    name: str = "matrix_factorization"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        U = rng.normal(size=(self.n_rows, self.rank))
        V = rng.normal(size=(self.n_cols, self.rank))
        self.M = U @ V.T
        self._n = (self.n_rows + self.n_cols) * self.rank

    def dim(self) -> int:
        return self._n

    def pack(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        return np.concatenate([U.ravel(), V.ravel()])

    def unpack(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.rank
        nu = self.n_rows * r
        U = x[:nu].reshape(self.n_rows, r)
        V = x[nu:].reshape(self.n_cols, r)
        return U, V

    def x0(self, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return 0.1 * rng.normal(size=self._n)

    def f(self, x: np.ndarray) -> float:
        U, V = self.unpack(x)
        R = U @ V.T - self.M
        return 0.5 * float(np.sum(R ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        U, V = self.unpack(x)
        R = U @ V.T - self.M
        gU = R @ V
        gV = R.T @ U
        return self.pack(gU, gV)


@dataclass
class FiniteSumQuadratics(Problem):
    """Heterogeneous finite-sum with some indefinite local Hessians."""

    n_terms: int = 40
    n: int = 50
    seed: int = 0
    name: str = "finite_sum_quadratics"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.A = np.zeros((self.n_terms, self.n, self.n))
        self.b = rng.normal(size=(self.n_terms, self.n))
        for i in range(self.n_terms):
            G = rng.normal(size=(self.n, self.n))
            S = 0.5 * (G + G.T) / np.sqrt(self.n)
            if i % 5 == 0:
                S -= 0.6 * np.eye(self.n)
            else:
                S += 0.4 * np.eye(self.n)
            self.A[i] = S
        self._n = self.n

    def dim(self) -> int:
        return self._n

    def x0(self, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(scale=0.5, size=self.n)

    def f(self, x: np.ndarray) -> float:
        val = 0.0
        for i in range(self.n_terms):
            val += 0.5 * x @ (self.A[i] @ x) + self.b[i] @ x
        return float(val / self.n_terms + 0.05 * np.sum(x ** 4) / 4.0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        g = np.zeros(self.n)
        for i in range(self.n_terms):
            g += self.A[i] @ x + self.b[i]
        return g / self.n_terms + 0.05 * x ** 3

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        hv = np.zeros(self.n)
        for i in range(self.n_terms):
            hv += self.A[i] @ v
        return hv / self.n_terms + 0.15 * (x ** 2) * v

    def grad_i(self, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
        g = np.zeros(self.n)
        for i in idx:
            g += self.A[i] @ x + self.b[i]
        return g / max(len(idx), 1) + 0.05 * x ** 3

    def hvp_i(self, x: np.ndarray, v: np.ndarray, idx: np.ndarray) -> np.ndarray:
        hv = np.zeros(self.n)
        for i in idx:
            hv += self.A[i] @ v
        return hv / max(len(idx), 1) + 0.15 * (x ** 2) * v
