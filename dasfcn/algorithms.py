"""Family of second-order algorithms: DA-SFCN and extensions plus baselines."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .core import (
    Trace,
    dynamic_m,
    lanczos,
    model_value,
    saddle_free,
    solve_cubic_subspace,
    tridiag,
)


def _make_hvp(problem, x):
    return lambda v: problem.hvp(x, v)


class _Runner:
    name = "base"

    def __init__(self, maxiter: int = 80, tol: float = 1e-8, verbose: bool = False):
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    def _record(self, tr: Trace, problem, x, n_hvp, n_grad, m, t0, lam_min=0.0, **extra):
        g = problem.grad(x)
        n_grad += 1
        gn = float(np.linalg.norm(g))
        tr.append(
            f=float(problem.f(x)),
            grad_norm=gn,
            m=int(m),
            n_hvp=int(n_hvp),
            n_grad=int(n_grad),
            lam_min=float(lam_min),
            time=time.perf_counter() - t0,
            **extra,
        )
        return g, gn, n_grad


class DA_SFCN(_Runner):
    name = "DA-SFCN"

    def __init__(
        self,
        m_min: int = 4,
        m_max: int = 24,
        alpha: float = 3.0,
        M0: float = 1.0,
        eta1: float = 0.1,
        eta2: float = 0.75,
        gamma1: float = 2.0,
        gamma2: float = 0.5,
        saddle_free_on: bool = True,
        dynamic: bool = True,
        fixed_m: Optional[int] = None,
        **kw,
    ):
        super().__init__(**kw)
        self.m_min = m_min
        self.m_max = m_max
        self.alpha = alpha
        self.M0 = M0
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.saddle_free_on = saddle_free_on
        self.dynamic = dynamic
        self.fixed_m = fixed_m

    def choose_m(self, gn, g0, n):
        if not self.dynamic:
            m = self.fixed_m if self.fixed_m is not None else self.m_max
            return int(min(m, n))
        return min(n, dynamic_m(gn, g0, self.m_min, min(self.m_max, n), self.alpha))

    def step(self, problem, x, g, M, m):
        hvp_count = [0]

        def hvp(v):
            hvp_count[0] += 1
            return problem.hvp(x, v)

        Q, al, be, beta0 = lanczos(hvp, g, m)
        n_hvp = hvp_count[0]
        if Q.shape[1] == 0:
            return x, M, n_hvp, 0.0, m, False
        T = tridiag(al, be)
        if self.saddle_free_on:
            A, w, _ = saddle_free(T)
        else:
            A, w, _ = T, np.linalg.eigvalsh(T), None
            # shift to keep cubic model bounded below
            shift = max(0.0, -np.min(w) + 1e-8)
            A = T + shift * np.eye(T.shape[0])
        g_loc = Q.T @ g
        y = solve_cubic_subspace(g_loc, A, M)
        s = Q @ y
        pred = -model_value(g_loc, A, y, M)
        fx = problem.f(x)
        fnew = problem.f(x + s)
        actual = fx - fnew
        rho = actual / max(pred, 1e-16) if pred > 0 else -1.0
        lam_min = float(np.min(w)) if w is not None and len(w) else 0.0
        if rho >= self.eta1 and pred > 0:
            x_new = x + s
            M_new = M * self.gamma2 if rho >= self.eta2 else M
            M_new = float(np.clip(M_new, 1e-8, 1e8))
            return x_new, M_new, n_hvp, lam_min, Q.shape[1], True
        M_new = float(np.clip(M * self.gamma1, 1e-8, 1e8))
        return x, M_new, n_hvp, lam_min, Q.shape[1], False

    def minimize(self, problem, x0: np.ndarray) -> Trace:
        x = np.array(x0, dtype=float)
        tr = Trace()
        M = self.M0
        n_hvp = 0
        n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        g0 = gn
        for _ in range(self.maxiter):
            if gn < self.tol:
                break
            m = self.choose_m(gn, g0, x.size)
            x, M, dh, lam_min, m_eff, _ok = self.step(problem, x, g, M, m)
            n_hvp += dh
            g, gn, n_grad = self._record(
                tr, problem, x, n_hvp, n_grad, m_eff, t0, lam_min=lam_min
            )
        return tr


class KrylovCRN(DA_SFCN):
    name = "Krylov-CRN"

    def __init__(self, m: int = 16, **kw):
        super().__init__(saddle_free_on=False, dynamic=False, fixed_m=m, **kw)


class SaddleFreeNewton(_Runner):
    name = "SFN"

    def __init__(self, m: int = 16, damping: float = 1e-4, **kw):
        super().__init__(**kw)
        self.m = m
        self.damping = damping

    def minimize(self, problem, x0):
        x = np.array(x0, dtype=float)
        tr = Trace()
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        for _ in range(self.maxiter):
            if gn < self.tol:
                break
            hcount = [0]

            def hvp(v):
                hcount[0] += 1
                return problem.hvp(x, v)

            Q, al, be, _ = lanczos(hvp, g, min(self.m, x.size))
            n_hvp += hcount[0]
            if Q.shape[1] == 0:
                break
            T = tridiag(al, be)
            A, w, _ = saddle_free(T)
            A = A + self.damping * np.eye(A.shape[0])
            y = -np.linalg.solve(A, Q.T @ g)
            s = Q @ y
            # Armijo
            fx = problem.f(x)
            step = 1.0
            for _ls in range(20):
                if problem.f(x + step * s) <= fx - 1e-4 * step * abs(g @ s):
                    break
                step *= 0.5
            x = x + step * s
            g, gn, n_grad = self._record(
                tr, problem, x, n_hvp, n_grad, Q.shape[1], t0, lam_min=float(np.min(w))
            )
        return tr


class CubicNewton(DA_SFCN):
    """Full cubic Newton using dense Hessian via n HVPs (small-n only)."""

    name = "CRN-full"

    def __init__(self, **kw):
        super().__init__(dynamic=False, saddle_free_on=False, **kw)

    def choose_m(self, gn, g0, n):
        return int(n)


class GradientDescent(_Runner):
    name = "GD"

    def __init__(self, lr: float = 0.05, **kw):
        super().__init__(**kw)
        self.lr = lr

    def minimize(self, problem, x0):
        x = np.array(x0, dtype=float)
        tr = Trace()
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        lr = self.lr
        for _ in range(self.maxiter):
            if gn < self.tol:
                break
            fx = problem.f(x)
            step = lr
            for _ls in range(12):
                if problem.f(x - step * g) <= fx - 1e-4 * step * (gn ** 2):
                    break
                step *= 0.5
            x = x - step * g
            g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        return tr


class LBFGS(_Runner):
    name = "L-BFGS"

    def __init__(self, mem: int = 10, **kw):
        super().__init__(**kw)
        self.mem = mem

    def minimize(self, problem, x0):
        x = np.array(x0, dtype=float)
        tr = Trace()
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        S, Y = [], []
        I = np.eye(x.size)
        for _ in range(self.maxiter):
            if gn < self.tol:
                break
            q = g.copy()
            rho, alpha = [], []
            for s, y in zip(reversed(S), reversed(Y)):
                r = 1.0 / max(y @ s, 1e-16)
                rho.append(r)
                a = r * (s @ q)
                alpha.append(a)
                q = q - a * y
            if S:
                gamma = (S[-1] @ Y[-1]) / max(Y[-1] @ Y[-1], 1e-16)
            else:
                gamma = 1.0 / max(gn, 1.0)
            rvec = gamma * q
            for s, y, r, a in zip(S, Y, reversed(rho), reversed(alpha)):
                b = r * (y @ rvec)
                rvec = rvec + s * (a - b)
            p = -rvec
            fx = problem.f(x)
            step = 1.0
            gp = float(g @ p)
            if gp >= 0:
                p = -g
                gp = float(g @ p)
            for _ls in range(20):
                if problem.f(x + step * p) <= fx + 1e-4 * step * gp:
                    break
                step *= 0.5
            s = step * p
            x_new = x + s
            g_new = problem.grad(x_new)
            n_grad += 1
            y = g_new - g
            if y @ s > 1e-12:
                S.append(s)
                Y.append(y)
                if len(S) > self.mem:
                    S.pop(0)
                    Y.pop(0)
            x, g = x_new, g_new
            gn = float(np.linalg.norm(g))
            tr.append(
                f=float(problem.f(x)),
                grad_norm=gn,
                m=0,
                n_hvp=n_hvp,
                n_grad=n_grad,
                lam_min=0.0,
                time=time.perf_counter() - t0,
            )
        return tr


class RandomSubspaceCubic(DA_SFCN):
    name = "SSCN-coord"

    def __init__(self, tau: int = 12, **kw):
        super().__init__(**kw)
        self.tau = tau
        self.rng = np.random.default_rng(0)

    def step(self, problem, x, g, M, m):
        n = x.size
        tau = min(self.tau, n)
        idx = self.rng.choice(n, size=tau, replace=False)
        Q = np.zeros((n, tau))
        Q[idx, np.arange(tau)] = 1.0
        # projected Hessian via HVPs on basis
        T = np.zeros((tau, tau))
        n_hvp = 0
        for j in range(tau):
            hv = problem.hvp(x, Q[:, j])
            n_hvp += 1
            T[:, j] = Q.T @ hv
        T = 0.5 * (T + T.T)
        A, w, _ = saddle_free(T) if self.saddle_free_on else (T + 1e-8 * np.eye(tau), np.linalg.eigvalsh(T), None)
        if not self.saddle_free_on:
            shift = max(0.0, -np.min(w) + 1e-8)
            A = T + shift * np.eye(tau)
        y = solve_cubic_subspace(Q.T @ g, A, M)
        s = Q @ y
        pred = -model_value(Q.T @ g, A, y, M)
        fx = problem.f(x)
        actual = fx - problem.f(x + s)
        rho = actual / max(pred, 1e-16) if pred > 0 else -1.0
        lam_min = float(np.min(w)) if len(w) else 0.0
        if rho >= self.eta1 and pred > 0:
            M_new = M * self.gamma2 if rho >= self.eta2 else M
            return x + s, float(np.clip(M_new, 1e-8, 1e8)), n_hvp, lam_min, tau, True
        return x, float(np.clip(M * self.gamma1, 1e-8, 1e8)), n_hvp, lam_min, tau, False


class VR_DA_SFCN(DA_SFCN):
    """SVRG-style variance-reduced DA-SFCN for finite-sum problems."""

    name = "VR-DA-SFCN"

    def __init__(self, batch: int = 32, snapshot_every: int = 5, **kw):
        super().__init__(**kw)
        self.batch = batch
        self.snapshot_every = snapshot_every

    def minimize(self, problem, x0):
        x = np.array(x0, dtype=float)
        tr = Trace()
        M = self.M0
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        g0 = gn
        x_snap = x.copy()
        g_snap = g.copy()
        n_terms = getattr(problem, "n_terms", getattr(problem, "n_samples", 100))
        rng = np.random.default_rng(0)
        for k in range(self.maxiter):
            if gn < self.tol:
                break
            if k % self.snapshot_every == 0:
                x_snap = x.copy()
                g_snap = problem.grad(x_snap)
                n_grad += 1
            idx = rng.choice(n_terms, size=min(self.batch, n_terms), replace=False)
            g_b = problem.grad_i(x, idx)
            g_b_snap = problem.grad_i(x_snap, idx)
            n_grad += 2
            g_vr = g_b - g_b_snap + g_snap
            m = self.choose_m(float(np.linalg.norm(g_vr)), g0, x.size)

            hcount = [0]

            def hvp(v):
                hcount[0] += 1
                hv_b = problem.hvp_i(x, v, idx)
                hv_s = problem.hvp_i(x_snap, v, idx)
                return hv_b - hv_s + problem.hvp(x_snap, v)

            Q, al, be, _ = lanczos(hvp, g_vr, m)
            n_hvp += hcount[0]
            if Q.shape[1] == 0:
                break
            T = tridiag(al, be)
            A, w, _ = saddle_free(T)
            y = solve_cubic_subspace(Q.T @ g_vr, A, M)
            s = Q @ y
            pred = -model_value(Q.T @ g_vr, A, y, M)
            actual = problem.f(x) - problem.f(x + s)
            rho = actual / max(pred, 1e-16) if pred > 0 else -1.0
            if rho >= self.eta1 and pred > 0:
                x = x + s
                M = float(np.clip(M * (self.gamma2 if rho >= self.eta2 else 1.0), 1e-8, 1e8))
            else:
                M = float(np.clip(M * self.gamma1, 1e-8, 1e8))
            g, gn, n_grad = self._record(
                tr, problem, x, n_hvp, n_grad, Q.shape[1], t0, lam_min=float(np.min(w))
            )
        return tr


class LK_Newton(DA_SFCN):
    """Lazy Krylov-Newton: reuse Lanczos basis for tau inner steps."""

    name = "LK-Newton"

    def __init__(self, reuse: int = 4, **kw):
        super().__init__(**kw)
        self.reuse = reuse

    def minimize(self, problem, x0):
        x = np.array(x0, dtype=float)
        tr = Trace()
        M = self.M0
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        g0 = gn
        Q = None
        A = None
        reuse_left = 0
        for _ in range(self.maxiter):
            if gn < self.tol:
                break
            m = self.choose_m(gn, g0, x.size)
            if reuse_left <= 0 or Q is None or Q.shape[1] != m:
                hcount = [0]

                def hvp(v):
                    hcount[0] += 1
                    return problem.hvp(x, v)

                Q, al, be, _ = lanczos(hvp, g, m)
                n_hvp += hcount[0]
                T = tridiag(al, be)
                A, w, _ = saddle_free(T)
                reuse_left = self.reuse
            else:
                w = np.linalg.eigvalsh(A)
            if Q.shape[1] == 0:
                break
            g_loc = Q.T @ g
            y = solve_cubic_subspace(g_loc, A, M)
            s = Q @ y
            pred = -model_value(g_loc, A, y, M)
            actual = problem.f(x) - problem.f(x + s)
            rho = actual / max(pred, 1e-16) if pred > 0 else -1.0
            if rho >= self.eta1 and pred > 0:
                x = x + s
                M = float(np.clip(M * (self.gamma2 if rho >= self.eta2 else 1.0), 1e-8, 1e8))
                reuse_left -= 1
            else:
                M = float(np.clip(M * self.gamma1, 1e-8, 1e8))
                reuse_left = 0
            g, gn, n_grad = self._record(
                tr, problem, x, n_hvp, n_grad, Q.shape[1], t0, lam_min=float(np.min(w))
            )
        return tr


class WQK_Newton(_Runner):
    """L-BFGS warmup then DA-SFCN (optionally L-BFGS-preconditioned Lanczos)."""

    name = "WQK-Newton"

    def __init__(self, warmup: int = 12, **kw):
        super().__init__(**kw)
        self.warmup = warmup
        self.lbfgs = LBFGS(maxiter=warmup, tol=self.tol)
        self.dasfcn = DA_SFCN(maxiter=self.maxiter, tol=self.tol)

    def minimize(self, problem, x0):
        # Run L-BFGS for warmup iterations by hijacking maxiter, then continue.
        x = np.array(x0, dtype=float)
        x = np.array(x0, dtype=float)
        dummy = LBFGS(maxiter=self.warmup, tol=self.tol)
        # intercept: copy minimize but keep x
        tr = Trace()
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = dummy._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        S, Y = [], []
        for k in range(self.warmup):
            if gn < self.tol:
                return tr
            q = g.copy()
            rho, alpha = [], []
            for s, y in zip(reversed(S), reversed(Y)):
                r = 1.0 / max(y @ s, 1e-16)
                rho.append(r)
                a = r * (s @ q)
                alpha.append(a)
                q = q - a * y
            gamma = (S[-1] @ Y[-1]) / max(Y[-1] @ Y[-1], 1e-16) if S else 1.0 / max(gn, 1.0)
            rvec = gamma * q
            for s, y, r, a in zip(S, Y, reversed(rho), reversed(alpha)):
                b = r * (y @ rvec)
                rvec = rvec + s * (a - b)
            p = -rvec
            fx = problem.f(x)
            step = 1.0
            gp = float(g @ p)
            if gp >= 0:
                p = -g
                gp = float(g @ p)
            for _ls in range(20):
                if problem.f(x + step * p) <= fx + 1e-4 * step * gp:
                    break
                step *= 0.5
            s = step * p
            x = x + s
            g_new = problem.grad(x)
            n_grad += 1
            y = g_new - g
            if y @ s > 1e-12:
                S.append(s)
                Y.append(y)
                if len(S) > 10:
                    S.pop(0)
                    Y.pop(0)
            g = g_new
            gn = float(np.linalg.norm(g))
            tr.append(
                f=float(problem.f(x)),
                grad_norm=gn,
                m=0,
                n_hvp=n_hvp,
                n_grad=n_grad,
                lam_min=0.0,
                time=time.perf_counter() - t0,
                phase=0,
            )
        inner = DA_SFCN(maxiter=self.maxiter - self.warmup, tol=self.tol)
        tr2 = inner.minimize(problem, x)
        # concat with shifted costs
        t_off = tr.time[-1] if tr.time else 0.0
        h_off = tr.n_hvp[-1] if tr.n_hvp else 0
        g_off = tr.n_grad[-1] if tr.n_grad else 0
        for i in range(1, len(tr2.f)):
            tr.append(
                f=tr2.f[i],
                grad_norm=tr2.grad_norm[i],
                m=tr2.m[i],
                n_hvp=tr2.n_hvp[i] + h_off,
                n_grad=tr2.n_grad[i] + g_off,
                lam_min=tr2.lam_min[i],
                time=tr2.time[i] + t_off,
                phase=1,
            )
        return tr


class Fed_DA_SFCN(_Runner):
    name = "Fed-DA-SFCN"

    def __init__(self, n_clients: int = 8, local_steps: int = 3, participate: float = 1.0, **kw):
        super().__init__(**kw)
        self.n_clients = n_clients
        self.local_steps = local_steps
        self.participate = participate
        self.local = DA_SFCN(maxiter=local_steps, tol=0.0, m_min=3, m_max=12, alpha=2.0)

    def minimize(self, problem, x0):
        if not hasattr(problem, "split_clients"):
            raise ValueError("Fed-DA-SFCN needs a problem with split_clients")
        chunks = problem.split_clients(self.n_clients)
        x = np.array(x0, dtype=float)
        tr = Trace()
        n_hvp = n_grad = 0
        t0 = time.perf_counter()
        g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, 0, t0)
        rng = np.random.default_rng(0)

        class LocalView:
            def __init__(self, parent, idx):
                self.parent = parent
                self.idx = idx
                self.name = "local"

            def f(self, w):
                return float(np.mean(
                    np.log1p(np.exp(np.clip(self.parent.X[self.idx] @ w, -40, 40)))
                    - self.parent.y[self.idx] * (self.parent.X[self.idx] @ w)
                ) + 0.5 * self.parent.alpha * np.sum((w ** 2) / (1.0 + w ** 2)))

            def grad(self, w):
                return self.parent.grad_i(w, self.idx)

            def hvp(self, w, v):
                return self.parent.hvp_i(w, v, self.idx)

        for _round in range(self.maxiter):
            if gn < self.tol:
                break
            chosen = [i for i in range(self.n_clients) if rng.random() <= self.participate]
            if not chosen:
                chosen = [0]
            acc = np.zeros_like(x)
            dh = dg = 0
            for i in chosen:
                loc = LocalView(problem, chunks[i])
                inner = DA_SFCN(maxiter=self.local_steps, tol=0.0, m_min=3, m_max=10, alpha=2.0, M0=1.0)
                # run local steps, recover x by a custom loop
                xi = x.copy()
                gi = loc.grad(xi)
                dg += 1
                Mi = 1.0
                g0i = float(np.linalg.norm(gi))
                for _ls in range(self.local_steps):
                    m = inner.choose_m(float(np.linalg.norm(gi)), max(g0i, 1e-12), xi.size)
                    xi, Mi, dhi, _lam, _me, _ok = inner.step(loc, xi, gi, Mi, m)
                    dh += dhi
                    gi = loc.grad(xi)
                    dg += 1
                acc += xi
            x = acc / len(chosen)
            n_hvp += dh
            n_grad += dg
            g, gn, n_grad = self._record(tr, problem, x, n_hvp, n_grad, self.local_steps, t0)
        return tr


class SC_Newton(DA_SFCN):
    """Spectral-certificate Newton: stop Lanczos when Ritz residual is small."""

    name = "SC-Newton"

    def __init__(self, theta: float = 1e-2, **kw):
        super().__init__(**kw)
        self.theta = theta

    def step(self, problem, x, g, M, m):
        # grow Lanczos until certificate or m
        hcount = [0]

        def hvp(v):
            hcount[0] += 1
            return problem.hvp(x, v)

        # incremental: try increasing dimensions
        m_try = max(self.m_min, 2)
        Q = al = be = None
        w = np.array([0.0])
        last_m = -1
        while m_try <= m:
            Q, al, be, _ = lanczos(hvp, g, m_try)
            if Q.shape[1] == 0:
                return x, M, hcount[0], 0.0, 0, False
            T = tridiag(al, be)
            w, U = np.linalg.eigh(T)
            if Q.shape[1] >= 2 and len(be) >= 1:
                cert = abs(be[-1]) * np.max(np.abs(U[-1, :]))
            else:
                cert = 0.0
            rel = cert / max(np.max(np.abs(w)), 1e-12)
            if rel <= self.theta:
                break
            if Q.shape[1] <= last_m or Q.shape[1] >= m:
                break
            last_m = Q.shape[1]
            m_try = min(m, m_try + 2)
        T = tridiag(al, be)
        if self.saddle_free_on:
            A, w, _ = saddle_free(T)
        else:
            A = T
            w = np.linalg.eigvalsh(T)
            A = T + max(0.0, -np.min(w) + 1e-8) * np.eye(T.shape[0])
        y = solve_cubic_subspace(Q.T @ g, A, M)
        s = Q @ y
        pred = -model_value(Q.T @ g, A, y, M)
        actual = problem.f(x) - problem.f(x + s)
        rho = actual / max(pred, 1e-16) if pred > 0 else -1.0
        lam_min = float(np.min(w))
        n_hvp = hcount[0]
        if rho >= self.eta1 and pred > 0:
            M_new = M * self.gamma2 if rho >= self.eta2 else M
            return x + s, float(np.clip(M_new, 1e-8, 1e8)), n_hvp, lam_min, Q.shape[1], True
        return x, float(np.clip(M * self.gamma1, 1e-8, 1e8)), n_hvp, lam_min, Q.shape[1], False
