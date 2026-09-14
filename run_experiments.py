"""Run the DA-SFCN experimental suite and export publication figures."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dasfcn.algorithms import (
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
from dasfcn.core import dynamic_m, lanczos, saddle_free, solve_cubic_subspace, tridiag
from dasfcn.problems import (
    FiniteSumQuadratics,
    MatrixFactorization,
    MonkeySaddle,
    NonconvexLogistic,
    QuadraticSaddle,
    Rosenbrock,
)

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
    }
)

COLORS = {
    "GD": "#7f8c8d",
    "L-BFGS": "#8e44ad",
    "SFN": "#e67e22",
    "Krylov-CRN": "#2980b9",
    "SSCN-coord": "#16a085",
    "DA-SFCN": "#c0392b",
    "DA-SFCN (no SF)": "#e74c3c",
    "DA-SFCN (fixed m)": "#d35400",
    "VR-DA-SFCN": "#27ae60",
    "LK-Newton": "#2c3e50",
    "WQK-Newton": "#1abc9c",
    "Fed-DA-SFCN": "#9b59b6",
    "SC-Newton": "#34495e",
    "FedAvg-GD": "#95a5a6",
}


def savefig(name: str):
    pdf = FIG / f"{name}.pdf"
    png = FIG / f"{name}.png"
    plt.savefig(pdf)
    plt.savefig(png)
    plt.close()
    print("wrote", pdf.name)


def run_path(method, problem, x0, maxiter=60):
    """Return (trace, xs) by wrapping problem.f to keep iterates via a proxy."""
    xs = [np.array(x0, dtype=float)]
    orig = method.minimize

    # We cannot easily hook x; reimplement a thin path tracker for DA-like methods.
    class PathProblem:
        def __init__(self, inner):
            self.inner = inner
            self.name = inner.name

        def f(self, x):
            return self.inner.f(x)

        def grad(self, x):
            return self.inner.grad(x)

        def hvp(self, x, v):
            return self.inner.hvp(x, v)

    pp = PathProblem(problem)
    # monkey-patch _record to store x... we store via wrapping minimize of DA_SFCN
    from dasfcn.algorithms import DA_SFCN as DAS

    if isinstance(method, (DAS, KrylovCRN, SaddleFreeNewton, GradientDescent, LK_Newton, SC_Newton)):
        x = np.array(x0, dtype=float)
        xs = [x.copy()]
        tr = method.minimize(pp, x)
        # reconstruct path by rerunning with logging
        xs = _replay_path(method, problem, x0)
        return tr, xs
    tr = method.minimize(pp, x0)
    return tr, xs


def _replay_path(method, problem, x0):
    from dasfcn.algorithms import DA_SFCN, GradientDescent, KrylovCRN, LK_Newton, SC_Newton, SaddleFreeNewton

    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    if isinstance(method, GradientDescent):
        g = problem.grad(x)
        lr = method.lr
        for _ in range(method.maxiter):
            fx = problem.f(x)
            gn = np.linalg.norm(g)
            step = lr
            for _ls in range(12):
                if problem.f(x - step * g) <= fx - 1e-4 * step * gn ** 2:
                    break
                step *= 0.5
            x = x - step * g
            xs.append(x.copy())
            g = problem.grad(x)
            if np.linalg.norm(g) < method.tol:
                break
        return xs
    # generic DA-like
    if hasattr(method, "step"):
        g = problem.grad(x)
        M = getattr(method, "M0", 1.0)
        g0 = float(np.linalg.norm(g))
        for _ in range(method.maxiter):
            gn = float(np.linalg.norm(g))
            if gn < method.tol:
                break
            m = method.choose_m(gn, g0, x.size) if hasattr(method, "choose_m") else getattr(method, "m", 8)
            if isinstance(method, SaddleFreeNewton):
                from dasfcn.core import lanczos, saddle_free, tridiag

                Q, al, be, _ = lanczos(lambda v: problem.hvp(x, v), g, min(method.m, x.size))
                T = tridiag(al, be)
                A, w, _ = saddle_free(T)
                A = A + method.damping * np.eye(A.shape[0])
                y = -np.linalg.solve(A, Q.T @ g)
                s = Q @ y
                x = x + s
            else:
                x, M, _dh, _lam, _me, _ok = method.step(problem, x, g, M, m)
            xs.append(x.copy())
            g = problem.grad(x)
        return xs
    return xs


def fig_saddle_landscape():
    prob = MonkeySaddle()
    X, Y, Z = prob.grid()
    x0 = np.array([0.08, 0.05])
    methods = [
        GradientDescent(lr=0.08, maxiter=40),
        SaddleFreeNewton(m=2, maxiter=40),
        KrylovCRN(m=2, maxiter=40),
        DA_SFCN(m_min=1, m_max=2, alpha=2.0, maxiter=40),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cs = ax.contour(X, Y, Z, levels=18, cmap="viridis", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    styles = ["--", "-.", ":", "-"]
    for method, ls in zip(methods, styles):
        xs = _replay_path(method, prob, x0)
        P = np.array(xs)
        ax.plot(
            P[:, 0],
            P[:, 1],
            ls,
            color=COLORS.get(method.name, "k"),
            label=method.name,
            marker="o",
            markersize=3,
        )
    ax.scatter([0], [0], c="k", s=40, zorder=5, label="saddle")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Escape from a monkey saddle")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    savefig("fig_saddle_escape")


def collect(methods, problem, x0):
    out = {}
    for m in methods:
        try:
            tr = m.minimize(problem, x0)
            out[m.name] = {
                "f": tr.f,
                "grad": tr.grad_norm,
                "hvp": tr.n_hvp,
                "m": tr.m,
                "time": tr.time,
                "lam_min": tr.lam_min,
            }
            print(f"  {m.name:18s}  iters={len(tr.f)-1:3d}  ||g||={tr.grad_norm[-1]:.3e}  f={tr.f[-1]:.4e}")
        except Exception as e:
            print("  FAIL", m.name, e)
            traceback.print_exc()
    return out


def plot_curves(data, xkey, ykey, xlabel, ylabel, title, fname, logy=True, logx=False):
    plt.figure(figsize=(6.4, 4.4))
    for name, d in data.items():
        y = np.array(d[ykey], dtype=float)
        if xkey == "iter":
            x = np.arange(len(y))
        else:
            x = np.array(d[xkey], dtype=float)
            if len(x) != len(y):
                x = np.arange(len(y))
        plt.plot(x, np.maximum(y, 1e-16), color=COLORS.get(name, None), label=name)
    if logy:
        plt.yscale("log")
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(fname)


def fig_saddle_highdim():
    print("=== quadratic saddle n=80 ===")
    prob = QuadraticSaddle(n=80, n_neg=10)
    x0 = prob.x0(0)
    methods = [
        GradientDescent(lr=0.15, maxiter=80),
        LBFGS(maxiter=80),
        SaddleFreeNewton(m=12, maxiter=80),
        KrylovCRN(m=12, maxiter=80),
        DA_SFCN(m_min=4, m_max=20, alpha=3.0, maxiter=80),
        SC_Newton(m_min=3, m_max=20, alpha=3.0, theta=2e-2, maxiter=80),
        LK_Newton(m_min=4, m_max=16, reuse=3, maxiter=80),
    ]
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "iteration $k$", r"$\|\nabla f(x_k)\|$",
        "High-dimensional saddle (n=80)", "fig_saddle_highdim",
    )
    # HVP cost
    plt.figure(figsize=(6.4, 4.4))
    for name, d in data.items():
        hv = np.array(d["hvp"], dtype=float)
        g = np.maximum(np.array(d["grad"]), 1e-16)
        if np.all(hv == 0):
            continue
        plt.plot(hv, g, color=COLORS.get(name), label=name)
    plt.yscale("log")
    plt.xlabel("Hessian-vector products")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title("Cost to escape (HVP oracle)")
    plt.legend()
    savefig("fig_saddle_hvp")
    return data


def fig_rosenbrock():
    print("=== Rosenbrock n=40 ===")
    prob = Rosenbrock(n=40)
    x0 = prob.x0(2)
    methods = [
        GradientDescent(lr=1e-3, maxiter=100),
        LBFGS(maxiter=100),
        KrylovCRN(m=10, maxiter=80),
        DA_SFCN(m_min=4, m_max=18, alpha=3.5, maxiter=80),
        WQK_Newton(warmup=15, maxiter=80),
        LK_Newton(m_min=4, m_max=16, reuse=4, maxiter=80),
        SC_Newton(m_min=3, m_max=18, theta=1e-2, maxiter=80),
    ]
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "iteration $k$", r"$\|\nabla f(x_k)\|$",
        "Extended Rosenbrock ($n=40$)", "fig_rosenbrock",
    )
    plot_curves(
        data, "iter", "f", "iteration $k$", r"$f(x_k)$",
        "Rosenbrock objective", "fig_rosenbrock_f",
    )
    return data


def fig_ablation():
    print("=== ablation on saddle ===")
    prob = QuadraticSaddle(n=60, n_neg=8)
    x0 = prob.x0(1)
    methods = [
        DA_SFCN(m_min=4, m_max=16, alpha=3.0, maxiter=70),
        DA_SFCN(m_min=4, m_max=16, alpha=3.0, maxiter=70, saddle_free_on=False),
        DA_SFCN(m_min=16, m_max=16, dynamic=False, fixed_m=16, maxiter=70),
        DA_SFCN(m_min=6, m_max=6, dynamic=False, fixed_m=6, maxiter=70),
        KrylovCRN(m=12, maxiter=70),
    ]
    methods[0].name = "DA-SFCN"
    methods[1].name = "DA-SFCN (no SF)"
    methods[2].name = "DA-SFCN (fixed m)"
    methods[3].name = "DA-SFCN (m=6)"
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "iteration $k$", r"$\|\nabla f(x_k)\|$",
        "Ablation: saddle-free and dynamic $m_k$", "fig_ablation",
    )
    plt.figure(figsize=(6.4, 4.4))
    for name, d in data.items():
        plt.plot(d["m"], color=COLORS.get(name), label=name, drawstyle="steps-post")
    plt.xlabel("iteration $k$")
    plt.ylabel(r"subspace dimension $m_k$")
    plt.title("Dynamic Krylov dimension")
    plt.legend()
    savefig("fig_mk_schedule")
    return data


def fig_logistic():
    print("=== nonconvex logistic ===")
    prob = NonconvexLogistic(n_samples=350, n_features=50, alpha=0.5, seed=0)
    x0 = prob.x0(3)
    methods = [
        GradientDescent(lr=0.8, maxiter=70),
        LBFGS(maxiter=70),
        RandomSubspaceCubic(tau=10, maxiter=70),
        KrylovCRN(m=10, maxiter=70),
        DA_SFCN(m_min=4, m_max=16, alpha=3.0, maxiter=70),
        VR_DA_SFCN(batch=48, snapshot_every=6, m_min=4, m_max=14, maxiter=70),
        WQK_Newton(warmup=10, maxiter=70),
    ]
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "iteration $k$", r"$\|\nabla f(x_k)\|$",
        "Nonconvex logistic regression", "fig_logistic",
    )
    plt.figure(figsize=(6.4, 4.4))
    for name, d in data.items():
        hv = np.array(d["hvp"], dtype=float)
        g = np.maximum(np.array(d["grad"]), 1e-16)
        if np.all(hv == 0):
            continue
        plt.plot(hv, g, color=COLORS.get(name), label=name)
    plt.yscale("log")
    plt.xlabel("Hessian-vector products")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title("Oracle complexity on logistic")
    plt.legend()
    savefig("fig_logistic_hvp")
    return data


def fig_vr():
    print("=== VR finite-sum ===")
    prob = FiniteSumQuadratics(n_terms=30, n=40, seed=1)
    x0 = prob.x0(0)
    methods = [
        DA_SFCN(m_min=4, m_max=14, maxiter=60),
        VR_DA_SFCN(batch=6, snapshot_every=5, m_min=4, m_max=14, maxiter=60),
        LK_Newton(m_min=4, m_max=14, reuse=3, maxiter=60),
        GradientDescent(lr=0.05, maxiter=80),
    ]
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "iteration $k$", r"$\|\nabla f(x_k)\|$",
        "Finite-sum nonconvex quadratics", "fig_vr",
    )
    return data


def fig_factorization():
    print("=== matrix factorization ===")
    prob = MatrixFactorization(n_rows=18, n_cols=14, rank=3, seed=0)
    x0 = prob.x0(4)
    methods = [
        GradientDescent(lr=0.02, maxiter=80),
        LBFGS(maxiter=80),
        SaddleFreeNewton(m=10, maxiter=70),
        DA_SFCN(m_min=4, m_max=16, alpha=3.0, maxiter=70),
        KrylovCRN(m=12, maxiter=70),
    ]
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "f", "iteration $k$", r"$f(x_k)$",
        r"Matrix factorization $\|X-UV^\top\|_F^2$", "fig_factor",
    )
    return data


def fig_fed():
    print("=== federated logistic ===")
    prob = NonconvexLogistic(n_samples=400, n_features=40, alpha=0.35, seed=2)
    x0 = prob.x0(5)
    methods = [
        DA_SFCN(m_min=4, m_max=12, maxiter=40),
        Fed_DA_SFCN(n_clients=8, local_steps=2, maxiter=25),
        GradientDescent(lr=0.5, maxiter=40),
    ]
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "communication / iteration", r"$\|\nabla f(x_k)\|$",
        "Federated nonconvex logistic", "fig_fed",
    )
    return data


def fig_local_quadratic():
    print("=== local quadratic rate ===")
    # strongly convex bowl: start near the minimizer of a shifted Rosenbrock-like convex quadratic
    n = 30
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    H = A.T @ A / n + 0.5 * np.eye(n)
    xs = rng.normal(size=n)

    class ConvQuad:
        name = "convex_quadratic"

        def f(self, x):
            d = x - xs
            return 0.5 * float(d @ (H @ d))

        def grad(self, x):
            return H @ (x - xs)

        def hvp(self, x, v):
            return H @ v

    prob = ConvQuad()
    x0 = xs + 0.4 * rng.normal(size=n)
    methods = [
        KrylovCRN(m=8, maxiter=25, M0=1e-6),
        DA_SFCN(m_min=4, m_max=20, alpha=4.0, maxiter=25, M0=1e-6),
        GradientDescent(lr=0.15, maxiter=40),
    ]
    data = collect(methods, prob, x0)
    plt.figure(figsize=(6.4, 4.4))
    for name, d in data.items():
        err = np.maximum(np.array(d["grad"]), 1e-16)
        plt.semilogy(err, color=COLORS.get(name), label=name)
    plt.xlabel("iteration $k$")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title("Local phase on a strongly convex quadratic")
    plt.legend()
    savefig("fig_local")

    # Chebyshev prediction: m vs residual
    plt.figure(figsize=(6.4, 4.4))
    g0 = 1.0
    gs = np.logspace(0, -8, 40)
    for alpha, mmin, mmax, lab in [
        (2.0, 4, 24, r"$\alpha=2$"),
        (3.0, 4, 24, r"$\alpha=3$"),
        (5.0, 4, 24, r"$\alpha=5$"),
    ]:
        ms = [dynamic_m(g, g0, mmin, mmax, alpha) for g in gs]
        plt.semilogx(gs, ms, label=lab)
    plt.gca().invert_xaxis()
    plt.xlabel(r"$\|g_k\|/\|g_0\|$")
    plt.ylabel(r"$m_k$")
    plt.title("Logarithmic dimension schedule")
    plt.legend()
    savefig("fig_schedule_theory")
    return data


def fig_sc_certificate():
    print("=== spectral certificate ===")
    prob = Rosenbrock(n=35)
    x0 = prob.x0(1)
    methods = [
        DA_SFCN(m_min=4, m_max=18, alpha=3.0, maxiter=50),
        SC_Newton(m_min=3, m_max=18, theta=5e-2, maxiter=50),
        SC_Newton(m_min=3, m_max=18, theta=5e-3, maxiter=50),
        KrylovCRN(m=18, maxiter=50),
    ]
    methods[1].name = "SC-Newton ($\\theta=5\\times10^{-2}$)"
    methods[2].name = "SC-Newton ($\\theta=5\\times10^{-3}$)"
    # names with latex might be messy in dict; use plain
    methods[1].name = "SC-Newton (loose)"
    methods[2].name = "SC-Newton (tight)"
    data = collect(methods, prob, x0)
    plot_curves(
        data, "iter", "grad", "iteration $k$", r"$\|\nabla f(x_k)\|$",
        "Spectral certificate vs fixed / dynamic $m$", "fig_sc",
    )
    plt.figure(figsize=(6.4, 4.4))
    for name, d in data.items():
        plt.plot(d["m"], label=name, drawstyle="steps-post", color=COLORS.get(name))
    plt.xlabel("iteration $k$")
    plt.ylabel(r"$m_k$ (Lanczos steps)")
    plt.title("Work per iteration under the spectral certificate")
    plt.legend()
    savefig("fig_sc_m")
    return data


def fig_success_rate():
    print("=== saddle escape success rate ===")
    n_trials = 12
    names = ["GD", "L-BFGS", "SFN", "Krylov-CRN", "DA-SFCN"]
    success = {n: 0 for n in names}
    iters = {n: [] for n in names}
    thresh = 1e-3
    for t in range(n_trials):
        prob = QuadraticSaddle(n=50, n_neg=8, quartic=0.08)
        x0 = prob.x0(10 + t)
        makers = [
            GradientDescent(lr=0.12, maxiter=60),
            LBFGS(maxiter=60),
            SaddleFreeNewton(m=10, maxiter=50),
            KrylovCRN(m=10, maxiter=50),
            DA_SFCN(m_min=4, m_max=14, maxiter=50),
        ]
        for m in makers:
            tr = m.minimize(prob, x0)
            ok = tr.grad_norm[-1] < thresh and tr.f[-1] < 0.05
            if ok:
                success[m.name] += 1
            # first time below thresh
            hit = next((i for i, g in enumerate(tr.grad_norm) if g < thresh), None)
            if hit is not None:
                iters[m.name].append(hit)
        print(f"  trial {t+1}/{n_trials} done")
    plt.figure(figsize=(6.4, 4.4))
    xs = np.arange(len(names))
    rates = [100.0 * success[n] / n_trials for n in names]
    bars = plt.bar(xs, rates, color=[COLORS[n] for n in names], edgecolor="k", linewidth=0.4)
    plt.xticks(xs, names, rotation=15)
    plt.ylabel("escape success (%)")
    plt.title(rf"Saddle escape over {n_trials} random starts")
    plt.ylim(0, 110)
    for b, r in zip(bars, rates):
        plt.text(b.get_x() + b.get_width() / 2, r + 2, f"{r:.0f}%", ha="center", fontsize=9)
    savefig("fig_success")

    plt.figure(figsize=(6.4, 4.4))
    means = [np.mean(iters[n]) if iters[n] else np.nan for n in names]
    plt.bar(xs, means, color=[COLORS[n] for n in names], edgecolor="k", linewidth=0.4)
    plt.xticks(xs, names, rotation=15)
    plt.ylabel("iterations to $\\|g\\| < 10^{-3}$")
    plt.title("Mean escape time (successful runs)")
    savefig("fig_escape_time")
    return {"success": success, "n_trials": n_trials}


def fig_family():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (3.3, 4.6, 3.4, 1.0, "DA-SFCN\nKrylov + |T| + dynamic $m_k$"),
        (0.3, 2.4, 2.6, 1.1, "VR-DA-SFCN\nSVRG oracles"),
        (3.7, 2.4, 2.6, 1.1, "LK-Newton\nlazy Lanczos"),
        (7.1, 2.4, 2.6, 1.1, "WQK-Newton\nL-BFGS warmup"),
        (1.9, 0.4, 2.6, 1.1, "Fed-DA-SFCN\nlocal cubic steps"),
        (5.5, 0.4, 2.6, 1.1, "SC-Newton\nspectral certificate"),
    ]
    for x, y, w, h, t in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#fdebd0", edgecolor="#922b21", lw=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, zorder=3)
    for x, y, w, h, _t in boxes[1:]:
        ax.annotate(
            "",
            xy=(x + w / 2, y + h),
            xytext=(5.0, 4.6),
            arrowprops=dict(arrowstyle="->", color="#7b241c", lw=1.2),
        )
    ax.set_title("The DA-SFCN family", pad=8)
    savefig("fig_family")


def fig_complexity_table(all_data: dict):
    """Bar chart of final gradient and HVP totals."""
    # pick comparable methods on logistic if present
    pass


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def latex_table(rows, path: Path):
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Iters & Final $\|\nabla f\|$ & HVPs & Time (s) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['name']} & {r['iters']} & {r['grad']:.2e} & {r['hvp']} & {r['time']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(tag, data):
    rows = []
    for name, d in data.items():
        rows.append(
            {
                "name": name,
                "iters": max(len(d["grad"]) - 1, 0),
                "grad": d["grad"][-1],
                "f": d["f"][-1],
                "hvp": d["hvp"][-1] if d["hvp"] else 0,
                "time": d["time"][-1] if d["time"] else 0.0,
            }
        )
    latex_table(rows, RES / f"table_{tag}.tex")
    return rows


def main():
    fig_family()
    fig_saddle_landscape()
    all_rows = {}
    d = fig_saddle_highdim()
    all_rows["saddle"] = summarize("saddle", d)
    d = fig_rosenbrock()
    all_rows["rosenbrock"] = summarize("rosenbrock", d)
    d = fig_ablation()
    all_rows["ablation"] = summarize("ablation", d)
    d = fig_logistic()
    all_rows["logistic"] = summarize("logistic", d)
    d = fig_vr()
    all_rows["vr"] = summarize("vr", d)
    d = fig_factorization()
    all_rows["factor"] = summarize("factor", d)
    d = fig_fed()
    all_rows["fed"] = summarize("fed", d)
    d = fig_local_quadratic()
    all_rows["local"] = summarize("local", d)
    d = fig_sc_certificate()
    all_rows["sc"] = summarize("sc", d)
    suc = fig_success_rate()
    payload = {"tables": all_rows, "success": suc}
    (RES / "summary.json").write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    print("DONE. Figures in", FIG)


if __name__ == "__main__":
    main()
