"""Light CPU smoke / multi-seed saddle escape (device-friendly)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dasfcn.api import minimize
from dasfcn.problems import QuadraticSaddle, NonconvexLogistic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "repro_smoke.json"


def saddle_escape_trial(seed: int, maxiter: int = 40, thresh: float = 1e-3) -> dict:
    prob = QuadraticSaddle(n=40, n_neg=6, quartic=0.08)
    x0 = prob.x0(10 + seed)
    methods = {
        "gd": dict(method="gd", lr=0.12, maxiter=maxiter, tol=0.0),
        "lbfgs": dict(method="lbfgs", maxiter=maxiter, tol=0.0),
        "sfn": dict(method="sfn", m=8, maxiter=maxiter, tol=0.0),
        "da-sfcn": dict(method="da-sfcn", m_min=3, m_max=12, alpha=2.5, maxiter=maxiter, tol=0.0),
    }
    row = {"seed": seed}
    for name, kw in methods.items():
        tr = minimize(prob, x0.copy(), **kw)
        ok = tr.grad_norm[-1] < thresh and tr.f[-1] < 0.05
        hit = next((i for i, g in enumerate(tr.grad_norm) if g < thresh), None)
        row[name] = {
            "success": bool(ok),
            "final_grad": float(tr.grad_norm[-1]),
            "final_f": float(tr.f[-1]),
            "iters_to_tol": hit,
            "hvp": int(tr.n_hvp[-1]) if tr.n_hvp else 0,
        }
    return row


def logistic_smoke(seed: int = 0) -> dict:
    prob = NonconvexLogistic(n_samples=200, n_features=30, alpha=0.4, seed=seed)
    x0 = prob.x0(seed + 1)
    tr = minimize(prob, x0, method="da-sfcn", maxiter=30, tol=1e-8, m_min=3, m_max=12, alpha=2.5)
    return {
        "iters": max(len(tr.grad_norm) - 1, 0),
        "final_grad": float(tr.grad_norm[-1]),
        "hvp": int(tr.n_hvp[-1]) if tr.n_hvp else 0,
        "grad_path": [float(g) for g in tr.grad_norm[:: max(1, len(tr.grad_norm) // 8)]],
        "hvp_path": [int(h) for h in tr.n_hvp[:: max(1, len(tr.n_hvp) // 8)]],
    }


def main(n_trials: int = 8):
    print(f"Saddle escape: {n_trials} seeds (n=40) ...")
    trials = [saddle_escape_trial(s) for s in range(n_trials)]
    methods = ["gd", "lbfgs", "sfn", "da-sfcn"]
    summary = {}
    for m in methods:
        succ = sum(1 for t in trials if t[m]["success"])
        hvps = [t[m]["hvp"] for t in trials if t[m]["success"]]
        summary[m] = {
            "success_rate": succ / n_trials,
            "successes": succ,
            "n_trials": n_trials,
            "mean_hvp_on_success": float(np.mean(hvps)) if hvps else None,
        }
        print(f"  {m:10s}  {succ}/{n_trials}  mean_hvp={summary[m]['mean_hvp_on_success']}")

    print("Logistic smoke ...")
    logi = logistic_smoke(0)
    print(f"  final_grad={logi['final_grad']:.3e}  hvp={logi['hvp']}")

    payload = {"saddle_summary": summary, "saddle_trials": trials, "logistic": logi}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
