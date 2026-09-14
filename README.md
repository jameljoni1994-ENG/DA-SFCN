# DA-SFCN Family

Dynamically-Adaptive Saddle-Free Cubic Newton and five extensions, with bilingual research papers (English / Arabic), reproducible experiments, and compiled PDFs.

## Algorithms

| Method | Role |
|--------|------|
| **DA-SFCN** | Parent: Krylov subspace + saddle-free `\|T\|` + logarithmic width `m_k` |
| **VR-DA-SFCN** | SVRG-style variance-reduced oracles for finite-sum problems |
| **LK-Newton** | Lazy reuse of Lanczos bases |
| **WQK-Newton** | L-BFGS warm start, then DA-SFCN |
| **Fed-DA-SFCN** | Federated local cubic steps + server averaging |
| **SC-Newton** | Lanczos stopped by a Ritz residual certificate |

Theory for the parent method: global rate `O(k^{-2/3})` (any `m_k ≥ 1`) and local quadratic convergence when `m_k` grows logarithmically in `1/‖∇f‖`.

## Repository layout

```
dasfcn/              Python package (algorithms, API, optional torch HVP)
tests/               Fast unit tests (pytest)
run_experiments.py   Full experiment suite → figures/ + results/
scripts/repro_smoke.py  Light multi-seed saddle escape + logistic smoke
figures/             Generated plots (PDF + PNG)
results/             Tables, summary.json, repro_smoke.json
papers/              LaTeX sources (EN/AR) + arabic-fonts.tex
pdfs/                Canonical compiled PDFs
docs/OPTIMIZER.md    Living contribution roadmap
CONTRIBUTING.md      How to contribute
LICENSE / CITATION.cff
```

## Roadmap

Strategic ideas and prioritized contributions (device-aware) live in **[docs/OPTIMIZER.md](docs/OPTIMIZER.md)**.  
Contribution rules: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Quick start

```bash
# Python 3.10+ (on Windows: py -3)
pip install -e ".[dev]"
pytest -q
py -3 scripts/repro_smoke.py
```

Unified API:

```python
from dasfcn import minimize
from dasfcn.problems import QuadraticSaddle

prob = QuadraticSaddle(n=40, n_neg=6)
tr = minimize(prob, prob.x0(0), method="da-sfcn", maxiter=40)
print(tr.grad_norm[-1], tr.n_hvp[-1])
```

Optional PyTorch CPU HVP helper: `pip install -e ".[torch]"` then see `dasfcn/torch_backend.py`.

Full plot suite (heavier): `py -3 run_experiments.py`.

### Compile papers

Requires [MiKTeX](https://miktex.org/) (or TeX Live). English uses `pdflatex`; Arabic uses `xelatex` with the **Amiri** font.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/compile_papers.ps1
```

Outputs are collected under `pdfs/`.

## PDFs

| File | Content |
|------|---------|
| `pdfs/00_unified_EN.pdf` / `_AR.pdf` | Full family monograph |
| `pdfs/01_DA-SFCN_*.pdf` … `06_SC-Newton_*.pdf` | One paper per algorithm |
| `pdfs/07_Results_EN.pdf` / `_AR.pdf` | Empirical results report (tables + figures) |

## Citation

Please cite this repository (`CITATION.cff`) and the DA-SFCN monograph in `papers/unified/` (September 2026).

## License

MIT — see [LICENSE](LICENSE).
