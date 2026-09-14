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
dasfcn/              Python package (algorithms, problems, Lanczos/cubic core)
run_experiments.py   Experiment suite → figures/ + results/
figures/             Generated plots (PDF + PNG)
results/             Tables and summary.json
papers/
  unified/           Full bilingual monographs (LaTeX)
  en/                Per-algorithm papers (English)
  ar/                Per-algorithm papers (Arabic)
  refs.bib
pdfs/                Canonical compiled PDFs (ready to read / share)
scripts/             Build helpers
docs/                Research plan and notes
```

## Quick start

```bash
# Python 3.10+ (on Windows use: py -3)
pip install -r requirements.txt
py -3 run_experiments.py
```

Plots are written to `figures/`; numeric summaries to `results/summary.json`.

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

## Citation

If you use this code or the papers, please cite the DA-SFCN monograph in `papers/unified/` (September 2026).

## License

Research / academic use. Add an explicit license file if you redistribute publicly.
