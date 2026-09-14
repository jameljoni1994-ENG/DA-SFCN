# Contributing to DA-SFCN

Thanks for contributing. Keep changes **device-friendly** (CPU, small `n`, short tests).

## Setup

```bash
pip install -e ".[dev]"
pytest -q
```

Optional PyTorch CPU path:

```bash
pip install -e ".[torch]"
```

## Guidelines

1. Prefer NumPy/SciPy for core algorithms in `dasfcn/`.
2. Add or update a unit test under `tests/` for any core change (Lanczos, cubic, API).
3. Do not commit large datasets, GPU-only notebooks, or long-running experiment dumps.
4. Update `docs/OPTIMIZER.md` status when you finish a roadmap item.
5. Keep PRs focused; separate theory notes from API changes when possible.

## Smoke reproducibility

```bash
py -3 scripts/repro_smoke.py
```

Writes `results/repro_smoke.json` (multi-seed saddle escape + logistic smoke).

## Code style

- Python 3.10+.
- No unnecessary new dependencies in the default install.
- Match existing naming: `DA_SFCN`, `minimize(..., method="da-sfcn")`.
