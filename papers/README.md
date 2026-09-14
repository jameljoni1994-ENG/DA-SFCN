# Papers

| Path | Content |
|------|---------|
| `unified/` | Full English and Arabic monographs (DA-SFCN family) |
| `en/` `01`–`06` | One English paper per algorithm |
| `ar/` `01`–`06` | One Arabic paper per algorithm |
| `en/07_empirical_results.tex` | Empirical results report (EN) |
| `ar/07_empirical_results.tex` | Empirical results report (AR) |
| `arabic-fonts.tex` | Amiri / XeLaTeX font setup for Arabic |
| `refs.bib` | Shared bibliography |

Canonical compiled PDFs live in [`../pdfs/`](../pdfs/) (`00_unified_*`, `01`–`06_*`, `07_Results_*`). Rebuild with:

```powershell
powershell -ExecutionPolicy Bypass -File ../scripts/compile_papers.ps1
```
