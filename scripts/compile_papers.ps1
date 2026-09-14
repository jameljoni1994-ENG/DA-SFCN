# Compile all DA-SFCN papers and refresh pdfs/
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pdfs = Join-Path $root "pdfs"
New-Item -ItemType Directory -Force -Path $pdfs | Out-Null

function Compile-PdfLatex($tex, $dir) {
    Push-Location $dir
    Write-Host "=== pdflatex $tex ==="
    pdflatex -interaction=nonstopmode $tex | Out-Null
    pdflatex -interaction=nonstopmode $tex | Out-Null
    $ok = Test-Path ([IO.Path]::ChangeExtension($tex, ".pdf"))
    if ($ok) { Write-Host "OK $tex" } else { Write-Host "FAIL $tex" }
    Pop-Location
    return $ok
}

function Compile-XeLatex($tex, $dir) {
    Push-Location $dir
    Write-Host "=== xelatex $tex ==="
    xelatex -interaction=nonstopmode $tex | Out-Null
    xelatex -interaction=nonstopmode $tex | Out-Null
    $ok = Test-Path ([IO.Path]::ChangeExtension($tex, ".pdf"))
    if ($ok) { Write-Host "OK $tex" } else { Write-Host "FAIL $tex" }
    Pop-Location
    return $ok
}

Compile-PdfLatex "paper_en.tex" (Join-Path $root "papers\unified") | Out-Null
Compile-XeLatex "paper_ar.tex" (Join-Path $root "papers\unified") | Out-Null

$names = @(
    "01_da_sfcn.tex",
    "02_vr_da_sfcn.tex",
    "03_lk_newton.tex",
    "04_wqk_newton.tex",
    "05_fed_da_sfcn.tex",
    "06_sc_newton.tex"
)
foreach ($f in $names) {
    Compile-PdfLatex $f (Join-Path $root "papers\en") | Out-Null
    Compile-XeLatex $f (Join-Path $root "papers\ar") | Out-Null
}

# Collect canonical PDFs
Copy-Item (Join-Path $root "papers\unified\paper_en.pdf") (Join-Path $pdfs "00_unified_EN.pdf") -Force
Copy-Item (Join-Path $root "papers\unified\paper_ar.pdf") (Join-Path $pdfs "00_unified_AR.pdf") -Force

$map = @{
    "01_da_sfcn.pdf"      = "01_DA-SFCN"
    "02_vr_da_sfcn.pdf"   = "02_VR-DA-SFCN"
    "03_lk_newton.pdf"    = "03_LK-Newton"
    "04_wqk_newton.pdf"   = "04_WQK-Newton"
    "05_fed_da_sfcn.pdf"  = "05_Fed-DA-SFCN"
    "06_sc_newton.pdf"    = "06_SC-Newton"
}
foreach ($src in $map.Keys) {
    $en = Join-Path $root "papers\en\$src"
    $ar = Join-Path $root "papers\ar\$src"
    if (Test-Path $en) { Copy-Item $en (Join-Path $pdfs "$($map[$src])_EN.pdf") -Force }
    if (Test-Path $ar) { Copy-Item $ar (Join-Path $pdfs "$($map[$src])_AR.pdf") -Force }
}

# Strip build artifacts after a successful compile pass
Get-ChildItem -Recurse $root -Include *.aux,*.log,*.out,*.toc,*.synctex.gz |
    Where-Object { $_.FullName -notmatch "\\figures\\" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "`nCanonical PDFs in $pdfs"
Get-ChildItem $pdfs | Format-Table Name, Length
