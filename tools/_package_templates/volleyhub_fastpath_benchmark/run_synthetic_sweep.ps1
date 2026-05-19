# Run the synthetic 4-camera sweep (no Basler hardware required).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[error] venv not found. Run RUN_ME_FIRST.ps1 first." -ForegroundColor Red
    exit 1
}
. .\.venv\Scripts\Activate.ps1

Write-Host "[sweep] synthetic mode, 60 s per variant" -ForegroundColor Cyan
python scripts\run_fastpath_sweep.py --mode synthetic --duration-s 60

if ($LASTEXITCODE -ne 0) {
    Write-Host "[sweep] failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[report] building comparison report" -ForegroundColor Cyan
python scripts\build_comparison_report.py `
    --input reports\results.json `
    --baseline configs\baseline_rtx2080super.json

Write-Host ""
Write-Host "Reports:" -ForegroundColor Green
Write-Host "  reports\fastpath_comparison_report.md" -ForegroundColor Yellow
Write-Host "  reports\fastpath_comparison_report.html" -ForegroundColor Yellow
Write-Host "  reports\results.json / results.csv" -ForegroundColor Yellow

if (Test-Path "reports") {
    Start-Process explorer.exe (Resolve-Path .\reports).Path
}
