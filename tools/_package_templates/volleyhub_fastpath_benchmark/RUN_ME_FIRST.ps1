# RUN_ME_FIRST.ps1 — bootstrap for the portable fast-path benchmark.
# Creates a venv if missing, installs requirements, runs environment diagnostics.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "[setup] creating venv at .venv" -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "[setup] activating venv" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "[setup] upgrading pip" -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

Write-Host "[setup] installing requirements (this may take a while)" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "[setup] running environment check" -ForegroundColor Cyan
python scripts\check_environment.py

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host " Environment ready. Next steps:" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Synthetic benchmark (no cameras needed):" -ForegroundColor White
Write-Host "    .\run_synthetic_sweep.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Real Basler benchmark:" -ForegroundColor White
Write-Host "    1) copy your camera_roles.json to configs/" -ForegroundColor Gray
Write-Host "    2) .\run_real_basler_sweep.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Rebuild report from existing results.json:" -ForegroundColor White
Write-Host "    python scripts\build_comparison_report.py --input reports\results.json --baseline configs\baseline_rtx2080super.json" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Engines diagnostics / build:" -ForegroundColor White
Write-Host "    python scripts\export_or_check_engines.py --check" -ForegroundColor Yellow
Write-Host "    python scripts\export_or_check_engines.py --build-missing" -ForegroundColor Yellow
Write-Host ""
