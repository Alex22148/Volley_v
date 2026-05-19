# CHECK_ENV.ps1 â€” quick environment sanity check for the lab benchmark.
#
# This script runs scripts/00_check_env.py with default arguments and
# prints a clear PASS/FAIL summary. It does NOT run any benchmark loops.
#
# Usage:
#   .\CHECK_ENV.ps1
#   .\CHECK_ENV.ps1 -Python "C:\Path\To\python.exe"

[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
Write-Host "fastpath_lab_benchmark / CHECK_ENV.ps1" -ForegroundColor Cyan
Write-Host "package root: $here"
Write-Host ""

& $Python (Join-Path $here "scripts\00_check_env.py")
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "[OK] environment looks fine. You can now run RUN_ALL.ps1" -ForegroundColor Green
} else {
    Write-Host "[FAIL] critical checks failed. See above and results\environment.md" -ForegroundColor Red
}
exit $code


