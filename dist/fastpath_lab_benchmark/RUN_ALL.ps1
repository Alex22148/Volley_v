# RUN_ALL.ps1 â€” run the entire fastpath lab benchmark, end to end.
#
# 1. environment check                        -> results/environment.*
# 2. inference-only for each resolution        -> results/inference_only_*.{json,csv}
# 3. color-only for each resolution            -> results/color_only_*.{json,csv}
# 4. full synthetic path for each resolution   -> results/full_synthetic_*.{json,csv}
#    (the sweep script does this automatically)
# 5. report                                    -> reports/fastpath_lab_report.{md,html}
#
# This script does NOT touch cameras, dataset, training, or the
# production app. All inputs are synthetic.
#
# Usage:
#   .\RUN_ALL.ps1
#   .\RUN_ALL.ps1 -DurationS 15 -Warmup 5
#   .\RUN_ALL.ps1 -Python "C:\envs\trt\Scripts\python.exe"

[CmdletBinding()]
param(
    [string]$Python = "python",
    [double]$DurationS = 30,
    [int]$Warmup = 10
)

$ErrorActionPreference = "Stop"
$here     = $PSScriptRoot
$scripts  = Join-Path $here "scripts"
$engines  = Join-Path $here "engines"
$results  = Join-Path $here "results"
$reports  = Join-Path $here "reports"

Write-Host "fastpath_lab_benchmark / RUN_ALL.ps1" -ForegroundColor Cyan
Write-Host "package root : $here"
Write-Host "duration     : $DurationS s per case"
Write-Host "warmup       : $Warmup iterations per case"
Write-Host ""

# ----------------------------------- 1) environment check
Write-Host "[1/5] environment check" -ForegroundColor Yellow
& $Python (Join-Path $scripts "00_check_env.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] environment check failed. Aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

# Engine -> (label, capture, inference) sweep
$sweep = @(
    @{ engine = "best__fp16_640_b4_static.engine";       label = "640x640";   inf = @(640, 640)   },
    @{ engine = "best__fp16_960_b4_static.engine";       label = "960x960";   inf = @(960, 960)   },
    @{ engine = "best__fp16_1280_b4_static.engine";      label = "1280x1280"; inf = @(1280, 1280) },
    @{ engine = "best__fp16_1088x1920_b4_static.engine"; label = "1088x1920"; inf = @(1088, 1920) }
)

# ----------------------------------- 2) inference-only per resolution
Write-Host ""
Write-Host "[2/5] inference-only benchmarks" -ForegroundColor Yellow
foreach ($case in $sweep) {
    $eng = Join-Path $engines $case.engine
    if (-not (Test-Path $eng)) {
        Write-Host "  [SKIP] $($case.label): engine missing at $eng"
        continue
    }
    Write-Host "  [RUN ] inference-only $($case.label)"
    & $Python (Join-Path $scripts "01_benchmark_inference_only.py") `
        --engine $eng `
        --input-shape $case.inf[0] $case.inf[1] `
        --batch 4 `
        --duration-s $DurationS `
        --warmup $Warmup
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] inference-only $($case.label) failed (rc=$LASTEXITCODE)" -ForegroundColor Red
    }
}

# ----------------------------------- 3) color-only per resolution
Write-Host ""
Write-Host "[3/5] color-only benchmarks (debayer + resize + normalize)" -ForegroundColor Yellow
foreach ($case in $sweep) {
    Write-Host "  [RUN ] color-only -> $($case.label)"
    & $Python (Join-Path $scripts "02_benchmark_color_only.py") `
        --capture-shape 1080 1920 `
        --output-shape $case.inf[0] $case.inf[1] `
        --batch 4 `
        --duration-s $DurationS `
        --warmup $Warmup
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] color-only $($case.label) failed (rc=$LASTEXITCODE)" -ForegroundColor Red
    }
}

# ----------------------------------- 4) full synthetic sweep
Write-Host ""
Write-Host "[4/5] full synthetic path sweep" -ForegroundColor Yellow
& $Python (Join-Path $scripts "04_resolution_sweep.py") `
    --duration-s $DurationS `
    --warmup $Warmup
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] sweep failed (rc=$LASTEXITCODE)" -ForegroundColor Red
}

# ----------------------------------- 5) report
Write-Host ""
Write-Host "[5/5] building report" -ForegroundColor Yellow
& $Python (Join-Path $scripts "05_make_report.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] report build failed (rc=$LASTEXITCODE)" -ForegroundColor Red
}

# ----------------------------------- final summary
$reportHtml = Join-Path $reports "fastpath_lab_report.html"
$sweepJson  = Join-Path $results "resolution_sweep.json"
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " Lab benchmark finished."
Write-Host " HTML report  : $reportHtml"
Write-Host " Sweep JSON   : $sweepJson"
if (Test-Path $sweepJson) {
    try {
        $sweep = Get-Content -Raw -Path $sweepJson | ConvertFrom-Json
        $fullhd = $sweep.rows | Where-Object { $_.label -eq "1088x1920" -and -not $_.skipped }
        $passes = $sweep.rows | Where-Object { -not $_.skipped -and $_.pass_50fps_safe_p95 }
        if ($passes) { $largest = ($passes | Sort-Object pixels_mpx -Descending)[0] } else { $largest = $null }
        Write-Host ""
        Write-Host " Q1. Full HD-like (1088x1920) safe FPS p95 = $(if ($fullhd) { "{0:N1}" -f $fullhd.fps_per_camera_safe_p95 } else { "(missing)" })"
        Write-Host " Q2. Largest input at safe 50 FPS p95      = $(if ($largest) { $largest.label } else { "none" })"
    } catch {
        Write-Host " (could not parse summary: $_)" -ForegroundColor Yellow
    }
}
Write-Host "=========================================" -ForegroundColor Cyan


