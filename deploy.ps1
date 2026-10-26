# One-step deploy for the Tool Wear dashboard.
# Usage:  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
#         (optional)  .\deploy.ps1 -Port 8502 -SkipInstall
param(
    [int]$Port = 8501,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
Write-Host "Deploying dashboard from: $PSScriptRoot" -ForegroundColor Cyan

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { Write-Error "Python was not found. Install Python and add it to PATH."; exit 1 }

if (-not $SkipInstall) {
    Write-Host "Checking dependencies..." -ForegroundColor Cyan
    & $python.Source -c "import streamlit, pandas, numpy, plotly, reportlab, sklearn, openpyxl, pdfplumber, PIL" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing requirements..." -ForegroundColor Yellow
        & $python.Source -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { Write-Error "Dependency install failed."; exit 1 }
    }
}

Write-Host "Compiling check..." -ForegroundColor Cyan
& $python.Source -m py_compile app.py optimizer.py calibration.py springer_database.py tool_physics.py cost_model.py gcode_optimizer.py ml_model.py readings_import.py
if ($LASTEXITCODE -ne 0) { Write-Error "Compile check failed."; exit 1 }

Write-Host "Running test suite..." -ForegroundColor Cyan
& $python.Source test_suite.py
if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed; not launching."; exit 1 }

# Free the port if an older instance is still running
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port $Port is in use; stopping the old instance..." -ForegroundColor Yellow
    $busy | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
}

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:NO_PROXY = "localhost,127.0.0.1"
Write-Host "Starting dashboard at http://127.0.0.1:$Port  (Ctrl+C to stop)" -ForegroundColor Green
Start-Process "http://127.0.0.1:$Port"
& $python.Source -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
