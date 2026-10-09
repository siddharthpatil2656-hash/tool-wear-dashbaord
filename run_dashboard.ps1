# Offline launcher for the Tool Wear Forecasting dashboard.
# Usage:  powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
#         (optionally:  .\run_dashboard.ps1 -Port 8502)
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Launching dashboard from: $PSScriptRoot"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python was not found. Install Python and add it to PATH."
    exit 1
}

# Offline hardening: no usage statistics, no telemetry, and keep localhost
# traffic out of any configured proxy.
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = "localhost,127.0.0.1"
$env:PYTHONNOUSERSITE = "1"

& $python.Source -c "import streamlit, pandas, numpy, plotly, reportlab, openpyxl, pdfplumber, pytesseract" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Required Python packages are missing."
    Write-Host "Install them once with: python -m pip install -r requirements.txt"
    exit 1
}

& $python.Source -c "from springer_database import MILLING_TOOLING_DATABASE, TOOL_HOLDER_DATABASE"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The Python files in this folder are from different or outdated versions."
    Write-Host "Download or update the complete dashboard project, then run this launcher again."
    exit 1
}

Write-Host ""
Write-Host "Starting dashboard (offline) at http://127.0.0.1:$Port"
Write-Host "The hybrid physics + ML engine trains locally on first load (~15 s);"
Write-Host "it is cached for the rest of the session. Press Ctrl+C to stop."
Write-Host ""

& $python.Source -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --server.enableXsrfProtection true --server.enableCORS true --browser.gatherUsageStats false
