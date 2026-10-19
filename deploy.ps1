# ==============================================================================
# PowerShell One-Click Deployment Script for Tool Wear Forecasting Dashboard
# Pushes code updates directly to GitHub to trigger auto-deployment on Streamlit Cloud
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "DEPLOYING TOOL WEAR DASHBOARD & ML SPRINGER ENGINE TO GITHUB" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Locate Git executable
$gitPath = ""
if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitPath = "git"
} else {
    $candidates = Get-ChildItem -Path "C:\Users\Siddharth Patil\AppData\Local\GitHubDesktop" -Filter "git.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    if ($candidates) {
        $gitPath = $candidates[0]
    } elseif (Test-Path "C:\Program Files\Git\cmd\git.exe") {
        $gitPath = "C:\Program Files\Git\cmd\git.exe"
    }
}

if (-not $gitPath) {
    Write-Error "Git executable could not be found. Please ensure Git is installed."
    exit 1
}

Write-Host "`n[1/5] Using Git binary at: $gitPath" -ForegroundColor Green

# 2. Check if git repository is initialized
if (-not (Test-Path ".git")) {
    Write-Host "`n[2/5] Initializing local git repository..." -ForegroundColor Yellow
    & $gitPath init
    & $gitPath branch -M main
    & $gitPath remote add origin "https://github.com/siddharthpatil2656-hash/tool-wear-dashbaord.git"
} else {
    Write-Host "`n[2/5] Git repository already initialized." -ForegroundColor Green
    # Verify remote URL
    $existingRemote = & $gitPath remote get-url origin 2>$null
    if (-not $existingRemote) {
        & $gitPath remote add origin "https://github.com/siddharthpatil2656-hash/tool-wear-dashbaord.git"
    }
}

# 3. Stage changes
Write-Host "`n[3/5] Staging files for commit..." -ForegroundColor Yellow
& $gitPath add .

# 4. Commit changes
Write-Host "`n[4/5] Committing changes..." -ForegroundColor Yellow
$commitMsg = "Integrate Springer Nature API client and ML literature analysis engine"
$status = & $gitPath status --porcelain
if ($status) {
    & $gitPath commit -m $commitMsg
    Write-Host " -> Changes committed successfully." -ForegroundColor Green
} else {
    Write-Host " -> No changes to commit (working tree clean)." -ForegroundColor Yellow
}

# 5. Push to GitHub
Write-Host "`n[5/5] Pushing updates to GitHub (main branch)..." -ForegroundColor Yellow
try {
    & $gitPath push -u origin main
    Write-Host "`n==================================================================" -ForegroundColor Green
    Write-Host "SUCCESS: Pushed to https://github.com/siddharthpatil2656-hash/tool-wear-dashbaord" -ForegroundColor Green
    Write-Host "Live App: https://siddharthpatil2656-hash-tool-wear-dashbaord.streamlit.app" -ForegroundColor Cyan
    Write-Host "Streamlit Cloud will automatically detect and deploy these changes in ~60s." -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Green
} catch {
    Write-Host "`n[NOTE] Standard push encountered an error: $_" -ForegroundColor Yellow
    Write-Host "Retrying with fetch/rebase..." -ForegroundColor Yellow
    & $gitPath fetch origin main
    & $gitPath rebase origin/main
    & $gitPath push origin main
}
