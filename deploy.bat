@echo off
echo ==================================================================
echo RUNNING TOOL WEAR DASHBOARD DEPLOYMENT
echo ==================================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1"
pause
