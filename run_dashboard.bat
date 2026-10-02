@echo off
setlocal
cd /d "%~dp0"

echo Launching dashboard from: %CD%

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python and add it to PATH.
    pause
    exit /b 1
)

python -c "import streamlit, pandas, numpy, plotly, reportlab" >nul 2>&1
if errorlevel 1 (
    echo Required Python packages are missing.
    echo Install them once with: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

python -c "from springer_database import MILLING_TOOLING_DATABASE, TOOL_HOLDER_DATABASE"
if errorlevel 1 (
    echo.
    echo The Python files in this folder are from different or outdated versions.
    echo Download or update the complete dashboard project, then run this launcher again.
    pause
    exit /b 1
)

python -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 pause
