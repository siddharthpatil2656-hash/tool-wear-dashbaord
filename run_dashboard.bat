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

python -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 pause
