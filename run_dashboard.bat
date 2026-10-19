@echo off
echo =======================================================================
echo Starting Machining Tool Wear & Life Forecasting Dashboard...
echo Calibrated with peer-reviewed empirical studies from Springer IJAMT
echo =======================================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
