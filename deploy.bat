@echo off
echo ==================================================================
echo DEPLOYING TOOL WEAR DASHBOARD TO GITHUB
echo ==================================================================
set "GIT_EXE=C:\Users\Siddharth Patil\AppData\Local\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe"

"%GIT_EXE%" push origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==================================================================
    echo SUCCESS: All files pushed to GitHub successfully!
    echo Live App: https://tool-wear-dashbaord-9cbu9j5jzzlkkrl6t4mmv3.streamlit.app/
    echo Streamlit Cloud will update automatically in ~60 seconds.
    echo ==================================================================
) else (
    echo.
    echo [ERROR] Push failed. If prompted, please click Authorize in your browser.
)
pause
