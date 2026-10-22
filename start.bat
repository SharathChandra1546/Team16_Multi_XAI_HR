@echo off
rem ---------------------------------------------------------------------------
rem  Start the app on Windows and open it in the browser.
rem    start.bat          http://localhost:8000
rem    start.bat 8080     use another port
rem  Runs setup.bat first if the app has not been set up yet.
rem  Stop the app with Ctrl+C in this window (or close the window).
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
chcp 65001 >nul
title Multi-XAI app
cd /d "%~dp0backend" || goto :fail

if not exist ".venv\Scripts\python.exe" (
  echo The app is not set up yet; running setup.bat first...
  call "%~dp0setup.bat" || goto :fail
  cd /d "%~dp0backend"
)
set "VPY=%CD%\.venv\Scripts\python.exe"

if not exist "models\hr_main.joblib" (
  echo Models are missing; training them first...
  "%VPY%" scripts\ensure_model.py || goto :fail
)

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  echo.
  echo  Port %PORT% is already in use. The app may already be running in another window:
  echo  open http://localhost:%PORT% or close that window. To use another port: start.bat 8080
  goto :fail
)

echo.
"%VPY%" scripts\configure_env.py --show
echo.
echo ===========================================================================
echo  App: http://localhost:%PORT%    (opens in your browser in a few seconds)
echo  Stop: press Ctrl+C in this window
echo  Cloud model chosen? Tick the consent box on the Explain page, or you get template text
echo  Change model or keys: close this window, run configure_llm.bat, start again
echo ===========================================================================
echo.
set "STATIC_DIR=%~dp0frontend\dist-local"
start "" /min cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:%PORT%/"
"%VPY%" -m uvicorn app.api:app --host 127.0.0.1 --port %PORT%
echo.
echo The app has stopped.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
