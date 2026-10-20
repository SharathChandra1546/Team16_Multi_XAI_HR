@echo off
setlocal
rem One-step setup and run on Windows. Needs Python 3.10 or newer on PATH.
cd /d "%~dp0backend"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
if not exist ".env" copy ".env.example" ".env" >nul
python scripts\ensure_model.py
if errorlevel 1 exit /b 1
python -m pytest -q
set "STATIC_DIR=%~dp0frontend\dist-local"
echo.
echo Opening http://localhost:8000  - press Ctrl+C in this window to stop.
start "" "http://localhost:8000"
python -m uvicorn app.api:app --port 8000
