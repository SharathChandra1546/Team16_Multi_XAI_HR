@echo off
rem ---------------------------------------------------------------------------
rem  One-time setup on Windows (run again after pulling new code or adding data)
rem    - finds Python 3.10+
rem    - creates backend\.venv and installs the requirements (only when they change)
rem    - creates backend\.env
rem    - trains the HR, finance and health models (real data if present in backend\data)
rem    - runs a quick check (one explanation per domain, offline)
rem    - optionally sets up a language model and API key
rem
rem  setup.bat          normal setup
rem  setup.bat --test   also run the full test suite (slow: about 40 minutes)
rem ---------------------------------------------------------------------------
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title Multi-XAI setup
cd /d "%~dp0backend" || goto :fail

echo.
echo [1/6] Looking for Python 3.10 or newer...
set "PY="
for %%V in (3.12 3.11 3.13 3.10 3.14) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo  Python 3.10 or newer was not found.
  echo  Install Python 3.12 from https://www.python.org/downloads/
  echo  and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
  goto :fail
)
for /f "delims=" %%A in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%A"
echo        Using %PY% (Python %PYVER%)

echo.
echo [2/6] Virtual environment...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul || (
    echo        The existing .venv is broken; recreating it.
    rmdir /s /q ".venv"
  )
)
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv || goto :fail
  echo        Created backend\.venv
) else (
  echo        backend\.venv already exists
)
set "VPY=%CD%\.venv\Scripts\python.exe"

echo.
echo [3/6] Python packages...
set "NEED_INSTALL=1"
if exist ".venv\requirements.installed" (
  fc /b "requirements.txt" ".venv\requirements.installed" >nul 2>nul && set "NEED_INSTALL=0"
)
if "!NEED_INSTALL!"=="1" (
  echo        Installing - the first time takes 3 to 10 minutes...
  "%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q || goto :fail
  "%VPY%" -m pip install -r requirements.txt --disable-pip-version-check || (
    echo.
    echo  Package installation failed. Check your internet connection and run setup.bat again.
    goto :fail
  )
  copy /y "requirements.txt" ".venv\requirements.installed" >nul
) else (
  echo        Already up to date
)

echo.
echo [4/6] Settings file...
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo        Created backend\.env - offline mock model by default
) else (
  echo        backend\.env already exists
)

echo.
echo [5/6] Models (trains only what is missing or out of date)...
"%VPY%" scripts\ensure_model.py || goto :fail

echo.
echo [6/6] Quick check (one explanation per domain, offline)...
"%VPY%" scripts\smoke_check.py || (
  echo.
  echo  The quick check failed. See the messages above.
  goto :fail
)

if /i "%~1"=="--test" (
  echo.
  echo Running the full test suite - about 40 minutes...
  "%VPY%" -m pytest -q
)

echo.
"%VPY%" scripts\configure_env.py --show
echo.
choice /c YN /n /m "Set up a language model / API key now? [Y/N] "
if errorlevel 2 goto :done
"%VPY%" scripts\configure_env.py

:done
echo.
echo ===========================================================================
echo  Setup finished. Start the app with start.bat
echo  Change the language model or keys any time with configure_llm.bat
echo ===========================================================================
echo.
pause
exit /b 0

:fail
echo.
echo  Setup stopped because of the error above.
pause
exit /b 1
