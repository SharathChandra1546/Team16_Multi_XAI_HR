@echo off
rem Choose the language model the app uses and store its API key in backend\.env (git ignores that file).
setlocal EnableExtensions
chcp 65001 >nul
title Multi-XAI language model
cd /d "%~dp0backend" || goto :fail
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  goto :fail
)
".venv\Scripts\python.exe" scripts\configure_env.py
echo.
echo Restart the app (close its window, run start.bat) for the change to take effect.
pause
exit /b 0

:fail
pause
exit /b 1
