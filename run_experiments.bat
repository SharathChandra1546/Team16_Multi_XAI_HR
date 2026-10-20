@echo off
cd /d "%~dp0backend"
call ".venv\Scripts\activate.bat"
python eval\run_all.py %*
