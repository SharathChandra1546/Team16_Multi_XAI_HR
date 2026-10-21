@echo off
rem Kept for older instructions: set up (if needed) and start the app.
rem The same steps are now split into setup.bat, start.bat and configure_llm.bat.
call "%~dp0setup.bat" || exit /b 1
call "%~dp0start.bat" %*
