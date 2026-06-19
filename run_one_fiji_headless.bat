@echo off
setlocal

set "PROJECT=C:\PERSONAL\ImageJ\IronCells_MVP"
set "PYTHON=python"

cd /d "%PROJECT%"
"%PYTHON%" "%PROJECT%\run_one_fiji_headless.py" %*
exit /b %ERRORLEVEL%
