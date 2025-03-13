@echo off
setlocal
set PYTHONHOME=%~dp0python-3.11.1.amd64
set PATH=%PYTHONHOME%;%PYTHONHOME%\Scripts;%PATH%
python "%~dp0MagnaVNA.py"
pause
