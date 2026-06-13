@echo off
setlocal
cd /d "%~dp0"
python launch_gui.py %*
if errorlevel 1 (
  echo.
  echo Failed to launch the GUI. Activate the conda environment first, then run:
  echo   conda activate preprocess
  echo   python launch_gui.py
  echo.
  pause
)
