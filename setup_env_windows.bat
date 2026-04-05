@echo off
setlocal

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3 "%~dp0scripts\setup_env.py" --platform windows %*
    exit /b %ERRORLEVEL%
)

python "%~dp0scripts\setup_env.py" --platform windows %*
exit /b %ERRORLEVEL%
