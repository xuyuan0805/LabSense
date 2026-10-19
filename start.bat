@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launcher.py
) else (
  echo The local environment is missing. Run setup.bat once first.
  pause
  exit /b 1
)
