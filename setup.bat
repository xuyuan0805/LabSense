@echo off
setlocal
cd /d "%~dp0"
echo [LabSense] Preparing the local Python environment...
where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.9 or newer first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.
echo Setup complete. You can now run start.bat offline.
pause
exit /b 0
:fail
echo.
echo Setup failed. Check your network connection and Python installation.
pause
exit /b 1
