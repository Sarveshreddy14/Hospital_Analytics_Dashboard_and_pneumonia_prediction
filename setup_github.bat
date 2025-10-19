@echo off
echo 🏥 Hospital Analytics - GitHub Setup
echo ====================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Run the setup script
echo 🚀 Starting GitHub setup...
python setup_github.py

echo.
echo Press any key to exit...
pause >nul
