@echo off
REM Setup script for local ERD detector testing on Windows
REM Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria

echo ========================================
echo ERD Detector - Local Test Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

echo [1/3] Python found:
python --version
echo.

REM Check if virtual environment exists
if exist .venv (
    echo [2/3] Virtual environment already exists at .venv
) else (
    echo [2/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

echo [3/3] Installing dependencies...
echo This may take a few minutes...
echo.

REM Activate virtual environment and install dependencies
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment:
echo     .venv\Scripts\activate
echo.
echo To run the local test:
echo     python test_local.py
echo.
echo To deactivate:
echo     deactivate
echo.
pause
