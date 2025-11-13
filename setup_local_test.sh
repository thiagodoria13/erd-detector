#!/bin/bash
# Setup script for local ERD detector testing on Linux/Mac/Git Bash
# Author: Lucas Pereira da Fonseca, Thiago Anversa Sampaio Doria

echo "========================================"
echo "ERD Detector - Local Test Setup"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] Python not found. Please install Python 3.10 or higher."
        exit 1
    else
        PYTHON_CMD=python3
    fi
else
    PYTHON_CMD=python
fi

echo "[1/3] Python found:"
$PYTHON_CMD --version
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "[2/3] Virtual environment already exists at .venv"
else
    echo "[2/3] Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully"
fi
echo ""

echo "[3/3] Installing dependencies..."
echo "This may take a few minutes..."
echo ""

# Activate virtual environment and install dependencies
source .venv/bin/activate || source .venv/Scripts/activate
$PYTHON_CMD -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to install dependencies"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate    # Linux/Mac"
echo "  source .venv/Scripts/activate  # Git Bash on Windows"
echo ""
echo "To run the local test:"
echo "  python test_local.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
