#!/bin/bash

# Setup script for VGG-WongWang LIM
# Install dependencies and verify environment

set -e

echo "=============================================="
echo "VGG-WongWang LIM Setup"
echo "=============================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VERSION}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run training:"
echo "  ./run_test.sh          # For quick test"
echo "  ./run_pipeline_background.sh  # For full training"
echo ""
