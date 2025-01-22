#!/bin/bash

# FTIR Analysis Project Setup Script
# Author: Dr. Priya's Research Team
# Version: 1.0.0
# Date: January 2025

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print error messages
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Function to print info messages
info() {
    echo -e "${GREEN}Info: $1${NC}"
}

# Function to print warning messages
warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Detect OS type
OS="unknown"
case "$(uname -s)" in
    Darwin*)    OS="macos";;
    Linux*)     OS="linux";;
    *)          warning "Unknown OS type. Some features might not work correctly.";;
esac

info "Detected operating system: $OS"

# Check for Python3
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    # Check if it's Python 3.x
    if python --version 2>&1 | grep -q "Python 3"; then
        PYTHON_CMD="python"
    else
        error "Python 3 is required but not found"
    fi
else
    error "Python 3 is required but not found"
fi

# Check Python version (requires Python 3.8+)
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    error "Python version must be >= 3.8 (found $PYTHON_VERSION)"
fi

info "Python version check passed ($PYTHON_VERSION)"

# Check for pip
if ! command -v pip3 >/dev/null 2>&1 && ! command -v pip >/dev/null 2>&1; then
    error "pip is not installed. Please install pip for Python 3"
fi

# Check for required system commands
for cmd in git; do
    if ! command -v $cmd >/dev/null 2>&1; then
        error "$cmd is required but not installed"
    fi
done

# Create and activate virtual environment
if [ -d "venv" ]; then
    warning "Virtual environment already exists. Recreating..."
    rm -rf venv
fi

info "Creating virtual environment..."
$PYTHON_CMD -m venv venv || error "Failed to create virtual environment"

# Activate virtual environment (works on both Linux and macOS)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate || error "Failed to activate virtual environment"
else
    error "Virtual environment activation script not found"
fi

info "Virtual environment activated"

# Upgrade pip
info "Upgrading pip..."
python -m pip install --upgrade pip || error "Failed to upgrade pip"

# Install dependencies
info "Installing dependencies..."
pip install -r requirements.txt || error "Failed to install dependencies"

# Create necessary directories
info "Creating project directories..."
for dir in data output docs tests; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" || error "Failed to create $dir directory"
        info "Created $dir directory"
    fi
done

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    info "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit" || warning "Failed to create initial commit"
fi

# Create example data
if [ ! -f "data/example_data.xlsx" ]; then
    info "Generating example data..."
    python scripts/generate_example_data.py || warning "Failed to generate example data"
fi

# Run tests
info "Running tests..."
python -m pytest tests/ || warning "Some tests failed"

# Setup complete
info "Setup complete! You can now run the analysis with:"
echo -e "${GREEN}python src/process_ftir.py${NC}"

# Cleanup
deactivate
info "Virtual environment deactivated. Remember to activate it before running the analysis:"
echo -e "${GREEN}source venv/bin/activate${NC}"

# OS-specific instructions
case "$OS" in
    "linux")
        info "Linux-specific notes:"
        echo "- If you encounter permission issues, you might need to use 'sudo' for system-wide installations"
        echo "- Make sure you have python3-venv installed (sudo apt-get install python3-venv on Debian/Ubuntu)"
        ;;
    "macos")
        info "macOS-specific notes:"
        echo "- If you encounter SSL certificate issues, you might need to install certificates:"
        echo "  /Applications/Python*/Install\ Certificates.command"
        ;;
esac
