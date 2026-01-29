#!/bin/bash
# Summit Housing Dashboard - Quick Start Script
# This script sets up and launches the dashboard with one command

set -e  # Exit on error

echo "🏔️  Summit Housing Dashboard - Quick Start"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "❌ Python $REQUIRED_VERSION or higher is required. You have $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"
echo ""

# Step 1: Setup
if [ ! -d ".venv" ]; then
    echo "📦 Setting up virtual environment and dependencies..."
    make setup
else
    echo "✅ Virtual environment already exists"
fi

echo ""

# Step 2: Check if database exists
if [ ! -f "summit_housing.db" ]; then
    echo "🗄️  Database not found. Running ETL pipeline..."
    echo "   (This may take a few minutes on first run)"
    make ingest
else
    echo "✅ Database already exists"
fi

echo ""
echo "Select Launch Mode:"
echo "  1) Streamlit App (Python Dynamic - For DS Exploration)"
echo "  2) Static Dashboard (HTML/JS - The Final Product)"
echo ""
read -p "Enter choice [1/2]: " choice

if [ "$choice" = "2" ]; then
    echo ""
    echo "🚀 Launching Static Dashboard..."
    echo "   Ensure you have run 'make export' recently to see latest data."
    echo ""
    make serve-static
else
    echo ""
    echo "🚀 Launching Streamlit App..."
    echo "   Dashboard will open at: http://localhost:8501"
    echo ""
    make run
fi
