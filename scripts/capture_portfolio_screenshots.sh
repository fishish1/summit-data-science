#!/bin/bash
#
# Portfolio Screenshot Generator
# Starts the local server and captures screenshots of the Summit Housing Dashboard
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🏔️  Summit Housing Dashboard${NC}"
echo -e "${BLUE}📸 Portfolio Screenshot Generator${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$PROJECT_DIR/static_dashboard"

# Check if static_dashboard directory exists
if [ ! -d "$DASHBOARD_DIR" ]; then
    echo -e "${RED}❌ Error: static_dashboard directory not found${NC}"
    exit 1
fi

# Check if Python virtual environment exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv "$PROJECT_DIR/.venv"
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source "$PROJECT_DIR/.venv/bin/activate"

# Check if playwright is installed
if ! python -c "import playwright" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing Playwright...${NC}"
    pip install playwright
    playwright install chromium
fi

# Start the local server in the background
echo -e "${BLUE}🚀 Starting local server on http://localhost:8000...${NC}"
cd "$DASHBOARD_DIR"
python3 -m http.server 8000 > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo -e "${BLUE}⏳ Waiting for server to be ready...${NC}"
sleep 3

# Check if server is running
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${RED}❌ Error: Server failed to start${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}✅ Server is running (PID: $SERVER_PID)${NC}"
echo ""

# Run the screenshot script
echo -e "${BLUE}📸 Taking screenshots...${NC}"
cd "$PROJECT_DIR"
python3 scripts/take_screenshots.py

# Capture the exit code
SCREENSHOT_EXIT_CODE=$?

# Stop the server
echo ""
echo -e "${BLUE}🛑 Stopping local server...${NC}"
kill $SERVER_PID 2>/dev/null || true

# Wait a moment for the server to stop
sleep 1

# Final status
echo ""
if [ $SCREENSHOT_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✨ Screenshot capture complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}📁 Screenshots saved to:${NC} $PROJECT_DIR/screenshots/"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review the screenshots in the screenshots/ directory"
    echo "  2. Use them in your portfolio"
    echo "  3. Consider adding them to your git repository"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Screenshot capture failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
