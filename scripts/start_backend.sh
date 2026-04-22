#!/bin/bash

# Start Paper2Slides Backend API Server
# Default port is 8001

PORT=${1:-8001}

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

cd "$PROJECT_ROOT/api"

echo "=========================================="
echo "Starting Paper2Slides Backend API"
echo "=========================================="
echo ""
echo "Server will run on: http://localhost:${PORT}"
echo "API endpoints: http://localhost:${PORT}/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

"$PYTHON_BIN" server.py ${PORT}
