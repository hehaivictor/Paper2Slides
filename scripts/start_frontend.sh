#!/bin/bash

# Start Paper2Slides Frontend Development Server

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/frontend"

echo "=========================================="
echo "Starting Paper2Slides Frontend"
echo "=========================================="
echo ""
echo "Frontend will run on: http://localhost:5173 (Vite default)"
echo "Or check the output below for the actual port"
echo ""
echo "Press Ctrl+C to stop"
echo ""

npm run dev

