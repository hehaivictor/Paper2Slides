#!/bin/bash

# Stop all Paper2Slides services

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Stopping All Paper2Slides Services"
echo "=========================================="
echo ""

# Kill all backend processes (server.py)
echo "Stopping backend processes..."
BACKEND_PIDS=$(pgrep -f "server.py" | xargs)
if [ ! -z "$BACKEND_PIDS" ]; then
    for pid in $BACKEND_PIDS; do
        echo "  Killing PID $pid..."
        kill -9 $pid 2>/dev/null
    done
    echo "  ✓ Backend stopped"
else
    echo "  ℹ No backend processes found"
fi

# Kill all frontend processes (vite)
echo ""
echo "Stopping frontend processes..."
VITE_PIDS=$(pgrep -f "vite" | xargs)
if [ ! -z "$VITE_PIDS" ]; then
    for pid in $VITE_PIDS; do
        echo "  Killing PID $pid..."
        kill -9 $pid 2>/dev/null
    done
    echo "  ✓ Frontend stopped"
else
    echo "  ℹ No frontend processes found"
fi

# Kill processes by port
echo ""
echo "Stopping processes on ports..."
for port in 8001 8002 8003 5173 3000; do
    PIDS=$(lsof -ti :$port 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
        echo "  Port $port: killing PIDs $PIDS"
        kill -9 $PIDS 2>/dev/null
    fi
done

echo ""
echo "=========================================="
echo "✓ All services stopped"
echo "=========================================="
echo ""
echo "To start fresh, run:"
echo "  ./scripts/start.sh"
echo ""

