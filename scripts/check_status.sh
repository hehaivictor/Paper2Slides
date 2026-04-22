#!/bin/bash

echo "=== Paper2Slides Service Status ==="
echo ""

# Check backend
echo "Backend (port 8001):"
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "  ✓ Running"
    curl -s http://localhost:8001/health | python3 -m json.tool 2>/dev/null || echo "  Response: $(curl -s http://localhost:8001/health)"
else
    echo "  ✗ Not running or not responding"
fi
echo ""

# Check frontend
echo "Frontend (port 5173):"
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "  ✓ Running"
    echo "  URL: http://localhost:5173"
else
    echo "  ✗ Not running or not responding"
fi
echo ""

# Check processes
echo "Running processes:"
ps aux | grep -E "(vite|server.py)" | grep -v grep | awk '{print "  PID:", $2, "|", $11, $12, $13, $14, $15}'
echo ""

# Check ports
echo "Port status:"
netstat -tuln 2>/dev/null | grep -E ":(5173|8001)" || ss -tuln 2>/dev/null | grep -E ":(5173|8001)"
echo ""

echo "=== End of Status Check ==="
