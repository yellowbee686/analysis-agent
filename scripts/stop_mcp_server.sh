#!/bin/bash
# Stop CbetaMCP server running in background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/log}"
PID_FILE="$LOG_DIR/mcp_server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "🛑 Stopping MCP server (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "✅ MCP server stopped"
    else
        echo "⚠️ MCP server process not found (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    LEGACY_PID_FILE="$PROJECT_ROOT/cache/mcp_server.pid"
    if [ -f "$LEGACY_PID_FILE" ]; then
        PID=$(cat "$LEGACY_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "🛑 Stopping MCP server (PID: $PID)..."
            kill "$PID"
            rm -f "$LEGACY_PID_FILE"
            echo "✅ MCP server stopped"
        else
            echo "⚠️ MCP server process not found (stale PID file)"
            rm -f "$LEGACY_PID_FILE"
        fi
    else
        echo "⚠️ No PID file found. MCP server may not be running."
    fi
fi
