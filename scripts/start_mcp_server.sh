#!/bin/bash
# Start CbetaMCP server
#
# Usage:
#   ./scripts/start_mcp_server.sh          # Run in foreground
#   ./scripts/start_mcp_server.sh --bg     # Run in background
#
# Environment variables:
#   MCP_PORT: Server port (default: 8001)
#   MCP_HOST: Server host (default: 0.0.0.0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MCP_DIR="$PROJECT_ROOT/mcp_servers/CbetaMCP"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/cache}"
PID_FILE="$LOG_DIR/mcp_server.pid"

# Default port (use 8001 to avoid conflict with Streamlit on 8501)
export APP_PORT="${MCP_PORT:-8001}"
export APP_BASE_URL="${APP_BASE_URL:-http://localhost:$APP_PORT}"

cd "$MCP_DIR"

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment for CbetaMCP..."
    python -m venv .venv
fi

echo "📦 Installing CbetaMCP dependencies..."
source .venv/bin/activate
pip install -q -r requirements.txt

echo "🚀 Starting CbetaMCP server on http://localhost:$APP_PORT/mcp"

if [ "$1" == "--bg" ]; then
    mkdir -p "$LOG_DIR"
    nohup python main.py > "$LOG_DIR/mcp_server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo "✅ MCP server started in background (PID: $(cat "$PID_FILE"))"
    echo "📝 Logs: $LOG_DIR/mcp_server.log"
else
    python main.py
fi
