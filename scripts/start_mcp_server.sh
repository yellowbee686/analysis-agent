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
    nohup python main.py > "$PROJECT_ROOT/cache/mcp_server.log" 2>&1 &
    echo $! > "$PROJECT_ROOT/cache/mcp_server.pid"
    echo "✅ MCP server started in background (PID: $(cat "$PROJECT_ROOT/cache/mcp_server.pid"))"
    echo "📝 Logs: $PROJECT_ROOT/cache/mcp_server.log"
else
    python main.py
fi
