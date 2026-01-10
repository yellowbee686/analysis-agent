#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STARTED_MCP_SERVER=false

cleanup() {
    if [ "$STARTED_MCP_SERVER" = true ]; then
        "$SCRIPT_DIR/stop_mcp_server.sh"
    fi
}

trap cleanup EXIT INT TERM

mkdir -p "$PROJECT_ROOT/cache"
echo "🚀 Starting MCP server..."
"$SCRIPT_DIR/start_mcp_server.sh" --bg
STARTED_MCP_SERVER=true

LOCAL_AGENT_MCP_ENABLED=true \
LOCAL_AGENT_MCP_CONFIG=config/mcp_config.json \
uv run streamlit run app.py
