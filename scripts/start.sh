#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/log"
STARTED_MCP_SERVER=false

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/app_${TIMESTAMP}.log"
exec >"$LOG_FILE" 2>&1

cleanup() {
    if [ "$STARTED_MCP_SERVER" = true ]; then
        "$SCRIPT_DIR/stop_mcp_server.sh"
    fi
}

trap cleanup EXIT INT TERM

mkdir -p "$PROJECT_ROOT/cache"
echo "🚀 Starting MCP server..."
LOG_DIR="$LOG_DIR" "$SCRIPT_DIR/start_mcp_server.sh" --bg
STARTED_MCP_SERVER=true

LOCAL_AGENT_MCP_ENABLED=true \
LOCAL_AGENT_MCP_CONFIG=config/mcp_config.json \
uv run streamlit run app.py
