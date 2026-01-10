"""MCP (Model Context Protocol) toolkit management.

This module provides utilities for connecting to MCP servers and
retrieving tools for use with CAMEL agents.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from camel.toolkits import FunctionTool
    from camel.toolkits.mcp_toolkit import MCPToolkit

logger = logging.getLogger(__name__)

# Global MCPToolkit instance for connection management
_mcp_toolkit: MCPToolkit | None = None


async def connect_mcp(config_path: Path | str) -> MCPToolkit:
    """Connect to MCP servers defined in the config file.

    Args:
        config_path: Path to MCP configuration JSON file.

    Returns:
        Connected MCPToolkit instance.
    """
    global _mcp_toolkit

    if _mcp_toolkit is not None:
        logger.info("MCP toolkit already connected, reusing existing instance")
        return _mcp_toolkit

    try:
        from camel.toolkits.mcp_toolkit import MCPToolkit
    except ImportError:
        logger.error(
            "MCPToolkit not available. Please install camel-ai with MCP support."
        )
        return None

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("MCP config file not found: %s", config_path)
        return None

    logger.info("Connecting to MCP servers from config: %s", config_path)
    try:
        _mcp_toolkit = MCPToolkit(config_path=str(config_path))
        await _mcp_toolkit.connect()
        logger.info("MCP toolkit connected successfully")
        return _mcp_toolkit
    except Exception as e:
        logger.error("Failed to connect MCP toolkit: %s", e)
        _mcp_toolkit = None
        return None


async def disconnect_mcp() -> None:
    """Disconnect from all MCP servers."""
    global _mcp_toolkit

    if _mcp_toolkit is not None:
        logger.info("Disconnecting MCP toolkit...")
        try:
            await _mcp_toolkit.disconnect()
            logger.info("MCP toolkit disconnected")
        except Exception as e:
            logger.error("Error disconnecting MCP toolkit: %s", e)
        finally:
            _mcp_toolkit = None


def get_mcp_tools() -> list[FunctionTool]:
    """Get tools from the connected MCP toolkit.

    Returns:
        List of FunctionTool instances from MCP servers,
        or empty list if not connected.
    """
    global _mcp_toolkit

    if _mcp_toolkit is None:
        logger.warning("MCP toolkit not connected, returning empty tool list")
        return []

    try:
        tools = _mcp_toolkit.get_tools()
        logger.info("Retrieved %d tools from MCP servers", len(tools))
        return tools
    except Exception as e:
        logger.error("Failed to get MCP tools: %s", e)
        return []


def is_mcp_connected() -> bool:
    """Check if MCP toolkit is connected."""
    return _mcp_toolkit is not None
