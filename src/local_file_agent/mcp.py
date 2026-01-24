"""MCP (Model Context Protocol) toolkit management.

This module provides utilities for connecting to MCP servers and
retrieving tools for use with CAMEL agents.

Key features:
- MCP server connection management
- Tool wrapping for large response handling (save to file)
- Session-based file storage integration
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from local_file_agent.session_storage import (
    DEFAULT_CONTENT_THRESHOLD,
    save_large_response,
)

if TYPE_CHECKING:
    from camel.toolkits import FunctionTool
    from camel.toolkits.mcp_toolkit import MCPToolkit

logger = logging.getLogger(__name__)

# Global MCPToolkit instance for connection management
_mcp_toolkit: MCPToolkit | None = None

# Current session ID for file storage (set by app.py)
_current_session_id: str | None = None

# Content threshold for saving to file (can be configured)
_content_threshold: int = DEFAULT_CONTENT_THRESHOLD


def set_session_id(session_id: str | None) -> None:
    """Set the current session ID for MCP response storage.

    Args:
        session_id: The session identifier.
    """
    global _current_session_id
    _current_session_id = session_id
    logger.debug("MCP session ID set to: %s", session_id)


def get_session_id() -> str | None:
    """Get the current session ID."""
    return _current_session_id


def set_content_threshold(threshold: int) -> None:
    """Set the content size threshold for saving to file.

    Args:
        threshold: Size in characters above which to save to file.
    """
    global _content_threshold
    _content_threshold = threshold
    logger.debug("MCP content threshold set to: %d", threshold)


def _wrap_mcp_tool_func(
    original_func: Callable,
    tool_name: str,
) -> Callable:
    """Wrap an MCP tool function to handle large responses.

    This wrapper intercepts tool responses and saves them to file if they
    exceed the content threshold.

    Args:
        original_func: The original tool function.
        tool_name: The name of the tool (for logging and filenames).

    Returns:
        Wrapped function that handles large responses.
    """
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs) -> Any:
        # Call the original function
        result = original_func(*args, **kwargs)

        # Try to save large response to file
        try:
            metadata = save_large_response(
                session_id=_current_session_id,
                tool_name=tool_name,
                args=kwargs if kwargs else {"args": args},
                response=result,
                threshold=_content_threshold,
            )

            if metadata is not None:
                # Return metadata instead of full response
                logger.info(
                    "MCP tool '%s' response saved to file: %s",
                    tool_name, metadata.get("file_path")
                )
                return metadata

        except Exception as e:
            logger.warning(
                "Failed to process large response for tool '%s': %s",
                tool_name, e
            )

        # Return original result if not saved
        return result

    @functools.wraps(original_func)
    async def async_wrapper(*args, **kwargs) -> Any:
        # Call the original async function
        result = await original_func(*args, **kwargs)

        # Try to save large response to file
        try:
            metadata = save_large_response(
                session_id=_current_session_id,
                tool_name=tool_name,
                args=kwargs if kwargs else {"args": args},
                response=result,
                threshold=_content_threshold,
            )

            if metadata is not None:
                logger.info(
                    "MCP tool '%s' response saved to file: %s",
                    tool_name, metadata.get("file_path")
                )
                return metadata

        except Exception as e:
            logger.warning(
                "Failed to process large response for tool '%s': %s",
                tool_name, e
            )

        return result

    # Check if original is async
    import asyncio
    if asyncio.iscoroutinefunction(original_func):
        return async_wrapper
    return wrapper


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
        # Use a 60s timeout for MCP tool execution
        # The default 10s is too short for slow APIs like CBETA
        _mcp_toolkit = MCPToolkit(config_path=str(config_path), timeout=60.0)
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


def get_mcp_tools(wrap_for_large_response: bool = True) -> list[FunctionTool]:
    """Get tools from the connected MCP toolkit.

    Args:
        wrap_for_large_response: Whether to wrap tools to handle large
            responses by saving them to files. (default: True)

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

        if wrap_for_large_response:
            wrapped_tools = []
            for tool in tools:
                try:
                    # Get tool name from the function or schema
                    tool_name = getattr(tool.func, "__name__", "unknown_tool")
                    if hasattr(tool, "get_openai_tool_schema"):
                        schema = tool.get_openai_tool_schema()
                        if isinstance(schema, dict):
                            func_schema = schema.get("function", {})
                            tool_name = func_schema.get("name", tool_name)

                    # Wrap the function
                    wrapped_func = _wrap_mcp_tool_func(tool.func, tool_name)

                    # Create new FunctionTool with wrapped function
                    from camel.toolkits import FunctionTool
                    wrapped_tool = FunctionTool(
                        wrapped_func,
                        openai_tool_schema=tool.get_openai_tool_schema(),
                    )
                    wrapped_tools.append(wrapped_tool)
                    logger.debug("Wrapped MCP tool: %s", tool_name)

                except Exception as e:
                    logger.warning(
                        "Failed to wrap tool, using original: %s", e
                    )
                    wrapped_tools.append(tool)

            logger.info(
                "Wrapped %d MCP tools for large response handling",
                len(wrapped_tools)
            )
            return wrapped_tools

        return tools
    except Exception as e:
        logger.error("Failed to get MCP tools: %s", e)
        return []


def is_mcp_connected() -> bool:
    """Check if MCP toolkit is connected."""
    return _mcp_toolkit is not None


def get_mcp_toolkit() -> MCPToolkit | None:
    """Return the connected MCP toolkit instance, if any."""
    return _mcp_toolkit


def get_session_storage_path() -> Path | None:
    """Get the storage path for the current session.

    Returns:
        Path to session storage directory, or None if no session is set.
    """
    if _current_session_id is None:
        return None

    from local_file_agent.session_storage import get_session_storage_dir
    return get_session_storage_dir(_current_session_id)
