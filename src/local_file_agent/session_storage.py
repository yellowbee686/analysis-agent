"""Session-based file storage for large MCP responses.

This module provides utilities for storing large MCP tool responses to files
instead of returning them directly to the context. This helps avoid context
pollution and allows the LLM to use FileToolkit to search the content.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default threshold for saving response to file (in characters)
DEFAULT_CONTENT_THRESHOLD = 8000


def get_session_storage_dir(session_id: str | None = None) -> Path:
    """Get the storage directory for a session.

    Args:
        session_id: The session identifier. If None, uses a default directory.

    Returns:
        Path to the session's storage directory.
    """
    cache_dir = Path(os.environ.get("LOCAL_AGENT_CACHE_DIR", "cache"))
    if session_id:
        storage_dir = cache_dir / "sessions" / session_id / "mcp_responses"
    else:
        storage_dir = cache_dir / "sessions" / "_default" / "mcp_responses"

    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: The original string.

    Returns:
        A sanitized string safe for use as a filename.
    """
    # Replace unsafe characters with underscores
    safe = re.sub(r'[^\w\-.]', '_', name)
    # Truncate to reasonable length
    if len(safe) > 100:
        safe = safe[:100]
    return safe


def generate_response_filename(
    tool_name: str,
    args: dict[str, Any],
    extension: str = "txt",
) -> str:
    """Generate a unique filename for a tool response.

    Args:
        tool_name: The name of the MCP tool.
        args: The arguments passed to the tool.
        extension: File extension (default: txt).

    Returns:
        A unique filename for storing the response.
    """
    # Create a deterministic hash from tool name and args
    args_json = json.dumps(args, sort_keys=True, ensure_ascii=False)
    content_hash = hashlib.md5(args_json.encode()).hexdigest()[:8]

    # Extract key args for readable filename
    key_parts = []
    for key in ["work", "q", "query", "keyword", "title", "juan"]:
        if key in args:
            value = str(args[key])
            key_parts.append(f"{key}={sanitize_filename(value)[:20]}")

    if key_parts:
        readable_part = "_".join(key_parts)
    else:
        readable_part = "response"

    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{sanitize_filename(tool_name)}_{readable_part}_{content_hash}_{timestamp}.{extension}"

    return filename


def estimate_content_size(content: Any) -> int:
    """Estimate the size of content in characters.

    Args:
        content: The content to measure.

    Returns:
        Estimated character count.
    """
    if isinstance(content, str):
        return len(content)
    elif isinstance(content, dict):
        return len(json.dumps(content, ensure_ascii=False))
    elif isinstance(content, (list, tuple)):
        return sum(estimate_content_size(item) for item in content)
    else:
        return len(str(content))


def save_large_response(
    session_id: str | None,
    tool_name: str,
    args: dict[str, Any],
    response: Any,
    threshold: int = DEFAULT_CONTENT_THRESHOLD,
) -> dict[str, Any] | None:
    """Save a large response to file and return metadata.

    If the response exceeds the threshold, saves it to a file in the session's
    storage directory and returns metadata. Otherwise returns None.

    Args:
        session_id: The session identifier.
        tool_name: The name of the MCP tool.
        args: The arguments passed to the tool.
        response: The tool's response.
        threshold: Size threshold for saving to file.

    Returns:
        Metadata dict if saved, None if response is small enough.
    """
    content_size = estimate_content_size(response)

    if content_size <= threshold:
        logger.debug(
            "Response size %d <= threshold %d, not saving to file",
            content_size, threshold
        )
        return None

    logger.info(
        "Response size %d > threshold %d, saving to file",
        content_size, threshold
    )

    storage_dir = get_session_storage_dir(session_id)

    # Determine file format based on response type
    if isinstance(response, dict):
        extension = "json"
        content_str = json.dumps(response, ensure_ascii=False, indent=2)
    elif isinstance(response, str):
        extension = "txt"
        content_str = response
    else:
        extension = "json"
        content_str = json.dumps(response, ensure_ascii=False, indent=2)

    filename = generate_response_filename(tool_name, args, extension)
    file_path = storage_dir / filename

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_str)

        # Return metadata about the saved file
        metadata = {
            "status": "saved_to_file",
            "message": (
                f"Response too large ({content_size} chars), saved to file. "
                f"Use file search tools (search_files, read_file) to access the content."
            ),
            "file_path": str(file_path.resolve()),
            "file_size_chars": content_size,
            "file_size_bytes": file_path.stat().st_size,
            "tool_name": tool_name,
            "tool_args": args,
            "content_preview": content_str[:500] + "..." if len(content_str) > 500 else content_str,
        }

        logger.info("Saved large response to: %s", file_path)
        return metadata

    except Exception as e:
        logger.error("Failed to save response to file: %s", e)
        return None


def cleanup_session_storage(session_id: str, max_age_hours: int = 24) -> int:
    """Clean up old files in a session's storage directory.

    Args:
        session_id: The session identifier.
        max_age_hours: Maximum age of files to keep.

    Returns:
        Number of files deleted.
    """
    storage_dir = get_session_storage_dir(session_id)

    if not storage_dir.exists():
        return 0

    now = datetime.now()
    deleted = 0

    for file_path in storage_dir.iterdir():
        if file_path.is_file():
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_hours = (now - mtime).total_seconds() / 3600

            if age_hours > max_age_hours:
                try:
                    file_path.unlink()
                    deleted += 1
                    logger.debug("Deleted old file: %s", file_path)
                except Exception as e:
                    logger.warning("Failed to delete file %s: %s", file_path, e)

    return deleted


def list_session_files(session_id: str | None) -> list[dict[str, Any]]:
    """List all stored files for a session.

    Args:
        session_id: The session identifier.

    Returns:
        List of file metadata dicts.
    """
    storage_dir = get_session_storage_dir(session_id)

    files = []
    for file_path in sorted(storage_dir.iterdir()):
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "path": str(file_path.resolve()),
                "name": file_path.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    return files
