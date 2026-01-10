from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    index_dir: Path
    chunk_max_chars: int = 1200
    snippet_chars: int = 400
    default_top_k: int = 5
    model_platform: str | None = None
    model_type: str | None = None
    system_prompt: str | None = None
    max_tokens: int = 65535
    stream: bool = True
    # Comma-separated list of model name patterns that should NOT use streaming
    # e.g., "gemini,claude" will disable streaming for any model containing these strings
    non_stream_patterns: str = ""
    # MCP configuration
    mcp_config_path: Path | None = None
    mcp_enabled: bool = False


def load_config() -> AppConfig:
    data_dir = Path(os.environ.get("LOCAL_AGENT_DATA_DIR", "data"))
    index_dir = Path(
        os.environ.get("LOCAL_AGENT_INDEX_DIR", "cache/indices")
    )
    chunk_max_chars = int(
        os.environ.get("LOCAL_AGENT_CHUNK_MAX_CHARS", "1200")
    )
    snippet_chars = int(os.environ.get("LOCAL_AGENT_SNIPPET_CHARS", "400"))
    default_top_k = int(os.environ.get("LOCAL_AGENT_TOP_K", "5"))
    model_platform = os.environ.get("LOCAL_AGENT_MODEL_PLATFORM")
    model_type = os.environ.get("LOCAL_AGENT_MODEL_TYPE")
    system_prompt = os.environ.get("LOCAL_AGENT_SYSTEM_PROMPT")
    max_tokens = int(os.environ.get("LOCAL_AGENT_MAX_TOKENS", "65535"))
    stream = os.environ.get("LOCAL_AGENT_STREAM", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    # Default: gemini models use non-streaming due to compatibility issues
    non_stream_patterns = os.environ.get(
        "LOCAL_AGENT_NON_STREAM_PATTERNS", "gemini"
    )
    # MCP configuration
    mcp_config_path_str = os.environ.get("LOCAL_AGENT_MCP_CONFIG")
    mcp_config_path = Path(mcp_config_path_str) if mcp_config_path_str else None
    mcp_enabled = os.environ.get("LOCAL_AGENT_MCP_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    return AppConfig(
        data_dir=data_dir,
        index_dir=index_dir,
        chunk_max_chars=chunk_max_chars,
        snippet_chars=snippet_chars,
        default_top_k=default_top_k,
        model_platform=model_platform,
        model_type=model_type,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        stream=stream,
        non_stream_patterns=non_stream_patterns,
        mcp_config_path=mcp_config_path,
        mcp_enabled=mcp_enabled,
    )
